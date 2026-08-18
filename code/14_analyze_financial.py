from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeRegressor, export_text


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parent
OUTPUT = ROOT / "results" / "work" / "financial_analysis"
PROTOCOL = ROOT / "protocols" / "financial_core_protocol.json"
FINANCIAL = ROOT / "data" / "financial.csv"
SPLIT = ROOT / "data" / "financial_train_test_split.csv"
M0_BUNDLE = ROOT / "models" / "financial" / "financial_m0_reproduced.joblib"
BASE_MATRIX = (
    ROOT / "results" / "financial" / "features" / "term_observed_matrix.csv"
)
BASE_TAXONOMY = (
    ROOT / "results" / "financial" / "features" / "term_observed_taxonomy.json"
)
CANDIDATE_MATRIX = (
    ROOT / "results" / "financial" / "features" / "term_state_observed_matrix.csv"
)
CANDIDATE_TAXONOMY = (
    ROOT / "results" / "financial" / "features" / "term_state_observed_taxonomy.json"
)
SOURCE_EVALUATION_DIR = ROOT / "results" / "financial" / "features"
SOURCE_REPORT = SOURCE_EVALUATION_DIR / "incremental_evaluation_report.json"
SOURCE_SELECTIONS = SOURCE_EVALUATION_DIR / "nested_selection_rankings.csv"
SOURCE_PRODUCT_SUMMARY = SOURCE_EVALUATION_DIR / "repeated_oof_product_summary.csv"
FROZEN_M1_BUNDLE = ROOT / "models" / "financial" / "financial_frozen_m1_k24_decision_tree.joblib"

SEED = 20260813 + 2300
FOLDS = 5
REPEATS = 20
BASE_K = 12
CANDIDATE_K = 24
BOOTSTRAP_DRAWS = 20_000
RULE_BOOTSTRAP_DRAWS = 5_000
RULE_MAX_DEPTH = 4
RULE_MIN_LEAF = 50
CAPACITY_MAX_LEAVES = 128
STATES = [
    "bounded_limitation",
    "confirmed_property",
    "fit_positive",
    "mixed_evidence",
    "unresolved_listing_claim",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def mechanism_columns(canonical_id: str) -> list[str]:
    return [
        f"implicit_{canonical_id}_observed",
        f"implicit_{canonical_id}_score",
    ]


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - y
    return {
        "mae_log10_usd": float(mean_absolute_error(y, prediction)),
        "rmse_log10_usd": float(np.sqrt(mean_squared_error(y, prediction))),
        "median_absolute_error_log10_usd": float(np.median(np.abs(error))),
        "r2": float(r2_score(y, prediction)),
    }


def paired_product_bootstrap(
    baseline_loss: np.ndarray,
    model_loss: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    improvement = np.asarray(baseline_loss) - np.asarray(model_loss)
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for start in range(0, BOOTSTRAP_DRAWS, 500):
        width = min(500, BOOTSTRAP_DRAWS - start)
        indices = rng.integers(0, len(improvement), size=(width, len(improvement)))
        draws[start : start + width] = improvement[indices].mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "estimand": (
            "product-level mean reduction in absolute log10-price error, "
            "averaged across repeated outer folds"
        ),
        "point_estimate": float(improvement.mean()),
        "relative_mae_reduction": float(improvement.mean() / baseline_loss.mean()),
        "percentile_95_ci": [float(lower), float(upper)],
        "two_sided_bootstrap_p_value": min(
            1.0,
            2.0
            * min(float((draws <= 0).mean()), float((draws >= 0).mean())),
        ),
        "model_lower_error_product_fraction": float((improvement > 0).mean()),
        "draws": BOOTSTRAP_DRAWS,
    }


def bh_adjust(values: list[float]) -> list[float]:
    raw = np.asarray(values, dtype=float)
    order = np.argsort(raw)
    adjusted = np.empty(len(raw), dtype=float)
    running = 1.0
    for reverse_rank in range(len(raw) - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, raw[index] * len(raw) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def evidence_state(item: dict[str, Any]) -> str:
    key = str(item["aggregation_key"])
    for state in STATES:
        if key.endswith(f"__{state}"):
            return state
    raise ValueError(f"Unrecognized evidence state: {key}")


def required_inputs() -> list[Path]:
    return [
        PROTOCOL,
        FINANCIAL,
        SPLIT,
        M0_BUNDLE,
        BASE_MATRIX,
        BASE_TAXONOMY,
        CANDIDATE_MATRIX,
        CANDIDATE_TAXONOMY,
        SOURCE_REPORT,
        SOURCE_SELECTIONS,
        SOURCE_PRODUCT_SUMMARY,
        FROZEN_M1_BUNDLE,
    ]


def load_analysis_frame() -> tuple[
    pd.DataFrame,
    list[str],
    str,
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    pd.DataFrame,
    dict[str, Any],
]:
    missing = [str(path) for path in required_inputs() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required inputs: {missing}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_core_analysis_execution":
        raise RuntimeError("The core-analysis protocol is not frozen")
    source_report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    if not all(source_report["decision"].values()):
        raise RuntimeError("The inherited nested feature evaluation did not pass")
    if (
        int(source_report["base_term_k"]) != BASE_K
        or int(source_report["primary_candidate_k"]) != CANDIDATE_K
    ):
        raise RuntimeError("The inherited feature counts do not match the protocol")

    bundle = joblib.load(M0_BUNDLE)
    explicit = list(bundle["features"])
    target = str(bundle["target"])
    params = dict(bundle["best_params"])
    expected_params = {"ccp_alpha": 0.0, "max_depth": None, "min_samples_leaf": 10}
    if params != expected_params:
        raise RuntimeError(f"Unexpected frozen M0 parameters: {params}")

    base_taxonomy = json.loads(BASE_TAXONOMY.read_text(encoding="utf-8"))[
        "taxonomy"
    ]
    candidate_taxonomy = json.loads(
        CANDIDATE_TAXONOMY.read_text(encoding="utf-8")
    )["taxonomy"]
    base_ids = [item["canonical_id"] for item in base_taxonomy]
    candidate_ids = [item["canonical_id"] for item in candidate_taxonomy]
    base_columns = [column for item in base_ids for column in mechanism_columns(item)]
    candidate_columns = [
        column for item in candidate_ids for column in mechanism_columns(item)
    ]

    financial = pd.read_csv(
        FINANCIAL, usecols=["Id", target, *explicit], low_memory=False
    )
    split = pd.read_csv(SPLIT, usecols=["Id", "split", "split_stratum"])
    base_matrix = pd.read_csv(BASE_MATRIX, usecols=["product_id", *base_columns])
    candidate_matrix = pd.read_csv(
        CANDIDATE_MATRIX, usecols=["product_id", *candidate_columns]
    )
    train_ids = set(split.loc[split["split"].eq("train"), "Id"].astype(int))
    test_ids = set(split.loc[split["split"].eq("test"), "Id"].astype(int))
    for name, matrix in [("base", base_matrix), ("candidate", candidate_matrix)]:
        matrix_ids = set(matrix["product_id"].astype(int))
        if matrix_ids != train_ids or matrix_ids & test_ids:
            raise RuntimeError(f"{name} matrix is not exactly training-only")

    frame = (
        financial.merge(split, on="Id", validate="one_to_one")
        .loc[lambda value: value["split"].eq("train")]
        .merge(
            base_matrix,
            left_on="Id",
            right_on="product_id",
            validate="one_to_one",
        )
        .drop(columns="product_id")
        .merge(
            candidate_matrix,
            left_on="Id",
            right_on="product_id",
            validate="one_to_one",
        )
        .sort_values("Id")
        .reset_index(drop=True)
    )
    if len(frame) != 1912:
        raise RuntimeError(f"Expected 1,912 training rows, found {len(frame)}")
    if frame[[target, *explicit, *base_columns, *candidate_columns]].isna().any().any():
        raise RuntimeError("The core analysis frame contains missing model values")

    selections = pd.read_csv(SOURCE_SELECTIONS)
    selections[["repeat", "fold", "rank"]] = selections[
        ["repeat", "fold", "rank"]
    ].astype(int)
    return (
        frame,
        explicit,
        target,
        params,
        base_taxonomy,
        candidate_taxonomy,
        selections,
        source_report,
    )


def selected_ids(
    selections: pd.DataFrame,
    repeat: int,
    fold: int,
    stage: str,
    k: int,
) -> list[str]:
    result = (
        selections.loc[
            selections["repeat"].eq(repeat)
            & selections["fold"].eq(fold)
            & selections["stage"].eq(stage)
            & selections["rank"].le(k)
        ]
        .sort_values("rank")["canonical_id"]
        .tolist()
    )
    if len(result) != k or len(set(result)) != k:
        raise RuntimeError(
            f"Expected {k} unique {stage} IDs for repeat={repeat}, fold={fold}"
        )
    return result


def model_factory(
    family: str,
    seed: int,
    tree_params: dict[str, Any],
) -> Any:
    if family == "decision_tree":
        return DecisionTreeRegressor(random_state=seed, **tree_params)
    if family == "capacity_matched_tree":
        return DecisionTreeRegressor(
            random_state=seed,
            max_leaf_nodes=CAPACITY_MAX_LEAVES,
            **tree_params,
        )
    if family == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=10,
            max_features=1.0,
            random_state=seed,
            n_jobs=-1,
        )
    if family == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed,
        )
    raise ValueError(f"Unknown model family: {family}")


def evaluate_models(
    frame: pd.DataFrame,
    explicit: list[str],
    target: str,
    tree_params: dict[str, Any],
    candidate_taxonomy: list[dict[str, Any]],
    selections: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    y = frame[target].to_numpy(float)
    strata = frame["split_stratum"].astype(str).to_numpy()
    families = [
        "decision_tree",
        "capacity_matched_tree",
        "random_forest",
        "hist_gradient_boosting",
    ]
    primary_names = [
        f"{family}__{view}"
        for family in families
        for view in ["m0_explicit", "text_baseline", "m1_full"]
    ]
    ablation_names = [
        "decision_tree__no_source_terms",
        "decision_tree__permuted_mechanisms",
        *[f"decision_tree__drop_{state}" for state in STATES],
    ]
    predictions = {
        name: np.full((REPEATS, len(frame)), np.nan, dtype=float)
        for name in [*primary_names, *ablation_names]
    }
    fold_rows: list[dict[str, Any]] = []
    selected_state_rows: list[dict[str, Any]] = []
    state_lookup = {
        item["canonical_id"]: evidence_state(item) for item in candidate_taxonomy
    }
    splitter = RepeatedStratifiedKFold(
        n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED
    )

    for split_index, (fit_index, validation_index) in enumerate(
        splitter.split(frame, strata)
    ):
        repeat = split_index // FOLDS + 1
        fold = split_index % FOLDS + 1
        fold_seed = SEED + repeat * 1000 + fold * 10
        fit = frame.iloc[fit_index]
        validation = frame.iloc[validation_index]
        selected_base = selected_ids(
            selections, repeat, fold, "base_term", BASE_K
        )
        selected_candidates = selected_ids(
            selections, repeat, fold, "conditional_mechanism", CANDIDATE_K
        )
        base_columns = [
            column for item in selected_base for column in mechanism_columns(item)
        ]
        candidate_columns = [
            column
            for item in selected_candidates
            for column in mechanism_columns(item)
        ]
        views = {
            "m0_explicit": explicit,
            "text_baseline": [*explicit, *base_columns],
            "m1_full": [*explicit, *base_columns, *candidate_columns],
        }
        fitted: dict[tuple[str, str], Any] = {}
        for family in families:
            for view, features in views.items():
                name = f"{family}__{view}"
                model = model_factory(family, fold_seed, tree_params)
                model.fit(fit[features], y[fit_index])
                prediction = model.predict(validation[features])
                predictions[name][repeat - 1, validation_index] = prediction
                fitted[(family, view)] = model
                fold_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "model": name,
                        "feature_count": len(features),
                        **metrics(y[validation_index], prediction),
                    }
                )

        no_source_features = [*explicit, *candidate_columns]
        no_source = model_factory("decision_tree", fold_seed, tree_params)
        no_source.fit(fit[no_source_features], y[fit_index])
        no_source_prediction = no_source.predict(validation[no_source_features])
        predictions["decision_tree__no_source_terms"][
            repeat - 1, validation_index
        ] = no_source_prediction
        fold_rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "model": "decision_tree__no_source_terms",
                "feature_count": len(no_source_features),
                **metrics(y[validation_index], no_source_prediction),
            }
        )

        full_tree = fitted[("decision_tree", "m1_full")]
        permuted_validation = validation[views["m1_full"]].copy()
        rng = np.random.default_rng(fold_seed + 700_000)
        permuted_validation.loc[:, candidate_columns] = (
            permuted_validation[candidate_columns]
            .to_numpy()[rng.permutation(len(validation))]
        )
        permuted_prediction = full_tree.predict(permuted_validation)
        predictions["decision_tree__permuted_mechanisms"][
            repeat - 1, validation_index
        ] = permuted_prediction
        fold_rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "model": "decision_tree__permuted_mechanisms",
                "feature_count": len(views["m1_full"]),
                **metrics(y[validation_index], permuted_prediction),
            }
        )

        for state in STATES:
            state_ids = [
                item for item in selected_candidates if state_lookup[item] == state
            ]
            removed_columns = {
                column for item in state_ids for column in mechanism_columns(item)
            }
            ablated_features = [
                column for column in views["m1_full"] if column not in removed_columns
            ]
            model = model_factory("decision_tree", fold_seed, tree_params)
            model.fit(fit[ablated_features], y[fit_index])
            prediction = model.predict(validation[ablated_features])
            name = f"decision_tree__drop_{state}"
            predictions[name][repeat - 1, validation_index] = prediction
            fold_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "model": name,
                    "feature_count": len(ablated_features),
                    **metrics(y[validation_index], prediction),
                }
            )
            selected_state_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "state": state,
                    "selected_mechanisms": len(state_ids),
                }
            )

        if split_index % 10 == 9:
            print(f"completed {split_index + 1}/{FOLDS * REPEATS} outer folds", flush=True)

    for name, values in predictions.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Incomplete OOF predictions for {name}")
    return predictions, pd.DataFrame(fold_rows), pd.DataFrame(selected_state_rows)


def summarize_predictions(
    predictions: dict[str, np.ndarray],
    y: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeat_rows = []
    product = pd.DataFrame({"row_index": np.arange(len(y)), "actual": y})
    for name, values in predictions.items():
        for repeat in range(REPEATS):
            repeat_rows.append(
                {"repeat": repeat + 1, "model": name, **metrics(y, values[repeat])}
            )
        product[f"{name}__mean_prediction"] = values.mean(axis=0)
        product[f"{name}__mean_absolute_error"] = np.abs(values - y).mean(axis=0)
    return pd.DataFrame(repeat_rows), product


def comparison(
    predictions: dict[str, np.ndarray],
    repeat_metrics: pd.DataFrame,
    y: np.ndarray,
    baseline: str,
    model: str,
    seed: int,
) -> dict[str, Any]:
    baseline_values = predictions[baseline]
    model_values = predictions[model]
    baseline_loss = np.abs(baseline_values - y).mean(axis=0)
    model_loss = np.abs(model_values - y).mean(axis=0)
    paired = paired_product_bootstrap(baseline_loss, model_loss, seed)
    baseline_repeat = repeat_metrics.loc[
        repeat_metrics["model"].eq(baseline)
    ].set_index("repeat")
    model_repeat = repeat_metrics.loc[
        repeat_metrics["model"].eq(model)
    ].set_index("repeat")
    delta = baseline_repeat["mae_log10_usd"] - model_repeat["mae_log10_usd"]
    return {
        "baseline": baseline,
        "model": model,
        "baseline_mean_metrics": {
            metric: float(baseline_repeat[metric].mean())
            for metric in [
                "mae_log10_usd",
                "rmse_log10_usd",
                "median_absolute_error_log10_usd",
                "r2",
            ]
        },
        "model_mean_metrics": {
            metric: float(model_repeat[metric].mean())
            for metric in [
                "mae_log10_usd",
                "rmse_log10_usd",
                "median_absolute_error_log10_usd",
                "r2",
            ]
        },
        "repeats_model_better": int((delta > 0).sum()),
        "repeats_total": REPEATS,
        "repeat_delta_quantiles_2_5_50_97_5": [
            float(value) for value in np.quantile(delta, [0.025, 0.5, 0.975])
        ],
        "paired_product_bootstrap": paired,
    }


def bootstrap_rule_effect(
    residual: np.ndarray,
    mask: np.ndarray,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(RULE_BOOTSTRAP_DRAWS, dtype=float)
    n = len(residual)
    for draw in range(RULE_BOOTSTRAP_DRAWS):
        indices = rng.integers(0, n, n)
        sampled_mask = mask[indices]
        sampled = residual[indices]
        if sampled_mask.all() or (~sampled_mask).all():
            draws[draw] = np.nan
        else:
            draws[draw] = sampled[sampled_mask].mean() - sampled[~sampled_mask].mean()
    valid = draws[np.isfinite(draws)]
    if len(valid) < RULE_BOOTSTRAP_DRAWS * 0.95:
        return [float("nan"), float("nan")]
    return [float(value) for value in np.quantile(valid, [0.025, 0.975])]


def readable_condition(
    feature: str,
    operator: str,
    threshold: float,
    taxonomy_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    canonical_id = feature.removeprefix("implicit_").removesuffix("_observed")
    name = taxonomy_lookup[canonical_id]["canonical_name"]
    binary_state = "observed" if operator == ">" and threshold < 1 else "not observed"
    return {
        "canonical_id": canonical_id,
        "canonical_name": name,
        "state": binary_state,
        "operator": operator,
        "threshold": threshold,
    }


def extract_path_candidates(
    model: DecisionTreeRegressor,
    features: list[str],
    frame: pd.DataFrame,
    residual: np.ndarray,
    taxonomy_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    tree = model.tree_
    paths: list[tuple[int, list[tuple[str, str, float]]]] = []

    def walk(node: int, conditions: list[tuple[str, str, float]]) -> None:
        feature_index = int(tree.feature[node])
        if feature_index < 0:
            paths.append((node, conditions))
            return
        feature = features[feature_index]
        threshold = float(tree.threshold[node])
        walk(int(tree.children_left[node]), [*conditions, (feature, "<=", threshold)])
        walk(int(tree.children_right[node]), [*conditions, (feature, ">", threshold)])

    walk(0, [])
    rules = []
    for sequence, (node, conditions) in enumerate(paths, start=1):
        mask = np.ones(len(frame), dtype=bool)
        for feature, operator, threshold in conditions:
            values = frame[feature].to_numpy(float)
            mask &= values <= threshold if operator == "<=" else values > threshold
        support = int(mask.sum())
        outside = len(mask) - support
        if support < RULE_MIN_LEAF or outside < RULE_MIN_LEAF:
            continue
        effect = float(residual[mask].mean() - residual[~mask].mean())
        direction = "higher" if effect > 0 else "lower"
        confidence = float(
            (residual[mask] > 0).mean() if effect > 0 else (residual[mask] < 0).mean()
        )
        readable = [
            readable_condition(feature, operator, threshold, taxonomy_lookup)
            for feature, operator, threshold in conditions
        ]
        rules.append(
            {
                "rule_id": f"path_candidate_{sequence}",
                "conditions": readable,
                "support": support,
                "coverage": float(support / len(frame)),
                "confidence": confidence,
                "residual_effect_log10_usd": effect,
                "approximate_price_effect_percent": float((10**effect - 1) * 100),
                "bootstrap_95_ci": bootstrap_rule_effect(
                    residual, mask, SEED + 800_000 + sequence
                ),
                "direction": direction,
                "statement": (
                    "IF "
                    + " AND ".join(
                        f"{item['canonical_name']} is {item['state']}" for item in readable
                    )
                    + f" THEN price tends to be {direction} than the explicit-plus-text expectation"
                ),
                "status": "post-selection associational candidate",
            }
        )
    return sorted(
        rules,
        key=lambda item: (
            -abs(item["residual_effect_log10_usd"]),
            -item["support"],
            item["rule_id"],
        ),
    )


def mine_rule_candidates(
    frame: pd.DataFrame,
    target: str,
    candidate_taxonomy: list[dict[str, Any]],
    selections: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], str, dict[str, Any]]:
    summary = pd.read_csv(SOURCE_PRODUCT_SUMMARY)
    summary = frame[["Id"]].merge(summary, on="Id", validate="one_to_one")
    if not np.allclose(summary[target], frame[target]):
        raise RuntimeError("OOF summary target does not match the analysis frame")
    residual = (
        frame[target].to_numpy(float)
        - summary["text_baseline_k12_mean_oof_prediction"].to_numpy(float)
    )
    frozen_bundle = joblib.load(FROZEN_M1_BUNDLE)
    selected_candidates = list(frozen_bundle["mechanism_ids"])
    if len(selected_candidates) != CANDIDATE_K:
        raise RuntimeError("The frozen rule model does not contain 24 mechanisms")
    taxonomy_lookup = {
        item["canonical_id"]: item for item in candidate_taxonomy
    }
    observed_features = [mechanism_columns(item)[0] for item in selected_candidates]
    frequency = (
        selections.loc[
            selections["stage"].eq("conditional_mechanism")
            & selections["rank"].le(CANDIDATE_K)
        ]
        .groupby("canonical_id")
        .size()
        .to_dict()
    )

    univariate_rows = []
    raw_p_values = []
    for index, canonical_id in enumerate(selected_candidates):
        feature = mechanism_columns(canonical_id)[0]
        mask = frame[feature].to_numpy(float) > 0
        support = int(mask.sum())
        if support == 0 or support == len(mask):
            continue
        effect = float(residual[mask].mean() - residual[~mask].mean())
        statistic = ttest_ind(
            residual[mask], residual[~mask], equal_var=False, nan_policy="raise"
        )
        raw_p = float(statistic.pvalue)
        raw_p_values.append(raw_p)
        direction = "higher" if effect > 0 else "lower"
        confidence = float(
            (residual[mask] > 0).mean() if effect > 0 else (residual[mask] < 0).mean()
        )
        item = taxonomy_lookup[canonical_id]
        univariate_rows.append(
            {
                "canonical_id": canonical_id,
                "canonical_name": item["canonical_name"],
                "evidence_state": evidence_state(item),
                "support": support,
                "coverage": float(mask.mean()),
                "confidence": confidence,
                "residual_effect_log10_usd": effect,
                "approximate_price_effect_percent": float((10**effect - 1) * 100),
                "bootstrap_95_ci": json.dumps(
                    bootstrap_rule_effect(residual, mask, SEED + 900_000 + index),
                    ensure_ascii=False,
                ),
                "raw_welch_p_value_descriptive": raw_p,
                "outer_top24_selection_frequency": float(
                    frequency.get(canonical_id, 0) / (FOLDS * REPEATS)
                ),
                "direction": direction,
                "statement": (
                    f"IF {item['canonical_name']} is observed, THEN price tends to be "
                    f"{direction} than the explicit-plus-text expectation"
                ),
                "status": "post-selection associational candidate",
            }
        )
    q_values = bh_adjust(raw_p_values)
    for row, q_value in zip(univariate_rows, q_values):
        row["bh_q_value_descriptive"] = q_value
        row["ranking_score"] = float(
            abs(row["residual_effect_log10_usd"])
            * np.sqrt(row["coverage"])
            * (0.5 + row["outer_top24_selection_frequency"])
        )
    univariate = pd.DataFrame(univariate_rows).sort_values(
        ["ranking_score", "support", "canonical_id"],
        ascending=[False, False, True],
    )

    theme_rows = []
    for index, state in enumerate(STATES):
        ids = [
            item
            for item in selected_candidates
            if evidence_state(taxonomy_lookup[item]) == state
        ]
        if not ids:
            continue
        columns = [mechanism_columns(item)[0] for item in ids]
        mask = frame[columns].gt(0).any(axis=1).to_numpy()
        support = int(mask.sum())
        if support == 0 or support == len(mask):
            continue
        effect = float(residual[mask].mean() - residual[~mask].mean())
        theme_rows.append(
            {
                "evidence_state": state,
                "selected_mechanisms": len(ids),
                "support": support,
                "coverage": float(mask.mean()),
                "residual_effect_log10_usd": effect,
                "approximate_price_effect_percent": float((10**effect - 1) * 100),
                "bootstrap_95_ci": json.dumps(
                    bootstrap_rule_effect(residual, mask, SEED + 950_000 + index),
                    ensure_ascii=False,
                ),
                "status": "exploratory aggregate association",
            }
        )
    themes = pd.DataFrame(theme_rows).sort_values(
        "residual_effect_log10_usd", ascending=False
    )

    rule_tree = DecisionTreeRegressor(
        max_depth=RULE_MAX_DEPTH,
        min_samples_leaf=RULE_MIN_LEAF,
        random_state=SEED + 990_000,
    ).fit(frame[observed_features], residual)
    path_rules = extract_path_candidates(
        rule_tree,
        observed_features,
        frame,
        residual,
        taxonomy_lookup,
    )
    tree_text = export_text(rule_tree, feature_names=observed_features, decimals=5)
    metadata = {
        "selected_candidate_ids": selected_candidates,
        "rule_target": "mean repeated-OOF residual from explicit-plus-text baseline",
        "rule_tree_depth": int(rule_tree.get_depth()),
        "rule_tree_leaves": int(rule_tree.get_n_leaves()),
        "path_candidate_count": len(path_rules),
        "univariate_candidate_count": len(univariate),
        "causal_status": (
            "All rules are post-selection associational candidates. Controlled "
            "intervention is required for causal confirmation."
        ),
    }
    return univariate, themes, path_rules, tree_text, metadata


def build_report(
    predictions: dict[str, np.ndarray],
    repeat_metrics: pd.DataFrame,
    y: np.ndarray,
    source_report: dict[str, Any],
    rule_metadata: dict[str, Any],
) -> dict[str, Any]:
    families = [
        "decision_tree",
        "capacity_matched_tree",
        "random_forest",
        "hist_gradient_boosting",
    ]
    model_comparisons: dict[str, Any] = {}
    for index, family in enumerate(families):
        model_comparisons[family] = {
            "m1_vs_m0": comparison(
                predictions,
                repeat_metrics,
                y,
                f"{family}__m0_explicit",
                f"{family}__m1_full",
                SEED + 10_000 + index,
            ),
            "m1_vs_text_baseline": comparison(
                predictions,
                repeat_metrics,
                y,
                f"{family}__text_baseline",
                f"{family}__m1_full",
                SEED + 20_000 + index,
            ),
        }

    full_name = "decision_tree__m1_full"
    ablation_names = [
        "decision_tree__text_baseline",
        "decision_tree__no_source_terms",
        "decision_tree__permuted_mechanisms",
        *[f"decision_tree__drop_{state}" for state in STATES],
    ]
    ablations = {
        name.removeprefix("decision_tree__"): comparison(
            predictions,
            repeat_metrics,
            y,
            name,
            full_name,
            SEED + 30_000 + index,
        )
        for index, name in enumerate(ablation_names)
    }
    full_mae = {
        family: model_comparisons[family]["m1_vs_m0"]["model_mean_metrics"][
            "mae_log10_usd"
        ]
        for family in families
    }
    benchmark_winner = min(full_mae, key=full_mae.get)
    primary_tree = model_comparisons["decision_tree"]
    primary_supported = bool(
        primary_tree["m1_vs_m0"]["paired_product_bootstrap"]["percentile_95_ci"][0]
        > 0
        and primary_tree["m1_vs_text_baseline"]["paired_product_bootstrap"][
            "percentile_95_ci"
        ][0]
        > 0
    )
    report = {
        "status": "financial_core_analysis_complete",
        "protocol_version": "financial-core-analysis-v1",
        "rows": 1912,
        "test_rows_used": 0,
        "test_targets_used": False,
        "real_price_acceptance_gate_used": False,
        "target": "LogPriceMo (log10 real monthly transaction price in USD)",
        "outer_cv": {"folds": FOLDS, "repeats": REPEATS},
        "feature_configuration": {
            "m0_explicit_features": 35,
            "text_terms_selected_per_outer_fold": BASE_K,
            "evidence_state_mechanisms_selected_per_outer_fold": CANDIDATE_K,
            "selection_is_nested_inside_outer_training_fold": True,
        },
        "inherited_frozen_result_cross_check": {
            "m0_mae": source_report["m0_mean_mae_log10_usd"],
            "text_baseline_mae": source_report["text_baseline_mean_mae_log10_usd"],
            "m1_mae": source_report["candidate_results"][
                "m1_incremental_k24"
            ]["mean_mae_log10_usd"],
        },
        "model_comparisons": model_comparisons,
        "decision_tree_ablations": ablations,
        "model_role_decision": {
            "single_decision_tree_is_primary_explanatory_model": primary_supported,
            "reason": (
                "The single tree preserves directly extractable if-then rules. "
                "Ensembles are fixed predictive benchmarks, not rule generators."
            ),
            "lowest_absolute_m1_mae_model_family": benchmark_winner,
            "m1_mae_by_model_family": full_mae,
        },
        "rule_discovery": rule_metadata,
        "interpretation": {
            "m1_superiority_scope": (
                "Development evidence on repeated OOF predictions for the 1,912 "
                "training products; not a fresh held-out confirmation."
            ),
            "rule_scope": (
                "Post-selection associational pricing candidates, not causal rules."
            ),
        },
        "input_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in required_inputs()},
    }
    return report


def markdown_report(report: dict[str, Any], univariate: pd.DataFrame) -> str:
    tree = report["model_comparisons"]["decision_tree"]
    versus_m0 = tree["m1_vs_m0"]["paired_product_bootstrap"]
    versus_text = tree["m1_vs_text_baseline"]["paired_product_bootstrap"]
    lines = [
        "# Financial Core Analysis Results",
        "",
        "## M0 versus M1",
        "",
        f"- Decision-tree M0 MAE: {tree['m1_vs_m0']['baseline_mean_metrics']['mae_log10_usd']:.6f}.",
        f"- Decision-tree M1 MAE: {tree['m1_vs_m0']['model_mean_metrics']['mae_log10_usd']:.6f}.",
        f"- M1 relative MAE reduction versus M0: {versus_m0['relative_mae_reduction'] * 100:.2f}% "
        f"(95% CI for absolute reduction [{versus_m0['percentile_95_ci'][0]:.6f}, {versus_m0['percentile_95_ci'][1]:.6f}], "
        f"bootstrap p={versus_m0['two_sided_bootstrap_p_value']:.4f}).",
        f"- Evidence-state mechanisms reduce MAE by another {versus_text['relative_mae_reduction'] * 100:.2f}% versus the explicit-plus-text baseline "
        f"(95% CI [{versus_text['percentile_95_ci'][0]:.6f}, {versus_text['percentile_95_ci'][1]:.6f}], "
        f"bootstrap p={versus_text['two_sided_bootstrap_p_value']:.4f}).",
        "",
        "## Model roles",
        "",
        f"The primary explanatory model is the single decision tree: {report['model_role_decision']['single_decision_tree_is_primary_explanatory_model']}. "
        f"The lowest-MAE fixed benchmark is `{report['model_role_decision']['lowest_absolute_m1_mae_model_family']}`. "
        "Ensembles are retained only as predictive references.",
        "",
        "## Top exploratory rule candidates",
        "",
        "| Candidate | Direction | Support | Effect (log10) | Selection frequency |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in univariate.head(10).iterrows():
        lines.append(
            f"| {row['canonical_name']} | {row['direction']} | {int(row['support'])} | "
            f"{row['residual_effect_log10_usd']:+.4f} | {row['outer_top24_selection_frequency']:.0%} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "The comparison is a nested repeated-CV development analysis on 1,912 training products. "
            "No test rows, test targets, simulated outcomes, or real-price acceptance gate were used. "
            "The mined rules remain post-selection associational candidates until controlled intervention.",
            "",
        ]
    )
    return "\n".join(lines)


def fit_final_models(
    frame: pd.DataFrame,
    explicit: list[str],
    target: str,
    tree_params: dict[str, Any],
) -> None:
    frozen = joblib.load(FROZEN_M1_BUNDLE)
    base_ids = list(frozen["base_term_ids"])
    candidate_ids = list(frozen["mechanism_ids"])
    base_columns = [column for item in base_ids for column in mechanism_columns(item)]
    candidate_columns = [
        column for item in candidate_ids for column in mechanism_columns(item)
    ]
    full_features = [*explicit, *base_columns, *candidate_columns]
    y = frame[target].to_numpy(float)
    m0 = DecisionTreeRegressor(random_state=SEED, **tree_params).fit(
        frame[explicit], y
    )
    m1 = DecisionTreeRegressor(random_state=SEED, **tree_params).fit(
        frame[full_features], y
    )
    joblib.dump(
        {
            "model": m0,
            "features": explicit,
            "target": target,
            "params": tree_params,
            "training_ids": frame["Id"].astype(int).tolist(),
            "status": "development_full_training_model",
        },
        OUTPUT / "m0_decision_tree.joblib",
    )
    joblib.dump(
        {
            "model": m1,
            "features": full_features,
            "explicit_features": explicit,
            "base_term_ids": base_ids,
            "mechanism_ids": candidate_ids,
            "target": target,
            "params": tree_params,
            "training_ids": frame["Id"].astype(int).tolist(),
            "status": "development_full_training_model",
        },
        OUTPUT / "m1_decision_tree.joblib",
    )
    (OUTPUT / "m0_tree_rules.txt").write_text(
        export_text(m0, feature_names=explicit, decimals=5), encoding="utf-8"
    )
    (OUTPUT / "m1_tree_rules.txt").write_text(
        export_text(m1, feature_names=full_features, decimals=5), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "feature": full_features,
            "importance": m1.feature_importances_,
        }
    ).sort_values(["importance", "feature"], ascending=[False, True]).to_csv(
        OUTPUT / "m1_feature_importance.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (
        frame,
        explicit,
        target,
        tree_params,
        _base_taxonomy,
        candidate_taxonomy,
        selections,
        source_report,
    ) = load_analysis_frame()
    predictions, fold_metrics, selected_states = evaluate_models(
        frame,
        explicit,
        target,
        tree_params,
        candidate_taxonomy,
        selections,
    )
    y = frame[target].to_numpy(float)
    repeat_metrics, product_summary = summarize_predictions(predictions, y)
    product_summary.insert(0, "Id", frame["Id"].to_numpy(int))
    univariate, themes, path_rules, rule_tree_text, rule_metadata = (
        mine_rule_candidates(
            frame,
            target,
            candidate_taxonomy,
            selections,
        )
    )
    report = build_report(
        predictions,
        repeat_metrics,
        y,
        source_report,
        rule_metadata,
    )

    fold_metrics.to_csv(
        OUTPUT / "fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    repeat_metrics.to_csv(
        OUTPUT / "repeat_metrics.csv", index=False, encoding="utf-8-sig"
    )
    product_summary.to_csv(
        OUTPUT / "oof_product_summary.csv", index=False, encoding="utf-8-sig"
    )
    selected_states.to_csv(
        OUTPUT / "selected_state_counts_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    univariate.to_csv(
        OUTPUT / "univariate_rule_candidates.csv", index=False, encoding="utf-8-sig"
    )
    themes.to_csv(
        OUTPUT / "mechanism_theme_summary.csv", index=False, encoding="utf-8-sig"
    )
    write_json(OUTPUT / "path_rule_candidates.json", path_rules)
    (OUTPUT / "candidate_rule_tree.txt").write_text(
        rule_tree_text, encoding="utf-8"
    )
    write_json(OUTPUT / "core_analysis_report.json", report)
    (OUTPUT / "CORE_ANALYSIS_REPORT.md").write_text(
        markdown_report(report, univariate), encoding="utf-8"
    )
    fit_final_models(frame, explicit, target, tree_params)

    output_paths = [
        path for path in OUTPUT.iterdir() if path.is_file() and path.name != "manifest.json"
    ]
    write_json(
        OUTPUT / "manifest.json",
        {
            "status": report["status"],
            "protocol_sha256": sha256_file(PROTOCOL),
            "outputs": {
                path.name: sha256_file(path) for path in sorted(output_paths)
            },
            "documentation": {
                "RESULTS_2026-08-17_CN.md": sha256_file(
                    STAGE / "RESULTS_2026-08-17_CN.md"
                )
            },
            "test_rows_used": False,
            "test_targets_used": False,
            "real_price_acceptance_gate_used": False,
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
