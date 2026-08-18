from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor, export_text


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parent
WORK = ROOT / "results" / "work" / "telecom"
OUTPUT = WORK / "m1_evaluation"
SPLIT = ROOT / "data" / "telecom_train_test_split.csv"
M0_BUNDLE = ROOT / "models" / "telecom" / "telecom_m0_reproduced.joblib"
M0_OOF = ROOT / "results" / "work" / "telecom_m0" / "m0_train_oof_predictions.csv"
TRAIN_INPUT = WORK / "model_inputs" / "train_negotiation_inputs.csv"
TEST_INPUT = WORK / "model_inputs" / "test_model_inputs.csv"
TEST_OUTCOMES = (
    WORK / "model_inputs" / "test_outcomes_for_evaluation.csv"
)
FEATURE_MANIFEST = WORK / "features" / "telecom_feature_manifest.json"
TRAIN_MATRIX = (
    WORK / "features" / "telecom_train_frozen_feature_matrix.csv"
)
TEST_MATRIX = (
    WORK / "features" / "telecom_test_frozen_feature_matrix.csv"
)
BASE_TAXONOMY = (
    WORK / "features" / "telecom_base_term_taxonomy.json"
)
CANDIDATE_TAXONOMY = (
    WORK / "features" / "telecom_term_state_taxonomy.json"
)

SEED = 20260813 + 3200
FOLDS = 5
REPEATS = 20
INNER_FOLDS = 4
BASE_K = 12
CANDIDATE_K = 24
MIN_PREVALENCE_SUPPORT = 15 / 1912
MAX_PREVALENCE = 0.95
BOOTSTRAP_DRAWS = 20_000
SIGN_FLIP_DRAWS = 50_000


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


def feature_columns(canonical_ids: list[str]) -> list[str]:
    return [
        column
        for canonical_id in canonical_ids
        for column in (
            f"implicit_{canonical_id}_observed",
            f"implicit_{canonical_id}_score",
        )
    ]


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - y
    return {
        "mae_log10_usd": float(mean_absolute_error(y, prediction)),
        "rmse_log10_usd": float(np.sqrt(mean_squared_error(y, prediction))),
        "median_absolute_error_log10_usd": float(np.median(np.abs(error))),
        "r2": float(r2_score(y, prediction)),
    }


def support_threshold(n_rows: int) -> int:
    return max(5, int(np.ceil(MIN_PREVALENCE_SUPPORT * n_rows)))


def eligible_ids(frame: pd.DataFrame, canonical_ids: list[str]) -> list[str]:
    threshold = support_threshold(len(frame))
    result = []
    for canonical_id in canonical_ids:
        observed = f"implicit_{canonical_id}_observed"
        score = f"implicit_{canonical_id}_score"
        count = int(frame[observed].sum())
        if (
            count >= threshold
            and count / len(frame) <= MAX_PREVALENCE
            and frame[score].nunique() > 1
        ):
            result.append(canonical_id)
    return result


def inner_oof_residual(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    params: dict[str, Any],
    strata: np.ndarray,
    seed: int,
) -> np.ndarray:
    y = frame[target].to_numpy(float)
    prediction = np.full(len(frame), np.nan, dtype=float)
    splitter = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=seed
    )
    for fold, (fit_index, validation_index) in enumerate(
        splitter.split(frame, strata), start=1
    ):
        model = DecisionTreeRegressor(
            random_state=seed + fold, **params
        ).fit(frame.iloc[fit_index][features], y[fit_index])
        prediction[validation_index] = model.predict(
            frame.iloc[validation_index][features]
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete inner OOF residuals")
    return y - prediction


def rank_features(
    frame: pd.DataFrame,
    canonical_ids: list[str],
    residual: np.ndarray,
) -> pd.DataFrame:
    rows = []
    scale = max(float(np.std(residual, ddof=1)), 1e-9)
    for canonical_id in eligible_ids(frame, canonical_ids):
        observed = frame[f"implicit_{canonical_id}_observed"].to_numpy(float)
        score = frame[f"implicit_{canonical_id}_score"].to_numpy(float)
        rho_score, p_score = spearmanr(score, residual)
        rho_observed, p_observed = spearmanr(observed, residual)
        if not np.isfinite(rho_score):
            rho_score, p_score = 0.0, 1.0
        if not np.isfinite(rho_observed):
            rho_observed, p_observed = 0.0, 1.0
        mask = observed > 0
        gap = float(residual[mask].mean() - residual[~mask].mean())
        rows.append(
            {
                "canonical_id": canonical_id,
                "observed_count": int(mask.sum()),
                "prevalence": float(mask.mean()),
                "spearman_score_residual": float(rho_score),
                "spearman_score_p_value": float(p_score),
                "spearman_observed_residual": float(rho_observed),
                "spearman_observed_p_value": float(p_observed),
                "standardized_residual_gap": gap / scale,
                "selection_score": max(abs(float(rho_score)), abs(float(rho_observed))),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["canonical_id", "selection_score", "observed_count"]
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["selection_score", "observed_count", "canonical_id"],
        ascending=[False, False, True],
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking.reset_index(drop=True)


def paired_bootstrap(
    baseline_loss: np.ndarray, model_loss: np.ndarray, seed: int
) -> dict[str, Any]:
    improvement = np.asarray(baseline_loss) - np.asarray(model_loss)
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for start in range(0, BOOTSTRAP_DRAWS, 500):
        width = min(500, BOOTSTRAP_DRAWS - start)
        index = rng.integers(0, len(improvement), size=(width, len(improvement)))
        means[start : start + width] = improvement[index].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    p_value = min(
        1.0,
        2 * min(float((means <= 0).mean()), float((means >= 0).mean())),
    )
    return {
        "point_estimate": float(improvement.mean()),
        "relative_mae_reduction": float(improvement.mean() / baseline_loss.mean()),
        "percentile_95_ci": [float(lower), float(upper)],
        "two_sided_bootstrap_p_value": float(p_value),
        "model_lower_error_product_fraction": float((improvement > 0).mean()),
        "draws": BOOTSTRAP_DRAWS,
    }


def sign_flip_p_value(delta: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(float(delta.mean()))
    extreme = 0
    for start in range(0, SIGN_FLIP_DRAWS, 500):
        width = min(500, SIGN_FLIP_DRAWS - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(width, len(delta)))
        extreme += int((np.abs((signs * delta).mean(axis=1)) >= observed).sum())
    return float((extreme + 1) / (SIGN_FLIP_DRAWS + 1))


def compare(
    y: np.ndarray, baseline: np.ndarray, model: np.ndarray, seed: int
) -> dict[str, Any]:
    baseline_loss = np.abs(y - baseline)
    model_loss = np.abs(y - model)
    delta = baseline_loss - model_loss
    bootstrap = paired_bootstrap(baseline_loss, model_loss, seed)
    return {
        "baseline_metrics": metrics(y, baseline),
        "model_metrics": metrics(y, model),
        "absolute_mae_reduction": float(delta.mean()),
        "relative_mae_reduction": float(delta.mean() / baseline_loss.mean()),
        "model_lower_error_product_fraction": float((model_loss < baseline_loss).mean()),
        "paired_bootstrap": bootstrap,
        "two_sided_sign_flip_p_value": sign_flip_p_value(delta, seed + 1),
        "primary_support_rule_passed": bool(
            delta.mean() > 0
            and bootstrap["percentile_95_ci"][0] > 0
            and bootstrap["two_sided_bootstrap_p_value"] < 0.05
            and sign_flip_p_value(delta, seed + 1) < 0.05
        ),
    }


def run_development(
    frame: pd.DataFrame,
    base_ids: list[str],
    candidate_ids: list[str],
    explicit: list[str],
    target: str,
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = frame[target].to_numpy(float)
    strata = frame["split_stratum"].astype(str).to_numpy()
    splitter = RepeatedStratifiedKFold(
        n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED
    )
    model_names = ["m0_explicit", "text_baseline", "m1_full", "m1_permuted"]
    predictions = {
        name: np.full((REPEATS, len(frame)), np.nan, dtype=float)
        for name in model_names
    }
    selection_rows = []
    fold_rows = []
    base_frequency: Counter[str] = Counter()
    candidate_frequency: Counter[str] = Counter()
    for split_index, (fit_index, validation_index) in enumerate(
        splitter.split(frame, strata)
    ):
        repeat = split_index // FOLDS + 1
        fold = split_index % FOLDS + 1
        fit = frame.iloc[fit_index].copy()
        validation = frame.iloc[validation_index].copy()
        fit_strata = strata[fit_index]
        fold_seed = SEED + repeat * 1000 + fold * 10
        explicit_residual = inner_oof_residual(
            fit, explicit, target, params, fit_strata, fold_seed
        )
        base_ranking = rank_features(fit, base_ids, explicit_residual)
        if len(base_ranking) < BASE_K:
            raise RuntimeError(
                f"Only {len(base_ranking)} eligible base terms in repeat={repeat}, fold={fold}"
            )
        selected_base = base_ranking.head(BASE_K)["canonical_id"].tolist()
        base_frequency.update(selected_base)
        base_columns = feature_columns(selected_base)
        for rank, canonical_id in enumerate(selected_base, start=1):
            selection_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "stage": "base_term",
                    "rank": rank,
                    "canonical_id": canonical_id,
                }
            )
        text_features = [*explicit, *base_columns]
        conditional_residual = inner_oof_residual(
            fit, text_features, target, params, fit_strata, fold_seed + 5
        )
        candidate_ranking = rank_features(fit, candidate_ids, conditional_residual)
        if len(candidate_ranking) < CANDIDATE_K:
            raise RuntimeError(
                f"Only {len(candidate_ranking)} eligible mechanisms in repeat={repeat}, fold={fold}"
            )
        ordered_candidates = candidate_ranking["canonical_id"].tolist()
        candidate_frequency.update(ordered_candidates[:CANDIDATE_K])
        for rank, canonical_id in enumerate(ordered_candidates, start=1):
            selection_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "stage": "conditional_mechanism",
                    "rank": rank,
                    "canonical_id": canonical_id,
                }
            )
        candidate_columns = feature_columns(ordered_candidates[:CANDIDATE_K])
        m1_features = [*text_features, *candidate_columns]
        models = {}
        for name, features in [
            ("m0_explicit", explicit),
            ("text_baseline", text_features),
            ("m1_full", m1_features),
        ]:
            model = DecisionTreeRegressor(
                random_state=fold_seed, **params
            ).fit(fit[features], y[fit_index])
            models[name] = model
            prediction = model.predict(validation[features])
            predictions[name][repeat - 1, validation_index] = prediction
            fold_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "model": name,
                    "feature_count": len(features),
                    **metrics(y[validation_index], prediction),
                }
            )

        # A target-free permutation placebo keeps the selected feature block but
        # destroys product alignment in the validation portion.
        permuted = validation[m1_features].copy()
        rng = np.random.default_rng(fold_seed + 700_000)
        permuted.loc[:, candidate_columns] = permuted[candidate_columns].to_numpy()[
            rng.permutation(len(validation))
        ]
        placebo_prediction = models["m1_full"].predict(permuted)
        predictions["m1_permuted"][repeat - 1, validation_index] = placebo_prediction
        fold_rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "model": "m1_permuted",
                "feature_count": len(m1_features),
                **metrics(y[validation_index], placebo_prediction),
            }
        )
        if (split_index + 1) % 10 == 0:
            print(f"completed_development_folds={split_index + 1}/{FOLDS * REPEATS}", flush=True)

    for name, values in predictions.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Incomplete development predictions for {name}")
    repeat_rows = []
    product = frame[["Id", target]].copy()
    for name, values in predictions.items():
        for repeat in range(REPEATS):
            repeat_rows.append(
                {"repeat": repeat + 1, "model": name, **metrics(y, values[repeat])}
            )
        product[f"{name}_mean_oof_prediction"] = values.mean(axis=0)
        product[f"{name}_mean_absolute_error"] = np.abs(values - y).mean(axis=0)
    repeat_frame = pd.DataFrame(repeat_rows)
    m0_repeat = repeat_frame.loc[repeat_frame["model"].eq("m0_explicit")].set_index("repeat")
    text_repeat = repeat_frame.loc[repeat_frame["model"].eq("text_baseline")].set_index("repeat")
    m1_repeat = repeat_frame.loc[repeat_frame["model"].eq("m1_full")].set_index("repeat")
    m0_loss = np.abs(predictions["m0_explicit"] - y).mean(axis=0)
    text_loss = np.abs(predictions["text_baseline"] - y).mean(axis=0)
    m1_loss = np.abs(predictions["m1_full"] - y).mean(axis=0)
    placebo_loss = np.abs(predictions["m1_permuted"] - y).mean(axis=0)
    development_report = {
        "status": "telecom_training_nested_repeated_cv_complete",
        "training_rows": len(frame),
        "test_rows_used": 0,
        "test_targets_used": False,
        "outer_cv": {"folds": FOLDS, "repeats": REPEATS},
        "inner_residual_cv_folds": INNER_FOLDS,
        "same_tree_hyperparameters": True,
        "tree_hyperparameters": params,
        "support_rule": "max(5, ceil((15/1912) * n_fit))",
        "base_k": BASE_K,
        "candidate_k": CANDIDATE_K,
        "m0_mean_metrics": {
            key: float(m0_repeat[key].mean())
            for key in ["mae_log10_usd", "rmse_log10_usd", "median_absolute_error_log10_usd", "r2"]
        },
        "text_mean_metrics": {
            key: float(text_repeat[key].mean())
            for key in ["mae_log10_usd", "rmse_log10_usd", "median_absolute_error_log10_usd", "r2"]
        },
        "m1_mean_metrics": {
            key: float(m1_repeat[key].mean())
            for key in ["mae_log10_usd", "rmse_log10_usd", "median_absolute_error_log10_usd", "r2"]
        },
        "m1_vs_m0": compare(
            y,
            predictions["m0_explicit"].mean(axis=0),
            predictions["m1_full"].mean(axis=0),
            SEED + 30,
        ),
        "m1_vs_text": compare(
            y,
            predictions["text_baseline"].mean(axis=0),
            predictions["m1_full"].mean(axis=0),
            SEED + 31,
        ),
        "m1_vs_permuted": compare(
            y,
            predictions["m1_permuted"].mean(axis=0),
            predictions["m1_full"].mean(axis=0),
            SEED + 32,
        ),
        "repeat_mae_reductions": {
            "m1_vs_m0": [
                float(m0_repeat.loc[index, "mae_log10_usd"] - m1_repeat.loc[index, "mae_log10_usd"])
                for index in range(1, REPEATS + 1)
            ],
            "m1_vs_text": [
                float(text_repeat.loc[index, "mae_log10_usd"] - m1_repeat.loc[index, "mae_log10_usd"])
                for index in range(1, REPEATS + 1)
            ],
        },
        "repeats_m1_better": {
            "vs_m0": int((m0_repeat["mae_log10_usd"] > m1_repeat["mae_log10_usd"]).sum()),
            "vs_text": int((text_repeat["mae_log10_usd"] > m1_repeat["mae_log10_usd"]).sum()),
            "total": REPEATS,
        },
        "selection_stability": {
            "base_top": [
                {"canonical_id": key, "selected_folds": int(value)}
                for key, value in base_frequency.most_common(20)
            ],
            "candidate_top": [
                {"canonical_id": key, "selected_folds": int(value)}
                for key, value in candidate_frequency.most_common(30)
            ],
        },
    }
    return (
        development_report,
        predictions,
        pd.DataFrame(selection_rows),
        pd.DataFrame(fold_rows),
        product,
    )


def fit_full_selection(
    frame: pd.DataFrame,
    base_ids: list[str],
    candidate_ids: list[str],
    explicit: list[str],
    target: str,
    params: dict[str, Any],
) -> tuple[list[str], list[str], Any, Any]:
    strata = frame["split_stratum"].astype(str).to_numpy()
    explicit_residual = inner_oof_residual(
        frame, explicit, target, params, strata, SEED + 5000
    )
    selected_base = rank_features(frame, base_ids, explicit_residual).head(BASE_K)[
        "canonical_id"
    ].tolist()
    base_columns = feature_columns(selected_base)
    text_features = [*explicit, *base_columns]
    conditional_residual = inner_oof_residual(
        frame, text_features, target, params, strata, SEED + 5005
    )
    selected_candidates = rank_features(
        frame, candidate_ids, conditional_residual
    ).head(CANDIDATE_K)["canonical_id"].tolist()
    if len(selected_base) != BASE_K or len(selected_candidates) != CANDIDATE_K:
        raise RuntimeError("Full-training feature selection returned too few features")
    y = frame[target].to_numpy(float)
    m0 = DecisionTreeRegressor(random_state=SEED, **params).fit(frame[explicit], y)
    m1_features = [*text_features, *feature_columns(selected_candidates)]
    m1 = DecisionTreeRegressor(random_state=SEED, **params).fit(
        frame[m1_features], y
    )
    return selected_base, selected_candidates, m0, m1


def build_external_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    base_ids: list[str],
    candidate_ids: list[str],
    explicit: list[str],
    target: str,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    y = train[target].to_numpy(float)
    strata = train["split_stratum"].astype(str).to_numpy()
    splitter = RepeatedStratifiedKFold(
        n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED + 100
    )
    names = ["m0_explicit", "text_baseline", "m1_full"]
    predictions = {name: np.empty((FOLDS * REPEATS, len(test))) for name in names}
    base_frequency: Counter[str] = Counter()
    candidate_frequency: Counter[str] = Counter()
    for split_index, (fit_index, _) in enumerate(splitter.split(train, strata)):
        repeat = split_index // FOLDS + 1
        fold = split_index % FOLDS + 1
        fit = train.iloc[fit_index].copy()
        fit_strata = strata[fit_index]
        fold_seed = SEED + 10_000 + repeat * 1000 + fold * 10
        explicit_residual = inner_oof_residual(
            fit, explicit, target, params, fit_strata, fold_seed
        )
        selected_base = rank_features(fit, base_ids, explicit_residual).head(BASE_K)[
            "canonical_id"
        ].tolist()
        base_frequency.update(selected_base)
        text_features = [*explicit, *feature_columns(selected_base)]
        conditional_residual = inner_oof_residual(
            fit, text_features, target, params, fit_strata, fold_seed + 5
        )
        selected_candidates = rank_features(
            fit, candidate_ids, conditional_residual
        ).head(CANDIDATE_K)["canonical_id"].tolist()
        candidate_frequency.update(selected_candidates)
        m1_features = [*text_features, *feature_columns(selected_candidates)]
        for name, features in [
            ("m0_explicit", explicit),
            ("text_baseline", text_features),
            ("m1_full", m1_features),
        ]:
            model = DecisionTreeRegressor(
                random_state=fold_seed, **params
            ).fit(fit[features], y[fit_index])
            predictions[name][split_index] = model.predict(test[features])
        if (split_index + 1) % 10 == 0:
            print(f"completed_external_models={split_index + 1}/{FOLDS * REPEATS}", flush=True)
    output = pd.DataFrame({"Id": test["Id"].astype(int)})
    for name, values in predictions.items():
        output[f"{name}_ensemble_prediction"] = values.mean(axis=0)
        output[f"{name}_prediction_sd"] = values.std(axis=0, ddof=1)
    metadata = {
        "models_per_ensemble": FOLDS * REPEATS,
        "folds": FOLDS,
        "repeats": REPEATS,
        "feature_selection_inside_each_training_fold": True,
        "same_tree_hyperparameters": True,
        "test_targets_loaded": False,
        "test_targets_used": False,
        "base_selection_frequency": [
            {"canonical_id": key, "models_selected": int(value)}
            for key, value in base_frequency.most_common()
        ],
        "candidate_selection_frequency": [
            {"canonical_id": key, "models_selected": int(value)}
            for key, value in candidate_frequency.most_common()
        ],
    }
    return output, metadata


def main() -> None:
    required = [
        SPLIT,
        M0_BUNDLE,
        M0_OOF,
        TRAIN_INPUT,
        TEST_INPUT,
        TEST_OUTCOMES,
        FEATURE_MANIFEST,
        TRAIN_MATRIX,
        TEST_MATRIX,
        BASE_TAXONOMY,
        CANDIDATE_TAXONOMY,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing telecom evaluation inputs: {missing}")
    feature_manifest = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
    if feature_manifest["test_targets_used"] or feature_manifest["taxonomy_refit_on_test"]:
        raise RuntimeError("Feature boundary audit failed")
    bundle = joblib.load(M0_BUNDLE)
    explicit = list(bundle["features"])
    target = str(bundle["target"])
    params = dict(bundle["best_params"])
    split = pd.read_csv(SPLIT, usecols=["Id", "split", "split_stratum"])
    train_input = pd.read_csv(
        TRAIN_INPUT, usecols=["Id", *explicit], low_memory=False
    )
    if {target, "TransactionPriceUSD"} & set(train_input.columns):
        raise RuntimeError("Training model inputs unexpectedly contain outcome columns")
    train_outcomes = pd.read_csv(M0_OOF, usecols=["Id", target])
    train_matrix = pd.read_csv(TRAIN_MATRIX, low_memory=False)
    test_matrix = pd.read_csv(TEST_MATRIX, low_memory=False)
    base_taxonomy = json.loads(BASE_TAXONOMY.read_text(encoding="utf-8"))["taxonomy"]
    candidate_taxonomy = json.loads(CANDIDATE_TAXONOMY.read_text(encoding="utf-8"))["taxonomy"]
    base_ids = [str(item["canonical_id"]) for item in base_taxonomy]
    candidate_ids = [str(item["canonical_id"]) for item in candidate_taxonomy]
    train = (
        train_input.merge(
            split.loc[split["split"].eq("train")], on="Id", validate="one_to_one"
        )
        .merge(train_outcomes, on="Id", validate="one_to_one")
        .merge(train_matrix, left_on="Id", right_on="product_id", validate="one_to_one")
        .drop(columns="product_id")
        .sort_values("Id")
        .reset_index(drop=True)
    )
    test_input = pd.read_csv(TEST_INPUT, usecols=["Id", *explicit], low_memory=False)
    if {target, "TransactionPriceUSD"} & set(test_input.columns):
        raise RuntimeError("Test model inputs contain outcome columns")
    test = (
        test_input.merge(test_matrix, left_on="Id", right_on="product_id", validate="one_to_one")
        .drop(columns="product_id")
        .sort_values("Id")
        .reset_index(drop=True)
    )
    if len(train) != 420 or len(test) != 106:
        raise RuntimeError(f"Unexpected telecom cohort sizes train={len(train)}, test={len(test)}")
    model_columns = [target, *explicit, *feature_columns(base_ids), *feature_columns(candidate_ids)]
    if train[model_columns].isna().any().any() or test[model_columns[1:]].isna().any().any():
        raise RuntimeError("Missing model values in telecom frame")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    development_report, development_predictions, selection_rows, fold_rows, product = run_development(
        train, base_ids, candidate_ids, explicit, target, params
    )
    selection_rows.to_csv(OUTPUT / "development_nested_selection_rankings.csv", index=False, encoding="utf-8-sig")
    fold_rows.to_csv(OUTPUT / "development_fold_metrics.csv", index=False, encoding="utf-8-sig")
    product.to_csv(OUTPUT / "development_oof_product_summary.csv", index=False, encoding="utf-8-sig")
    write_json(OUTPUT / "development_report.json", development_report)

    # Build every test prediction before loading the held-out outcome file.
    external_predictions, external_metadata = build_external_predictions(
        train, test, base_ids, candidate_ids, explicit, target, params
    )
    target_free_path = OUTPUT / "external_predictions_without_targets.csv"
    external_predictions.to_csv(target_free_path, index=False, encoding="utf-8-sig")
    target_free_hash = sha256_file(target_free_path)
    write_json(
        OUTPUT / "external_prediction_manifest.json",
        {
            "status": "telecom_external_predictions_complete_without_test_targets",
            **external_metadata,
            "train_rows": len(train),
            "test_rows": len(test),
            "test_targets_loaded": False,
            "test_targets_used": False,
            "target_free_prediction_sha256": target_free_hash,
            "input_hashes": {
                path.name: sha256_file(path)
                for path in [FEATURE_MANIFEST, TRAIN_MATRIX, TEST_MATRIX, TRAIN_INPUT, TEST_INPUT]
            },
        },
    )

    # This is the first point at which held-out real transaction prices are read.
    outcomes = pd.read_csv(TEST_OUTCOMES, usecols=["Id", target, "TransactionPriceUSD"])
    evaluated = external_predictions.merge(outcomes, on="Id", validate="one_to_one")
    y_test = evaluated[target].to_numpy(float)
    m0_test = evaluated["m0_explicit_ensemble_prediction"].to_numpy(float)
    text_test = evaluated["text_baseline_ensemble_prediction"].to_numpy(float)
    m1_test = evaluated["m1_full_ensemble_prediction"].to_numpy(float)
    test_report = {
        "m1_vs_m0": compare(y_test, m0_test, m1_test, SEED + 60),
        "m1_vs_text": compare(y_test, text_test, m1_test, SEED + 61),
    }
    evaluated["price_decile"] = pd.qcut(
        evaluated[target], q=10, labels=False, duplicates="drop"
    )
    decile_rows = []
    for decile, group in evaluated.groupby("price_decile", sort=True):
        gy = group[target].to_numpy(float)
        gm0 = group["m0_explicit_ensemble_prediction"].to_numpy(float)
        gm1 = group["m1_full_ensemble_prediction"].to_numpy(float)
        decile_rows.append(
            {
                "price_decile": int(decile),
                "rows": len(group),
                "m0_mae": metrics(gy, gm0)["mae_log10_usd"],
                "m1_mae": metrics(gy, gm1)["mae_log10_usd"],
                "m1_better": bool(metrics(gy, gm1)["mae_log10_usd"] < metrics(gy, gm0)["mae_log10_usd"]),
            }
        )
    pd.DataFrame(decile_rows).to_csv(OUTPUT / "external_price_decile_results.csv", index=False, encoding="utf-8-sig")
    evaluated.to_csv(OUTPUT / "external_predictions_with_targets.csv", index=False, encoding="utf-8-sig")

    dev_predictions_frame = product.copy()
    blend_rows = []
    y_dev = train[target].to_numpy(float)
    m0_dev = dev_predictions_frame["m0_explicit_mean_oof_prediction"].to_numpy(float)
    m1_dev = dev_predictions_frame["m1_full_mean_oof_prediction"].to_numpy(float)
    for weight in np.linspace(0.0, 1.0, 101):
        blend = (1 - weight) * m0_dev + weight * m1_dev
        blend_rows.append({"m1_weight": float(weight), **metrics(y_dev, blend)})
    blend_frame = pd.DataFrame(blend_rows)
    best_row = blend_frame.loc[blend_frame["mae_log10_usd"].idxmin()]
    blend_weight = float(best_row["m1_weight"])
    blend_test = (1 - blend_weight) * m0_test + blend_weight * m1_test
    test_report["training_selected_blend"] = {
        "m1_weight": blend_weight,
        "test_targets_used_for_selection": False,
        "m1_blend_vs_m0": compare(y_test, m0_test, blend_test, SEED + 62),
    }
    blend_frame.to_csv(OUTPUT / "development_blend_weight_search.csv", index=False, encoding="utf-8-sig")

    selected_base, selected_candidates, final_m0, final_m1 = fit_full_selection(
        train, base_ids, candidate_ids, explicit, target, params
    )
    selected_features = [
        *explicit,
        *feature_columns(selected_base),
        *feature_columns(selected_candidates),
    ]
    joblib.dump(
        {
            "model": final_m0,
            "features": explicit,
            "target": target,
            "params": params,
            "training_ids": train["Id"].astype(int).tolist(),
        },
        OUTPUT / "telecom_m0_final.joblib",
    )
    joblib.dump(
        {
            "model": final_m1,
            "features": selected_features,
            "explicit_features": explicit,
            "base_term_ids": selected_base,
            "mechanism_ids": selected_candidates,
            "target": target,
            "params": params,
            "training_ids": train["Id"].astype(int).tolist(),
        },
        OUTPUT / "telecom_m1_final.joblib",
    )
    (OUTPUT / "telecom_m0_rules.txt").write_text(
        export_text(final_m0, feature_names=explicit, decimals=5), encoding="utf-8"
    )
    (OUTPUT / "telecom_m1_rules.txt").write_text(
        export_text(final_m1, feature_names=selected_features, decimals=5), encoding="utf-8"
    )
    pd.DataFrame(
        {"feature": selected_features, "importance": final_m1.feature_importances_}
    ).sort_values("importance", ascending=False).to_csv(
        OUTPUT / "telecom_m1_feature_importance.csv", index=False, encoding="utf-8-sig"
    )
    write_json(
        OUTPUT / "telecom_selected_taxonomy.json",
        {
            "status": "frozen_from_training_only",
            "base_term_ids": selected_base,
            "mechanism_ids": selected_candidates,
            "test_targets_used": False,
        },
    )
    report = {
        "status": "telecom_m1_replication_complete",
        "domain": "Telecom",
        "rows": {"training": len(train), "heldout_test": len(evaluated)},
        "target": "LogPriceMo (log10 real monthly transaction price in USD)",
        "primary_metric": "MAE in log10 real monthly transaction price",
        "m0_tree_hyperparameters": params,
        "same_tree_hyperparameters": True,
        "feature_configuration": {
            "explicit_features": len(explicit),
            "base_terms_per_fold": BASE_K,
            "mechanisms_per_fold": CANDIDATE_K,
            "base_taxonomy_size": len(base_ids),
            "candidate_taxonomy_size": len(candidate_ids),
            "support_rule": "max(5, ceil((15/1912) * n_fit))",
            "selection_nested_in_training_folds": True,
        },
        "development": development_report,
        "heldout_test": {
            "test_targets_used_for_feature_construction": False,
            "test_targets_used_for_hyperparameter_selection": False,
            "target_free_prediction_file_sha256": target_free_hash,
            "models_per_ensemble": external_metadata["models_per_ensemble"],
            **test_report,
            "price_deciles_where_m1_better": int(sum(row["m1_better"] for row in decile_rows)),
            "price_deciles_total": len(decile_rows),
        },
        "claim_scope": {
            "m1_vs_m0_primary_mae_supported": bool(
                test_report["m1_vs_m0"]["primary_support_rule_passed"]
            ),
            "m1_vs_text_primary_mae_supported": bool(
                test_report["m1_vs_text"]["primary_support_rule_passed"]
            ),
            "causal_rule_status": "Predictive M1 comparison does not by itself establish causal rules; controlled interventions remain separate evidence.",
        },
        "input_hashes": {path.name: sha256_file(path) for path in required},
    }
    write_json(OUTPUT / "telecom_m1_evaluation_report.json", report)
    lines = [
        "# Telecom M1 Replication Results",
        "",
        f"- Held-out M0 MAE: {test_report['m1_vs_m0']['baseline_metrics']['mae_log10_usd']:.6f}",
        f"- Held-out M1 MAE: {test_report['m1_vs_m0']['model_metrics']['mae_log10_usd']:.6f}",
        f"- Relative M1 MAE reduction: {test_report['m1_vs_m0']['relative_mae_reduction'] * 100:.2f}%",
        f"- Paired bootstrap 95% CI: {test_report['m1_vs_m0']['paired_bootstrap']['percentile_95_ci']}",
        f"- Paired sign-flip p-value: {test_report['m1_vs_m0']['two_sided_sign_flip_p_value']:.6f}",
        f"- M1 lower-error products: {test_report['m1_vs_m0']['model_lower_error_product_fraction']:.2%}",
        f"- M1 better price deciles: {sum(row['m1_better'] for row in decile_rows)}/{len(decile_rows)}",
        "",
        "All feature construction and prediction generation were completed before the held-out outcome file was loaded. The simulator-derived mechanisms are target-free predictive signals; they are not presented as real-market causal evidence without intervention validation.",
        "",
    ]
    (OUTPUT / "RESULTS_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    manifest_paths = [path for path in OUTPUT.iterdir() if path.is_file()]
    write_json(
        OUTPUT / "manifest.json",
        {
            "status": report["status"],
            "test_targets_used_for_final_evaluation": True,
            "test_targets_used_for_feature_construction": False,
            "outputs": {path.name: sha256_file(path) for path in sorted(manifest_paths)},
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
