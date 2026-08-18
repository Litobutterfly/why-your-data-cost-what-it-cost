from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor, export_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "financial.csv"
DEFAULT_SPLIT = ROOT / "data" / "financial_train_test_split.csv"
DEFAULT_M0_BUNDLE = ROOT / "models" / "financial" / "financial_m0_reproduced.joblib"
DEFAULT_IMPLICIT = ROOT / "results" / "work" / "implicit_features" / "release_m1_numeric_features_v3.csv"
DEFAULT_FEATURE_MANIFEST = ROOT / "results" / "work" / "implicit_features" / "release_FEATURE_MANIFEST.json"
DEFAULT_OUTPUT = ROOT / "results" / "work" / "m1_training"
FOLDS = 5
BOOTSTRAP_DRAWS = 10_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def metrics(y_log: np.ndarray, prediction_log: np.ndarray) -> dict[str, float]:
    y_usd = np.power(10.0, y_log)
    prediction_usd = np.power(10.0, prediction_log)
    error_log = prediction_log - y_log
    absolute_percentage = np.abs(prediction_usd - y_usd) / y_usd
    return {
        "mae_log10_usd": float(mean_absolute_error(y_log, prediction_log)),
        "rmse_log10_usd": float(np.sqrt(mean_squared_error(y_log, prediction_log))),
        "r2": float(r2_score(y_log, prediction_log)),
        "mean_error_log10_usd": float(error_log.mean()),
        "median_absolute_error_log10_usd": float(np.median(np.abs(error_log))),
        "median_absolute_percentage_error": float(np.median(absolute_percentage)),
        "mean_absolute_error_usd": float(mean_absolute_error(y_usd, prediction_usd)),
    }


def paired_bootstrap(
    y: np.ndarray,
    prediction_m0: np.ndarray,
    prediction_m1: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    improvement = np.abs(prediction_m0 - y) - np.abs(prediction_m1 - y)
    rng = np.random.default_rng(seed)
    draw_means = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for start in range(0, BOOTSTRAP_DRAWS, 500):
        width = min(500, BOOTSTRAP_DRAWS - start)
        indices = rng.integers(0, len(improvement), size=(width, len(improvement)))
        draw_means[start:start + width] = improvement[indices].mean(axis=1)
    lower, upper = np.quantile(draw_means, [0.025, 0.975])
    p_two_sided = min(
        1.0,
        2.0 * min(float((draw_means <= 0).mean()), float((draw_means >= 0).mean())),
    )
    return {
        "estimand": "paired mean reduction in absolute log10-price error (M0 minus M1)",
        "point_estimate": float(improvement.mean()),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "percentile_95_ci": [float(lower), float(upper)],
        "two_sided_bootstrap_p_value": p_two_sided,
        "m1_lower_absolute_error_fraction": float((improvement > 0).mean()),
        "equal_absolute_error_fraction": float((improvement == 0).mean()),
    }


def run(
    data_path: Path,
    split_path: Path,
    m0_bundle_path: Path,
    implicit_path: Path,
    feature_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(data_path, low_memory=False)
    split = pd.read_csv(split_path)
    implicit = pd.read_csv(implicit_path)
    bundle = joblib.load(m0_bundle_path)
    explicit_features = list(bundle["features"])
    target = str(bundle["target"])
    seed = int(bundle["seed"])
    fixed_params = dict(bundle["best_params"])
    implicit_features = [column for column in implicit.columns if column != "product_id"]

    if len(implicit) != 1892 or implicit["product_id"].duplicated().any():
        raise RuntimeError("Expected 1,892 unique approved training products")
    if implicit[implicit_features].isna().any().any():
        raise RuntimeError("Implicit feature matrix contains missing values")
    train_ids = set(split.loc[split["split"].eq("train"), "Id"].astype(int))
    test_ids = set(split.loc[split["split"].eq("test"), "Id"].astype(int))
    implicit_ids = set(implicit["product_id"].astype(int))
    if not implicit_ids < train_ids or implicit_ids & test_ids:
        raise RuntimeError("Implicit features must be a strict training-only subset")
    excluded_train_ids = sorted(train_ids - implicit_ids)
    if len(excluded_train_ids) != 20:
        raise RuntimeError("Expected exactly 20 protocol-excluded training dialogues")

    common = (
        data.loc[data["Id"].isin(implicit_ids), ["Id", target, *explicit_features]]
        .merge(implicit, left_on="Id", right_on="product_id", validate="one_to_one")
        .sort_values("Id")
        .reset_index(drop=True)
    )
    if len(common) != 1892 or common[[target, *explicit_features, *implicit_features]].isna().any().any():
        raise RuntimeError("Common-sample training frame failed validation")
    numeric = common[[target, *explicit_features, *implicit_features]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Common-sample training frame contains non-finite values")

    y = common[target].to_numpy(float)
    x_m0 = common[explicit_features].to_numpy(float)
    x_m1 = common[[*explicit_features, *implicit_features]].to_numpy(float)
    price_bin = pd.qcut(
        common[target].rank(method="first"), 10, labels=False, duplicates="raise"
    ).astype(int)
    strata = price_bin.astype(str) + "|api" + common["RESTAPI"].astype(int).astype(str)
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed + 1)

    oof_m0 = np.empty(len(common), dtype=float)
    oof_m1 = np.empty(len(common), dtype=float)
    fold_id = np.empty(len(common), dtype=int)
    fold_reports = []
    for fold, (fit_idx, validation_idx) in enumerate(cv.split(x_m0, strata), start=1):
        model_m0 = DecisionTreeRegressor(random_state=seed + fold, **fixed_params)
        model_m1 = DecisionTreeRegressor(random_state=seed + fold, **fixed_params)
        model_m0.fit(x_m0[fit_idx], y[fit_idx])
        model_m1.fit(x_m1[fit_idx], y[fit_idx])
        oof_m0[validation_idx] = model_m0.predict(x_m0[validation_idx])
        oof_m1[validation_idx] = model_m1.predict(x_m1[validation_idx])
        fold_id[validation_idx] = fold
        fold_reports.append(
            {
                "fold": fold,
                "fit_rows": len(fit_idx),
                "validation_rows": len(validation_idx),
                "m0": metrics(y[validation_idx], oof_m0[validation_idx]),
                "m1": metrics(y[validation_idx], oof_m1[validation_idx]),
            }
        )

    final_m0 = DecisionTreeRegressor(random_state=seed, **fixed_params).fit(x_m0, y)
    final_m1 = DecisionTreeRegressor(random_state=seed, **fixed_params).fit(x_m1, y)
    output = common[["Id", target]].copy()
    output["cv_fold"] = fold_id
    output["m0_oof_prediction_log10_usd"] = oof_m0
    output["m1_oof_prediction_log10_usd"] = oof_m1
    output["m0_absolute_error_log10_usd"] = np.abs(oof_m0 - y)
    output["m1_absolute_error_log10_usd"] = np.abs(oof_m1 - y)
    output["paired_absolute_error_reduction"] = (
        output["m0_absolute_error_log10_usd"] - output["m1_absolute_error_log10_usd"]
    )
    output.to_csv(output_dir / "common_sample_oof_predictions.csv", index=False, encoding="utf-8-sig")

    joblib.dump(
        {
            "model": final_m0,
            "features": explicit_features,
            "target": target,
            "seed": seed,
            "fixed_params": fixed_params,
            "training_ids": common["Id"].astype(int).tolist(),
        },
        output_dir / "m0_common_sample_decision_tree.joblib",
    )
    joblib.dump(
        {
            "model": final_m1,
            "features": [*explicit_features, *implicit_features],
            "explicit_features": explicit_features,
            "implicit_features": implicit_features,
            "target": target,
            "seed": seed,
            "fixed_params": fixed_params,
            "training_ids": common["Id"].astype(int).tolist(),
        },
        output_dir / "m1_decision_tree.joblib",
    )
    pd.DataFrame(
        {"feature": explicit_features, "importance": final_m0.feature_importances_}
    ).sort_values("importance", ascending=False).to_csv(
        output_dir / "m0_common_feature_importance.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        {
            "feature": [*explicit_features, *implicit_features],
            "importance": final_m1.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(
        output_dir / "m1_feature_importance.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "m0_common_tree_rules.txt").write_text(
        export_text(final_m0, feature_names=explicit_features, decimals=5), encoding="utf-8"
    )
    (output_dir / "m1_tree_rules.txt").write_text(
        export_text(final_m1, feature_names=[*explicit_features, *implicit_features], decimals=5),
        encoding="utf-8",
    )

    m0_metrics = metrics(y, oof_m0)
    m1_metrics = metrics(y, oof_m1)
    report = {
        "status": "common_sample_training_oof_complete_test_untouched",
        "comparison_role": "training-only diagnostic before frozen test evaluation",
        "common_training_rows": len(common),
        "original_training_rows": len(train_ids),
        "protocol_excluded_training_rows": len(excluded_train_ids),
        "protocol_excluded_training_ids": excluded_train_ids,
        "test_rows": len(test_ids),
        "test_ids_used": False,
        "test_targets_used": False,
        "target": target,
        "explicit_feature_count": len(explicit_features),
        "implicit_feature_count": len(implicit_features),
        "m0_feature_count": len(explicit_features),
        "m1_feature_count": len(explicit_features) + len(implicit_features),
        "same_tree_hyperparameters": True,
        "fixed_hyperparameters": fixed_params,
        "folds": FOLDS,
        "fold_assignment": "same stratified folds for M0 and M1",
        "m0_oof_metrics": m0_metrics,
        "m1_oof_metrics": m1_metrics,
        "relative_mae_change": float(
            (m1_metrics["mae_log10_usd"] - m0_metrics["mae_log10_usd"])
            / m0_metrics["mae_log10_usd"]
        ),
        "paired_bootstrap": paired_bootstrap(y, oof_m0, oof_m1, seed=seed + 90),
        "fold_metrics": fold_reports,
        "tree_structure": {
            "m0_depth": int(final_m0.get_depth()),
            "m0_leaves": int(final_m0.get_n_leaves()),
            "m1_depth": int(final_m1.get_depth()),
            "m1_leaves": int(final_m1.get_n_leaves()),
        },
        "implicit_matrix_frequency_columns_used": False,
        "inputs": {
            "financial_sha256": sha256_file(data_path),
            "split_sha256": sha256_file(split_path),
            "m0_bundle_sha256": sha256_file(m0_bundle_path),
            "implicit_matrix_sha256": sha256_file(implicit_path),
            "feature_manifest_sha256": sha256_file(feature_manifest_path),
        },
    }
    write_json(output_dir / "m0_m1_common_sample_report.json", report)
    write_json(
        output_dir / "CURRENT_M1_TRAINING_MANIFEST.json",
        {
            "status": report["status"],
            "report": "m0_m1_common_sample_report.json",
            "m0_model": "m0_common_sample_decision_tree.joblib",
            "m1_model": "m1_decision_tree.joblib",
            "oof_predictions": "common_sample_oof_predictions.csv",
            "test_evaluation_complete": False,
        },
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--m0-bundle", type=Path, default=DEFAULT_M0_BUNDLE)
    parser.add_argument("--implicit", type=Path, default=DEFAULT_IMPLICIT)
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                data_path=args.data,
                split_path=args.split,
                m0_bundle_path=args.m0_bundle,
                implicit_path=args.implicit,
                feature_manifest_path=args.feature_manifest,
                output_dir=args.output,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
