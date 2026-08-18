"""Train the fixed-capacity M0 decision tree for Financial or Telecom.

The final hyperparameters are fixed in this release. This script does not
perform hyperparameter search and never uses held-out outcomes for model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor, export_text


ROOT = Path(__file__).resolve().parents[1]
EXPLICIT_FEATURES = [
    "History", "people", "entities", "products", "records", "events",
    "symbols", "assets", "requests", "features", "locations", "USD",
    "sources", "units", "Limitations", "ProfServices", "IdIndividuals",
    "IdCompanies", "NCountries", "PercGDP", "DelMethod", "S3Bucket",
    "Download", "RESTAPI", "UIExport", "Email", "FeedAPI", "EnrichApp",
    "monthly", "weekly", "daily", "ondemand", "realtime", "csv", "json",
]
PARAMETERS = {
    "max_depth": 10,
    "min_samples_leaf": 10,
    "ccp_alpha": 0.0,
    "random_state": 20260813,
}


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae_log10_usd": float(mean_absolute_error(y_true, prediction)),
        "rmse_log10_usd": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "r2": float(r2_score(y_true, prediction)),
        "median_absolute_error_log10_usd": float(np.median(np.abs(y_true - prediction))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("financial", "telecom"), required=True)
    args = parser.parse_args()
    data = pd.read_csv(ROOT / "data" / f"{args.domain}.csv", low_memory=False)
    split = pd.read_csv(ROOT / "data" / f"{args.domain}_train_test_split.csv")
    frame = data.merge(split[["Id", "split"]], on="Id", validate="one_to_one")
    features = [column for column in EXPLICIT_FEATURES if column in frame.columns]
    train = frame[frame["split"] == "train"].copy()
    test = frame[frame["split"] == "test"].copy()
    model = DecisionTreeRegressor(**PARAMETERS)
    model.fit(train[features].fillna(0), train["LogPriceMo"])
    prediction = model.predict(test[features].fillna(0))
    model_dir = ROOT / "models" / args.domain
    result_dir = ROOT / "results" / args.domain
    work_dir = ROOT / "results" / "work" / f"{args.domain}_m0"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{args.domain}_m0_reproduced.joblib"
    joblib.dump({"model": model, "features": features, "parameters": PARAMETERS}, model_path)
    pd.DataFrame({
        "Id": test["Id"].astype(int),
        "LogPriceMo": test["LogPriceMo"].astype(float),
        "m0_prediction": prediction,
    }).to_csv(result_dir / "reproduced_m0_predictions.csv", index=False)
    oof = np.full(len(train), np.nan, dtype=float)
    fold_ids = np.full(len(train), -1, dtype=int)
    for fold, (fit_idx, holdout_idx) in enumerate(
        KFold(n_splits=5, shuffle=True, random_state=PARAMETERS["random_state"]).split(train)
    ):
        fold_model = DecisionTreeRegressor(**PARAMETERS)
        fold_model.fit(train.iloc[fit_idx][features].fillna(0), train.iloc[fit_idx]["LogPriceMo"])
        oof[holdout_idx] = fold_model.predict(train.iloc[holdout_idx][features].fillna(0))
        fold_ids[holdout_idx] = fold
    pd.DataFrame({
        "Id": train["Id"].astype(int),
        "m0_oof_fold": fold_ids,
        "m0_reference_log10_usd": oof,
        "m0_reference_usd": np.power(10.0, oof),
    }).to_csv(work_dir / "m0_train_oof_predictions.csv", index=False)
    pd.DataFrame({
        "Id": test["Id"].astype(int),
        "m0_prediction_log10_usd": prediction,
        "m0_prediction_usd": np.power(10.0, prediction),
    }).to_csv(work_dir / "m0_test_predictions.csv", index=False)
    (result_dir / "reproduced_m0_tree_rules.txt").write_text(
        export_text(model, feature_names=features), encoding="utf-8"
    )
    report = {
        "domain": args.domain,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": features,
        "parameters": PARAMETERS,
        "test_metrics": metrics(test["LogPriceMo"].to_numpy(float), prediction),
    }
    (result_dir / "reproduced_m0_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
