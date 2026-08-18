from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "financial.csv"
DEFAULT_SPLIT = ROOT / "data" / "financial_train_test_split.csv"
DEFAULT_OOF = ROOT / "results" / "work" / "financial_m0" / "m0_train_oof_predictions.csv"
DEFAULT_TEST = ROOT / "results" / "work" / "financial_m0" / "m0_test_predictions.csv"
DEFAULT_MODEL = ROOT / "models" / "financial" / "financial_m0_reproduced.joblib"
DEFAULT_OUTPUT = ROOT / "results" / "work" / "negotiation_prep"


def compact_value(value: float) -> int | float:
    """Keep generated prompt facts readable without changing source values."""
    value = float(value)
    if value.is_integer():
        return int(value)
    return round(value, 8)


def make_product_facts(row: pd.Series, features: list[str], word_features: list[str]) -> tuple[str, str]:
    explicit = {
        feature: compact_value(row[feature])
        for feature in features
        if float(row[feature]) != 0.0
    }
    lexical = {
        feature.removeprefix("word"): compact_value(row[feature])
        for feature in word_features
        if float(row[feature]) > 0.0
    }
    facts = {"explicit_fields_nonzero": explicit, "description_term_scores": lexical}
    facts_json = json.dumps(facts, ensure_ascii=True, sort_keys=True)

    explicit_text = ", ".join(f"{key}={value}" for key, value in explicit.items()) or "none"
    lexical_text = ", ".join(f"{key}={value}" for key, value in lexical.items()) or "none"
    facts_text = (
        f"Visible product fields: {explicit_text}. "
        f"Description-term signals from the source record: {lexical_text}."
    )
    return facts_json, facts_text


def run(
    data_path: Path,
    split_path: Path,
    oof_path: Path,
    test_path: Path,
    model_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(data_path, low_memory=False)
    split = pd.read_csv(split_path)
    oof = pd.read_csv(oof_path)
    test = pd.read_csv(test_path)
    bundle = joblib.load(model_path)
    features = list(bundle["features"])
    word_features = [column for column in data.columns if column.lower().startswith("word")]

    if len(data) not in {2390, 526} or data["Id"].duplicated().any():
        raise RuntimeError("Expected the released Financial or Telecom domain")
    if set(split["Id"]) != set(data["Id"]):
        raise RuntimeError("Split file does not cover the financial input exactly")
    train_ids = set(split.loc[split["split"].eq("train"), "Id"])
    test_ids = set(split.loc[split["split"].eq("test"), "Id"])
    expected_split = {2390: (1912, 478), 526: (420, 106)}[len(data)]
    if train_ids & test_ids or (len(train_ids), len(test_ids)) != expected_split:
        raise RuntimeError("Unexpected or overlapping train/test membership")
    if set(oof["Id"]) != train_ids or len(oof) != len(train_ids):
        raise RuntimeError("OOF file is not exactly the training set")
    if set(test["Id"]) != test_ids or len(test) != len(test_ids):
        raise RuntimeError("Test prediction file is not exactly the test set")

    base = data.set_index("Id")
    split_indexed = split.set_index("Id")
    facts_json: dict[int, str] = {}
    facts_text: dict[int, str] = {}
    for product_id, row in base.iterrows():
        facts_json[product_id], facts_text[product_id] = make_product_facts(
            row, features, word_features
        )

    # Negotiation inputs never contain the observed outcome. OOF references are
    # used for training products so each reference is produced without its own
    # target; test products carry the final M0 prediction only.
    train = base.loc[sorted(train_ids), features].copy().reset_index()
    train["split"] = "train"
    train = train.merge(
        oof[["Id", "m0_oof_fold", "m0_reference_log10_usd", "m0_reference_usd"]],
        on="Id",
        validate="one_to_one",
    )
    train["product_facts_json"] = train["Id"].map(facts_json)
    train["product_facts_text"] = train["Id"].map(facts_text)
    train = train.sort_values("Id")
    train.to_csv(output_dir / "train_negotiation_inputs.csv", index=False, encoding="utf-8-sig")

    test_model = base.loc[sorted(test_ids), features].copy().reset_index()
    test_model["split"] = "test"
    test_model = test_model.merge(
        test[["Id", "m0_prediction_log10_usd", "m0_prediction_usd"]],
        on="Id",
        validate="one_to_one",
    )
    test_model["product_facts_json"] = test_model["Id"].map(facts_json)
    test_model["product_facts_text"] = test_model["Id"].map(facts_text)
    test_model = test_model.sort_values("Id")
    test_model.to_csv(output_dir / "test_model_inputs.csv", index=False, encoding="utf-8-sig")

    # Outcomes are kept in a separate file and are only joined for final scoring.
    outcomes = base.loc[sorted(test_ids), ["LogPriceMo", "TransactionPriceUSD"]].copy().reset_index()
    outcomes = outcomes.merge(
        test[["Id", "m0_prediction_log10_usd", "m0_prediction_usd"]],
        on="Id",
        validate="one_to_one",
    ).sort_values("Id")
    outcomes.to_csv(output_dir / "test_outcomes_for_evaluation.csv", index=False, encoding="utf-8-sig")

    # A compact manifest makes the no-leakage boundary explicit for later stages.
    manifest = {
        "status": "prepared_without_api_calls",
        "source_rows": len(data),
        "train_rows": len(train),
        "test_rows": len(test_model),
        "explicit_features": features,
        "source_word_signal_count": len(word_features),
        "train_negotiation_contains_observed_outcome": False,
        "test_model_inputs_contain_observed_outcome": False,
        "test_outcomes_separate_from_model_inputs": True,
        "train_reference_protocol": "five-fold OOF M0 prediction",
        "test_reference_protocol": "final M0 fit on all training products",
        "files": {
            "train_negotiation_inputs": "train_negotiation_inputs.csv",
            "test_model_inputs": "test_model_inputs.csv",
            "test_outcomes": "test_outcomes_for_evaluation.csv",
        },
    }
    (output_dir / "negotiation_input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="data_path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", dest="split_path", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--oof", dest="oof_path", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--test", dest="test_path", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--model", dest="model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", dest="output_dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), ensure_ascii=False, indent=2))
