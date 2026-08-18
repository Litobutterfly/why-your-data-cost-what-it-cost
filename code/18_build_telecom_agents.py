from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parent
SOURCE = STAGE / "05_build_agent_profiles.py"
REGISTRY = ROOT / "protocols" / "persona_registry.json"
INPUTS = ROOT / "results" / "work" / "telecom" / "model_inputs"
DESCRIPTIONS = ROOT / "results" / "work" / "telecom" / "descriptions"
OUTPUT = ROOT / "results" / "work" / "telecom" / "agents"


def load_builder(name: str, expected_rows: int):
    source = SOURCE.read_text(encoding="utf-8")
    source = source.replace(
        'if len(inputs) != 1912 or len(descriptions) != 1912:',
        f'if len(inputs) != {expected_rows} or len(descriptions) != {expected_rows}:',
    )
    source = source.replace(
        'raise RuntimeError("Agent construction requires all 1,912 training products and descriptions")',
        f'raise RuntimeError("Agent construction requires all {expected_rows} products and descriptions")',
    )
    source = source.replace('len(output) == 1912', f'len(output) == {expected_rows}')
    source = source.replace(
        'output["product_id"].nunique() == 1912',
        f'output["product_id"].nunique() == {expected_rows}',
    )
    source = source.replace(
        'output["fixed_pair_id"].nunique() == 1912',
        f'output["fixed_pair_id"].nunique() == {expected_rows}',
    )
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    sys.modules[name] = module
    exec(compile(source, str(SOURCE), "exec"), module.__dict__)
    return module


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(INPUTS / "test_model_inputs.csv", low_memory=False)
    if {"LogPriceMo", "TransactionPriceUSD"} & set(test.columns):
        raise RuntimeError("Test agent input contains observed outcomes")
    test["m0_reference_log10_usd"] = test["m0_prediction_log10_usd"]
    test["m0_reference_usd"] = test["m0_prediction_usd"]
    test["m0_oof_fold"] = 0
    test_compatibility = OUTPUT / "test_agent_inputs.csv"
    test.to_csv(test_compatibility, index=False, encoding="utf-8-sig")

    reports = {}
    configurations = [
        (
            "train",
            420,
            INPUTS / "train_negotiation_inputs.csv",
            DESCRIPTIONS / "train" / "train_product_descriptions.csv",
        ),
        (
            "test",
            106,
            test_compatibility,
            DESCRIPTIONS / "test" / "train_product_descriptions.csv",
        ),
    ]
    for split, rows, input_path, description_path in configurations:
        builder = load_builder(f"symbiotrade_telecom_agents_{split}", rows)
        builder.INPUT_PATH = input_path
        builder.DESCRIPTION_PATH = description_path
        builder.REGISTRY_PATH = REGISTRY
        builder.OUTPUT_DIR = OUTPUT / split
        reports[split] = builder.run()
    payload = {
        "status": "telecom_agents_complete",
        "train_rows": reports["train"]["audit"]["rows"],
        "test_rows": reports["test"]["audit"]["rows"],
        "observed_transaction_outcome_used": False,
        "reports": reports,
    }
    (OUTPUT / "telecom_agent_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
