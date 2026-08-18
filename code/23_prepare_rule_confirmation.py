"""Validate the frozen rule-confirmation plan before optional API reruns."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "results" / "causal_rules" / "extension_plan.csv"
PATIENCE = ROOT / "results" / "causal_rules" / "extension_patience.csv"
PROTOCOL = ROOT / "protocols" / "causal_rule_extension_protocol.json"


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    plan = pd.read_csv(PLAN, low_memory=False)
    patience = pd.read_csv(PATIENCE, low_memory=False)
    if protocol["status"] != "frozen_before_extension_plan_and_api_execution":
        raise RuntimeError("Rule-confirmation protocol is not frozen")
    if plan["run_id"].duplicated().any() or set(plan["arm"]) != {"control", "treatment"}:
        raise RuntimeError("Invalid frozen rule-confirmation plan")
    if set(plan["product_id"].astype(str)) != set(patience["product_id"].astype(str)):
        raise RuntimeError("Patience assignments do not cover every frozen product")
    if patience["product_id"].duplicated().any():
        raise RuntimeError("Patience assignments must contain one row per frozen product")
    print(json.dumps({"runs": len(plan), "pairs": plan["pair_id"].nunique(), "status": "ready"}, indent=2))


if __name__ == "__main__":
    main()
