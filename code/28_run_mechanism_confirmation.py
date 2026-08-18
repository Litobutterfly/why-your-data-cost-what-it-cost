from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parent
RUNNER_PATH = STAGE / "support" / "run_rule_interventions.py"
PLAN = ROOT / "results" / "mechanisms" / "mechanism_confirmation_plan.csv"
PATIENCE = ROOT / "results" / "mechanisms" / "mechanism_confirmation_patience.csv"
RUNS = ROOT / "results" / "work" / "mechanism_confirmation"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if not PLAN.exists() or not PATIENCE.exists():
        raise RuntimeError("Generate the frozen mechanism confirmation plan first")
    runner = load_module("mechanism_confirmation_runner", RUNNER_PATH)
    runner.PATIENCE = PATIENCE
    runner.OUTPUT_DIR = RUNS
    result = runner.run(
        plan=PLAN,
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
        pairs_per_rule=None,
        output_stem="mechanism_confirmation_runs_v1",
    )
    result.update(
        {
            "protocol_status": "frozen_before_confirmation_plan_and_api_execution",
            "development_only": False,
            "thresholds_changed_after_results": False,
            "products_replaced_after_results": False,
        }
    )
    (RUNS / "mechanism_confirmation_runs_v1_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
