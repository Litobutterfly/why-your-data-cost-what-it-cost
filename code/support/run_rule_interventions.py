from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[1]
RUNNER_PATH = STAGE / "task_fit_runner.py"
DEFAULT_PLAN = ROOT / "results" / "causal_rules" / "extension_plan.csv"
PATIENCE = ROOT / "results" / "causal_rules" / "extension_patience.csv"
OUTPUT_DIR = ROOT / "results" / "work" / "rule_confirmation"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalize_checkpoint(path: Path) -> None:
    if not path.exists():
        return
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            item["contrast_version"] = "confirmed_vs_unresolved_rule_v1"
            records.append(item)
    path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            for item in records
        ),
        encoding="utf-8",
    )


def install_currency_aware_validator(runner: Any) -> None:
    original_loader = runner.load_v3_engine

    def as_cents(value: float) -> Decimal:
        return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def load_engine():
        engine = original_loader()
        original_validator = engine.validate_action

        def validate_action(actor: str, action: dict[str, Any], **state: Any):
            error = original_validator(actor, action, **state)
            if error == "buyer counter exceeds private ceiling":
                offer = action.get("offer_usd")
                ceiling = state.get("ceiling")
                if offer is not None and ceiling is not None and as_cents(offer) <= as_cents(ceiling):
                    return None
            if error != "the proposed price in message does not match offer_usd":
                return error
            offer = action.get("offer_usd")
            if action.get("action") != "counter" or offer is None:
                return error
            currency_amounts = [
                float(value.replace(",", ""))
                for value in re.findall(
                    r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
                    str(action.get("message", "")),
                )
            ]
            if any(
                abs(value - float(offer)) <= max(0.01, 0.001 * float(offer))
                for value in currency_amounts
            ):
                return None
            return error

        engine.validate_action = validate_action
        return engine

    runner.load_v3_engine = load_engine


def run(
    *,
    plan: Path,
    workers: int,
    timeout: float,
    retries: int,
    pairs_per_rule: int | None,
    output_stem: str,
) -> dict[str, Any]:
    if not PATIENCE.exists():
        raise RuntimeError("Generate the frozen intervention plan before running")
    runner = load_module("symbiotrade_supplementary_v3_runner", RUNNER_PATH)
    runner.PATIENCE = PATIENCE
    runner.OUTPUT_DIR = OUTPUT_DIR
    install_currency_aware_validator(runner)
    manifest = runner.run(
        plan_path=plan,
        workers=workers,
        timeout=timeout,
        retries=retries,
        pairs_per_stratum=pairs_per_rule,
        output_stem=output_stem,
    )
    checkpoint = OUTPUT_DIR / f"{output_stem}.jsonl"
    normalize_checkpoint(checkpoint)
    manifest.update(
        {
            "contrast_version": "confirmed_vs_unresolved_rule_v1",
            "patience_source": str(PATIENCE),
            "same_initial_patience_in_both_arms": True,
            "legacy_disclosure_patience_cost_disabled_in_both_arms": True,
            "currency_aware_price_text_validator": True,
        }
    )
    (OUTPUT_DIR / f"{output_stem}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--pairs-per-rule", type=int)
    parser.add_argument("--output-stem", default="supplementary_rule_runs_v1")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                plan=args.plan,
                workers=args.workers,
                timeout=args.timeout,
                retries=args.retries,
                pairs_per_rule=args.pairs_per_rule,
                output_stem=args.output_stem,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
