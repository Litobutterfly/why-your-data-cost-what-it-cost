from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import types
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd


V3 = Path(__file__).resolve().parent
ROOT = V3.parents[1]
ENGINE_PATH = V3 / "negotiation_engine.py"
PROMPT_PATH = V3 / "task_fit_prompts.py"
PATIENCE = ROOT / "results" / "work" / "negotiation" / "train_frozen_patience_assignments.csv"
DEFAULT_PLAN = ROOT / "results" / "causal_rules" / "extension_plan.csv"
OUTPUT_DIR = ROOT / "results" / "work" / "rule_confirmation"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Expected exactly one frozen-engine block for {label}")
    return source.replace(old, new)


def load_v3_engine():
    source = ENGINE_PATH.read_text(encoding="utf-8")
    source = replace_once(
        source,
        'RUN_VERSION = "m0-bilateral-negotiation-v11-predecision-buyer-assessment"',
        'RUN_VERSION = "task-fit-dynamic-wtp-v3"',
        "run version",
    )
    source = replace_once(
        source,
        'if set(value) != {"action", "offer_usd", "message"}:\n'
        '        raise ValueError("Response must contain exactly action, offer_usd, message")',
        'required_keys = {"action", "offer_usd", "message"}\n'
        '    optional_keys = {"value_update"}\n'
        '    if not required_keys.issubset(value) or not set(value).issubset(required_keys | optional_keys):\n'
        '        raise ValueError("Response must contain action, offer_usd, message and optional value_update")\n'
        '    value.setdefault("value_update", None)',
        "action schema",
    )
    source = replace_once(
        source,
        '    if value["offer_usd"] is not None:\n'
        '        value["offer_usd"] = float(value["offer_usd"])\n'
        '    return value',
        '    if value["offer_usd"] is not None:\n'
        '        value["offer_usd"] = float(value["offer_usd"])\n'
        '    if value["value_update"] is not None:\n'
        '        value["value_update"] = float(value["value_update"])\n'
        '    return value',
        "value update parsing",
    )
    source = replace_once(
        source,
        '    kind = action["action"]\n    offer = action["offer_usd"]\n',
        '    kind = action["action"]\n'
        '    offer = action["offer_usd"]\n'
        '    value_update = action.get("value_update")\n'
        '    if actor == "buyer" and kind == "assess":\n'
        '        if value_update is None or not -1.0 <= float(value_update) <= 1.0:\n'
        '            return "buyer assessment requires value_update in [-1,1]"\n'
        '    elif value_update is not None:\n'
        '        return "value_update must be null outside the buyer assessment turn"\n',
        "value update validation",
    )
    source = replace_once(
        source,
        '"counter" not in state.get("required_actions", [])',
        '"counter" not in (state.get("required_actions") or [])',
        "nullable required actions",
    )
    source = replace_once(
        source,
        '    reference = float(row["m0_platform_reference_usd"])\n'
        '    ceiling = float(row["buyer_ceiling_usd"])\n'
        '    floor = float(row["seller_floor_usd"])',
        '    reference = float(row["m0_platform_reference_usd"])\n'
        '    pre_evidence_wtp = float(row["buyer_pre_evidence_wtp_usd"])\n'
        '    budget_cap = float(row["buyer_budget_cap_usd"])\n'
        '    ceiling = pre_evidence_wtp\n'
        '    floor = float(row["seller_floor_usd"])\n'
        '    value_update_private = None\n'
        '    valuation_trace_private = []',
        "valuation state initialization",
    )
    source = replace_once(
        source,
        '    if proposed_amount and kind == "accept":',
        '    conditional_counter_prices = re.findall(\n'
        '        r"\\$\\s*[0-9][0-9,]*(?:\\.[0-9]+)?", action["message"]\n'
        '    )\n'
        '    if (\n'
        '        kind == "counter"\n'
        '        and len(conditional_counter_prices) > 1\n'
        '        and re.search(r"\\bif\\b", action["message"], flags=re.IGNORECASE)\n'
        '        and re.search(r"\\botherwise\\b", action["message"], flags=re.IGNORECASE)\n'
        '    ):\n'
        '        return "counter must state one unconditional current offer"\n'
        '    current_counter_commitment = re.search(\n'
        '        r"\\b(?:i|we)\\s+can only proceed at\\b[^$\\d]{0,20}"\n'
        '        r"\\$?\\s*([0-9][0-9,]*(?:\\.\\d+)?)",\n'
        '        action["message"], flags=re.IGNORECASE,\n'
        '    )\n'
        '    if kind == "counter" and current_counter_commitment:\n'
        '        committed = float(current_counter_commitment.group(1).replace(",", ""))\n'
        '        if abs(committed - float(offer)) > max(0.01, 0.001 * float(offer)):\n'
        '            return "current counter commitment does not match offer_usd"\n'
        '    if proposed_amount and kind == "accept":',
        "conditional counter consistency",
    )
    source = replace_once(
        source,
        '        cost = 1\n',
        '        if actor == "buyer" and index == 3:\n'
        '            value_update_private = float(action["value_update"])\n'
        '            previous_ceiling = ceiling\n'
        '            ceiling = min(\n'
        '                budget_cap,\n'
        '                pre_evidence_wtp\n'
        '                * float(np.exp(np.log1p(0.15) * value_update_private)),\n'
        '            )\n'
        '            valuation_trace_private.append({\n'
        '                "message_index": index + 1,\n'
        '                "value_update": value_update_private,\n'
        '                "previous_wtp_usd": previous_ceiling,\n'
        '                "effective_wtp_usd": ceiling,\n'
        '                "budget_cap_usd": budget_cap,\n'
        '            })\n'
        '        cost = 1\n',
        "valuation state update",
    )
    source = replace_once(
        source,
        '        "buyer_ceiling_usd_private": ceiling,\n'
        '        "seller_floor_usd_private": floor,',
        '        "buyer_ceiling_usd_private": ceiling,\n'
        '        "buyer_pre_evidence_wtp_usd_private": pre_evidence_wtp,\n'
        '        "buyer_budget_cap_usd_private": budget_cap,\n'
        '        "buyer_value_update_private": value_update_private,\n'
        '        "valuation_trace_private": valuation_trace_private,\n'
        '        "seller_floor_usd_private": floor,',
        "valuation audit output",
    )
    module = types.ModuleType("symbiotrade_task_fit_v3_engine")
    module.__file__ = str(ENGINE_PATH)
    sys.modules[module.__name__] = module
    exec(compile(source, str(ENGINE_PATH), "exec"), module.__dict__)
    return module


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if item["run_id"] in records:
                raise RuntimeError(f"Duplicate run ID: {item['run_id']}")
            records[item["run_id"]] = item
    return records


def run(
    *,
    plan_path: Path,
    workers: int,
    timeout: float,
    retries: int,
    pairs_per_stratum: int | None,
    output_stem: str,
) -> dict[str, Any]:
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("AI_EDGE_API_KEY") or ""
    if not api_key:
        raise RuntimeError("Set DEEPSEEK_API_KEY or AI_EDGE_API_KEY")
    plan = pd.read_csv(plan_path, low_memory=False)
    if pairs_per_stratum is not None:
        selected = (
            plan.drop_duplicates("pair_id")
            .sort_values(["field_group", "task_fit_stratum_audit_only", "pair_id"])
            .groupby(["field_group", "task_fit_stratum_audit_only"], group_keys=False)
            .head(pairs_per_stratum)
        )
        plan = plan.loc[plan["pair_id"].isin(selected["pair_id"])].copy()
    plan = plan.sort_values(["pair_id", "order_index"])
    patience = pd.read_csv(PATIENCE, low_memory=False).set_index("product_id")
    engine = load_v3_engine()
    prompts = load_module("symbiotrade_task_fit_v3_prompts", PROMPT_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = OUTPUT_DIR / f"{output_stem}.jsonl"
    errors_path = OUTPUT_DIR / f"{output_stem}_errors.jsonl"
    existing = load_checkpoint(checkpoint)
    requested_ids = set(plan["run_id"])
    if set(existing) - requested_ids:
        raise RuntimeError("Checkpoint contains runs outside the selected plan")
    lock = Lock()

    def execute_pair(group: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        completed: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for _, row in group.sort_values("order_index").iterrows():
            run_id = str(row["run_id"])
            if run_id in existing:
                continue
            try:
                product_id = int(row["product_id"])
                patience_row = patience.loc[product_id]
                scenario_row = row.copy()
                scenario_row["release_slot"] = run_id
                scenario_row["buyer_profile_id_selection"] = row["buyer_profile_id"]
                scenario_row["seller_profile_id_selection"] = row["seller_profile_id"]
                scenario_row["observable_readiness_level_selection"] = patience_row[
                    "observable_readiness_level"
                ]
                scenario_row["potential_price_overlap"] = bool(
                    float(row["buyer_budget_cap_usd"]) >= float(row["seller_floor_usd"])
                )
                result = engine.scenario(
                    scenario_row,
                    quality_ledger=json.loads(row["quality_ledger_json"]),
                    prompts=prompts,
                    api_key=api_key,
                    max_messages=engine.DEFAULT_MAX_MESSAGES,
                    timeout=timeout,
                    retries=retries,
                    patience_assignment=patience_row,
                    real_transaction_price_usd=None,
                )
                completed.append(
                    {
                        "run_id": run_id,
                        "pair_id": str(row["pair_id"]),
                        "phase": str(row["phase"]),
                        "arm": str(row["arm"]),
                        "order_index": int(row["order_index"]),
                        "field_group": str(row["field_group"]),
                        "task_fit_stratum": str(row["task_fit_stratum_audit_only"]),
                        "expected_direction": str(row["expected_direction"]),
                        "non_intervention_context_hash": str(row["non_intervention_context_hash"]),
                        "contrast_version": "confirmed_vs_unresolved_task_fit",
                        **result,
                    }
                )
            except Exception as exc:
                errors.append(
                    {"run_id": run_id, "pair_id": str(row["pair_id"]), "error": repr(exc)}
                )
        return completed, errors

    grouped = [group for _, group in plan.groupby("pair_id", sort=True)]
    new_errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute_pair, group) for group in grouped]
        for future in as_completed(futures):
            completed, errors = future.result()
            with lock:
                for item in completed:
                    with checkpoint.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
                    existing[item["run_id"]] = item
                for error in errors:
                    with errors_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(error, ensure_ascii=False) + "\n")
                    new_errors.append(error)
    records = [existing[run_id] for run_id in plan["run_id"] if run_id in existing]
    usage = Counter()
    for item in records:
        for call in item.get("api_calls", []):
            for attempt in call.get("attempts", []):
                for key in [
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "prompt_cache_hit_tokens",
                    "prompt_cache_miss_tokens",
                ]:
                    usage[key] += int(attempt.get("usage", {}).get(key, 0) or 0)
    manifest = {
        "status": "completed" if len(records) == len(plan) else "incomplete",
        "requested_runs": len(plan),
        "completed_runs": len(records),
        "new_errors": len(new_errors),
        "agreements": sum(item.get("outcome") == "agreement" for item in records),
        "real_transaction_price_visible_to_agents": False,
        "real_price_gate_used": False,
        "same_engine_prompts_and_valuation_formula_in_both_arms": True,
        "plan_path": str(plan_path),
        "usage": dict(usage),
        "updated_unix_time": time.time(),
    }
    (OUTPUT_DIR / f"{output_stem}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--pairs-per-stratum", type=int)
    parser.add_argument("--output-stem", default="development_task_fit_runs_v1")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                plan_path=args.plan,
                workers=args.workers,
                timeout=args.timeout,
                retries=args.retries,
                pairs_per_stratum=args.pairs_per_stratum,
                output_stem=args.output_stem,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
