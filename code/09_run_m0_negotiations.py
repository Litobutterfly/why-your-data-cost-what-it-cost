from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd

from support import negotiation_engine as engine


ROOT = Path(__file__).resolve().parents[1]
BASE_OUTPUT_DIR = ROOT / "results" / "work" / "negotiation"
OUTPUT_DIR = Path(os.getenv("SYMBIOTRADE_OUTPUT_DIR", str(BASE_OUTPUT_DIR)))
SELECTION_PATH = Path(
    os.getenv("SYMBIOTRADE_SELECTION_PATH", str(BASE_OUTPUT_DIR / "m0_full_train_selection.csv"))
)
PATIENCE_PATH = BASE_OUTPUT_DIR / "train_frozen_patience_assignments.csv"
OUTPUT_STEM = os.getenv("SYMBIOTRADE_OUTPUT_STEM", "m0_full_train")
OUTPUT_VERSION = os.getenv("SYMBIOTRADE_OUTPUT_VERSION", "v3")
CHECKPOINT_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_dialogues_{OUTPUT_VERSION}.jsonl"
SUMMARY_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_dialogue_summary_{OUTPUT_VERSION}.csv"
MANIFEST_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_run_manifest_{OUTPUT_VERSION}.json"
ERROR_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_errors_{OUTPUT_VERSION}.jsonl"
RUN_VERSION = os.getenv(
    "SYMBIOTRADE_RUN_VERSION", "m0-full-train-v3-predecision-buyer-assessment"
)
M1_REAL_PRICE_COLUMN = os.getenv("SYMBIOTRADE_M1_REAL_PRICE_COLUMN", "").strip()
M1_PRICE_GATE_TOLERANCE = float(
    os.getenv(
        "SYMBIOTRADE_M1_PRICE_GATE_TOLERANCE",
        str(engine.DEFAULT_M1_PRICE_GATE_TOLERANCE),
    )
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_existing() -> dict[int, dict[str, Any]]:
    existing: dict[int, dict[str, Any]] = {}
    if not CHECKPOINT_PATH.exists():
        return existing
    for line_number, line in enumerate(
        CHECKPOINT_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        product_id = int(item["product_id"])
        if product_id in existing:
            raise RuntimeError(f"Duplicate product {product_id} at checkpoint line {line_number}")
        existing[product_id] = item
    return existing


def append_record(record: dict[str, Any], lock: Lock) -> None:
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with lock:
        with CHECKPOINT_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def append_error(record: dict[str, Any], lock: Lock) -> None:
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with lock:
        with ERROR_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def resolved_models(records: list[dict[str, Any]]) -> dict[str, int]:
    values = Counter()
    for record in records:
        for call in record.get("api_calls", []):
            for attempt in call.get("attempts", []):
                raw = attempt.get("raw_response", "")
                if not raw:
                    continue
                try:
                    model = json.loads(raw).get("model")
                except Exception:
                    model = None
                if model:
                    values[str(model)] += 1
    return dict(values)


def usage_totals(records: list[dict[str, Any]]) -> dict[str, int]:
    totals = Counter()
    for record in records:
        for call in record.get("api_calls", []):
            for attempt in call.get("attempts", []):
                usage = attempt.get("usage", {})
                for key in [
                    "prompt_tokens", "completion_tokens", "total_tokens",
                    "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
                ]:
                    totals[key] += int(usage.get(key, 0) or 0)
    return dict(totals)


def export_state(
    selected: pd.DataFrame,
    existing: dict[int, dict[str, Any]],
    prompts: Any,
    failed_product_ids: list[int] | None = None,
) -> dict[str, Any]:
    records = [existing[int(product_id)] for product_id in selected["product_id"] if int(product_id) in existing]
    pd.DataFrame(
        [
            {
                "product_id": item["product_id"],
                "condition": item["controlled_condition_id_audit_only"],
                "outcome": item["outcome"],
                "negotiated_price_usd": item["negotiated_price_usd"],
                "message_count": item["message_count"],
            }
            for item in records
        ]
    ).to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    manifest = {
        "status": (
            "completed" if len(records) == len(selected)
            else "completed_with_product_errors" if failed_product_ids
            else "running_checkpointed"
        ),
        "run_version": RUN_VERSION,
        "engine_run_version": engine.RUN_VERSION,
        "prompt_version": prompts.PROMPT_VERSION,
        "requested_model_alias": engine.MODEL,
        "resolved_model_response_counts": resolved_models(records),
        "requested_products": len(selected),
        "completed_products": len(records),
        "agreements": sum(item["outcome"] == "agreement" for item in records),
        "buyer_walkaways": sum(item["outcome"] == "buyer_walked_away" for item in records),
        "seller_walkaways": sum(item["outcome"] == "seller_walked_away" for item in records),
        "m1_price_gate_rejections": sum(
            item["outcome"] == "m1_price_gate_rejected" for item in records
        ),
        "minimum_messages": min((item["message_count"] for item in records), default=None),
        "maximum_messages": max((item["message_count"] for item in records), default=None),
        "message_limit_outcomes": sum(item["outcome"] == "message_limit" for item in records),
        "failed_product_count_this_run": len(failed_product_ids or []),
        "failed_product_ids_this_run": sorted(failed_product_ids or []),
        "usage": usage_totals(records),
        "real_transaction_price_visible_to_agents": False,
        "m1_real_price_gate_enabled": bool(M1_REAL_PRICE_COLUMN),
        "m1_real_price_column_private": M1_REAL_PRICE_COLUMN or None,
        "m1_price_gate_relative_error_tolerance": (
            M1_PRICE_GATE_TOLERANCE if M1_REAL_PRICE_COLUMN else None
        ),
        "buyer_received_quality_ledger": False,
        "test_products_used": False,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if CHECKPOINT_PATH.exists():
        manifest["checkpoint_sha256"] = sha256_file(CHECKPOINT_PATH)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def run(
    api_key: str,
    *,
    workers: int = 12,
    limit: int | None = None,
    timeout: float = 120.0,
    retries: int = 3,
    product_retries: int = 1,
) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("Missing key: set DEEPSEEK_API_KEY or AI_EDGE_API_KEY")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(SELECTION_PATH, low_memory=False)
    if M1_REAL_PRICE_COLUMN:
        if M1_REAL_PRICE_COLUMN not in selected:
            raise RuntimeError(
                f"M1 price gate column {M1_REAL_PRICE_COLUMN!r} is absent from the selection"
            )
        real_prices = pd.to_numeric(selected[M1_REAL_PRICE_COLUMN], errors="coerce")
        if real_prices.isna().any() or (real_prices <= 0).any():
            raise RuntimeError("M1 real-price gate column contains missing or non-positive values")
        if not 0 <= M1_PRICE_GATE_TOLERANCE < 1:
            raise RuntimeError("M1 price gate tolerance must be in [0, 1)")
    patience = pd.read_csv(PATIENCE_PATH, low_memory=False)
    selected = selected.merge(
        patience, on="product_id", validate="one_to_one", suffixes=("_selection", "_patience")
    )
    if limit is not None:
        selected = selected.head(limit).copy()
    if set(selected["product_id"]) & set(
        pd.read_csv(ROOT / "results" / "work" / "negotiation_prep" / "test_model_inputs.csv", usecols=["Id"])["Id"]
    ):
        raise RuntimeError("Full training negotiation selection includes held-out test products")
    existing = read_existing()
    selected_ids = set(selected["product_id"].astype(int))
    unexpected = set(existing) - selected_ids
    if unexpected and limit is None:
        raise RuntimeError(f"Checkpoint contains products outside the frozen selection: {sorted(unexpected)[:5]}")
    ledgers = engine.load_ledgers()
    prompts = engine.load_prompts()
    pending = [row for _, row in selected.iterrows() if int(row["product_id"]) not in existing]
    lock = Lock()

    def execute(row: pd.Series) -> dict[str, Any]:
        product_id = int(row["product_id"])
        errors = []
        for product_attempt in range(1, product_retries + 2):
            try:
                item = engine.scenario(
                    row,
                    quality_ledger=ledgers[product_id],
                    prompts=prompts,
                    api_key=api_key,
                    max_messages=engine.DEFAULT_MAX_MESSAGES,
                    timeout=timeout,
                    retries=retries,
                    patience_assignment=row,
                    real_transaction_price_usd=(
                        float(row[M1_REAL_PRICE_COLUMN]) if M1_REAL_PRICE_COLUMN else None
                    ),
                    m1_price_gate_tolerance=M1_PRICE_GATE_TOLERANCE,
                )
                item["run_version"] = RUN_VERSION
                item["product_attempt"] = product_attempt
                item["prior_product_attempt_errors"] = errors
                return item
            except Exception as exc:
                errors.append(
                    {
                        "product_attempt": product_attempt,
                        "error": repr(exc),
                        "occurred_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                if product_attempt <= product_retries:
                    time.sleep(min(10.0, 2.0 ** (product_attempt - 1)))
        raise RuntimeError(json.dumps({"product_id": product_id, "attempt_errors": errors}))

    completed_this_run = 0
    failed_product_ids: list[int] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(execute, row): int(row["product_id"]) for row in pending}
        for future in as_completed(futures):
            product_id = futures[future]
            try:
                item = future.result()
                append_record(item, lock)
                existing[product_id] = item
                completed_this_run += 1
            except Exception as exc:
                failed_product_ids.append(product_id)
                append_error(
                    {
                        "product_id": product_id,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(limit=8),
                        "occurred_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    lock,
                )
                print(f"product_error product={product_id} error={exc!r}", flush=True)
            finished_this_run = completed_this_run + len(failed_product_ids)
            if finished_this_run % 10 == 0 or finished_this_run == len(pending):
                manifest = export_state(selected, existing, prompts, failed_product_ids)
                print(
                    f"progress finished={finished_this_run}/{len(pending)} "
                    f"new={completed_this_run} failed={len(failed_product_ids)} "
                    f"total={manifest['completed_products']}/{len(selected)} "
                    f"agreements={manifest['agreements']}",
                    flush=True,
                )
    return export_state(selected, existing, prompts, failed_product_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--product-retries", type=int, default=1)
    args = parser.parse_args()
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("AI_EDGE_API_KEY", "")
    print(json.dumps(run(key, **vars(args)), ensure_ascii=False, indent=2))
