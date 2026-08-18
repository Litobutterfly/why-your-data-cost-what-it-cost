from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "work" / "negotiation_prep" / "train_negotiation_inputs.csv"
DEFAULT_OUTPUT = ROOT / "results" / "work" / "product_descriptions"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_ENDPOINT = "https://api.deepseek.com/chat/completions"
PROMPT_VERSION = "listing-v1-source-grounded"

FIELD_LABELS = {
    "History": "historical coverage field",
    "people": "people coverage field",
    "entities": "entity coverage field",
    "products": "product coverage field",
    "records": "record coverage field",
    "events": "event coverage field",
    "symbols": "financial-symbol coverage field",
    "assets": "asset coverage field",
    "requests": "request coverage field",
    "features": "feature coverage field",
    "locations": "location coverage field",
    "USD": "USD-related field",
    "sources": "source coverage field",
    "units": "unit coverage field",
    "Limitations": "limitations field",
    "ProfServices": "professional-services field",
    "IdIndividuals": "individual-identifier field",
    "IdCompanies": "company-identifier field",
    "NCountries": "country coverage field",
    "PercGDP": "GDP-coverage field",
    "DelMethod": "delivery-method field",
    "S3Bucket": "S3-bucket access field",
    "Download": "download access field",
    "RESTAPI": "REST API access field",
    "UIExport": "UI export field",
    "Email": "email delivery field",
    "FeedAPI": "feed API access field",
    "EnrichApp": "enrichment-application field",
    "monthly": "monthly update field",
    "weekly": "weekly update field",
    "daily": "daily update field",
    "ondemand": "on-demand update field",
    "realtime": "real-time update field",
    "csv": "CSV format field",
    "json": "JSON format field",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_prompt(row: pd.Series) -> str:
    facts = json.loads(row["product_facts_json"])
    explicit = facts.get("explicit_fields_nonzero", {})
    terms = facts.get("description_term_scores", {})
    labeled = {
        FIELD_LABELS.get(name, name): value for name, value in explicit.items()
    }
    payload = {
        "explicit_source_fields": labeled,
        "source_description_term_scores": terms,
    }
    payload_text = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return (
        "Write a short, factual English data-market listing for the product described by the source record below.\n"
        "Use only information present in the source record. The numeric values are normalized source-field scores; "
        "do not treat them as prices, counts, guarantees, quality ratings, legal claims, or performance results.\n"
        "Do not invent a product name, provider, industry, geography, time period, sample size, accuracy, privacy property, "
        "use case, license, or delivery guarantee. Do not mention an observed or predicted transaction price.\n"
        "Mention only the nonzero source fields that can be stated without guessing, and omit unsupported details. "
        "Use 2-3 clear sentences, no bullet list, no heading, no markdown, and no preface.\n\n"
        f"Source record (product ID {int(row['Id'])}):\n{payload_text}"
    )


def request_one(
    row: pd.Series,
    api_key: str,
    endpoint: str,
    model: str,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    product_id = int(row["Id"])
    prompt = build_prompt(row)
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful data-market catalog editor. Stay strictly grounded in the supplied source fields.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 180,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = ""
    for attempt in range(retries + 1):
        started = time.time()
        try:
            response = requests.post(endpoint, headers=headers, json=body, timeout=timeout)
            raw_text = response.text
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {raw_text[:500]}")
            payload = response.json()
            choices = payload.get("choices") or []
            if not choices or not choices[0].get("message", {}).get("content"):
                raise RuntimeError("API response did not contain choices[0].message.content")
            description = str(choices[0]["message"]["content"]).strip()
            return {
                "Id": product_id,
                "status": "ok",
                "description": description,
                "prompt_sha256": sha256_text(prompt),
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "raw_response": raw_text,
                "elapsed_seconds": round(time.time() - started, 3),
                "attempt": attempt + 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": "",
            }
        except Exception as exc:  # retain errors for resumable audit
            last_error = repr(exc)
            if attempt < retries:
                time.sleep(min(30.0, 2.0 ** attempt))
    return {
        "Id": product_id,
        "status": "error",
        "description": "",
        "prompt_sha256": sha256_text(prompt),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "raw_response": "",
        "elapsed_seconds": 0.0,
        "attempt": retries + 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "error": last_error,
    }


def append_jsonl(path: Path, record: dict[str, Any], lock: Lock) -> None:
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def run(
    input_path: Path,
    output_dir: Path,
    api_key: str,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    workers: int = 2,
    limit: int | None = None,
    timeout: float = 120.0,
    retries: int = 4,
) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("Missing API key; pass --api-key or set DEEPSEEK_API_KEY")
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_path.resolve()
    frame = pd.read_csv(input_path, low_memory=False)
    if "TransactionPriceUSD" in frame.columns or "LogPriceMo" in frame.columns:
        raise RuntimeError("Description input must not contain observed transaction outcomes")
    required = {"Id", "product_facts_json", "product_facts_text"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Description input missing required columns: {sorted(missing)}")
    if frame["Id"].duplicated().any():
        raise RuntimeError("Description input contains duplicate product IDs")
    if limit is not None:
        frame = frame.head(limit).copy()

    jsonl_path = output_dir / "train_product_descriptions.jsonl"
    existing: dict[int, dict[str, Any]] = {}
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                existing[int(record["Id"])] = record
    pending = [row for _, row in frame.iterrows() if int(row["Id"]) not in existing or existing[int(row["Id"])].get("status") != "ok"]
    lock = Lock()
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(request_one, row, api_key, endpoint, model, timeout, retries): int(row["Id"])
            for row in pending
        }
        for future in as_completed(futures):
            record = future.result()
            append_jsonl(jsonl_path, record, lock)
            existing[int(record["Id"])] = record
            completed += 1
            if completed % 25 == 0 or completed == len(pending):
                ok = sum(record.get("status") == "ok" for record in existing.values())
                print(f"progress={completed}/{len(pending)} ok_total={ok}", flush=True)

    selected = [existing[int(product_id)] for product_id in frame["Id"].tolist() if int(product_id) in existing]
    pd.DataFrame(selected).sort_values("Id").to_csv(
        output_dir / "train_product_descriptions.csv", index=False, encoding="utf-8-sig"
    )
    manifest = {
        "status": "completed" if all(record.get("status") == "ok" for record in selected) else "completed_with_errors",
        "input_rows_requested": len(frame),
        "records_saved": len(selected),
        "successful": sum(record.get("status") == "ok" for record in selected),
        "errors": sum(record.get("status") != "ok" for record in selected),
        "input_sha256": sha256_text(input_path.read_bytes().decode("utf-8", errors="replace")),
        "model": model,
        "endpoint": endpoint,
        "prompt_version": PROMPT_VERSION,
        "temperature": 0.0,
        "max_tokens": 180,
        "workers": workers,
        "test_rows_called": 0,
        "observed_outcome_in_input": False,
    }
    (output_dir / "description_generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input_path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", dest="output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), ensure_ascii=False, indent=2))
