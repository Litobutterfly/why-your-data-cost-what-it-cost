from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
M0_OUTPUT = ROOT / "results" / "work" / "negotiation"
DEFAULT_MANIFEST = M0_OUTPUT / "CURRENT_DATASET_MANIFEST.json"
OUTPUT_DIR = ROOT / "results" / "work" / "implicit_features"
ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
PIPELINE_VERSION = "implicit-feature-release-v3-gated-scoring"
EXTRACT_PROMPT_VERSION = "implicit-extract-v2-public-dialogue-only"
GROUP_PROMPT_VERSION = "implicit-group-v2-product-property-only"
SCORE_PROMPT_VERSION = "implicit-score-v2-continuous-evidence-impact"
ADJUDICATE_PROMPT_VERSION = "implicit-adjudicate-v1-public-evidence-boundary"
DEFAULT_TOP_K = 12
DEFAULT_WORKERS = 6
GROUP_BATCH_SIZE = 48
DEFAULT_INPUT = M0_OUTPUT / "m0_full_train_public_dialogues_approved_v2.jsonl"
RUN_TAG = os.getenv("IMPLICIT_RUN_TAG", "release")


def output_path(stem: str, suffix: str) -> Path:
    return OUTPUT_DIR / f"{RUN_TAG}_{stem}_{suffix}"

CURRENCY_PRICE_PATTERN = re.compile(
    r"(?ix)(?:\$\s*\d[\d,]*(?:\.\d+)?|\bUSD\s*\d[\d,]*(?:\.\d+)?|"
    r"\b\d[\d,]*(?:\.\d+)?\s*(?:USD|dollars?)\b)"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_usage(total: dict[str, Any], usage: dict[str, Any]) -> None:
    """Recursively add DeepSeek token counters, including cache detail objects."""
    for key, value in usage.items():
        if isinstance(value, dict):
            nested = total.setdefault(key, {})
            if not isinstance(nested, dict):
                raise TypeError(f"Usage field changed type: {key}")
            add_usage(nested, value)
        elif isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: dict[str, Any], lock: Lock | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def write() -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
    if lock is None:
        write()
    else:
        with lock:
            write()


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in model response")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    return value


def call_json(
    api_key: str,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    last_error = ""
    for attempt in range(1, retries + 2):
        started = time.time()
        raw = ""
        try:
            response = requests.post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt + " Return a valid json object and no surrounding prose.",
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                    "stream": False,
                },
                timeout=timeout,
            )
            raw = response.text
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {raw[:500]}")
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            value = parse_json_object(content)
            return value, {
                "attempt": attempt,
                "elapsed_seconds": round(time.time() - started, 3),
                "usage": payload.get("usage", {}),
                "response_sha256": sha256_text(content),
            }
        except Exception as exc:
            last_error = repr(exc)
            if attempt <= retries:
                time.sleep(min(12.0, 2.0 ** (attempt - 1)))
    raise RuntimeError(f"DeepSeek request failed after {retries + 1} attempts: {last_error}")


def redact_prices(text: str, known_offers: list[float]) -> str:
    text = CURRENCY_PRICE_PATTERN.sub(" [PRICE] ", text)
    variants = set()
    for offer in known_offers:
        variants.update({f"{offer:g}", f"{offer:.1f}", f"{offer:.2f}"})
    for value in sorted(variants, key=len, reverse=True):
        text = re.sub(rf"(?<![A-Za-z0-9.]){re.escape(value)}(?![A-Za-z0-9.])", " [PRICE] ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalized_evidence_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def quote_is_grounded(quote: str, message: str) -> bool:
    quote_tokens = normalized_evidence_text(quote).split()
    message_tokens = normalized_evidence_text(message).split()
    if len(quote_tokens) < 2 or len(quote_tokens) > len(message_tokens):
        return False
    width = len(quote_tokens)
    return any(message_tokens[index:index + width] == quote_tokens for index in range(len(message_tokens) - width + 1))


def align_evidence_quote(quote: str, message: str) -> str | None:
    """Return an exact source span when a model quote is exact or a near-verbatim typo."""
    if re.search(r"(?:\.{3,}|…|\[\s*\.{3}\s*\])", quote):
        segments = [
            segment.strip(" \t\r\n\"'")
            for segment in re.split(r"(?:\.{3,}|…|\[\s*\.{3}\s*\])", quote)
        ]
        segments.sort(key=lambda value: len(normalized_evidence_text(value).split()), reverse=True)
        for segment in segments:
            if len(normalized_evidence_text(segment).split()) < 2:
                continue
            aligned_segment = align_evidence_quote(segment, message)
            if aligned_segment is not None:
                return aligned_segment
    sentence_segments = [
        segment.strip(" \t\r\n\"'")
        for segment in re.split(r"(?<=[.!?])\s+", quote)
        if segment.strip()
    ]
    if len(sentence_segments) > 1:
        sentence_segments.sort(
            key=lambda value: len(normalized_evidence_text(value).split()), reverse=True
        )
        for segment in sentence_segments:
            if len(normalized_evidence_text(segment).split()) < 2:
                continue
            aligned_segment = align_evidence_quote(segment, message)
            if aligned_segment is not None:
                return aligned_segment
    token_matches = list(re.finditer(r"[A-Za-z0-9]+", unicodedata.normalize("NFKC", message)))
    message_tokens = [match.group(0).lower() for match in token_matches]
    quote_tokens = normalized_evidence_text(quote).split()
    if len(quote_tokens) < 2 or not message_tokens:
        return None
    best: tuple[float, int, int] | None = None
    minimum = max(2, len(quote_tokens) - 2)
    maximum = min(len(message_tokens), len(quote_tokens) + 2)
    for width in range(minimum, maximum + 1):
        for start in range(0, len(message_tokens) - width + 1):
            candidate = message_tokens[start:start + width]
            score = difflib.SequenceMatcher(None, quote_tokens, candidate).ratio()
            if best is None or score > best[0]:
                best = (score, start, start + width)
    if best is None or best[0] < 0.88:
        return None
    _, start, end = best
    normalized_message = unicodedata.normalize("NFKC", message)
    return normalized_message[token_matches[start].start():token_matches[end - 1].end()]


def public_model_packet(record: dict[str, Any]) -> dict[str, Any]:
    packet = record.get("public_product_packet", {})
    listing = {
        "platform_listing": packet.get("platform_listing", ""),
        "listed_delivery_channels": packet.get("delivery_channels_listed", []),
        "listed_formats": packet.get("formats_listed", []),
        "listed_update_options": packet.get("update_options_listed", []),
        "company_identifiers_indicated": packet.get("company_identifiers_indicated", False),
        "individual_identifiers_indicated": packet.get("individual_identifiers_indicated", False),
        "countries_covered": packet.get("countries_covered", 0),
        "use_limitations_flag_indicated": packet.get("use_limitations_flag_indicated", False),
        "professional_services_indicated": packet.get("professional_services_indicated", False),
    }
    known_offers = [
        float(item["offer_usd"])
        for item in record.get("dialogue", [])
        if item.get("offer_usd") is not None
    ]
    dialogue = [
        {
            "message_index": int(item["message_index"]),
            "actor": item["actor"],
            "action": item["action"],
            "message": redact_prices(str(item["message"]), known_offers),
        }
        for item in record.get("dialogue", [])
    ]
    return {"product_id": int(record["product_id"]), "platform_listing": listing, "dialogue": dialogue}


def extraction_prompt(packet: dict[str, Any]) -> str:
    return (
        "Extract price-relevant implicit PRODUCT features from one public data-market negotiation.\n\n"
        "An admissible implicit feature is a property of the data product, its evidence, or its delivery that is "
        "disclosed or strongly evidenced in the dialogue but is not a direct restatement of the platform listing. "
        "It must explain why a buyer might raise or lower valuation, verification effort, integration cost, usable "
        "scope, or operational risk.\n\n"
        "Exclude buyer/seller identity or persona, budget, reservation price, patience, bargaining style, willingness "
        "to concede, the platform reference price, offers, and the final transaction outcome. Do not infer a defect "
        "from a missing platform field alone. Do not turn a buyer's unsupported concern into a product fact. Do not "
        "repeat listed delivery channel, format, update option, identifiers, or country count as a new feature.\n\n"
        "Return zero to three candidates. Returning an empty list is correct when evidence is insufficient. Every "
        "candidate must have a concise reusable noun-phrase name, a product-level definition, direction "
        "(upward/downward/conditional/unclear), a concrete price mechanism, extraction_confidence from 0.00 to 1.00, "
        "negotiation_impact from 0.00 to 1.00, and one or more evidence objects containing a valid message_index and "
        "a short verbatim quote. Quotes must appear in the supplied dialogue.\n\n"
        "Return exactly this JSON shape: "
        '{"features":[{"name":"...","definition":"...","direction":"downward",'
        '"mechanism":"...","extraction_confidence":0.83,"negotiation_impact":0.71,'
        '"evidence":[{"message_index":3,"quote":"..."}]}]}\n\n'
        "INPUT:\n" + json.dumps(packet, ensure_ascii=False, sort_keys=True)
    )


def validate_extraction(value: dict[str, Any], packet: dict[str, Any]) -> list[dict[str, Any]]:
    features = value.get("features")
    if not isinstance(features, list) or len(features) > 3:
        raise ValueError("features must be a list with no more than three items")
    message_by_index = {item["message_index"]: item["message"] for item in packet["dialogue"]}
    valid_directions = {"upward", "downward", "conditional", "unclear"}
    result = []
    for item in features:
        required = {
            "name", "definition", "direction", "mechanism", "extraction_confidence",
            "negotiation_impact", "evidence",
        }
        if set(item) != required:
            raise ValueError(f"candidate fields differ from required schema: {set(item)}")
        if item["direction"] not in valid_directions:
            raise ValueError("invalid direction")
        for key in ["extraction_confidence", "negotiation_impact"]:
            item[key] = float(item[key])
            if not 0 <= item[key] <= 1:
                raise ValueError(f"{key} outside 0-1")
        if not item["evidence"]:
            raise ValueError("candidate has no evidence")
        for evidence in item["evidence"]:
            index = int(evidence["message_index"])
            quote = str(evidence["quote"]).strip()
            if index not in message_by_index:
                raise ValueError(f"unverifiable quote at message {index}: {quote!r}")
            aligned = align_evidence_quote(quote, message_by_index[index])
            if aligned is None:
                raise ValueError(f"unverifiable quote at message {index}: {quote!r}")
            evidence["message_index"] = index
            evidence["quote"] = aligned
        result.append(item)
    return result


def load_existing_by_product(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    return {int(item["product_id"]): item for item in read_jsonl(path) if item.get("status") == "ok"}


def extract_candidates(
    records: list[dict[str, Any]], api_key: str, *, timeout: float, retries: int,
    workers: int = DEFAULT_WORKERS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = output_path("candidate_features", "v2.jsonl")
    existing = load_existing_by_product(output)
    usage: dict[str, Any] = {}
    lock = Lock()
    pending = [record for record in records if int(record["product_id"]) not in existing]

    def extract_one(record: dict[str, Any]) -> dict[str, Any]:
        product_id = int(record["product_id"])
        packet = public_model_packet(record)
        prompt = extraction_prompt(packet)
        features, meta = call_and_validate(
            api_key,
            system_prompt=(
                "You are an evidence-constrained data-market research coder. Extract only product properties grounded "
                "in public dialogue. It is better to return no feature than to invent one."
            ),
            user_prompt=prompt,
            max_tokens=900,
            validator=lambda value, p=packet: validate_extraction(value, p),
            timeout=timeout,
            retries=retries,
        )
        saved = {
            "product_id": product_id,
            "status": "ok",
            "features": features,
            "feature_count": len(features),
            "prompt_sha256": sha256_text(prompt),
            "prompt_version": EXTRACT_PROMPT_VERSION,
            "model": MODEL,
            "created_at_utc": utc_now(),
            "request_meta": meta,
        }
        return saved

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(extract_one, record): int(record["product_id"]) for record in pending}
        for future in as_completed(futures):
            saved = future.result()
            product_id = int(saved["product_id"])
            append_jsonl(output, saved, lock)
            existing[product_id] = saved
            add_usage(usage, saved["request_meta"].get("usage", {}))
            completed += 1
            if completed % 25 == 0 or completed == len(pending):
                print(
                    f"extract progress new={completed}/{len(pending)} total={len(existing)}/{len(records)}",
                    flush=True,
                )
    ordered = [existing[int(record["product_id"])] for record in records]
    return ordered, usage


def flatten_candidates(extractions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for extraction in extractions:
        for index, feature in enumerate(extraction["features"], start=1):
            rows.append(
                {
                    "candidate_id": f"p{int(extraction['product_id'])}_f{index}",
                    "product_id": int(extraction["product_id"]),
                    **feature,
                }
            )
    return rows


def adjudication_prompt(packet: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    compact = [
        {
            "candidate_id": item["candidate_id"],
            "name": item["name"],
            "definition": item["definition"],
            "direction": item["direction"],
            "mechanism": item["mechanism"],
            "extraction_confidence": item["extraction_confidence"],
            "negotiation_impact": item["negotiation_impact"],
            "evidence": item["evidence"],
        }
        for item in candidates
    ]
    return (
        "Independently adjudicate proposed implicit data-product features against the public platform listing and "
        "negotiation evidence. Each candidate must receive exactly one decision.\n\n"
        "KEEP only a product or delivery property disclosed by the seller or jointly established in the dialogue, "
        "not already explicit in the platform listing, with a clear valuation, verification, integration, usability, "
        "or operational mechanism. REJECT direct listing restatements; buyer/seller personas, budgets, patience, "
        "offers or bargaining style; buyer-only speculation; generic absence of certification; and statements that "
        "there was no additional recent internal quality observation.\n\n"
        "A missing document is an evidence-availability feature only when the dialogue makes it relevant to valuation. "
        "Do not rewrite 'not documented/not supplied' as proof that the underlying property does not exist. Correct "
        "overstated names or definitions before keeping them. Do not use the final transaction outcome as evidence.\n\n"
        "For a kept item, return a corrected_candidate with exactly name, definition, direction, mechanism, "
        "extraction_confidence, negotiation_impact, and evidence. Evidence quotes must remain verbatim. For a rejected "
        "item, corrected_candidate must be null. Return exactly: "
        '{"decisions":[{"candidate_id":"p1_f1","decision":"keep","reason":"...",'
        '"corrected_candidate":{"name":"...","definition":"...","direction":"downward",'
        '"mechanism":"...","extraction_confidence":0.82,"negotiation_impact":0.73,'
        '"evidence":[{"message_index":3,"quote":"..."}]}}]}\n\n'
        "INPUT PACKET:\n" + json.dumps(packet, ensure_ascii=False, sort_keys=True)
        + "\n\nCANDIDATES:\n" + json.dumps(compact, ensure_ascii=False, sort_keys=True)
    )


def validate_adjudication(
    value: dict[str, Any], packet: dict[str, Any], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    decisions = value.get("decisions")
    expected = {item["candidate_id"] for item in candidates}
    if not isinstance(decisions, list):
        raise ValueError("decisions must be a list")
    ids = [item.get("candidate_id") for item in decisions]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("adjudication must assign every candidate exactly once")
    message_by_index = {item["message_index"]: item["message"] for item in packet["dialogue"]}
    original_by_id = {item["candidate_id"]: item for item in candidates}
    output = []
    for item in decisions:
        if set(item) != {"candidate_id", "decision", "reason", "corrected_candidate"}:
            raise ValueError("adjudication decision schema mismatch")
        if item["decision"] not in {"keep", "reject"}:
            raise ValueError("adjudication decision must be keep or reject")
        if not str(item["reason"]).strip():
            raise ValueError("adjudication reason must be nonempty")
        corrected = item["corrected_candidate"]
        if item["decision"] == "reject":
            if corrected is not None:
                raise ValueError("rejected candidate must have corrected_candidate=null")
        else:
            if not isinstance(corrected, dict):
                raise ValueError("kept candidate requires corrected_candidate")
            required = {
                "name", "definition", "direction", "mechanism", "extraction_confidence",
                "negotiation_impact", "evidence",
            }
            if set(corrected) != required:
                raise ValueError("corrected candidate schema mismatch")
            if corrected["direction"] not in {"upward", "downward", "conditional", "unclear"}:
                raise ValueError("invalid corrected direction")
            for key in ["extraction_confidence", "negotiation_impact"]:
                corrected[key] = float(corrected[key])
                if not 0 <= corrected[key] <= 1:
                    raise ValueError(f"corrected {key} outside 0-1")
            if not corrected["evidence"]:
                raise ValueError("kept candidate requires evidence")
            for evidence in corrected["evidence"]:
                index = int(evidence["message_index"])
                quote = str(evidence["quote"]).strip()
                if index not in message_by_index:
                    raise ValueError(f"unverifiable adjudication quote at message {index}: {quote!r}")
                aligned = align_evidence_quote(quote, message_by_index[index])
                if aligned is None:
                    raise ValueError(f"unverifiable adjudication quote at message {index}: {quote!r}")
                evidence["message_index"] = index
                evidence["quote"] = aligned
        output.append(
            {
                "candidate_id": item["candidate_id"],
                "product_id": int(original_by_id[item["candidate_id"]]["product_id"]),
                "decision": item["decision"],
                "reason": str(item["reason"]).strip(),
                "corrected_candidate": corrected,
            }
        )
    return output


def call_and_validate(
    api_key: str,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    validator: Any,
    timeout: float,
    retries: int,
) -> tuple[Any, dict[str, Any]]:
    errors = []
    total_usage: dict[str, Any] = {}
    for validation_attempt in range(1, retries + 2):
        feedback = ""
        if errors:
            feedback = (
                "\n\nYour previous response failed validation: " + errors[-1]
                + ". Return a complete corrected JSON object; do not omit or duplicate any required ID."
            )
        value, meta = call_json(
            api_key,
            system_prompt=system_prompt,
            user_prompt=user_prompt + feedback,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=1,
        )
        add_usage(total_usage, meta.get("usage", {}))
        try:
            validated = validator(value)
            meta["validation_attempt"] = validation_attempt
            meta["validation_errors"] = errors
            meta["cumulative_usage"] = total_usage
            return validated, meta
        except Exception as exc:
            errors.append(repr(exc))
    raise RuntimeError(f"Model output failed semantic validation: {errors}")


def adjudicate_candidates(
    records: list[dict[str, Any]], candidates: list[dict[str, Any]], api_key: str,
    *, timeout: float, retries: int, workers: int = DEFAULT_WORKERS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    output = output_path("candidate_adjudication", "v2.jsonl")
    existing = load_existing_by_product(output)
    candidates_by_product: dict[int, list[dict[str, Any]]] = {}
    for item in candidates:
        candidates_by_product.setdefault(int(item["product_id"]), []).append(item)
    usage: dict[str, Any] = {}
    record_by_id = {int(item["product_id"]): item for item in records}
    lock = Lock()
    pending_ids = [product_id for product_id in record_by_id if product_id not in existing]

    def adjudicate_one(product_id: int) -> dict[str, Any]:
        product_candidates = candidates_by_product.get(product_id, [])
        packet = public_model_packet(record_by_id[product_id])
        if not product_candidates:
            saved = {
                "product_id": product_id, "status": "ok", "decisions": [],
                "prompt_version": ADJUDICATE_PROMPT_VERSION, "model": MODEL,
                "created_at_utc": utc_now(), "request_meta": {"skipped_no_candidates": True},
            }
        else:
            prompt = adjudication_prompt(packet, product_candidates)
            decisions, meta = call_and_validate(
                api_key,
                system_prompt=(
                    "You are an independent evidence auditor for data-market research. Reject leakage, restatements, "
                    "unsupported conjecture, and overclaiming even when it would make results look stronger."
                ),
                user_prompt=prompt,
                max_tokens=1500,
                validator=lambda value, p=packet, c=product_candidates: validate_adjudication(value, p, c),
                timeout=timeout,
                retries=retries,
            )
            saved = {
                "product_id": product_id, "status": "ok", "decisions": decisions,
                "prompt_sha256": sha256_text(prompt), "prompt_version": ADJUDICATE_PROMPT_VERSION,
                "model": MODEL, "created_at_utc": utc_now(), "request_meta": meta,
            }
        return saved

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(adjudicate_one, product_id): product_id for product_id in pending_ids}
        for future in as_completed(futures):
            saved = future.result()
            product_id = int(saved["product_id"])
            append_jsonl(output, saved, lock)
            existing[product_id] = saved
            meta = saved.get("request_meta", {})
            add_usage(usage, meta.get("cumulative_usage", meta.get("usage", {})))
            completed += 1
            if completed % 25 == 0 or completed == len(pending_ids):
                print(
                    f"adjudicate progress new={completed}/{len(pending_ids)} "
                    f"total={len(existing)}/{len(record_by_id)}",
                    flush=True,
                )
    ordered_records = [existing[int(record["product_id"])] for record in records]
    accepted = []
    for record in ordered_records:
        for decision in record["decisions"]:
            if decision["decision"] == "keep":
                accepted.append(
                    {
                        "candidate_id": decision["candidate_id"],
                        "product_id": decision["product_id"],
                        **decision["corrected_candidate"],
                    }
                )
    return ordered_records, accepted, usage


def deterministic_rejection_reason(
    candidate: dict[str, Any], packet: dict[str, Any]
) -> str | None:
    """Enforce explicit/implicit boundaries that an LLM adjudicator may miss."""
    text = normalized_evidence_text(
        f"{candidate.get('name', '')} {candidate.get('definition', '')}"
    )
    listing = packet["platform_listing"]
    platform_text = normalized_evidence_text(listing.get("platform_listing", ""))

    if not listing.get("listed_update_options"):
        update_restatements = [
            "absence of update schedule", "no update schedule", "update schedule not",
            "update frequency not", "absence of update frequency",
        ]
        if any(term in text for term in update_restatements):
            return "direct restatement of the platform listing's explicit update-schedule absence"
    if not listing.get("listed_formats"):
        format_restatements = [
            "absence of format", "no format", "format specification", "format details",
            "csv or json", "schema compatibility",
        ]
        if any(term in text for term in format_restatements):
            return "direct restatement of the platform listing's explicit format absence"
    generic_no_observation = [
        "absence of internal quality observations", "no additional internal quality observation",
        "absence of additional quality evidence", "no additional quality evidence",
        "additional quality evidence", "no additional recent internal",
        "evidence limited to listing", "evidence is limited to the platform listing",
        "documented evidence boundaries", "limited to the platform listing and structured fields",
    ]
    if any(term in text for term in generic_no_observation):
        return "generic absence of an additional internal observation is not a product property"
    if "no update schedule is listed" in platform_text and "latency commitment" not in text:
        if "update schedule" in text or "update frequency" in text:
            return "update-schedule availability is already explicit in the public listing"
    return None


def apply_deterministic_gate(
    records: list[dict[str, Any]],
    adjudications: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    record_by_id = {int(item["product_id"]): item for item in records}
    accepted_by_id = {item["candidate_id"]: item for item in accepted}
    audit_rows = []
    final = []
    for adjudication in adjudications:
        product_id = int(adjudication["product_id"])
        packet = public_model_packet(record_by_id[product_id])
        for decision in adjudication["decisions"]:
            candidate_id = decision["candidate_id"]
            candidate = accepted_by_id.get(candidate_id)
            reason = deterministic_rejection_reason(candidate, packet) if candidate else None
            final_decision = "reject" if decision["decision"] == "reject" or reason else "keep"
            audit_rows.append(
                {
                    "candidate_id": candidate_id,
                    "product_id": product_id,
                    "llm_decision": decision["decision"],
                    "deterministic_decision": final_decision,
                    "deterministic_reason": reason or "",
                }
            )
            if final_decision == "keep" and candidate is not None:
                final.append(candidate)
    pd.DataFrame(audit_rows).to_csv(
        output_path("candidate_deterministic_gate", "v3.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return final, audit_rows


def grouping_prompt(candidates: list[dict[str, Any]]) -> str:
    compact = [
        {
            "candidate_id": item["candidate_id"],
            "product_id": item["product_id"],
            "name": item["name"],
            "definition": item["definition"],
            "direction": item["direction"],
            "mechanism": item["mechanism"],
        }
        for item in candidates
    ]
    return (
        "Group semantically equivalent implicit product-feature candidates into a canonical taxonomy. Merge only "
        "when candidates describe the same underlying product property and price mechanism. Keep materially distinct "
        "properties separate. Canonical names must be short product-property noun phrases, not buyer behavior, seller "
        "behavior, price movements, or negotiation outcomes.\n\n"
        f"There are exactly {len(compact)} candidate IDs. Before returning, verify that every candidate_id appears "
        "exactly once, with no omission and no duplicate. Return exactly: "
        '{"groups":[{"canonical_id":"C01","canonical_name":"...","definition":"...",'
        '"price_mechanism":"...","candidate_ids":["p1_f1"]}]}\n\n'
        "CANDIDATES:\n" + json.dumps(compact, ensure_ascii=False, sort_keys=True)
    )


def validate_groups(value: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = value.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups must be a nonempty list")
    expected = {item["candidate_id"] for item in candidates}
    candidate_by_id = {item["candidate_id"]: item for item in candidates}
    seen: set[str] = set()
    repaired = []
    for group in groups:
        required = {"canonical_id", "canonical_name", "definition", "price_mechanism", "candidate_ids"}
        if set(group) != required:
            raise ValueError("group schema mismatch")
        valid_ids = []
        for candidate_id in group["candidate_ids"]:
            if candidate_id in expected and candidate_id not in seen:
                valid_ids.append(candidate_id)
                seen.add(candidate_id)
        if valid_ids:
            repaired.append({**group, "candidate_ids": valid_ids})

    # An omitted ID is kept as a singleton rather than being guessed into a group.
    # This is conservative: later hierarchy levels may still merge it semantically.
    for candidate_id in sorted(expected - seen):
        candidate = candidate_by_id[candidate_id]
        repaired.append(
            {
                "canonical_id": "",
                "canonical_name": candidate["name"],
                "definition": candidate["definition"],
                "price_mechanism": candidate["mechanism"],
                "candidate_ids": [candidate_id],
            }
        )
    for index, group in enumerate(repaired, start=1):
        group["canonical_id"] = f"C{index:02d}"
    return repaired


def group_and_rank(
    candidates: list[dict[str, Any]], api_key: str, *, top_k: int, timeout: float, retries: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    taxonomy_path = output_path("canonical_taxonomy", "v3.json")
    meta: dict[str, Any] = {}
    if taxonomy_path.exists():
        taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        groups = taxonomy["groups"]
    else:
        if not candidates:
            raise RuntimeError("No candidates were extracted; taxonomy cannot be formed")
        usage: dict[str, Any] = {}
        request_count = 0

        def group_batch(batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
            prompt = grouping_prompt(batch)
            batch_groups, request_meta = call_and_validate(
                api_key,
                system_prompt=(
                    "You build conservative, auditable taxonomies for data-market product properties. Do not merge "
                    "merely because two features both lower price."
                ),
                user_prompt=prompt,
                max_tokens=5000,
                validator=lambda value, b=batch: validate_groups(value, b),
                timeout=timeout,
                retries=retries,
            )
            return batch_groups, request_meta, sha256_text(prompt)

        # First group bounded batches of original candidates. Each completed batch is
        # checkpointed independently so a later interruption does not repeat paid calls.
        checkpoint_path = output_path("grouping_batches", "v4.jsonl")
        checkpoints = {
            (int(item["round"]), int(item["batch_index"])): item
            for item in read_jsonl(checkpoint_path)
        } if checkpoint_path.exists() else {}
        current_nodes: list[dict[str, Any]] = []
        ordered_candidates = sorted(
            candidates,
            key=lambda item: normalized_evidence_text(
                f"{item.get('name', '')} {item.get('definition', '')}"
            ),
        )
        original_batches = [
            ordered_candidates[index:index + GROUP_BATCH_SIZE]
            for index in range(0, len(ordered_candidates), GROUP_BATCH_SIZE)
        ]
        for batch_index, batch in enumerate(original_batches):
            key = (0, batch_index)
            source_hash = sha256_text("\n".join(item["candidate_id"] for item in batch))
            saved = checkpoints.get(key)
            if saved is None or saved.get("source_ids_sha256") != source_hash:
                batch_groups, request_meta, prompt_hash = group_batch(batch)
                saved = {
                    "round": 0,
                    "batch_index": batch_index,
                    "source_ids_sha256": source_hash,
                    "groups": batch_groups,
                    "prompt_sha256": prompt_hash,
                    "request_meta": request_meta,
                    "created_at_utc": utc_now(),
                }
                append_jsonl(checkpoint_path, saved)
                checkpoints[key] = saved
                add_usage(usage, request_meta.get("cumulative_usage", request_meta.get("usage", {})))
                request_count += 1
            for local_index, group in enumerate(saved["groups"]):
                current_nodes.append({
                    "node_id": f"R00B{batch_index:04d}G{local_index:04d}",
                    "canonical_name": group["canonical_name"],
                    "definition": group["definition"],
                    "price_mechanism": group["price_mechanism"],
                    "candidate_ids": group["candidate_ids"],
                })

        # Merge local canonical groups across batches. Sorting by name makes related
        # representatives likely to meet in the same bounded request. Original
        # candidate membership is propagated exactly through every hierarchy level.
        round_index = 1
        reached_fixed_point = False
        while len(current_nodes) > GROUP_BATCH_SIZE:
            current_nodes.sort(key=lambda item: normalized_evidence_text(item["canonical_name"]))
            next_nodes: list[dict[str, Any]] = []
            round_batches = [
                current_nodes[index:index + GROUP_BATCH_SIZE]
                for index in range(0, len(current_nodes), GROUP_BATCH_SIZE)
            ]
            for batch_index, node_batch in enumerate(round_batches):
                batch = [
                    {
                        "candidate_id": node["node_id"],
                        "product_id": len(node["candidate_ids"]),
                        "name": node["canonical_name"],
                        "definition": node["definition"],
                        "direction": "unclear",
                        "mechanism": node["price_mechanism"],
                    }
                    for node in node_batch
                ]
                key = (round_index, batch_index)
                source_hash = sha256_text("\n".join(item["candidate_id"] for item in batch))
                saved = checkpoints.get(key)
                if saved is None or saved.get("source_ids_sha256") != source_hash:
                    batch_groups, request_meta, prompt_hash = group_batch(batch)
                    saved = {
                        "round": round_index,
                        "batch_index": batch_index,
                        "source_ids_sha256": source_hash,
                        "groups": batch_groups,
                        "prompt_sha256": prompt_hash,
                        "request_meta": request_meta,
                        "created_at_utc": utc_now(),
                    }
                    append_jsonl(checkpoint_path, saved)
                    checkpoints[key] = saved
                    add_usage(usage, request_meta.get("cumulative_usage", request_meta.get("usage", {})))
                    request_count += 1
                node_by_id = {node["node_id"]: node for node in node_batch}
                for local_index, group in enumerate(saved["groups"]):
                    original_ids = [
                        candidate_id
                        for node_id in group["candidate_ids"]
                        for candidate_id in node_by_id[node_id]["candidate_ids"]
                    ]
                    next_nodes.append({
                        "node_id": f"R{round_index:02d}B{batch_index:04d}G{local_index:04d}",
                        "canonical_name": group["canonical_name"],
                        "definition": group["definition"],
                        "price_mechanism": group["price_mechanism"],
                        "candidate_ids": original_ids,
                    })
            if len(next_nodes) >= len(current_nodes):
                current_nodes = next_nodes
                reached_fixed_point = True
                break
            current_nodes = next_nodes
            round_index += 1

        # A final bounded cross-batch pass ensures every surviving representative can
        # be compared with every other representative before Top K is frozen.
        if len(current_nodes) > 1 and not reached_fixed_point:
            current_nodes.sort(key=lambda item: normalized_evidence_text(item["canonical_name"]))
            batch = [
                {
                    "candidate_id": node["node_id"],
                    "product_id": len(node["candidate_ids"]),
                    "name": node["canonical_name"],
                    "definition": node["definition"],
                    "direction": "unclear",
                    "mechanism": node["price_mechanism"],
                }
                for node in current_nodes
            ]
            key = (round_index, 0)
            source_hash = sha256_text("\n".join(item["candidate_id"] for item in batch))
            saved = checkpoints.get(key)
            if saved is None or saved.get("source_ids_sha256") != source_hash:
                final_groups, request_meta, prompt_hash = group_batch(batch)
                saved = {
                    "round": round_index,
                    "batch_index": 0,
                    "source_ids_sha256": source_hash,
                    "groups": final_groups,
                    "prompt_sha256": prompt_hash,
                    "request_meta": request_meta,
                    "created_at_utc": utc_now(),
                }
                append_jsonl(checkpoint_path, saved)
                checkpoints[key] = saved
                add_usage(usage, request_meta.get("cumulative_usage", request_meta.get("usage", {})))
                request_count += 1
            node_by_id = {node["node_id"]: node for node in current_nodes}
            groups = []
            for index, group in enumerate(saved["groups"], start=1):
                groups.append({
                    "canonical_id": f"C{index:02d}",
                    "canonical_name": group["canonical_name"],
                    "definition": group["definition"],
                    "price_mechanism": group["price_mechanism"],
                    "candidate_ids": [
                        candidate_id
                        for node_id in group["candidate_ids"]
                        for candidate_id in node_by_id[node_id]["candidate_ids"]
                    ],
                })
        else:
            groups = [
                {
                    "canonical_id": f"C{index:02d}",
                    "canonical_name": node["canonical_name"],
                    "definition": node["definition"],
                    "price_mechanism": node["price_mechanism"],
                    "candidate_ids": node["candidate_ids"],
                }
                for index, node in enumerate(current_nodes, start=1)
            ]

        expected_ids = {item["candidate_id"] for item in candidates}
        grouped_ids = [candidate_id for group in groups for candidate_id in group["candidate_ids"]]
        if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != expected_ids:
            raise RuntimeError("Hierarchical grouping did not preserve every candidate exactly once")
        taxonomy = {
            "pipeline_version": PIPELINE_VERSION,
            "prompt_version": GROUP_PROMPT_VERSION,
            "model": MODEL,
            "groups": groups,
            "group_batch_size": GROUP_BATCH_SIZE,
            "hierarchical_grouping": True,
            "reached_fixed_point_without_forced_merging": reached_fixed_point,
            "grouping_id_repair": (
                "Unknown and duplicate model-emitted IDs are discarded; omitted IDs are conservatively retained "
                "as singleton groups before the next hierarchy level"
            ),
            "created_at_utc": utc_now(),
            "request_meta": {"new_request_count": request_count, "usage": usage},
        }
        write_json(taxonomy_path, taxonomy)
        meta = taxonomy["request_meta"]

    candidate_by_id = {item["candidate_id"]: item for item in candidates}
    ranked = []
    for group in groups:
        members = [candidate_by_id[candidate_id] for candidate_id in group["candidate_ids"]]
        products = sorted({int(item["product_id"]) for item in members})
        ranked.append(
            {
                **group,
                "product_frequency": len(products),
                "product_ids": products,
                "mean_extraction_confidence": round(
                    sum(float(item["extraction_confidence"]) for item in members) / len(members), 6
                ),
                "mean_negotiation_impact": round(
                    sum(float(item["negotiation_impact"]) for item in members) / len(members), 6
                ),
            }
        )
    ranked.sort(
        key=lambda item: (
            -item["product_frequency"],
            -item["mean_extraction_confidence"],
            -item["mean_negotiation_impact"],
            item["canonical_name"].lower(),
        )
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    selected = ranked[: min(top_k, len(ranked))]
    selection = {
        "status": (
            "frozen_training_only_topk"
        ),
        "selection_rule": (
            "Descending unique training-product frequency; mean extraction confidence and mean negotiation impact "
            "are deterministic tie-breakers"
        ),
        "requested_top_k": top_k,
        "selected_count": len(selected),
        "ranked_groups": ranked,
        "selected_features": selected,
    }
    write_json(output_path("topk_features", "v3.json"), selection)
    return selection, meta


def scoring_prompt(packet: dict[str, Any], selected: list[dict[str, Any]]) -> str:
    taxonomy = [
        {
            "canonical_id": item["canonical_id"],
            "canonical_name": item["canonical_name"],
            "definition": item["definition"],
            "price_mechanism": item["price_mechanism"],
        }
        for item in selected
    ]
    return (
        "Score the textual evidence for each frozen implicit product feature in this one negotiation. Do not score "
        "general plausibility. Use only the supplied dialogue. No evidence means observed=false and all four dimensions "
        "must be 0. A concern unsupported by seller disclosure is not product evidence.\n\n"
        "For observed features, give integer values from 1 to 100 for: evidence_coverage (how much relevant dialogue "
        "supports it), evidence_specificity (how concrete the disclosed fact is), negotiation_salience (how much it "
        "changes valuation, verification/integration effort, offers, or a decision), and cross_turn_consistency (whether "
        "the same implication remains coherent across turns). Cite exact message indices and short verbatim quotes. "
        "Do not use corpus frequency in these product-level dimensions.\n\n"
        "Return exactly: "
        '{"scores":[{"canonical_id":"C01","observed":true,"evidence_coverage":73,'
        '"evidence_specificity":81,"negotiation_salience":68,"cross_turn_consistency":75,'
        '"direction":"downward","rationale":"...","evidence":[{"message_index":3,"quote":"..."}]}]} '
        "with exactly one entry for every supplied canonical_id.\n\n"
        "FROZEN FEATURES:\n" + json.dumps(taxonomy, ensure_ascii=False, sort_keys=True)
        + "\n\nNEGOTIATION:\n" + json.dumps(packet, ensure_ascii=False, sort_keys=True)
    )


def validate_scores(
    value: dict[str, Any], packet: dict[str, Any], selected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    scores = value.get("scores")
    expected = [item["canonical_id"] for item in selected]
    if not isinstance(scores, list) or sorted(item.get("canonical_id") for item in scores) != sorted(expected):
        raise ValueError("scores must contain exactly one entry per selected canonical feature")
    message_by_index = {item["message_index"]: item["message"] for item in packet["dialogue"]}
    dimensions = [
        "evidence_coverage", "evidence_specificity", "negotiation_salience", "cross_turn_consistency"
    ]
    by_id = {}
    for item in scores:
        required = {
            "canonical_id", "observed", *dimensions, "direction", "rationale", "evidence"
        }
        if set(item) != required:
            raise ValueError("score schema mismatch")
        item["observed"] = bool(item["observed"])
        for dimension in dimensions:
            item[dimension] = int(item[dimension])
            if not 0 <= item[dimension] <= 100:
                raise ValueError(f"{dimension} outside 0-100")
        if not item["observed"]:
            if any(item[dimension] != 0 for dimension in dimensions) or item["evidence"]:
                raise ValueError("unobserved feature must have zero dimensions and no evidence")
        else:
            if any(item[dimension] == 0 for dimension in dimensions) or not item["evidence"]:
                raise ValueError("observed feature must have positive dimensions and evidence")
            for evidence in item["evidence"]:
                index = int(evidence["message_index"])
                quote = str(evidence["quote"]).strip()
                if index not in message_by_index:
                    raise ValueError(f"unverifiable scoring quote at message {index}: {quote!r}")
                aligned = align_evidence_quote(quote, message_by_index[index])
                if aligned is None:
                    raise ValueError(f"unverifiable scoring quote at message {index}: {quote!r}")
                evidence["message_index"] = index
                evidence["quote"] = aligned
        raw = (
            0.35 * item["evidence_coverage"]
            + 0.25 * item["evidence_specificity"]
            + 0.30 * item["negotiation_salience"]
            + 0.10 * item["cross_turn_consistency"]
        )
        item["continuous_score"] = round(raw / 100.0, 4)
        by_id[item["canonical_id"]] = item
    return [by_id[canonical_id] for canonical_id in expected]


def score_products(
    records: list[dict[str, Any]], selected: list[dict[str, Any]], api_key: str,
    *, timeout: float, retries: int, workers: int = DEFAULT_WORKERS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = output_path("feature_scores", "v3.jsonl")
    existing = load_existing_by_product(output)
    usage: dict[str, Any] = {}
    lock = Lock()
    pending = [record for record in records if int(record["product_id"]) not in existing]

    def score_one(record: dict[str, Any]) -> dict[str, Any]:
        product_id = int(record["product_id"])
        packet = public_model_packet(record)
        eligible = [item for item in selected if product_id in item["product_ids"]]
        ineligible_ids = {
            item["canonical_id"] for item in selected if product_id not in item["product_ids"]
        }
        if eligible:
            prompt = scoring_prompt(packet, eligible)
            eligible_scores, meta = call_and_validate(
                api_key,
                system_prompt=(
                    "You are a conservative evidence coder. Score product properties only from quoted public "
                    "negotiation evidence. Absence of evidence is not low quality. A feature is supplied only after "
                    "candidate extraction, independent adjudication, and canonical grouping established semantic "
                    "eligibility for this product; do not broaden its definition."
                ),
                user_prompt=prompt,
                max_tokens=1000,
                validator=lambda value, p=packet, s=eligible: validate_scores(value, p, s),
                timeout=timeout,
                retries=retries,
            )
        else:
            prompt = ""
            eligible_scores = []
            meta = {"skipped_no_semantically_eligible_topk_feature": True, "usage": {}}
        eligible_by_id = {item["canonical_id"]: item for item in eligible_scores}
        scores = []
        for feature in selected:
            canonical_id = feature["canonical_id"]
            if canonical_id in eligible_by_id:
                scores.append(eligible_by_id[canonical_id])
            else:
                scores.append(
                    {
                        "canonical_id": canonical_id,
                        "observed": False,
                        "evidence_coverage": 0,
                        "evidence_specificity": 0,
                        "negotiation_salience": 0,
                        "cross_turn_consistency": 0,
                        "direction": "unclear",
                        "rationale": "No accepted candidate from this product was grouped into this canonical feature.",
                        "evidence": [],
                        "continuous_score": 0.0,
                    }
                )
        saved = {
            "product_id": product_id,
            "status": "ok",
            "scores": scores,
            "semantically_eligible_feature_ids": [item["canonical_id"] for item in eligible],
            "semantically_ineligible_feature_ids": sorted(ineligible_ids),
            "prompt_sha256": sha256_text(prompt),
            "prompt_version": SCORE_PROMPT_VERSION,
            "model": MODEL,
            "created_at_utc": utc_now(),
            "request_meta": meta,
        }
        return saved

    completed = 0
    batch_size = max(1, workers)
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start:batch_start + batch_size]
        # Keep only one in-flight batch. If the API becomes unavailable, executor
        # shutdown waits for at most `workers` calls instead of the entire corpus.
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = {pool.submit(score_one, record): int(record["product_id"]) for record in batch}
            for future in as_completed(futures):
                saved = future.result()
                product_id = int(saved["product_id"])
                append_jsonl(output, saved, lock)
                existing[product_id] = saved
                meta = saved.get("request_meta", {})
                add_usage(usage, meta.get("cumulative_usage", meta.get("usage", {})))
                completed += 1
                if completed % 25 == 0 or completed == len(pending):
                    observed = sum(item["observed"] for item in saved["scores"])
                    print(
                        f"score progress new={completed}/{len(pending)} total={len(existing)}/{len(records)} "
                        f"last_product_id={product_id} observed={observed}",
                        flush=True,
                    )
    ordered = [existing[int(record["product_id"])] for record in records]
    return ordered, usage


def export_matrix(
    scores: list[dict[str, Any]], selected: list[dict[str, Any]]
) -> pd.DataFrame:
    name_by_id = {item["canonical_id"]: item["canonical_name"] for item in selected}
    frequency_by_id = {item["canonical_id"]: item["product_frequency"] for item in selected}
    rows = []
    for record in scores:
        row: dict[str, Any] = {"product_id": int(record["product_id"])}
        for item in record["scores"]:
            feature_id = item["canonical_id"].lower()
            row[f"implicit_{feature_id}_observed"] = int(item["observed"])
            row[f"implicit_{feature_id}_score"] = float(item["continuous_score"])
            row[f"implicit_{feature_id}_training_frequency"] = frequency_by_id[item["canonical_id"]]
            row[f"implicit_{feature_id}_name"] = name_by_id[item["canonical_id"]]
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("product_id")
    frame.to_csv(output_path("feature_matrix", "v3.csv"), index=False, encoding="utf-8-sig")
    model_columns = [
        column
        for column in frame.columns
        if column == "product_id" or column.endswith("_observed") or column.endswith("_score")
    ]
    frame[model_columns].to_csv(
        output_path("m1_numeric_features", "v3.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return frame


def run(
    api_key: str,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    input_file: Path = DEFAULT_INPUT,
    run_tag: str = "release",
    top_k: int = DEFAULT_TOP_K,
    limit: int | None = None,
    timeout: float = 120.0,
    retries: int = 3,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("Missing key: set DEEPSEEK_API_KEY or AI_EDGE_API_KEY")
    global RUN_TAG
    RUN_TAG = run_tag
    input_path = input_file.resolve()
    if not input_path.exists():
        raise RuntimeError(f"Input dialogue file does not exist: {input_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    expected_hash = manifest.get("sha256", {}).get(input_path.name)
    if not expected_hash:
        full_manifest = input_path.parent / "FULL_PUBLIC_DIALOGUE_MANIFEST.json"
        if full_manifest.exists():
            expected_hash = json.loads(full_manifest.read_text(encoding="utf-8")).get("approved_sha256")
    actual_hash = sha256_file(input_path)
    if expected_hash and expected_hash != actual_hash:
        raise RuntimeError("Approved M0 public-dialogue hash does not match the manifest")
    records = read_jsonl(input_path)
    if limit is not None:
        records = records[:limit]
    if not records:
        raise RuntimeError("No approved public dialogue records")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extractions, extraction_usage = extract_candidates(
        records, api_key, timeout=timeout, retries=retries, workers=workers
    )
    candidates = flatten_candidates(extractions)
    pd.DataFrame(candidates).to_csv(
        output_path("candidate_features_flat", "v2.csv"), index=False, encoding="utf-8-sig"
    )
    adjudications, llm_accepted_candidates, adjudication_usage = adjudicate_candidates(
        records, candidates, api_key, timeout=timeout, retries=retries, workers=workers
    )
    accepted_candidates, deterministic_gate_rows = apply_deterministic_gate(
        records, adjudications, llm_accepted_candidates
    )
    if not accepted_candidates:
        raise RuntimeError("All candidates were rejected during adjudication")
    pd.DataFrame(accepted_candidates).to_csv(
        output_path("accepted_candidate_features_flat", "v3.csv"), index=False, encoding="utf-8-sig"
    )
    selection, grouping_usage = group_and_rank(
        accepted_candidates, api_key, top_k=top_k, timeout=timeout, retries=retries
    )
    selected = selection["selected_features"]
    scores, scoring_usage = score_products(
        records, selected, api_key, timeout=timeout, retries=retries, workers=workers
    )
    matrix = export_matrix(scores, selected)

    audit = {
        "status": "completed_feature_pipeline",
        "pipeline_version": PIPELINE_VERSION,
        "input_file": str(input_path.resolve()),
        "input_sha256": actual_hash,
        "approved_manifest": str(manifest_path.resolve()) if manifest_path.exists() else "FULL_PUBLIC_DIALOGUE_MANIFEST.json",
        "products": len(records),
        "candidate_count": len(candidates),
        "products_with_candidates": sum(bool(item["features"]) for item in extractions),
        "accepted_candidate_count": len(accepted_candidates),
        "rejected_candidate_count": len(candidates) - len(accepted_candidates),
        "llm_accepted_candidate_count": len(llm_accepted_candidates),
        "deterministic_gate_rejection_count": sum(
            item["llm_decision"] == "keep" and item["deterministic_decision"] == "reject"
            for item in deterministic_gate_rows
        ),
        "canonical_group_count": len(selection["ranked_groups"]),
        "top_k": len(selected),
        "selected_feature_names": [item["canonical_name"] for item in selected],
        "matrix_rows": len(matrix),
        "m1_numeric_feature_columns": [
            column for column in matrix.columns
            if column.endswith("_observed") or column.endswith("_score")
        ],
        "training_frequency_exported_as_m1_predictor": False,
        "test_products_used": False,
        "real_transaction_prices_sent_to_model": False,
        "simulated_outcome_field_sent_to_model": False,
        "private_agent_fields_sent_to_model": False,
        "patience_fields_sent_to_model": False,
        "numeric_negotiation_prices_redacted": True,
        "frequency_used_for_selection_not_product_score": True,
        "scoring_semantically_gated_by_accepted_canonical_membership": True,
        "api_usage_new_calls": {
            "extraction": extraction_usage,
            "adjudication": adjudication_usage,
            "grouping": grouping_usage,
            "scoring": scoring_usage,
        },
        "output_sha256": {
            path.name: sha256_file(path)
            for path in sorted(OUTPUT_DIR.iterdir())
            if path.is_file() and path.name.startswith(f"{RUN_TAG}_")
        },
        "created_at_utc": utc_now(),
        "workers": workers,
    }
    write_json(output_path("pipeline_manifest", "v3.json"), audit)
    current = {
        "status": "feature_pipeline_complete_not_final_m1_training" if run_tag == "release" else "full_training_feature_pipeline_complete",
        "pipeline_version": PIPELINE_VERSION,
        "source_dialogues": str(input_path.resolve()),
        "source_dialogues_sha256": actual_hash,
        "topk_file": output_path("topk_features", "v3.json").name,
        "audit_matrix_file": output_path("feature_matrix", "v3.csv").name,
        "numeric_matrix_file": output_path("m1_numeric_features", "v3.csv").name,
        "manifest_file": output_path("pipeline_manifest", "v3.json").name,
    }
    write_json(OUTPUT_DIR / f"{RUN_TAG}_FEATURE_MANIFEST.json", current)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY") or os.getenv("AI_EDGE_API_KEY", ""))
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run-tag", default="release")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), ensure_ascii=False, indent=2))
