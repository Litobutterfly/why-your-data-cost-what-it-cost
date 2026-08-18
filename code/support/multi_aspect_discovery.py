from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "09_m0_negotiation" / "outputs" / "m0_full_train_selection.csv"
FACTS = ROOT / "05_negotiation_prep" / "outputs" / "train_negotiation_inputs.csv"
OUTPUT_DIR = ROOT / "23_multi_aspect_discovery" / "outputs"
ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
PROMPT_VERSION = "multi-aspect-description-discovery-v1"
TERM_LABELS = {
    "analysi": "analysis",
    "analyt": "analytics",
    "attribut": "attributes",
    "avail": "availability",
    "busi": "business",
    "compani": "company",
    "consum": "consumer",
    "countri": "country",
    "cover": "cover",
    "coverag": "coverage",
    "custom": "customization",
    "detail": "detail",
    "global": "global scope",
    "includ": "included content",
    "industri": "industry",
    "inform": "information",
    "insight": "insights",
    "lead": "leads",
    "locat": "location",
    "manag": "management",
    "million": "scale wording",
    "provid": "provided content",
    "qualiti": "quality wording",
    "requir": "requirements",
    "research": "research",
    "sale": "sales",
    "segment": "segments",
    "servic": "service",
    "sourc": "sources",
    "store": "storage",
    "use": "use",
    "user": "users",
    "websit": "websites",
    "worldwid": "worldwide scope",
    "year": "year wording",
}
FAMILIES = {
    "task_fit",
    "verification_cost",
    "integration_cost",
    "coverage_value",
    "timeliness_value",
    "governance_risk",
    "delivery_reliability",
    "other",
}
EVIDENCE_STATES = {
    "confirmed_property",
    "unresolved_listing_claim",
    "bounded_limitation",
    "fit_positive",
    "mixed_evidence",
}
DIRECTIONS = {"upward", "downward", "mixed", "conditional"}
FORBIDDEN = {
    "real transaction price",
    "controlled condition",
    "private ceiling",
    "private floor",
    "simulation",
    "synthetic condition",
    "m0",
    "m1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(value: str) -> str:
    return " ".join(
        re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", str(value)).casefold())
    )


def term_signals(product_facts_json: str) -> list[str]:
    facts = json.loads(product_facts_json)
    scores = facts.get("description_term_scores", {})
    ranked = sorted(scores.items(), key=lambda item: (-float(item[1]), item[0]))[:6]
    return [TERM_LABELS.get(stem, stem) for stem, _ in ranked]


def focus_groups(terms: list[str]) -> list[list[str]]:
    return [terms[index : index + 2] for index in range(0, len(terms), 2)]


def prepare(limit: int | None) -> list[dict[str, Any]]:
    selection = pd.read_csv(SELECTION, low_memory=False)
    facts = pd.read_csv(FACTS, usecols=["Id", "product_facts_json"], low_memory=False)
    rows = selection.merge(
        facts, left_on="product_id", right_on="Id", validate="one_to_one"
    ).sort_values("product_id")
    if limit is not None:
        rows = rows.head(limit)
    result = []
    for _, row in rows.iterrows():
        packet = json.loads(row["public_product_packet_json"])
        buyer = json.loads(row["buyer_private_context_json"])
        seller = json.loads(row["seller_private_context_json"])
        terms = term_signals(row["product_facts_json"])
        result.append(
            {
                "product_id": int(row["product_id"]),
                "source_generated_listing": str(packet.get("platform_listing", "")),
                "required_focus_groups": focus_groups(terms),
                "structured_fields": {
                    "delivery_channels": packet.get("delivery_channels_listed", []),
                    "formats": packet.get("formats_listed", []),
                    "update_options": packet.get("update_options_listed", []),
                    "company_identifiers_indicated": packet.get(
                        "company_identifiers_indicated", False
                    ),
                    "individual_identifiers_indicated": packet.get(
                        "individual_identifiers_indicated", False
                    ),
                    "countries_covered": packet.get("countries_covered", 0),
                    "use_limitations_flag_indicated": packet.get(
                        "use_limitations_flag_indicated", False
                    ),
                    "professional_services_indicated": packet.get(
                        "professional_services_indicated", False
                    ),
                },
                "buyer_context": {
                    key: buyer.get(key)
                    for key in [
                        "intended_use",
                        "integration_capacity",
                        "quality_priorities",
                        "risk_and_evidence_posture",
                        "time_context",
                        "decision_consequence",
                    ]
                },
                "seller_context": {
                    key: seller.get(key)
                    for key in [
                        "evidence_policy",
                        "response_policy",
                        "support_scope",
                        "reputation_priority",
                    ]
                },
            }
        )
    ids = [item["product_id"] for item in result]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Prepared rows contain duplicate product IDs")
    return result


def prompt(records: list[dict[str, Any]]) -> str:
    return (
        "Generate evidence-bounded multi-aspect data-product discovery exchanges. For every product, create exactly "
        "one exchange for each required_focus_groups entry, in the supplied order. Each exchange has three turns: "
        "(1) the buyer asks what concrete product content or evidence supports exactly those concepts; (2) the "
        "seller answers only from the supplied listing and structured fields, distinguishing supported facts from "
        "unspecified wording; (3) the buyer explains the resulting task-fit, utility, verification-cost, "
        "integration-cost, risk, coverage, timeliness, governance, or reliability implication. Missing detail is "
        "uncertainty, never proof of a defect.\n\n"
        "For each exchange, extract one or two reusable product-information mechanisms. Each mechanism must use the "
        "exchange's exact focus_terms and include one verbatim evidence_quote copied from either seller_answer or "
        "buyer_assessment. Exclude buyer/seller identity, budgets, patience, bargaining behavior, prices, offers, "
        "outcomes, experiments, and invented product facts.\n\n"
        "family must be one of: "
        + ", ".join(sorted(FAMILIES))
        + ". evidence_state must be one of: "
        + ", ".join(sorted(EVIDENCE_STATES))
        + ". direction must be upward, downward, mixed, or conditional. signed_effect is continuous in [-1,1]; "
        "impact_magnitude and evidence_confidence are continuous in [0,1].\n\n"
        "Return JSON only with exact product coverage and this shape: "
        '{"products":[{"product_id":1,"exchanges":[{"exchange_index":1,'
        '"focus_terms":["term a","term b"],"buyer_question":"...",'
        '"seller_answer":"...","buyer_assessment":"...","features":[{'
        '"name":"schema granularity uncertainty","definition":"...",'
        '"family":"verification_cost","evidence_state":"unresolved_listing_claim",'
        '"direction":"downward","signed_effect":-0.62,"impact_magnitude":0.71,'
        '"evidence_confidence":0.90,"evidence_speaker":"buyer",'
        '"evidence_quote":"verbatim quote"}]}]}]}\n\nINPUT:\n'
        + json.dumps(records, ensure_ascii=False, sort_keys=True)
    )


def validate_text(value: Any, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} is empty")
    lowered = text.casefold()
    leaked = sorted(term for term in FORBIDDEN if term in lowered)
    monetary_amount = re.search(
        r"(?:\$\s*\d|\b\d[\d,]*(?:\.\d+)?\s*(?:usd|dollars?)\b)", lowered
    )
    if leaked or monetary_amount:
        raise ValueError(f"{field} leaks forbidden information: {leaked}")
    return text


def validate_response(
    value: dict[str, Any], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    products = value.get("products")
    if not isinstance(products, list):
        raise ValueError("products must be a list")
    expected = {item["product_id"]: item for item in records}
    received = [int(item.get("product_id", -1)) for item in products]
    if set(received) != set(expected) or len(received) != len(set(received)):
        raise ValueError("product coverage mismatch")
    output = []
    for product in products:
        product_id = int(product["product_id"])
        source = expected[product_id]
        exchanges = product.get("exchanges")
        groups = source["required_focus_groups"]
        if not isinstance(exchanges, list) or len(exchanges) != len(groups):
            raise ValueError("exchange coverage mismatch")
        checked_exchanges = []
        flat_features = []
        for exchange_index, (exchange, group) in enumerate(
            zip(exchanges, groups), start=1
        ):
            required = {
                "exchange_index",
                "focus_terms",
                "buyer_question",
                "seller_answer",
                "buyer_assessment",
                "features",
            }
            if set(exchange) != required or int(exchange["exchange_index"]) != exchange_index:
                raise ValueError("exchange schema or index mismatch")
            terms = [str(term).strip() for term in exchange["focus_terms"]]
            if [term.casefold() for term in terms] != [term.casefold() for term in group]:
                raise ValueError("exchange focus terms differ from required group")
            question = validate_text(exchange["buyer_question"], "buyer_question")
            seller_answer = validate_text(exchange["seller_answer"], "seller_answer")
            buyer_assessment = validate_text(
                exchange["buyer_assessment"], "buyer_assessment"
            )
            if "?" not in question:
                raise ValueError("buyer question is not a question")
            if not all(term.casefold() in question.casefold() for term in terms):
                question = f"Regarding {' and '.join(terms)}, {question}"
            features = exchange["features"]
            if not isinstance(features, list) or not 1 <= len(features) <= 2:
                raise ValueError("each exchange requires one or two features")
            checked_features = []
            for feature in features:
                feature_required = {
                    "name",
                    "definition",
                    "family",
                    "evidence_state",
                    "direction",
                    "signed_effect",
                    "impact_magnitude",
                    "evidence_confidence",
                    "evidence_speaker",
                    "evidence_quote",
                }
                if set(feature) != feature_required:
                    raise ValueError("feature schema mismatch")
                feature["name"] = validate_text(feature["name"], "feature name")
                feature["definition"] = validate_text(
                    feature["definition"], "feature definition"
                )
                if feature["family"] not in FAMILIES:
                    raise ValueError("invalid family")
                if feature["evidence_state"] not in EVIDENCE_STATES:
                    raise ValueError("invalid evidence_state")
                if feature["direction"] not in DIRECTIONS:
                    raise ValueError("invalid direction")
                for field, low, high in [
                    ("signed_effect", -1.0, 1.0),
                    ("impact_magnitude", 0.0, 1.0),
                    ("evidence_confidence", 0.0, 1.0),
                ]:
                    feature[field] = float(feature[field])
                    if not low <= feature[field] <= high:
                        raise ValueError(f"{field} out of range")
                if feature["direction"] == "upward" and feature["signed_effect"] <= 0:
                    raise ValueError("upward feature has nonpositive effect")
                if feature["direction"] == "downward" and feature["signed_effect"] >= 0:
                    raise ValueError("downward feature has nonnegative effect")
                speaker = str(feature["evidence_speaker"]).casefold()
                if speaker not in {"seller", "buyer"}:
                    raise ValueError("evidence_speaker must be seller or buyer")
                quote = str(feature["evidence_quote"]).strip()
                evidence_text = seller_answer if speaker == "seller" else buyer_assessment
                if not quote or normalized(quote) not in normalized(evidence_text):
                    quote = evidence_text
                checked = {
                    **feature,
                    "evidence_speaker": speaker,
                    "evidence_quote": quote,
                    "exchange_index": exchange_index,
                    "source_terms": terms,
                }
                checked_features.append(checked)
                flat_features.append(checked)
            checked_exchanges.append(
                {
                    "exchange_index": exchange_index,
                    "focus_terms": terms,
                    "buyer_question": question,
                    "seller_answer": seller_answer,
                    "buyer_assessment": buyer_assessment,
                    "features": checked_features,
                }
            )
        output.append(
            {
                "product_id": product_id,
                "source_description_term_signals": [
                    term for group in groups for term in group
                ],
                "exchanges": checked_exchanges,
                "features": flat_features,
            }
        )
    return sorted(output, key=lambda item: item["product_id"])


def call_batch(
    records: list[dict[str, Any]], api_key: str, timeout: float, retries: int
) -> list[dict[str, Any]]:
    user_prompt = prompt(records)
    feedback = ""
    last_error = ""
    for attempt in range(1, retries + 2):
        started = time.time()
        try:
            response = requests.post(
                ENDPOINT,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a conservative evidence-constrained data-market discovery coder. "
                                "Return valid JSON only, preserve exact product and focus-group coverage, "
                                "and never invent defects."
                            ),
                        },
                        {"role": "user", "content": user_prompt + feedback},
                    ],
                    "temperature": 0.0,
                    "max_tokens": min(8000, max(2600, 2100 * len(records))),
                    "response_format": {"type": "json_object"},
                    "stream": False,
                },
                timeout=timeout,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
            payload = response.json()
            parsed = json.loads(payload["choices"][0]["message"]["content"])
            checked = validate_response(parsed, records)
            elapsed = round(time.time() - started, 3)
            return [
                {
                    "status": "ok",
                    **item,
                    "prompt_version": PROMPT_VERSION,
                    "model": payload.get("model", MODEL),
                    "attempt": attempt,
                    "elapsed_seconds": elapsed,
                    "usage_batch": payload.get("usage", {}),
                    "batch_products": len(records),
                    "real_transaction_price_visible": False,
                    "negotiated_outcome_visible": False,
                }
                for item in checked
            ]
        except Exception as exc:
            last_error = repr(exc)
            feedback = (
                "\n\nThe previous response failed validation: "
                + str(exc)
                + ". Return corrected complete JSON with exact products, focus groups, and verbatim quotes."
            )
            if attempt <= retries:
                time.sleep(min(5.0, 2.0 ** (attempt - 1)))
    if len(records) > 1:
        midpoint = len(records) // 2
        return call_batch(records[:midpoint], api_key, timeout, retries) + call_batch(
            records[midpoint:], api_key, timeout, retries
        )
    raise RuntimeError(
        f"Multi-aspect discovery failed for product {records[0]['product_id']}: {last_error}"
    )


def main(
    api_key: str,
    output_stem: str,
    selection_path: Path,
    facts_path: Path,
    output_dir: Path,
    limit: int | None,
    batch_size: int,
    workers: int,
    timeout: float,
    retries: int,
) -> None:
    global SELECTION, FACTS, OUTPUT_DIR
    SELECTION = selection_path
    FACTS = facts_path
    OUTPUT_DIR = output_dir
    if not api_key:
        raise RuntimeError("Set DEEPSEEK_API_KEY or AI_EDGE_API_KEY")
    if not 1 <= batch_size <= 4:
        raise ValueError("batch_size must be between 1 and 4")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"{output_stem}.jsonl"
    manifest_path = OUTPUT_DIR / f"{output_stem}_manifest.json"
    error_path = OUTPUT_DIR / f"{output_stem}_errors.jsonl"
    records = prepare(limit)
    expected = {item["product_id"]: item for item in records}
    existing: dict[int, dict[str, Any]] = {}
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if int(item["product_id"]) in expected and item.get("status") == "ok":
                    existing[int(item["product_id"])] = item
    pending = [item for item in records if item["product_id"] not in existing]
    batches = [pending[index : index + batch_size] for index in range(0, len(pending), batch_size)]
    if batches and not existing:
        first = batches.pop(0)
        result = call_batch(first, api_key, timeout, retries)
        with output.open("a", encoding="utf-8") as handle:
            for item in result:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                existing[int(item["product_id"])] = item
        print(f"progress completed={len(existing)}/{len(records)}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(call_batch, batch, api_key, timeout, retries): batch
            for batch in batches
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                failed_batch = futures[future]
                with error_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "product_ids": [item["product_id"] for item in failed_batch],
                                "error": repr(exc),
                                "timestamp": time.time(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                print(
                    "batch_failed product_ids="
                    + repr([item["product_id"] for item in failed_batch]),
                    flush=True,
                )
                continue
            with output.open("a", encoding="utf-8") as handle:
                for item in result:
                    product_id = int(item["product_id"])
                    if product_id not in existing:
                        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                        existing[product_id] = item
            print(f"progress completed={len(existing)}/{len(records)}", flush=True)
    missing = sorted(set(expected) - set(existing))
    if missing:
        partial = {
            "status": "partial",
            "requested_products": len(records),
            "completed_products": len(existing),
            "missing_products": len(missing),
            "missing_product_ids": missing,
            "prompt_version": PROMPT_VERSION,
            "real_transaction_price_visible": False,
            "negotiated_outcome_visible": False,
        }
        manifest_path.write_text(
            json.dumps(partial, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(partial, ensure_ascii=False, indent=2), flush=True)
        return
    ordered = [existing[item["product_id"]] for item in records]
    usage = Counter()
    seen_batches = set()
    for item in ordered:
        signature = json.dumps(item.get("usage_batch", {}), sort_keys=True) + "|" + str(
            item.get("elapsed_seconds")
        )
        if item.get("usage_batch") and signature not in seen_batches:
            seen_batches.add(signature)
            for key, value in item["usage_batch"].items():
                if isinstance(value, (int, float)):
                    usage[key] += int(value)
    manifest = {
        "status": "completed",
        "products": len(ordered),
        "exchanges": sum(len(item["exchanges"]) for item in ordered),
        "features": sum(len(item["features"]) for item in ordered),
        "mean_features_per_product": sum(len(item["features"]) for item in ordered)
        / len(ordered),
        "family_counts": dict(
            Counter(
                feature["family"]
                for item in ordered
                for feature in item["features"]
            )
        ),
        "prompt_version": PROMPT_VERSION,
        "model_counts": dict(Counter(item["model"] for item in ordered)),
        "batch_size": batch_size,
        "workers": workers,
        "usage_approximate_deduplicated_by_batch": dict(usage),
        "real_transaction_price_visible": False,
        "negotiated_outcome_visible": False,
        "selection_sha256": sha256_file(SELECTION),
        "facts_sha256": sha256_file(FACTS),
        "output_sha256": sha256_file(output),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-stem", default="multi_aspect_full_train_v1")
    parser.add_argument("--selection-path", type=Path, default=SELECTION)
    parser.add_argument("--facts-path", type=Path, default=FACTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("AI_EDGE_API_KEY", "")
    main(key, **vars(args))
