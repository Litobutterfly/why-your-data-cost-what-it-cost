from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V11_INPUT = ROOT / "09_m0_negotiation" / "outputs" / "m0_release_dialogues_v11.jsonl"
V10_INPUT = ROOT / "09_m0_negotiation" / "outputs" / "m0_release_dialogues_v10.jsonl"
V9_INPUT = ROOT / "09_m0_negotiation" / "outputs" / "m0_release_dialogues_v9.jsonl"
DEFAULT_INPUT = V11_INPUT if V11_INPUT.exists() else V10_INPUT if V10_INPUT.exists() else V9_INPUT
OUTPUT_DIR = ROOT / "09_m0_negotiation" / "outputs"

SIGNATURES = {
    "completeness_variation": [["optional field", "optional-field"], ["populat", "missingness"]],
    "temporal_lag": [["refresh", "schedule"], ["delay", "later", "latency"]],
    "cross_batch_consistency": [["batch"], ["label", "categorical"], ["mapping", "version"]],
    "coverage_imbalance": [["coverage", "segment"], ["fewer", "imbalance", "dominant"], ["benchmark"]],
    "accuracy_validation_gap": [["benchmark", "validation"], ["error rate", "measured rate", "accuracy rate"]],
    "governance_traceability_gap": [["authorization", "source-to-use", "provenance"], ["trail", "documentation"]],
    "delivery_reliability": [["retry"], ["uptime", "reliability log", "delivery record"]],
}

FORBIDDEN = [
    "low quality", "high risk", "defective", "bad data", "hidden issue",
    "synthetic condition", "controlled treatment", "controlled observation",
    "synthetic observation", "experimental status", "condition id", "real product",
    "simulation", "prompt rule", "hidden variable", "m0", "m1",
    "lowest i can go", "lowest price i can", "minimum i can accept",
    "minimum acceptable", "my floor", "below my floor", "at my floor",
]
DUE_DILIGENCE = [
    "internal observation", "evidence limitation", "material caveat", "known issue", "recent check",
    "internal check", "recent observation",
]


def condition_disclosed(condition: str, text: str) -> bool:
    if condition == "none":
        return False
    lowered = text.lower()
    return all(any(token in lowered for token in alternatives) for alternatives in SIGNATURES[condition])


def audit_record(record: dict[str, Any]) -> dict[str, Any]:
    condition = record["controlled_condition_id_audit_only"]
    dialogue = record["dialogue"]
    seller_events = [event for event in dialogue if event["actor"] == "seller"]
    buyer_events = [event for event in dialogue if event["actor"] == "buyer"]
    opening_text = seller_events[0]["message"] if seller_events else ""
    full_text = " ".join(event["message"] for event in dialogue).lower()
    due_indices = [
        event["message_index"] for event in buyer_events
        if any(token in event["message"].lower() for token in DUE_DILIGENCE)
    ]
    disclosure_indices = [
        event["message_index"] for event in seller_events
        if condition_disclosed(condition, event["message"])
    ]
    first_disclosure = min(disclosure_indices) if disclosure_indices else None
    post_disclosure_buyer = [
        event for event in buyer_events
        if first_disclosure is not None and event["message_index"] > first_disclosure
    ]
    buyer_reasoned = any(
        any(token in event["message"].lower() for token in [
            "uncertainty", "need", "affect", "limit", "risk", "validation", "integration",
            "analysis", "use", "offer", "price", "budget", "fit", "confidence", "credible",
            "manageable", "resolve", "value", "time-to-value",
        ])
        for event in post_disclosure_buyer
    )
    if condition == "none":
        target_ok = not disclosure_indices
    else:
        target_ok = bool(
            disclosure_indices and due_indices and min(disclosure_indices) > min(due_indices)
        )
    checks = {
        "final_outcome_is_decided": record["outcome"] in {
            "agreement",
            "buyer_walked_away",
            "seller_walked_away",
            "m1_price_gate_rejected",
        },
        "seller_opening_did_not_disclose_condition": not condition_disclosed(condition, opening_text),
        "buyer_asked_neutral_due_diligence": bool(due_indices),
        "neutral_due_diligence_is_first_buyer_turn": bool(due_indices and min(due_indices) == 2),
        "target_condition_disclosed_only_after_due_diligence": target_ok,
        "forbidden_quality_labels_absent": not any(token in full_text for token in FORBIDDEN),
        "buyer_reasoned_after_disclosure": buyer_reasoned if condition != "none" else True,
        "buyer_message_4_is_nonprice_assessment": (
            len(dialogue) >= 4
            and dialogue[3]["actor"] == "buyer"
            and dialogue[3]["action"] == "assess"
            and dialogue[3].get("offer_usd") is None
        ) if str(record.get("prompt_version", "")).endswith("predecision-buyer-assessment") else True,
    }
    return {
        "product_id": record["product_id"],
        "condition": condition,
        "outcome": record["outcome"],
        "due_diligence_message_indices": due_indices,
        "disclosure_message_indices": disclosure_indices,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def run(input_path: Path) -> dict[str, Any]:
    records = [
        json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    audits = [audit_record(record) for record in records]
    issue_audits = [item for item in audits if item["condition"] != "none"]
    report = {
        "input": str(input_path),
        "records": len(audits),
        "records_passing_all_checks": sum(item["all_checks_passed"] for item in audits),
        "condition_disclosure_rate_for_issue_products": (
            sum(item["checks"]["target_condition_disclosed_only_after_due_diligence"] for item in issue_audits)
            / max(1, len(issue_audits))
        ),
        "audits": audits,
    }
    suffix = input_path.stem.replace("m0_release_dialogues", "") or "_audit"
    (OUTPUT_DIR / f"m0_release_dialogue_audit{suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.input), ensure_ascii=False, indent=2))
