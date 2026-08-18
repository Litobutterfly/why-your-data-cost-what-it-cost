from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "work" / "negotiation" / "m0_full_train_dialogues_v2.jsonl"
OUTPUT = ROOT / "results" / "work" / "negotiation" / "m0_full_train_public_dialogues_v2.jsonl"


def run() -> dict[str, object]:
    records = [
        json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    public_records = []
    for record in records:
        public_records.append({
            "product_id": record["product_id"],
            "run_version": record["run_version"],
            "prompt_version": record["prompt_version"],
            "platform_reference_price_usd": record["m0_platform_reference_usd"],
            "public_product_packet": record["agent_public_packet"],
            "dialogue": record["dialogue"],
            "outcome": record["outcome"],
            "negotiated_price_usd": record["negotiated_price_usd"],
            "message_count": record["message_count"],
        })
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for record in public_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    checks = {
        "records": len(public_records),
        "all_outcomes_decided": all(
            record["outcome"] in {"agreement", "buyer_walked_away", "seller_walked_away"}
            for record in public_records
        ),
        "private_fields_absent": all(
            not ({"api_calls", "initial_patience_private", "patience_trace_private",
                  "buyer_ceiling_usd_private", "seller_floor_usd_private",
                  "controlled_condition_id_audit_only"} & set(record))
            for record in public_records
        ),
    }
    if not all(value for key, value in checks.items() if key != "records"):
        raise RuntimeError(f"Public-dialogue export failed validation: {checks}")
    return checks


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
