from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from support import audit_dialogues


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "work" / "negotiation" / "m0_full_train_dialogues_v2.jsonl"
PUBLIC = ROOT / "results" / "work" / "negotiation" / "m0_full_train_public_dialogues_v2.jsonl"
OUTPUT_DIR = ROOT / "results" / "work" / "negotiation"
APPROVED = OUTPUT_DIR / "m0_full_train_public_dialogues_approved_v2.jsonl"
MANIFEST = OUTPUT_DIR / "FULL_PUBLIC_DIALOGUE_MANIFEST_V2.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_projection(record: dict[str, Any]) -> dict[str, Any]:
    private = {
        "api_calls", "initial_patience_private", "patience_trace_private",
        "buyer_ceiling_usd_private", "seller_floor_usd_private",
        "controlled_condition_id_audit_only", "observable_readiness_level_audit_only",
        "potential_price_overlap_audit_only", "buyer_profile_id_audit_only",
        "seller_profile_id_audit_only",
    }
    return {
        key: value for key, value in record.items() if key not in private
    }


def run() -> dict[str, Any]:
    raw = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]
    public = [json.loads(line) for line in PUBLIC.read_text(encoding="utf-8").splitlines() if line.strip()]
    public_by_id = {int(item["product_id"]): item for item in public}
    approved = []
    excluded = []
    for record in raw:
        audit = audit_dialogues.audit_record(record)
        product_id = int(record["product_id"])
        if audit["all_checks_passed"]:
            item = public_by_id[product_id]
            forbidden = {
                "api_calls", "initial_patience_private", "patience_trace_private",
                "buyer_ceiling_usd_private", "seller_floor_usd_private",
                "controlled_condition_id_audit_only", "observable_readiness_level_audit_only",
                "potential_price_overlap_audit_only", "buyer_profile_id_audit_only",
                "seller_profile_id_audit_only",
            }
            if forbidden & set(item):
                raise RuntimeError(f"Private fields leaked in public record {product_id}")
            approved.append(item)
        else:
            excluded.append({
                "product_id": product_id,
                "condition": record.get("controlled_condition_id_audit_only"),
                "outcome": record.get("outcome"),
                "failed_checks": [key for key, value in audit["checks"].items() if not value],
            })
    with APPROVED.open("w", encoding="utf-8") as handle:
        for item in approved:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "status": "approved_public_dialogues_ready_for_feature_discovery",
        "source_raw_file": RAW.name,
        "source_public_file": PUBLIC.name,
        "approved_file": APPROVED.name,
        "approved_records": len(approved),
        "excluded_records": len(excluded),
        "excluded_product_ids": [item["product_id"] for item in excluded],
        "excluded_records_file": "m0_full_train_public_dialogues_excluded_v2.jsonl",
        "excluded_reason": "Dialogue-level protocol audit failure; retained for audit, not feature discovery.",
        "source_raw_sha256": sha256_file(RAW),
        "source_public_sha256": sha256_file(PUBLIC),
        "approved_sha256": sha256_file(APPROVED),
    }
    (OUTPUT_DIR / "m0_full_train_public_dialogues_excluded_v2.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in excluded) + "\n",
        encoding="utf-8",
    )
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
