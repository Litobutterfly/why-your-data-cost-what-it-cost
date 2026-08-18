from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAIR_PATH = ROOT / "results" / "work" / "agents" / "train_fixed_agent_pairs.csv"
CONDITION_PATH = ROOT / "results" / "work" / "quality_conditions" / "controlled_quality_conditions.csv"
OUTPUT_DIR = ROOT / "results" / "work" / "negotiation"
VERSION = "finite-patience-v1-preregistered"

BUYER_BASE = {
    "B1_governance_review": 8,
    "B2_budget_guarded": 7,
    "B3_deadline_operations": 6,
    "B4_exploratory_research": 8,
    "B5_integration_constrained": 7,
    "B6_strategic_repeat": 8,
}

SELLER_BASE = {
    "S1_evidence_first": 8,
    "S2_balanced_market": 7,
    "S3_current_revenue": 6,
    "S4_value_defender": 8,
    "S5_lean_service": 7,
    "S6_repeat_account": 7,
}

READINESS_ADJUSTMENT = {"low": -1, "medium": 0, "high": 1}


def clamp(value: int) -> int:
    return max(6, min(8, value))


def run() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(PAIR_PATH, low_memory=False)
    conditions = pd.read_csv(CONDITION_PATH, low_memory=False)
    frame = pairs.merge(
        conditions[[
            "product_id", "observable_readiness_level", "controlled_condition_id",
            "controlled_condition_present",
        ]],
        on="product_id",
        validate="one_to_one",
    )
    rows = []
    for row in frame.itertuples(index=False):
        buyer_base = BUYER_BASE[row.buyer_profile_id]
        seller_base = SELLER_BASE[row.seller_profile_id]
        readiness_delta = READINESS_ADJUSTMENT[row.observable_readiness_level]
        evidence_delta = -1 if bool(row.controlled_condition_present) else 1
        rows.append({
            "product_id": int(row.product_id),
            "buyer_profile_id": row.buyer_profile_id,
            "seller_profile_id": row.seller_profile_id,
            "buyer_base_patience": buyer_base,
            "buyer_pre_disclosure_adjustment": 0,
            "buyer_initial_patience": buyer_base,
            "seller_base_patience": seller_base,
            "seller_readiness_adjustment": readiness_delta,
            "seller_private_evidence_adjustment": evidence_delta,
            "seller_initial_patience": clamp(seller_base + readiness_delta + evidence_delta),
            "buyer_post_disclosure_issue_cost": 1 if bool(row.controlled_condition_present) else 0,
            "observable_readiness_level": row.observable_readiness_level,
            "controlled_condition_id_audit_only": row.controlled_condition_id,
            "controlled_condition_present_audit_only": bool(row.controlled_condition_present),
            "patience_version": VERSION,
            "minimum_initial_patience": 6,
            "maximum_initial_patience": 8,
        })
    output = pd.DataFrame(rows).sort_values("product_id")
    output.to_csv(
        OUTPUT_DIR / "train_frozen_patience_assignments.csv", index=False, encoding="utf-8-sig"
    )
    checks = {
        "covers_all_training_pairs": bool(set(output["product_id"]) == set(pairs["product_id"])),
        "one_assignment_per_product": bool(not output["product_id"].duplicated().any()),
        "buyer_initial_patience_in_6_8": bool(output["buyer_initial_patience"].between(6, 8).all()),
        "seller_initial_patience_in_6_8": bool(output["seller_initial_patience"].between(6, 8).all()),
        "buyer_not_adjusted_by_hidden_quality_before_disclosure": bool(output[
            "buyer_pre_disclosure_adjustment"
        ].eq(0).all()),
    }
    audit = {
        "version": VERSION,
        "rows": len(output),
        "buyer_initial_distribution": output["buyer_initial_patience"].value_counts().sort_index().to_dict(),
        "seller_initial_distribution": output["seller_initial_patience"].value_counts().sort_index().to_dict(),
        "design": {
            "buyer": "profile-based only before disclosure",
            "seller": "profile base plus observable readiness and private evidence adjustment, clamped to 6-8",
            "dynamic": "one point per own action; buyer additionally spends one point after an issue disclosure",
            "termination": "an exhausted party makes a final offer or final decision; the counterparty must accept or walk away",
        },
        "interpretation": (
            "Patience values are preregistered experimental mechanisms motivated by alternating-offer bargaining; "
            "they are not estimates of real market participant patience."
        ),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    (OUTPUT_DIR / "patience_assignment_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"Patience assignment validation failed: {checks}")
    return audit


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
