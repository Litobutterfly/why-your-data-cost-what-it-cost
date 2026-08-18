from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAIR_PATH = ROOT / "results" / "work" / "agents" / "train_fixed_agent_pairs.csv"
CONDITION_PATH = ROOT / "results" / "work" / "quality_conditions" / "controlled_quality_conditions.csv"
TEST_PATH = ROOT / "results" / "work" / "negotiation_prep" / "test_model_inputs.csv"
OUTPUT_DIR = ROOT / "results" / "work" / "negotiation"
VERSION = "full-training-selection-v1-frozen"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(PAIR_PATH, low_memory=False)
    conditions = pd.read_csv(CONDITION_PATH, low_memory=False)
    test_ids = set(pd.read_csv(TEST_PATH, usecols=["Id"])["Id"].astype(int))
    condition_columns = [
        "product_id", "controlled_condition_id", "controlled_condition_present",
        "observable_readiness_level",
    ]
    selected = pairs.merge(
        conditions[condition_columns], on="product_id", validate="one_to_one"
    )
    selected["release_slot"] = selected["product_id"].map(lambda value: f"full-{int(value)}")
    selected["potential_price_overlap"] = (
        selected["buyer_ceiling_usd"] >= selected["seller_floor_usd"]
    )
    selected["buyer_condition_relevant"] = True
    keep = [
        "release_slot", "product_id", "controlled_condition_id",
        "controlled_condition_present", "observable_readiness_level",
        "potential_price_overlap", "buyer_condition_relevant", "buyer_profile_id",
        "seller_profile_id", "fixed_pair_id", "m0_platform_reference_usd",
        "buyer_ceiling_usd", "seller_floor_usd", "public_product_packet_json",
        "buyer_private_context_json", "seller_private_context_json",
    ]
    selected = selected[keep].sort_values("product_id").reset_index(drop=True)
    output = OUTPUT_DIR / "m0_full_train_selection.csv"
    selected.to_csv(output, index=False, encoding="utf-8-sig")
    checks = {
        "exactly_1912_training_products": len(selected) == 1912,
        "unique_products": selected["product_id"].nunique() == len(selected),
        "same_ids_as_frozen_pairs": set(selected["product_id"]) == set(pairs["product_id"]),
        "test_products_excluded": set(selected["product_id"]).isdisjoint(test_ids),
        "one_condition_per_product": not selected["product_id"].duplicated().any(),
        "private_bounds_positive": bool(
            selected["buyer_ceiling_usd"].gt(0).all()
            and selected["seller_floor_usd"].gt(0).all()
        ),
    }
    audit = {
        "status": "full_training_selection_frozen",
        "version": VERSION,
        "products": len(selected),
        "condition_counts": selected["controlled_condition_id"].value_counts().sort_index().to_dict(),
        "buyer_profile_counts": selected["buyer_profile_id"].value_counts().sort_index().to_dict(),
        "seller_profile_counts": selected["seller_profile_id"].value_counts().sort_index().to_dict(),
        "potential_overlap_rate": float(selected["potential_price_overlap"].mean()),
        "real_transaction_price_used": False,
        "dialogue_outcome_used": False,
        "m1_result_used": False,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "sha256": {
            PAIR_PATH.name: sha256_file(PAIR_PATH),
            CONDITION_PATH.name: sha256_file(CONDITION_PATH),
            output.name: sha256_file(output),
        },
    }
    (OUTPUT_DIR / "m0_full_train_selection_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"Full selection checks failed: {checks}")
    return audit


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
