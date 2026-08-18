from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "results" / "work" / "negotiation_prep" / "train_negotiation_inputs.csv"
DESCRIPTION_PATH = ROOT / "results" / "work" / "product_descriptions" / "train_product_descriptions.csv"
REGISTRY_PATH = ROOT / "protocols" / "persona_registry.json"
OUTPUT_DIR = ROOT / "results" / "work" / "agents"
SEED = 20260813
PAIRING_VERSION = "balanced-random-fixed-pairs-v1"

DELIVERY_FIELDS = {
    "S3Bucket": "S3 bucket",
    "Download": "direct download",
    "RESTAPI": "REST API",
    "UIExport": "UI export",
    "Email": "email delivery",
    "FeedAPI": "feed API",
    "EnrichApp": "enrichment application",
}
UPDATE_FIELDS = {
    "monthly": "monthly",
    "weekly": "weekly",
    "daily": "daily",
    "ondemand": "on demand",
    "realtime": "real time",
}
FORMAT_FIELDS = {"csv": "CSV", "json": "JSON"}
OPAQUE_RELATIVE_FIELDS = [
    "History", "people", "entities", "products", "records", "events",
    "symbols", "assets", "requests", "features", "locations", "USD",
    "sources", "units",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def money_round(value: float) -> float:
    if value < 1:
        return round(max(value, 0.0001), 4)
    if value < 100:
        return round(value, 2)
    return float(round(value))


def balanced_random_ids(ids: list[str], n: int, rng: np.random.Generator) -> list[str]:
    values = (ids * ((n + len(ids) - 1) // len(ids)))[:n]
    rng.shuffle(values)
    return values


def relative_signal(value: float, reference: pd.Series) -> str:
    if value <= 0:
        return "not indicated"
    positive = reference[reference > 0]
    if positive.empty:
        return "not indicated"
    percentile = float((positive <= value).mean())
    if percentile < 1 / 3:
        return "lower catalog-relative signal"
    if percentile < 2 / 3:
        return "middle catalog-relative signal"
    return "higher catalog-relative signal"


def public_packet(row: pd.Series, description: str, reference: pd.DataFrame) -> dict[str, Any]:
    return {
        "product_id": int(row["Id"]),
        "platform_listing": description,
        "listing_provenance_note": (
            "Synthetic platform-style text generated only from source record fields; "
            "it is not an original vendor description or independent quality evidence."
        ),
        "countries_covered": int(round(float(row["NCountries"]))),
        "gdp_coverage_share_field": round(float(row["PercGDP"]), 6),
        "delivery_channels_listed": [
            label for field, label in DELIVERY_FIELDS.items() if int(round(float(row[field]))) == 1
        ],
        "update_options_listed": [
            label for field, label in UPDATE_FIELDS.items() if int(round(float(row[field]))) == 1
        ],
        "formats_listed": [
            label for field, label in FORMAT_FIELDS.items() if int(round(float(row[field]))) == 1
        ],
        "company_identifiers_indicated": bool(round(float(row["IdCompanies"]))),
        "individual_identifiers_indicated": bool(round(float(row["IdIndividuals"]))),
        "professional_services_indicated": bool(round(float(row["ProfServices"]))),
        "use_limitations_flag_indicated": bool(round(float(row["Limitations"]))),
        "opaque_catalog_relative_signals": {
            field: relative_signal(float(row[field]), reference[field])
            for field in OPAQUE_RELATIVE_FIELDS
            if float(row[field]) > 0
        },
        "interpretation_constraint": (
            "Opaque catalog-relative signals have undocumented units and must not be translated "
            "into record counts, time periods, quality ratings, or guarantees."
        ),
    }


def evidence_ledger(packet: dict[str, Any]) -> dict[str, Any]:
    supplied = [
        {"evidence_id": "platform:listing", "topic": "platform listing", "status": "supplied", "fact": packet["platform_listing"]},
        {"evidence_id": "record:delivery", "topic": "listed delivery channels", "status": "supplied", "fact": packet["delivery_channels_listed"]},
        {"evidence_id": "record:update", "topic": "listed update options", "status": "supplied", "fact": packet["update_options_listed"]},
        {"evidence_id": "record:format", "topic": "listed formats", "status": "supplied", "fact": packet["formats_listed"]},
        {"evidence_id": "record:coverage", "topic": "coverage fields", "status": "supplied", "fact": {"countries_covered": packet["countries_covered"], "gdp_coverage_share_field": packet["gdp_coverage_share_field"]}},
        {"evidence_id": "record:identifiers", "topic": "identifier flags", "status": "supplied", "fact": {"companies": packet["company_identifiers_indicated"], "individuals": packet["individual_identifiers_indicated"]}},
    ]
    missing_topics = [
        "data provenance documentation",
        "collection authorization documentation",
        "license or permitted-use text",
        "privacy or regulatory compliance evidence",
        "external certification or audit",
        "measured accuracy or error rate",
        "sample records or schema documentation",
        "service-level guarantee",
        "customer or adoption counts",
    ]
    missing = [
        {
            "evidence_id": f"missing:{index}",
            "topic": topic,
            "status": "not supplied in the source record",
            "fact": None,
        }
        for index, topic in enumerate(missing_topics, start=1)
    ]
    return {
        "ledger_version": "source-bounded-ledger-v1",
        "items": supplied + missing,
        "interpretation_constraint": (
            "Not supplied means unavailable in this experiment's source record; it does not prove "
            "that the product fails the criterion. Never invent a fact to fill a gap."
        ),
    }


def run() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = pd.read_csv(INPUT_PATH, low_memory=False)
    descriptions = pd.read_csv(DESCRIPTION_PATH, low_memory=False)
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if len(inputs) != 1912 or len(descriptions) != 1912:
        raise RuntimeError("Agent construction requires all 1,912 training products and descriptions")
    if set(inputs["Id"]) != set(descriptions["Id"]):
        raise RuntimeError("Description IDs do not exactly cover the training products")
    if {"LogPriceMo", "TransactionPriceUSD"} & set(inputs.columns):
        raise RuntimeError("Observed outcomes must not enter agent construction")
    if not descriptions["status"].eq("ok").all():
        raise RuntimeError("All descriptions must be successful before agent construction")

    descriptions = descriptions.set_index("Id")
    buyer_profiles = {item["profile_id"]: item for item in registry["buyer_profiles"]}
    seller_profiles = {item["profile_id"]: item for item in registry["seller_profiles"]}
    rng = np.random.default_rng(SEED)
    buyer_ids = balanced_random_ids(list(buyer_profiles), len(inputs), rng)
    seller_ids = balanced_random_ids(list(seller_profiles), len(inputs), rng)
    reference = inputs[OPAQUE_RELATIVE_FIELDS]

    rows: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(inputs.sort_values("Id").reset_index(drop=True).iterrows()):
        product_id = int(row["Id"])
        buyer = buyer_profiles[buyer_ids[position]]
        seller = seller_profiles[seller_ids[position]]
        anchor = float(row["m0_reference_usd"])
        buyer_low, buyer_high = map(float, buyer["ceiling_multiplier_range"])
        seller_low, seller_high = map(float, seller["floor_multiplier_range"])
        buyer_multiplier = float(rng.uniform(buyer_low, buyer_high))
        seller_multiplier = float(rng.uniform(seller_low, seller_high))
        buyer_ceiling = money_round(anchor * buyer_multiplier)
        seller_floor = money_round(anchor * seller_multiplier)
        packet = public_packet(row, str(descriptions.loc[product_id, "description"]), reference)
        ledger = evidence_ledger(packet)
        rows.append(
            {
                "product_id": product_id,
                "fixed_pair_id": f"PAIR-{product_id}",
                "pairing_seed": SEED,
                "pairing_version": PAIRING_VERSION,
                "buyer_profile_id": buyer["profile_id"],
                "seller_profile_id": seller["profile_id"],
                "buyer_private_context_json": json.dumps(buyer["private_context"], ensure_ascii=False, sort_keys=True),
                "seller_private_context_json": json.dumps(seller["private_context"], ensure_ascii=False, sort_keys=True),
                "buyer_ceiling_usd": buyer_ceiling,
                "seller_floor_usd": seller_floor,
                "buyer_ceiling_multiplier": round(buyer_multiplier, 8),
                "seller_floor_multiplier": round(seller_multiplier, 8),
                "m0_platform_reference_usd": anchor,
                "m0_oof_fold": int(row["m0_oof_fold"]),
                "public_product_packet_json": json.dumps(packet, ensure_ascii=False, sort_keys=True),
                "seller_evidence_ledger_json": json.dumps(ledger, ensure_ascii=False, sort_keys=True),
                "m0_m1_pair_lock": "same buyer context, seller context, ceiling, floor, and product; only the platform reference and dialogue may change",
                "known_context_exclusion": "persona variables are registered controls and cannot be claimed as discovered implicit product features",
            }
        )
    output = pd.DataFrame(rows).sort_values("product_id")
    output_path = OUTPUT_DIR / "train_fixed_agent_pairs.csv"
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    overlap = output["buyer_ceiling_usd"] >= output["seller_floor_usd"]
    output_with_overlap = output.assign(potential_price_overlap=overlap)
    audit = {
        "rows": len(output),
        "unique_products": int(output["product_id"].nunique()),
        "unique_pair_ids": int(output["fixed_pair_id"].nunique()),
        "buyer_profile_counts": output["buyer_profile_id"].value_counts().sort_index().to_dict(),
        "seller_profile_counts": output["seller_profile_id"].value_counts().sort_index().to_dict(),
        "profile_pair_cell_count": int(output.groupby(["buyer_profile_id", "seller_profile_id"]).ngroups),
        "private_bound_overlap_count_diagnostic_only": int(overlap.sum()),
        "private_bound_overlap_rate_diagnostic_only": float(overlap.mean()),
        "overlap_rate_by_buyer_profile": {
            key: float(value)
            for key, value in output_with_overlap.groupby("buyer_profile_id")[
                "potential_price_overlap"
            ].mean().sort_index().items()
        },
        "overlap_rate_by_seller_profile": {
            key: float(value)
            for key, value in output_with_overlap.groupby("seller_profile_id")[
                "potential_price_overlap"
            ].mean().sort_index().items()
        },
        "registered_buyer_multiplier_ranges": {
            profile_id: profile["ceiling_multiplier_range"]
            for profile_id, profile in buyer_profiles.items()
        },
        "registered_seller_multiplier_ranges": {
            profile_id: profile["floor_multiplier_range"]
            for profile_id, profile in seller_profiles.items()
        },
        "test_products_assigned": 0,
        "observed_transaction_outcome_used": False,
        "profile_frequencies_are_market_estimates": False,
        "numeric_ranges_are_literature_estimates": False,
        "same_agents_locked_for_m0_m1": True,
        "all_checks_passed": bool(
            len(output) == 1912
            and output["product_id"].nunique() == 1912
            and output["fixed_pair_id"].nunique() == 1912
            and output["buyer_ceiling_usd"].gt(0).all()
            and output["seller_floor_usd"].gt(0).all()
        ),
    }
    (OUTPUT_DIR / "agent_pair_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "status": "agents_constructed_without_api_calls",
        "seed": SEED,
        "pairing_version": PAIRING_VERSION,
        "training_products": len(output),
        "buyer_profiles": len(buyer_profiles),
        "seller_profiles": len(seller_profiles),
        "pairing_protocol": "independent balanced random permutation of registered buyer and seller profiles",
        "bound_protocol": "private bounds calibrated once from leakage-free M0 OOF anchor and then frozen for M0/M1",
        "experimental_status": "controlled theoretical contexts, not digital replicas of market participants",
        "input_sha256": sha256_file(INPUT_PATH),
        "description_sha256": sha256_file(DESCRIPTION_PATH),
        "registry_sha256": sha256_file(REGISTRY_PATH),
        "output_sha256": sha256_file(output_path),
    }
    (OUTPUT_DIR / "agent_construction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"manifest": manifest, "audit": audit}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
