from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parent
WORK = ROOT / "results" / "work" / "telecom"
OUTPUT = WORK / "features"
DISCOVERY = WORK / "discovery"
SPLIT = ROOT / "data" / "telecom_train_test_split.csv"
TRAIN_DISCOVERY = DISCOVERY / "telecom_train_discovery_v1.jsonl"
TEST_DISCOVERY = DISCOVERY / "telecom_test_discovery_v1.jsonl"
REFERENCE_PREVALENCE = 15 / 1912
MIN_FULL_TRAIN_SUPPORT = max(5, math.ceil(REFERENCE_PREVALENCE * 420))
MAX_PREVALENCE = 0.95


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")[:70]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_discovery(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [int(record["product_id"]) for record in records]
    if len(records) != expected_rows or len(set(ids)) != expected_rows:
        raise RuntimeError(f"{path.name} does not contain {expected_rows} unique products")
    if any(record.get("status") != "ok" for record in records):
        raise RuntimeError(f"{path.name} contains unsuccessful discovery records")
    if any(
        record.get("real_transaction_price_visible")
        or record.get("negotiated_outcome_visible")
        for record in records
    ):
        raise RuntimeError(f"{path.name} failed the target-visibility audit")
    return records


def record_keys(record: dict[str, Any]) -> tuple[set[str], set[str]]:
    terms: set[str] = set()
    term_states: set[str] = set()
    for feature in record["features"]:
        state = slug(str(feature["evidence_state"]))
        for raw_term in feature["source_terms"]:
            term = slug(str(raw_term))
            if not term:
                continue
            terms.add(term)
            term_states.add(f"{term}__{state}")
    return terms, term_states


def taxonomy_from_training(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    term_products: dict[str, set[int]] = defaultdict(set)
    state_products: dict[str, set[int]] = defaultdict(set)
    display_terms: dict[str, str] = {}
    for record in records:
        product_id = int(record["product_id"])
        terms, term_states = record_keys(record)
        for term in terms:
            term_products[term].add(product_id)
            display_terms.setdefault(term, term.replace("_", " "))
        for key in term_states:
            state_products[key].add(product_id)

    def eligible(products: set[int]) -> bool:
        prevalence = len(products) / len(records)
        return (
            len(products) >= MIN_FULL_TRAIN_SUPPORT
            and prevalence <= MAX_PREVALENCE
        )

    base_taxonomy = []
    for term, products in sorted(term_products.items()):
        if not eligible(products):
            continue
        base_taxonomy.append(
            {
                "canonical_id": f"context_term__{term}",
                "canonical_name": f"{display_terms[term]} evidence context",
                "definition": f"The target-free exchange discusses '{display_terms[term]}'.",
                "feature_type": "source_term",
                "aggregation_key": term,
                "product_count": len(products),
                "product_prevalence": len(products) / len(records),
            }
        )

    candidate_taxonomy = []
    for key, products in sorted(state_products.items()):
        if not eligible(products):
            continue
        term, state = key.rsplit("__", 1)
        candidate_taxonomy.append(
            {
                "canonical_id": f"score__term_state__{term}_{state}",
                "canonical_name": (
                    f"{display_terms.get(term, term.replace('_', ' '))} with "
                    f"{state.replace('_', ' ')}"
                ),
                "definition": (
                    f"The '{display_terms.get(term, term.replace('_', ' '))}' mechanism "
                    f"has {state.replace('_', ' ')} evidence."
                ),
                "family": "evidence_state_mechanism",
                "feature_type": "term_state",
                "aggregation_key": key,
                "product_count": len(products),
                "product_prevalence": len(products) / len(records),
            }
        )
    if len(base_taxonomy) < 12 or len(candidate_taxonomy) < 24:
        raise RuntimeError(
            "Training-only taxonomy is too small for the frozen 12+24 protocol: "
            f"base={len(base_taxonomy)}, candidate={len(candidate_taxonomy)}"
        )
    return base_taxonomy, candidate_taxonomy


def feature_pair(canonical_id: str) -> tuple[str, str]:
    return (
        f"implicit_{canonical_id}_observed",
        f"implicit_{canonical_id}_score",
    )


def encode(
    records: list[dict[str, Any]],
    base_taxonomy: list[dict[str, Any]],
    candidate_taxonomy: list[dict[str, Any]],
) -> tuple[pd.DataFrame, set[str], set[str]]:
    base_lookup = {
        str(item["aggregation_key"]): str(item["canonical_id"])
        for item in base_taxonomy
    }
    candidate_lookup = {
        str(item["aggregation_key"]): str(item["canonical_id"])
        for item in candidate_taxonomy
    }
    unseen_terms: set[str] = set()
    unseen_states: set[str] = set()
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: int(item["product_id"])):
        terms, states = record_keys(record)
        unseen_terms.update(terms - set(base_lookup))
        unseen_states.update(states - set(candidate_lookup))
        row: dict[str, Any] = {"product_id": int(record["product_id"])}
        for term, canonical_id in base_lookup.items():
            value = int(term in terms)
            observed, score = feature_pair(canonical_id)
            row[observed] = value
            row[score] = float(value)
        for key, canonical_id in candidate_lookup.items():
            value = int(key in states)
            observed, score = feature_pair(canonical_id)
            row[observed] = value
            row[score] = float(value)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.isna().any().any():
        raise RuntimeError("Encoded feature matrix contains missing values")
    return frame, unseen_terms, unseen_states


def main() -> None:
    required = [SPLIT, TRAIN_DISCOVERY, TEST_DISCOVERY]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing inputs: {missing}")
    split = pd.read_csv(SPLIT, usecols=["Id", "split"])
    train_records = read_discovery(TRAIN_DISCOVERY, 420)
    test_records = read_discovery(TEST_DISCOVERY, 106)
    train_ids = {int(record["product_id"]) for record in train_records}
    test_ids = {int(record["product_id"]) for record in test_records}
    expected_train = set(split.loc[split["split"].eq("train"), "Id"].astype(int))
    expected_test = set(split.loc[split["split"].eq("test"), "Id"].astype(int))
    if train_ids != expected_train or test_ids != expected_test or train_ids & test_ids:
        raise RuntimeError("Discovery IDs do not match the frozen train/test split")

    base_taxonomy, candidate_taxonomy = taxonomy_from_training(train_records)
    train, _, _ = encode(train_records, base_taxonomy, candidate_taxonomy)
    test, unseen_terms, unseen_states = encode(
        test_records, base_taxonomy, candidate_taxonomy
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_matrix": OUTPUT / "telecom_train_frozen_feature_matrix.csv",
        "test_matrix": OUTPUT / "telecom_test_frozen_feature_matrix.csv",
        "base_taxonomy": OUTPUT / "telecom_base_term_taxonomy.json",
        "candidate_taxonomy": OUTPUT / "telecom_term_state_taxonomy.json",
    }
    train.to_csv(paths["train_matrix"], index=False, encoding="utf-8-sig")
    test.to_csv(paths["test_matrix"], index=False, encoding="utf-8-sig")
    write_json(
        paths["base_taxonomy"],
        {
            "status": "frozen_from_telecom_training_discovery_only",
            "taxonomy": base_taxonomy,
            "test_rows_used": False,
            "test_targets_used": False,
        },
    )
    write_json(
        paths["candidate_taxonomy"],
        {
            "status": "frozen_from_telecom_training_discovery_only",
            "encoding": "term-by-evidence-state observed indicator",
            "taxonomy": candidate_taxonomy,
            "test_rows_used": False,
            "test_targets_used": False,
        },
    )
    manifest = {
        "status": "telecom_target_free_feature_matrices_complete",
        "train_rows": len(train),
        "test_rows": len(test),
        "base_term_features": len(base_taxonomy),
        "candidate_mechanism_features": len(candidate_taxonomy),
        "minimum_full_training_support": MIN_FULL_TRAIN_SUPPORT,
        "support_rule": (
            "max(5, ceil((15/1912) * n)); fixed by prevalence transfer from Financial"
        ),
        "taxonomy_fit_cohort": "420 training products only",
        "taxonomy_refit_on_test": False,
        "test_targets_used": False,
        "real_transaction_price_used": False,
        "negotiated_outcome_used": False,
        "unseen_test_terms_ignored": sorted(unseen_terms),
        "unseen_test_term_state_keys_ignored_count": len(unseen_states),
        "feature_type_counts": {
            "source_term": len(base_taxonomy),
            "term_by_evidence_state": len(candidate_taxonomy),
        },
        "input_hashes": {path.name: sha256_file(path) for path in required},
        "output_hashes": {name: sha256_file(path) for name, path in paths.items()},
    }
    write_json(OUTPUT / "telecom_feature_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
