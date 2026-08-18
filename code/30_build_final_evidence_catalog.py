from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parent
OUTPUT = ROOT / "results" / "work" / "mechanism_confirmation"
PRIOR_CATALOG = (
    ROOT / "results" / "causal_rules" / "combined_causal_rule_catalog.csv"
)
PROTOCOL = ROOT / "protocols" / "mechanism_confirmation_protocol.json"
NEW_RESULTS = OUTPUT / "mechanism_confirmation_rule_results.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rules = {item["rule_id"]: item for item in protocol["rules"]}
    alpha = float(protocol["primary_confirmation"]["alpha_per_rule"])
    detail_rows: list[dict[str, Any]] = []

    for row in read_csv(PRIOR_CATALOG):
        if row.get("catalog_status") != "active":
            continue
        detail_rows.append(
            {
                "rule_id": row["rule_id"],
                "mechanism_family": row["mechanism_family"],
                "rule_statement": row["rule_statement"],
                "source_experiment": row["source_experiment"],
                "evidence_tier": row["evidence_tier"],
                "causal_wtp_confirmed": as_bool(row["causal_wtp_confirmed"]),
                "causal_agreement_confirmed": as_bool(
                    row["causal_agreement_confirmed"]
                ),
                "causal_negotiated_price_confirmed": as_bool(
                    row["causal_negotiated_price_confirmed"]
                ),
                "pairs": int(row["pairs"]),
                "both_arm_agreement_pairs": int(row["both_arm_agreement_pairs"]),
                "mean_relative_wtp_effect": float(row["mean_relative_wtp_effect"]),
                "agreement_rate_effect": float(row["agreement_rate_effect"]),
                "mean_relative_negotiated_price_effect": float(
                    row["mean_relative_negotiated_price_effect"]
                ),
                "price_sign_flip_p_value": float(row["price_sign_flip_p_value"]),
                "scope": "Controlled LLM-agent simulator only",
            }
        )

    new_rows = read_csv(NEW_RESULTS)
    new_by_mechanism: dict[str, list[dict[str, Any]]] = {}
    for row in new_rows:
        rule = rules[row["rule_id"]]
        wtp_ci = json.loads(row["wtp_log10_95_ci"])
        wtp_confirmed = (
            float(row["mean_relative_wtp_effect"]) > 0
            and float(row["wtp_sign_flip_p_value"]) <= alpha
            and float(wtp_ci[0]) > 0
            and float(row["control_integrity"]) == 1.0
        )
        price_confirmed = row["confirmation_status"].startswith("strict_causal")
        evidence_tier = (
            "strict_causal_negotiated_price_rule"
            if price_confirmed
            else "causal_pricing_rule_wtp_level_price_not_confirmed"
        )
        statement = (
            f"IF the buyer's intended task requires {rule['topic']} and "
            f"{rule['treatment_fact'][0].lower() + rule['treatment_fact'][1:]} "
            "THEN effective WTP increases relative to leaving that condition unresolved."
        )
        item = {
            "rule_id": row["rule_id"],
            "mechanism_family": row["mechanism_id"],
            "rule_statement": statement,
            "source_experiment": "independent_mechanism_confirmation_v1",
            "evidence_tier": evidence_tier,
            "causal_wtp_confirmed": wtp_confirmed,
            "causal_agreement_confirmed": False,
            "causal_negotiated_price_confirmed": price_confirmed,
            "pairs": int(row["pairs"]),
            "both_arm_agreement_pairs": int(row["both_arm_agreement_pairs"]),
            "mean_relative_wtp_effect": float(row["mean_relative_wtp_effect"]),
            "agreement_rate_effect": float(row["agreement_rate_effect"]),
            "mean_relative_negotiated_price_effect": float(
                row["mean_relative_negotiated_price_effect"]
            ),
            "price_sign_flip_p_value": float(row["price_sign_flip_p_value"]),
            "scope": protocol["causal_scope"],
        }
        detail_rows.append(item)
        new_by_mechanism.setdefault(row["mechanism_id"], []).append(item)

    mechanism_rows: list[dict[str, Any]] = []
    prior_active = [
        item
        for item in detail_rows
        if item["mechanism_family"] == "grounded_task_fit_confirmation"
    ]
    mechanism_rows.append(
        {
            "mechanism_id": "grounded_task_fit_confirmation",
            "summary_rule": (
                "IF grounded evidence resolves whether a product property fits the "
                "buyer's prespecified task, THEN task fit changes effective WTP and "
                "can change agreement and negotiated price."
            ),
            "detail_rules": len(prior_active),
            "wtp_confirmed_detail_rules": sum(
                bool(item["causal_wtp_confirmed"]) for item in prior_active
            ),
            "strict_price_detail_rules": sum(
                bool(item["causal_negotiated_price_confirmed"])
                for item in prior_active
            ),
            "strict_mechanism_confirmed": "not_evaluated_under_two-rule_criterion",
            "mechanism_evidence_tier": (
                "established_mechanism_family_with_strict_detail_price_rules"
            ),
            "scope": protocol["causal_scope"],
        }
    )

    for mechanism_id in ("verification_cost", "integration_cost", "governance_risk"):
        selected = new_by_mechanism[mechanism_id]
        wtp_count = sum(bool(item["causal_wtp_confirmed"]) for item in selected)
        price_count = sum(
            bool(item["causal_negotiated_price_confirmed"]) for item in selected
        )
        strict_mechanism = price_count == len(selected)
        if strict_mechanism:
            tier = "strict_mechanism_confirmed"
        elif wtp_count == len(selected) and price_count > 0:
            tier = "mechanism_wtp_supported_partial_price_confirmation"
        elif wtp_count == len(selected):
            tier = "mechanism_wtp_supported_price_not_confirmed"
        else:
            tier = "mechanism_not_confirmed"
        mechanism_rows.append(
            {
                "mechanism_id": mechanism_id,
                "summary_rule": protocol["mechanism_summary_rules"][mechanism_id],
                "detail_rules": len(selected),
                "wtp_confirmed_detail_rules": wtp_count,
                "strict_price_detail_rules": price_count,
                "strict_mechanism_confirmed": strict_mechanism,
                "mechanism_evidence_tier": tier,
                "scope": protocol["causal_scope"],
            }
        )

    write_csv(OUTPUT / "final_causal_rule_evidence_catalog.csv", detail_rows)
    write_csv(OUTPUT / "final_mechanism_evidence_catalog.csv", mechanism_rows)
    (OUTPUT / "final_causal_rule_evidence_catalog.json").write_text(
        json.dumps(detail_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "final_mechanism_evidence_catalog.json").write_text(
        json.dumps(mechanism_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": "final_mechanism_evidence_catalog_complete",
        "active_detail_rule_records": len(detail_rows),
        "strict_causal_negotiated_price_rule_records": sum(
            bool(item["causal_negotiated_price_confirmed"]) for item in detail_rows
        ),
        "mechanism_families": len(mechanism_rows),
        "strict_new_mechanisms": [
            item["mechanism_id"]
            for item in mechanism_rows
            if item["strict_mechanism_confirmed"] is True
        ],
        "risk_mechanism_evidence_tier": next(
            item["mechanism_evidence_tier"]
            for item in mechanism_rows
            if item["mechanism_id"] == "governance_risk"
        ),
        "causal_scope": protocol["causal_scope"],
        "input_hashes": {
            str(PRIOR_CATALOG.relative_to(ROOT)): sha256(PRIOR_CATALOG),
            str(PROTOCOL.relative_to(ROOT)): sha256(PROTOCOL),
            str(NEW_RESULTS.relative_to(ROOT)): sha256(NEW_RESULTS),
        },
    }
    (OUTPUT / "final_evidence_catalog_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
