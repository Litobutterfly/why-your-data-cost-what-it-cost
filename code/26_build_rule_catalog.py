from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parent
OUTPUT = ROOT / "results" / "work" / "rule_confirmation"
PRIOR_CATALOG = (
    ROOT / "results" / "causal_rules" / "causal_pricing_rule_catalog.json"
)
EXTENSION_RESULTS = OUTPUT / "extension_rule_results.csv"
EXTENSION_DECISIONS = OUTPUT / "extension_confirmation_decisions.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_json_cell(value: str) -> Any:
    return json.loads(str(value))


def new_rule(row: pd.Series, decision: dict[str, Any]) -> dict[str, Any]:
    rule_id = str(row["rule_id"])
    if rule_id == "coverage_task_fit_price_premium":
        catalog_id = "causal_coverage_match_price_premium"
        field_scope = "coverage"
        statement = (
            "IF grounded evidence confirms that product coverage meets the "
            "buyer's prespecified task requirement, THEN effective willingness "
            "to pay and agreement probability increase, and the negotiated price "
            "among pairs agreeing under both arms rises relative to leaving "
            "coverage unresolved."
        )
        supersedes = "causal_coverage_match"
    elif rule_id == "update_cadence_task_fit_price_premium":
        catalog_id = "causal_update_cadence_match_price_premium"
        field_scope = "update cadence"
        statement = (
            "IF grounded evidence confirms that the product's update cadence "
            "meets the buyer's prespecified refresh requirement, THEN effective "
            "willingness to pay and agreement probability increase, and the "
            "negotiated price among pairs agreeing under both arms rises relative "
            "to leaving update cadence unresolved."
        )
        supersedes = None
    else:
        raise ValueError(f"Unexpected extension rule: {rule_id}")

    result = {
        "rule_id": catalog_id,
        "rule_statement": statement,
        "paper_safe_claim": (
            "Within the controlled LLM-agent simulator, confirming this "
            "prespecified task-fit property causally increases effective WTP and "
            "agreement probability and raises negotiated price conditional on "
            "agreement under both arms."
        ),
        "evidence_tier": "strict_causal_negotiated_price_rule",
        "catalog_status": "active",
        "mechanism_family": "grounded_task_fit_confirmation",
        "field_scope": field_scope,
        "contrast": "confirmed grounded property versus unresolved property",
        "source_experiment": "independent_extension_confirmation_v1",
        "pairs": int(row["pairs"]),
        "both_arm_agreement_pairs": int(row["both_arm_agreement_pairs"]),
        "control_agreement_rate": float(row["control_agreement_rate"]),
        "treatment_agreement_rate": float(row["treatment_agreement_rate"]),
        "agreement_rate_effect": float(row["agreement_rate_effect"]),
        "mean_relative_wtp_effect": float(row["mean_relative_wtp_effect"]),
        "wtp_sign_flip_p_value": float(row["wtp_p_value"]),
        "wtp_log10_95_ci": parse_json_cell(row["wtp_log10_95_ci"]),
        "mean_relative_negotiated_price_effect": float(
            row["mean_relative_negotiated_price_effect"]
        ),
        "price_sign_flip_p_value": float(row["price_p_value"]),
        "price_log10_95_ci": parse_json_cell(row["price_log10_95_ci"]),
        "agreement_mcnemar": {
            "treatment_only": int(row["agreement_mcnemar.treatment_only"]),
            "control_only": int(row["agreement_mcnemar.control_only"]),
            "discordant": int(row["agreement_mcnemar.discordant"]),
            "two_sided_exact_p_value": float(
                row["agreement_mcnemar.two_sided_exact_p_value"]
            ),
        },
        "bonferroni_alpha": float(decision["bonferroni_alpha"]),
        "control_integrity": float(decision["control_integrity"]),
        "causal_wtp_confirmed": bool(decision["wtp_prerequisite_pass"]),
        "causal_agreement_confirmed": bool(
            float(row["agreement_mcnemar.two_sided_exact_p_value"])
            <= float(decision["bonferroni_alpha"])
            and float(row["agreement_rate_effect"]) > 0
        ),
        "causal_negotiated_price_confirmed": True,
        "conditional_price_estimand": True,
    }
    if supersedes:
        result["supersedes_rule_id"] = supersedes
    return result


def main() -> None:
    required = [PRIOR_CATALOG, EXTENSION_RESULTS, EXTENSION_DECISIONS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing catalog inputs: {missing}")

    prior = json.loads(PRIOR_CATALOG.read_text(encoding="utf-8"))
    decisions = {
        item["rule_id"]: item
        for item in json.loads(EXTENSION_DECISIONS.read_text(encoding="utf-8"))
    }
    results = pd.read_csv(EXTENSION_RESULTS)
    if set(results["rule_id"]) != set(decisions):
        raise RuntimeError("Extension results and decisions do not align")
    if not all(
        item["status"]
        == "strict_causal_negotiated_price_rule_confirmed_in_simulator"
        for item in decisions.values()
    ):
        raise RuntimeError("Only confirmed extension rules may enter the catalog")

    combined = []
    for item in prior:
        copied = dict(item)
        copied.setdefault("mechanism_family", "grounded_task_fit_confirmation")
        if copied["rule_id"] == "causal_coverage_match":
            copied["catalog_status"] = "superseded_by_strict_confirmation"
            copied["superseded_by_rule_id"] = (
                "causal_coverage_match_price_premium"
            )
        else:
            copied["catalog_status"] = "active"
        combined.append(copied)

    combined.extend(
        new_rule(row, decisions[str(row["rule_id"])] )
        for _, row in results.iterrows()
    )
    active = [item for item in combined if item["catalog_status"] == "active"]
    strict = [
        item
        for item in active
        if item["evidence_tier"] == "strict_causal_negotiated_price_rule"
    ]
    summary = {
        "status": "combined_causal_rule_catalog_complete",
        "historical_records": len(combined),
        "active_rules": len(active),
        "strict_causal_negotiated_price_rule_records": len(strict),
        "strict_rule_mechanism_families": len(
            {item["mechanism_family"] for item in strict}
        ),
        "causal_scope": "Controlled LLM-agent simulator only",
        "conditional_price_effect_note": (
            "Negotiated-price effects are estimated only among product-persona "
            "pairs that agree under both treatment and control."
        ),
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in required
        },
    }
    write_json(OUTPUT / "combined_causal_rule_catalog.json", combined)
    pd.json_normalize(combined).to_csv(
        OUTPUT / "combined_causal_rule_catalog.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_json(OUTPUT / "combined_causal_rule_catalog_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
