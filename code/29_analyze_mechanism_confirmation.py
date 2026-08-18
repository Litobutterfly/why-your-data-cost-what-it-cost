from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STAGE = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "work" / "mechanism_confirmation"
PROTOCOL = ROOT / "protocols" / "mechanism_confirmation_protocol.json"
RUNS = ROOT / "results" / "mechanisms" / "mechanism_confirmation_runs.jsonl"
RUN_MANIFEST = ROOT / "results" / "mechanisms" / "mechanism_confirmation_runs_manifest.json"
SEED = 20264518
UNRESOLVED = re.compile(
    r"(?:unresolved|does not (?:establish|confirm|resolve)|"
    r"neither confirm(?:s)? nor rule(?:s)? out|cannot (?:confirm|verify))",
    re.I,
)
GENERIC_UNRESOLVED = re.compile(
    r"(?:unresolved|does not (?:establish|confirm|resolve)|"
    r"neither confirm(?:s)? nor rule(?:s)? out|cannot (?:confirm|verify))",
    re.I,
)


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 1.0
    observed = abs(float(values.mean()))
    if len(values) <= 18:
        simulated = []
        for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
            simulated.append(abs(float(np.mean(values * np.asarray(signs)))))
        return float(np.mean(np.asarray(simulated) >= observed - 1e-15))
    rng = np.random.default_rng(20264520 + len(values))
    exceed = 0
    draws = 100_000
    for start in range(0, draws, 1000):
        width = min(1000, draws - start)
        signs = rng.choice((-1.0, 1.0), size=(width, len(values)))
        simulated = np.abs((signs * values).mean(axis=1))
        exceed += int((simulated >= observed - 1e-15).sum())
    return float((exceed + 1) / (draws + 1))


def bootstrap(values: np.ndarray, seed: int) -> list[float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(10_000, len(values)))
    means = values[index].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def load_runs() -> pd.DataFrame:
    rows = []
    for line in RUNS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        seller_text = " ".join(
            event["message"] for event in item["dialogue"] if event["actor"] == "seller"
        )
        rows.append(
            {
                "run_id": item["run_id"],
                "pair_id": item["pair_id"],
                "arm": item["arm"],
                "product_id": int(item["product_id"]),
                "rule_id": item["controlled_condition_id_audit_only"],
                "context_hash": item["non_intervention_context_hash"],
                "reference": float(item["m0_platform_reference_usd"]),
                "pre_wtp": float(item["buyer_pre_evidence_wtp_usd_private"]),
                "effective_wtp": float(item["buyer_ceiling_usd_private"]),
                "agreement": item["outcome"] == "agreement",
                "price": item.get("negotiated_price_usd"),
                "control_unresolved": bool(
                    item["arm"] != "control"
                    or GENERIC_UNRESOLVED.search(seller_text)
                ),
                "real_price_gate_enabled": bool(
                    item.get("m1_real_price_gate_private", {}).get("enabled", False)
                ),
            }
        )
    return pd.DataFrame(rows)


def analyze_rule(frame: pd.DataFrame, seed: int) -> dict[str, Any]:
    wide = frame.pivot(index="pair_id", columns="arm")
    control_wtp = wide[("effective_wtp", "control")].to_numpy(float)
    treatment_wtp = wide[("effective_wtp", "treatment")].to_numpy(float)
    wtp_relative = treatment_wtp / control_wtp - 1.0
    wtp_log = np.log10(treatment_wtp / control_wtp)
    control_agreement = wide[("agreement", "control")].to_numpy(bool)
    treatment_agreement = wide[("agreement", "treatment")].to_numpy(bool)
    both = control_agreement & treatment_agreement
    control_price = pd.to_numeric(
        wide[("price", "control")], errors="coerce"
    ).to_numpy(float)
    treatment_price = pd.to_numeric(
        wide[("price", "treatment")], errors="coerce"
    ).to_numpy(float)
    valid_price = both & np.isfinite(control_price) & np.isfinite(treatment_price)
    price_relative = treatment_price[valid_price] / control_price[valid_price] - 1.0
    price_log = np.log10(treatment_price[valid_price] / control_price[valid_price])
    return {
        "pairs": int(len(wide)),
        "mean_relative_wtp_effect": float(wtp_relative.mean()),
        "positive_wtp_pair_share": float((wtp_relative > 0).mean()),
        "wtp_sign_flip_p_value": exact_sign_flip_p(wtp_log),
        "wtp_log10_95_ci": bootstrap(wtp_log, seed),
        "control_agreement_rate": float(control_agreement.mean()),
        "treatment_agreement_rate": float(treatment_agreement.mean()),
        "agreement_rate_effect": float(
            treatment_agreement.mean() - control_agreement.mean()
        ),
        "both_arm_agreement_pairs": int(valid_price.sum()),
        "mean_relative_negotiated_price_effect": (
            float(price_relative.mean()) if len(price_relative) else None
        ),
        "price_sign_flip_p_value": (
            exact_sign_flip_p(price_log) if len(price_log) else None
        ),
        "price_log10_95_ci": (
            bootstrap(price_log, seed + 1) if len(price_log) else None
        ),
    }


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    expected_runs = int(protocol["sample"]["total_runs"])
    expected_pairs = int(protocol["sample"]["total_pairs"])
    if manifest["status"] != "completed" or manifest["completed_runs"] != expected_runs:
        raise RuntimeError("Mechanism confirmation execution is incomplete")
    frame = load_runs()
    if len(frame) != expected_runs or frame["run_id"].nunique() != expected_runs:
        raise RuntimeError("Unexpected mechanism confirmation run count")
    if frame["real_price_gate_enabled"].any():
        raise RuntimeError("Real-price gate found in mechanism confirmation")
    pair_audit = frame.groupby("pair_id").agg(
        arms=("arm", lambda values: set(values)),
        products=("product_id", "nunique"),
        contexts=("context_hash", "nunique"),
        references=("reference", "nunique"),
        pre_wtp=("pre_wtp", "nunique"),
    )
    if not pair_audit["arms"].eq({"control", "treatment"}).all():
        raise RuntimeError("Confirmation pairs are incomplete")
    if not pair_audit.drop(columns="arms").eq(1).all().all():
        raise RuntimeError("Non-intervention state changed within a confirmation pair")

    thresholds = protocol["primary_confirmation"]
    rule_to_mechanism = {
        rule["rule_id"]: rule["mechanism_id"] for rule in protocol["rules"]
    }
    results = []
    for index, rule in enumerate(protocol["rules"]):
        rule_id = rule["rule_id"]
        subset = frame.loc[frame["rule_id"].eq(rule_id)]
        effect = analyze_rule(subset, SEED + index * 10)
        control_integrity = float(
            subset.loc[subset["arm"].eq("control"), "control_unresolved"].mean()
        )
        price_effect = effect["mean_relative_negotiated_price_effect"]
        screen_pass = bool(
            effect["pairs"] == int(protocol["sample"]["pairs_per_detail_rule"])
            and control_integrity >= thresholds["require_control_integrity"]
            and effect["mean_relative_wtp_effect"] > 0
            and effect["wtp_log10_95_ci"][0] > 0
            and effect["wtp_sign_flip_p_value"] <= thresholds["alpha_per_rule"]
            and effect["both_arm_agreement_pairs"] >= thresholds["minimum_both_arm_agreements"]
            and price_effect is not None
            and price_effect > 0
            and effect["price_log10_95_ci"][0] > 0
            and effect["price_sign_flip_p_value"] <= thresholds["alpha_per_rule"]
        )
        results.append(
            {
                "rule_id": rule_id,
                "mechanism_id": rule_to_mechanism[rule_id],
                "confirmation_status": (
                    "strict_causal_negotiated_price_rule_confirmed_in_simulator"
                    if screen_pass
                    else "not_confirmed_as_strict_negotiated_price_rule"
                ),
                "control_integrity": control_integrity,
                **effect,
            }
        )

    decisions = []
    for mechanism_id in sorted(set(rule_to_mechanism.values())):
        selected = [item for item in results if item["mechanism_id"] == mechanism_id]
        advanced = [
            item["rule_id"]
            for item in selected
            if item["confirmation_status"]
            == "strict_causal_negotiated_price_rule_confirmed_in_simulator"
        ]
        decisions.append(
            {
                "mechanism_id": mechanism_id,
                "detail_rules_screened": len(selected),
                "detail_rules_advanced": advanced,
                "mechanism_status": (
                    "strict_mechanism_confirmed"
                    if len(advanced) == 2
                    else "mechanism_not_confirmed"
                ),
                "causal_confirmation_status": (
                    "confirmed_in_controlled_simulator" if len(advanced) == 2 else "not_confirmed"
                ),
            }
        )

    pd.json_normalize(results).to_csv(
        OUTPUT / "mechanism_confirmation_rule_results.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT / "mechanism_confirmation_decisions.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "mechanism_confirmation_analysis_complete",
        "runs": len(frame),
        "pairs": int(frame["pair_id"].nunique()),
        "rules": results,
        "mechanisms": decisions,
        "causal_scope": "Controlled LLM-agent simulator only",
        "development_only": False,
        "independent_confirmation_completed": True,
        "real_price_gate_used": False,
        "products_deleted": 0,
    }
    (OUTPUT / "mechanism_confirmation_analysis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
