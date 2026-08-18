from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


STAGE = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "work" / "rule_confirmation"
PROTOCOL = ROOT / "protocols" / "causal_rule_extension_protocol.json"
PLAN = ROOT / "results" / "causal_rules" / "extension_plan.csv"
RUNS = ROOT / "results" / "causal_rules" / "extension_rule_runs.jsonl"
RUN_MANIFEST = ROOT / "results" / "causal_rules" / "extension_rule_runs_manifest.json"
SEED = 20264517
SIGN_FLIP_DRAWS = 100_000
BOOTSTRAP_DRAWS = 20_000

UNRESOLVED = {
    "coverage": re.compile(
        r"coverage.*(?:unresolved|unconfirmed|not resolve)|"
        r"neither confirms nor rules out.*coverage|"
        r"does not resolve whether.*coverage",
        re.I,
    ),
    "update": re.compile(
        r"(?:update|refresh|cadence).{0,400}"
        r"(?:unresolved|unconfirmed|does not (?:confirm|resolve)|"
        r"neither confirm(?:s)? nor rule(?:s)? out)|"
        r"(?:unresolved|unconfirmed|does not (?:confirm|resolve)|"
        r"neither confirm(?:s)? nor rule(?:s)? out).{0,400}"
        r"(?:update|refresh|cadence)",
        re.I,
    ),
}

LEGACY_UPDATE_UNRESOLVED = re.compile(
    r"update (?:cadence|requirement).*(?:unresolved|unconfirmed|not resolve)|"
    r"neither confirms nor rules out.*update|"
    r"does not resolve whether.*update",
    re.I,
)


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


def sign_flip_p(values: np.ndarray, seed: int) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 1.0
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    completed = 0
    for start in range(0, SIGN_FLIP_DRAWS, 1000):
        width = min(1000, SIGN_FLIP_DRAWS - start)
        signs = rng.choice([-1.0, 1.0], size=(width, len(values)))
        simulated = np.abs((signs * values).mean(axis=1))
        exceed += int((simulated >= observed).sum())
        completed += width
    return float((exceed + 1) / (completed + 1))


def bootstrap(values: np.ndarray, seed: int) -> list[float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for start in range(0, BOOTSTRAP_DRAWS, 500):
        width = min(500, BOOTSTRAP_DRAWS - start)
        index = rng.integers(0, len(values), size=(width, len(values)))
        draws[start : start + width] = values[index].mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def mcnemar(control: np.ndarray, treatment: np.ndarray) -> dict[str, Any]:
    control = np.asarray(control, dtype=bool)
    treatment = np.asarray(treatment, dtype=bool)
    treatment_only = int((~control & treatment).sum())
    control_only = int((control & ~treatment).sum())
    discordant = treatment_only + control_only
    return {
        "treatment_only": treatment_only,
        "control_only": control_only,
        "discordant": discordant,
        "two_sided_exact_p_value": (
            float(stats.binomtest(treatment_only, discordant, 0.5).pvalue)
            if discordant
            else 1.0
        ),
    }


def load_runs() -> pd.DataFrame:
    rows = []
    for line in RUNS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        seller_text = " ".join(
            event["message"]
            for event in item["dialogue"]
            if event["actor"] == "seller"
        )
        rows.append(
            {
                "run_id": item["run_id"],
                "pair_id": item["pair_id"],
                "arm": item["arm"],
                "product_id": int(item["product_id"]),
                "rule_id": item["controlled_condition_id_audit_only"],
                "field_group": item["field_group"],
                "context_hash": item["non_intervention_context_hash"],
                "reference": float(item["m0_platform_reference_usd"]),
                "pre_wtp": float(item["buyer_pre_evidence_wtp_usd_private"]),
                "effective_wtp": float(item["buyer_ceiling_usd_private"]),
                "value_update": float(item["buyer_value_update_private"]),
                "agreement": item["outcome"] == "agreement",
                "price": item.get("negotiated_price_usd"),
                "control_unresolved": bool(
                    item["arm"] != "control"
                    or UNRESOLVED[item["field_group"]].search(seller_text)
                ),
                "legacy_control_unresolved": bool(
                    item["arm"] != "control"
                    or item["field_group"] != "update"
                    or LEGACY_UPDATE_UNRESOLVED.search(seller_text)
                ),
                "real_price_gate_enabled": bool(
                    item.get("m1_real_price_gate_private", {}).get("enabled", False)
                ),
            }
        )
    return pd.DataFrame(rows)


def analyze_rule(frame: pd.DataFrame, seed: int) -> dict[str, Any]:
    wide = frame.pivot(index="pair_id", columns="arm")
    value_delta = (
        wide[("value_update", "treatment")].to_numpy(float)
        - wide[("value_update", "control")].to_numpy(float)
    )
    wtp_log = np.log10(
        wide[("effective_wtp", "treatment")].to_numpy(float)
        / wide[("effective_wtp", "control")].to_numpy(float)
    )
    control_agreement = wide[("agreement", "control")].to_numpy(bool)
    treatment_agreement = wide[("agreement", "treatment")].to_numpy(bool)
    both = control_agreement & treatment_agreement
    control_price = pd.to_numeric(
        wide.loc[both, ("price", "control")], errors="coerce"
    ).to_numpy(float)
    treatment_price = pd.to_numeric(
        wide.loc[both, ("price", "treatment")], errors="coerce"
    ).to_numpy(float)
    valid = (
        np.isfinite(control_price)
        & np.isfinite(treatment_price)
        & (control_price > 0)
        & (treatment_price > 0)
    )
    price_log = np.log10(treatment_price[valid] / control_price[valid])
    control_value = np.where(
        control_agreement,
        pd.to_numeric(wide[("price", "control")], errors="coerce").fillna(0).to_numpy(float)
        / wide[("reference", "control")].to_numpy(float),
        0.0,
    )
    treatment_value = np.where(
        treatment_agreement,
        pd.to_numeric(wide[("price", "treatment")], errors="coerce").fillna(0).to_numpy(float)
        / wide[("reference", "treatment")].to_numpy(float),
        0.0,
    )
    transaction_delta = treatment_value - control_value
    return {
        "pairs": int(len(wide)),
        "mean_value_update_effect": float(value_delta.mean()),
        "value_update_p_value": sign_flip_p(value_delta, seed + 1),
        "value_update_95_ci": bootstrap(value_delta, seed + 2),
        "mean_relative_wtp_effect": float(10 ** wtp_log.mean() - 1),
        "wtp_log10_effect": float(wtp_log.mean()),
        "wtp_p_value": sign_flip_p(wtp_log, seed + 3),
        "wtp_log10_95_ci": bootstrap(wtp_log, seed + 4),
        "control_agreement_rate": float(control_agreement.mean()),
        "treatment_agreement_rate": float(treatment_agreement.mean()),
        "agreement_rate_effect": float(
            treatment_agreement.mean() - control_agreement.mean()
        ),
        "agreement_mcnemar": mcnemar(control_agreement, treatment_agreement),
        "both_arm_agreement_pairs": int(valid.sum()),
        "mean_relative_negotiated_price_effect": (
            float(10 ** price_log.mean() - 1) if len(price_log) else None
        ),
        "price_log10_effect": float(price_log.mean()) if len(price_log) else None,
        "price_p_value": sign_flip_p(price_log, seed + 5),
        "price_log10_95_ci": bootstrap(price_log, seed + 6),
        "mean_reference_normalized_transaction_value_effect": float(
            transaction_delta.mean()
        ),
        "transaction_value_p_value": sign_flip_p(transaction_delta, seed + 7),
        "transaction_value_95_ci": bootstrap(transaction_delta, seed + 8),
    }


def main() -> None:
    required = [PROTOCOL, PLAN, RUNS, RUN_MANIFEST]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing extension outputs: {missing}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    run_manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    if run_manifest["status"] != "completed" or run_manifest["completed_runs"] != 320:
        raise RuntimeError("Extension API execution is incomplete")
    frame = load_runs()
    if len(frame) != 320 or frame["run_id"].nunique() != 320:
        raise RuntimeError("Unexpected extension run count")
    arms = frame.groupby("pair_id")["arm"].apply(set)
    if len(arms) != 160 or not arms.eq({"control", "treatment"}).all():
        raise RuntimeError("Incomplete extension pairs")
    invariants = frame.groupby("pair_id").agg(
        products=("product_id", "nunique"),
        rules=("rule_id", "nunique"),
        fields=("field_group", "nunique"),
        contexts=("context_hash", "nunique"),
        references=("reference", "nunique"),
        pre_wtp=("pre_wtp", "nunique"),
    )
    if not invariants.eq(1).all().all():
        raise RuntimeError("Non-intervention state changed within a pair")
    if frame["real_price_gate_enabled"].any():
        raise RuntimeError("Real-price gate found in extension runs")
    control_integrity = (
        frame.loc[frame["arm"].eq("control")]
        .groupby("rule_id")["control_unresolved"]
        .mean()
        .to_dict()
    )
    legacy_control_integrity = (
        frame.loc[frame["arm"].eq("control")]
        .groupby("rule_id")["legacy_control_unresolved"]
        .mean()
        .to_dict()
    )
    results = []
    decisions = []
    alpha = float(protocol["primary_confirmation"]["alpha_per_rule"])
    minimum_both = int(
        protocol["primary_confirmation"]["minimum_both_arm_agreements"]
    )
    for index, rule in enumerate(protocol["rules"]):
        rule_id = str(rule["rule_id"])
        effect = analyze_rule(frame.loc[frame["rule_id"].eq(rule_id)], SEED + index * 100)
        results.append({"rule_id": rule_id, "field_group": rule["field_group"], **effect})
        wtp_pass = bool(
            effect["mean_relative_wtp_effect"] > 0
            and effect["wtp_p_value"] <= alpha
            and effect["wtp_log10_95_ci"][0] > 0
        )
        price_pass = bool(
            effect["both_arm_agreement_pairs"] >= minimum_both
            and effect["mean_relative_negotiated_price_effect"] is not None
            and effect["mean_relative_negotiated_price_effect"] > 0
            and effect["price_p_value"] <= alpha
            and effect["price_log10_95_ci"][0] > 0
        )
        integrity_pass = bool(
            effect["pairs"] == 80 and control_integrity.get(rule_id) == 1.0
        )
        confirmed = bool(wtp_pass and price_pass and integrity_pass)
        decisions.append(
            {
                "rule_id": rule_id,
                "status": (
                    "strict_causal_negotiated_price_rule_confirmed_in_simulator"
                    if confirmed
                    else "not_confirmed_as_strict_negotiated_price_rule"
                ),
                "bonferroni_alpha": alpha,
                "complete_pairs": effect["pairs"],
                "control_integrity": control_integrity.get(rule_id),
                "legacy_control_integrity_before_parser_fix": (
                    legacy_control_integrity.get(rule_id)
                ),
                "wtp_prerequisite_pass": wtp_pass,
                "minimum_both_arm_agreements_pass": (
                    effect["both_arm_agreement_pairs"] >= minimum_both
                ),
                "price_test_pass": price_pass,
                "mean_relative_wtp_effect": effect["mean_relative_wtp_effect"],
                "mean_relative_negotiated_price_effect": effect[
                    "mean_relative_negotiated_price_effect"
                ],
                "price_p_value": effect["price_p_value"],
                "price_log10_95_ci": effect["price_log10_95_ci"],
            }
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.json_normalize(results).to_csv(
        OUTPUT / "extension_rule_results.csv", index=False, encoding="utf-8-sig"
    )
    write_json(OUTPUT / "extension_confirmation_decisions.json", decisions)
    report = {
        "status": "causal_price_rule_extension_analysis_complete",
        "protocol_version": protocol["protocol_version"],
        "runs": len(frame),
        "pairs": int(frame["pair_id"].nunique()),
        "strict_rules_confirmed": sum(
            item["status"].startswith("strict_causal") for item in decisions
        ),
        "decisions": decisions,
        "causal_scope": "Controlled LLM-agent simulator only",
        "real_price_gate_used": False,
        "products_deleted": 0,
        "analysis_revision": {
            "type": "control_integrity_parser_bug_fix",
            "raw_runs_changed": False,
            "samples_or_thresholds_changed": False,
            "reason": (
                "The original update-field regular expression did not recognize "
                "daily, weekly, or monthly refresh as synonyms for update cadence, "
                "even when the control-arm seller explicitly said the capability "
                "was unresolved or could not be confirmed or ruled out."
            ),
        },
        "input_hashes": {
            str(path.relative_to(STAGE)): sha256_file(path) for path in required
        },
    }
    write_json(OUTPUT / "extension_analysis_report.json", report)
    lines = [
        "# 因果价格规则扩展确认结果",
        "",
        f"- 完成运行：{len(frame)}",
        f"- 完成配对：{frame['pair_id'].nunique()}",
        f"- 严格确认规则：{report['strict_rules_confirmed']}/2",
        "",
        "| 规则 | 支付意愿效应 | 双臂成交对 | 协商价效应 | 价格 p | 判定 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    result_by_rule = {item["rule_id"]: item for item in results}
    for decision in decisions:
        effect = result_by_rule[decision["rule_id"]]
        lines.append(
            f"| {decision['rule_id']} | {effect['mean_relative_wtp_effect']:+.2%} | "
            f"{effect['both_arm_agreement_pairs']} | "
            f"{effect['mean_relative_negotiated_price_effect']:+.2%} | "
            f"{effect['price_p_value']:.5f} | {decision['status']} |"
        )
    lines.extend(
        [
            "",
            "所有结论仅适用于受控 LLM 代理仿真，不能解释为真实数据市场随机实验。",
        ]
    )
    (STAGE / "RESULTS_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
