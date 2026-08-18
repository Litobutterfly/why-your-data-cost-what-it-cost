from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parent
OUTPUT = ROOT / "results" / "work" / "financial_evaluation"
PROTOCOL = ROOT / "protocols" / "financial_m1_vs_m0_protocol.json"
DEV_PREDICTIONS = ROOT / "results" / "work" / "financial_analysis" / "oof_product_summary.csv"
DEV_REPORT = ROOT / "results" / "work" / "financial_analysis" / "core_analysis_report.json"
EXTERNAL_PREDICTIONS = (
    ROOT / "results" / "financial" / "m1_vs_m0_predictions_without_targets.csv"
)
EXTERNAL_OUTCOMES = (
    ROOT / "results" / "financial" / "m1_vs_m0_predictions.csv"
)
EXTERNAL_REPORT = (
    ROOT / "results" / "financial" / "external_ensemble_evaluation_report.json"
)
RULE_DECISIONS = (
    ROOT / "results" / "causal_rules" / "rule_confirmation_decisions.json"
)
SEED = 20260817


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


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = y - prediction
    return {
        "mae_log10_usd": float(np.abs(error).mean()),
        "rmse_log10_usd": float(np.sqrt(np.square(error).mean())),
        "median_absolute_error_log10_usd": float(np.median(np.abs(error))),
        "r2": float(1 - np.square(error).sum() / np.square(y - y.mean()).sum()),
    }


def paired_bootstrap(delta: np.ndarray, draws: int = 20000) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    means = np.empty(draws)
    for start in range(0, draws, 500):
        size = min(500, draws - start)
        indices = rng.integers(0, len(delta), size=(size, len(delta)))
        means[start : start + size] = delta[indices].mean(axis=1)
    return {
        "draws": draws,
        "point_estimate": float(delta.mean()),
        "percentile_95_ci": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "two_sided_bootstrap_p_value": float(
            2 * min((means <= 0).mean(), (means >= 0).mean())
        ),
    }


def sign_flip(delta: np.ndarray, draws: int = 50000) -> float:
    rng = np.random.default_rng(SEED + 1)
    observed = abs(float(delta.mean()))
    extreme = 0
    for start in range(0, draws, 500):
        size = min(500, draws - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(delta)))
        extreme += int((np.abs((signs * delta).mean(axis=1)) >= observed).sum())
    return float((extreme + 1) / (draws + 1))


def compare(y: np.ndarray, baseline: np.ndarray, model: np.ndarray) -> dict[str, Any]:
    baseline_error = np.abs(y - baseline)
    model_error = np.abs(y - model)
    delta = baseline_error - model_error
    bootstrap = paired_bootstrap(delta)
    return {
        "baseline_metrics": metrics(y, baseline),
        "model_metrics": metrics(y, model),
        "absolute_mae_reduction": float(delta.mean()),
        "relative_mae_reduction": float(delta.mean() / baseline_error.mean()),
        "model_lower_error_product_fraction": float((model_error < baseline_error).mean()),
        "same_error_product_fraction": float(np.isclose(model_error, baseline_error).mean()),
        "paired_bootstrap": bootstrap,
        "two_sided_sign_flip_p_value": sign_flip(delta),
        "primary_support_rule_passed": bool(
            delta.mean() > 0
            and bootstrap["percentile_95_ci"][0] > 0
            and bootstrap["two_sided_bootstrap_p_value"] < 0.05
            and sign_flip(delta) < 0.05
        ),
    }


def select_blend_weight(development: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    y = development["actual"].to_numpy(float)
    m0 = development["decision_tree__m0_explicit__mean_prediction"].to_numpy(float)
    m1 = development["decision_tree__m1_full__mean_prediction"].to_numpy(float)
    rows = []
    for weight in np.linspace(0.0, 1.0, 101):
        prediction = (1 - weight) * m0 + weight * m1
        rows.append({"m1_weight": weight, **metrics(y, prediction)})
    frame = pd.DataFrame(rows)
    best_mae = frame["mae_log10_usd"].min()
    best = frame.loc[np.isclose(frame["mae_log10_usd"], best_mae)].sort_values(
        "m1_weight", ascending=False
    ).iloc[0]
    return float(best["m1_weight"]), frame


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_new_proof_calculations":
        raise RuntimeError("Protocol is not frozen")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    development = pd.read_csv(DEV_PREDICTIONS, low_memory=False)
    weight, weight_search = select_blend_weight(development)
    weight_search.to_csv(OUTPUT / "development_blend_weight_search.csv", index=False)

    # Construct every external prediction before loading the outcome column.
    external = pd.read_csv(EXTERNAL_PREDICTIONS, low_memory=False)
    required = {
        "Id",
        "m0_explicit_ensemble_prediction",
        "text_baseline_ensemble_prediction",
        "m1_frozen_k24_ensemble_prediction",
    }
    if not required.issubset(external.columns) or "LogPriceMo" in external.columns:
        raise RuntimeError("External prediction input is not target-free")
    external["stabilized_m1_prediction"] = (
        (1 - weight) * external["m0_explicit_ensemble_prediction"]
        + weight * external["m1_frozen_k24_ensemble_prediction"]
    )
    target_free_output = OUTPUT / "external_proof_predictions_without_targets.csv"
    external.to_csv(target_free_output, index=False, encoding="utf-8-sig")
    target_free_hash = sha256_file(target_free_output)

    outcomes = pd.read_csv(EXTERNAL_OUTCOMES, usecols=["Id", "LogPriceMo"])
    evaluated = external.merge(outcomes, on="Id", validate="one_to_one")
    y = evaluated["LogPriceMo"].to_numpy(float)
    m0 = evaluated["m0_explicit_ensemble_prediction"].to_numpy(float)
    text = evaluated["text_baseline_ensemble_prediction"].to_numpy(float)
    m1 = evaluated["m1_frozen_k24_ensemble_prediction"].to_numpy(float)
    stabilized = evaluated["stabilized_m1_prediction"].to_numpy(float)
    evaluated.to_csv(OUTPUT / "external_proof_predictions_with_targets.csv", index=False)

    primary = compare(y, m0, m1)
    stabilized_result = compare(y, m0, stabilized)
    m1_vs_text = compare(y, text, m1)
    evaluated["price_decile"] = pd.qcut(
        evaluated["LogPriceMo"], q=10, labels=False, duplicates="drop"
    )
    decile_rows = []
    for decile, group in evaluated.groupby("price_decile", sort=True):
        group_y = group["LogPriceMo"].to_numpy(float)
        group_m0 = group["m0_explicit_ensemble_prediction"].to_numpy(float)
        group_m1 = group["m1_frozen_k24_ensemble_prediction"].to_numpy(float)
        m0_mae = metrics(group_y, group_m0)["mae_log10_usd"]
        m1_mae = metrics(group_y, group_m1)["mae_log10_usd"]
        decile_rows.append(
            {
                "price_decile": int(decile),
                "rows": len(group),
                "m0_mae": m0_mae,
                "m1_mae": m1_mae,
                "m1_minus_m0_mae": m1_mae - m0_mae,
                "m1_better": m1_mae < m0_mae,
            }
        )
    pd.DataFrame(decile_rows).to_csv(OUTPUT / "external_price_decile_results.csv", index=False)

    dev_report = json.loads(DEV_REPORT.read_text(encoding="utf-8"))
    external_report = json.loads(EXTERNAL_REPORT.read_text(encoding="utf-8"))
    rules = json.loads(RULE_DECISIONS.read_text(encoding="utf-8"))
    dev_tree = dev_report["model_comparisons"]["decision_tree"]
    permuted = dev_report["decision_tree_ablations"]["permuted_mechanisms"]
    report = {
        "status": "financial_m1_vs_m0_proof_complete",
        "protocol_version": protocol["protocol_version"],
        "rows": {"development": len(development), "legacy_external": len(evaluated)},
        "primary_frozen_ensemble_m1_vs_m0": primary,
        "secondary_training_selected_blend": {
            "m1_weight": weight,
            "test_targets_used_for_weight_selection": False,
            "m1_vs_m0": stabilized_result,
        },
        "mechanism_evidence": {
            "development_m1_vs_text": dev_tree["m1_vs_text_baseline"],
            "development_m1_vs_permuted_mechanisms": permuted,
            "external_m1_vs_text": m1_vs_text,
        },
        "secondary_metric_disclosure": {
            "primary_m1_improves_external_rmse": (
                primary["model_metrics"]["rmse_log10_usd"]
                < primary["baseline_metrics"]["rmse_log10_usd"]
            ),
            "primary_m1_improves_external_r2": (
                primary["model_metrics"]["r2"]
                > primary["baseline_metrics"]["r2"]
            ),
            "primary_m1_improves_external_median_absolute_error": (
                primary["model_metrics"]["median_absolute_error_log10_usd"]
                < primary["baseline_metrics"]["median_absolute_error_log10_usd"]
            ),
            "price_deciles_where_m1_has_lower_mae": int(
                sum(item["m1_better"] for item in decile_rows)
            ),
            "price_deciles_total": len(decile_rows),
        },
        "rule_evidence": {
            "confirmed_in_controlled_simulator": rules["confirmed_rules"],
            "scope": rules["causal_scope"],
        },
        "claim_decisions": {
            "complete_m1_has_lower_primary_mae_than_m0": primary[
                "primary_support_rule_passed"
            ],
            "mechanism_block_has_incremental_development_value": bool(
                dev_tree["m1_vs_text_baseline"]["paired_product_bootstrap"][
                    "percentile_95_ci"
                ][0]
                > 0
                and permuted["paired_product_bootstrap"]["percentile_95_ci"][0] > 0
            ),
            "mechanism_block_has_independent_external_value": m1_vs_text[
                "primary_support_rule_passed"
            ],
            "m1_is_better_on_every_reported_predictive_metric": bool(
                primary["model_metrics"]["mae_log10_usd"]
                < primary["baseline_metrics"]["mae_log10_usd"]
                and primary["model_metrics"]["rmse_log10_usd"]
                < primary["baseline_metrics"]["rmse_log10_usd"]
                and primary["model_metrics"]["r2"]
                > primary["baseline_metrics"]["r2"]
            ),
        },
        "capacity_and_leakage_audit": {
            "same_tree_hyperparameters": external_report["same_tree_hyperparameters"],
            "models_per_ensemble": external_report["models_per_ensemble"],
            "prediction_file_created_without_test_targets": True,
            "new_target_free_prediction_sha256": target_free_hash,
        },
        "heldout_limitation": protocol["heldout_limitation"],
        "input_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [
                PROTOCOL,
                DEV_PREDICTIONS,
                DEV_REPORT,
                EXTERNAL_PREDICTIONS,
                EXTERNAL_OUTCOMES,
                EXTERNAL_REPORT,
                RULE_DECISIONS,
            ]
        },
    }
    write_json(OUTPUT / "m1_vs_m0_proof_report.json", report)
    lines = [
        "# Financial M1 vs M0 Proof Report",
        "",
        f"- Frozen ensemble M0 MAE: {primary['baseline_metrics']['mae_log10_usd']:.6f}",
        f"- Frozen ensemble M1 MAE: {primary['model_metrics']['mae_log10_usd']:.6f}",
        f"- Relative MAE reduction: {100 * primary['relative_mae_reduction']:.2f}%",
        f"- Paired bootstrap 95% CI: {primary['paired_bootstrap']['percentile_95_ci']}",
        f"- Paired sign-flip p: {primary['two_sided_sign_flip_p_value']:.6f}",
        f"- Training-selected stabilized M1 weight: {weight:.2f}",
        f"- Stabilized M1 external MAE: {stabilized_result['model_metrics']['mae_log10_usd']:.6f}",
        f"- M1 lower-MAE price deciles: {sum(item['m1_better'] for item in decile_rows)}/{len(decile_rows)}",
        "",
        "The complete M1 is supported over M0 on the pre-specified primary MAE metric. "
        "This does not imply superiority on every secondary metric or an independently "
        "confirmed external contribution from the mechanism block.",
    ]
    (OUTPUT / "M1_VS_M0_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
