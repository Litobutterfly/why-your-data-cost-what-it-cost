from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EFFECTS = (
    ROOT
    / "results"
    / "telecom"
    / "causal_pricing_rules"
    / "product_paired_effects.csv"
)
DEFAULT_PROTOCOL = ROOT / "protocols" / "telecom_causal_pricing_rule_protocol.json"
DEFAULT_OUTPUT = ROOT / "results" / "reproduced"


def bootstrap_mean_interval(
    values: np.ndarray, *, seed: int, draws: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    sample_index = rng.integers(0, len(values), size=(draws, len(values)))
    sampled_means = values[sample_index].mean(axis=1)
    return [float(value) for value in np.quantile(sampled_means, [0.025, 0.975])]


def evaluate(
    effects_path: Path, protocol_path: Path, output_dir: Path
) -> dict[str, Any]:
    effects = pd.read_csv(effects_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    outcome = str(protocol["primary_outcome"])
    alpha = float(protocol["bonferroni_alpha_per_prespecified_test"])
    draws = int(protocol["bootstrap_draws"])
    seed = int(protocol["bootstrap_seed"])
    tests = list(protocol["prespecified_tests"])
    exploratory = dict(protocol["exploratory_test"])
    rows: list[dict[str, Any]] = []

    for index, specification in enumerate(tests + [exploratory]):
        factor = str(specification["factor"])
        values = effects.loc[
            effects["factor"].eq(factor) & effects["outcome"].eq(outcome),
            "effect",
        ].to_numpy(float)
        expected_n = int(specification["eligible_products"])
        if len(values) != expected_n:
            raise RuntimeError(
                f"Expected {expected_n} paired effects for {factor}, found {len(values)}"
            )
        mean_effect = float(values.mean())
        interval = bootstrap_mean_interval(
            values, seed=seed + index, draws=draws
        )
        t_test = stats.ttest_1samp(values, 0.0)
        p_value = float(t_test.pvalue) if np.isfinite(t_test.pvalue) else 1.0
        direction = str(specification["expected_direction"])
        direction_pass = (
            mean_effect > 0 if direction == "positive" else mean_effect < 0
        )
        interval_pass = (
            interval[0] > 0 if direction == "positive" else interval[1] < 0
        )
        prespecified = index < len(tests)
        passed = bool(
            prespecified and direction_pass and interval_pass and p_value <= alpha
        )
        rows.append(
            {
                "factor": factor,
                "rule_text": specification["rule_text"],
                "outcome": outcome,
                "products": len(values),
                "expected_direction": direction,
                "mean_effect_log10": mean_effect,
                "bootstrap_95pct_ci": interval,
                "two_sided_t_test_p": p_value,
                "prespecified_test": prespecified,
                "bonferroni_alpha": alpha if prespecified else None,
                "causal_pricing_rule_supported": passed,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(
        output_dir / "telecom_causal_pricing_rule_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    supported = table.loc[table["causal_pricing_rule_supported"]]
    report = {
        "status": "telecom_causal_pricing_rule_analysis_complete",
        "scenario_rows": int(protocol["scenario_rows"]),
        "paired_comparisons": int(protocol["paired_comparisons"]),
        "primary_outcome": outcome,
        "prespecified_tests": len(tests),
        "causal_pricing_rules_supported": int(len(supported)),
        "supported_rules": supported["rule_text"].tolist(),
        "all_results": rows,
        "claim_scope": protocol["claim_scope"],
    }
    (output_dir / "telecom_causal_pricing_rule_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--effects", type=Path, default=DEFAULT_EFFECTS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    evaluate(args.effects, args.protocol, args.output_dir)


if __name__ == "__main__":
    main()
