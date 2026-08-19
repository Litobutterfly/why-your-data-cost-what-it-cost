# Why Your Data Cost What It Cost

This repository contains the anonymous reproducibility materials for **Why Your Data Cost What It Cost**. The release contains only two domains: **financial** and **telecom**.

## Experimental objective

This study aims to explain how prices are formed in data markets by combining explicit product attributes with semantic information that emerges during buyer-seller negotiations. We first use a decision-tree baseline to represent the information visible on a data platform, then extract implicit pricing factors from negotiation records and incorporate them into an augmented model. We evaluate whether these factors provide useful additional price-prediction information and use the resulting interpretable model to discover causal pricing rules across domains.

## Contents

- `data/`: the two released domain tables and their fixed train/test split files.
- `models/`: serialized M0 and M1 decision-tree models for both domains.
- `protocols/`: frozen analysis protocols, persona definitions, and confirmation plans.
- `results/`: released predictions, feature matrices, reports, causal-pricing-rule evidence, and mechanism-confirmation evidence.
- `results/telecom/causal_pricing_rules/`: paired Telecom intervention outcomes, product-level effects, and the three supported causal pricing rules.
- `code/`: the complete numbered workflow and its small support modules.

## Complete workflow

The numbered scripts describe the full experiment in execution order:

- `01_prepare_domain_data.py` validates the two released domains and their fixed splits.
- `02_train_m0.py` trains the fixed-capacity baseline decision tree for either domain.
- `03_prepare_negotiation_inputs.py` prepares the financial training/test inputs and the M0 reference prices.
- `04_generate_product_descriptions.py` generates platform-style product descriptions (API required).
- `05_build_agent_profiles.py` builds buyer/seller profiles and fixed pairs.
- `06_assign_quality_conditions.py` assigns evidence-constrained quality conditions.
- `07_assign_patience.py` assigns finite buyer/seller patience values.
- `08_prepare_negotiation_batch.py` freezes the financial training negotiation batch.
- `09_run_m0_negotiations.py` runs the M0 buyer-seller negotiations (API required).
- `10_export_public_dialogues.py` exports the public dialogue view.
- `11_filter_public_dialogues.py` applies dialogue-quality filters.
- `12_extract_implicit_features.py` extracts and scores implicit features from the filtered dialogues (API required).
- `13_train_m1.py` trains M1 using explicit and released implicit-feature matrices.
- `14_analyze_financial.py` and `15_evaluate_financial.py` produce the financial comparison reports.
- `16_prepare_telecom.py` through `21_evaluate_telecom.py` run the corresponding telecom replication.
- `22_mine_causal_pricing_rules.py` through `26_build_rule_catalog.py` mine and summarize causal-pricing rules.
- `27_prepare_mechanism_confirmation.py` through `30_build_final_evidence_catalog.py` confirm mechanisms and assemble the final evidence catalog.
- `31_evaluate_telecom_causal_pricing_rules.py` reproduces the Telecom paired-effect tests and Bonferroni screening offline.

Scripts that call the language-model service require `DEEPSEEK_API_KEY` in the environment. The package never stores a key. All other stages can be run offline from the released inputs and results.

## Reproduction examples

Run from this directory:

```text
python code/01_prepare_domain_data.py
python code/02_train_m0.py --domain financial
python code/02_train_m0.py --domain telecom
python code/23_prepare_rule_confirmation.py
python code/27_prepare_mechanism_confirmation.py
python code/31_evaluate_telecom_causal_pricing_rules.py
```

The first two commands perform the data and baseline-model checks without an API. The later numbered scripts consume the fixed plans and released results; API-dependent stages are explicitly marked above.

## M1 versus M0 comparison

M0 uses explicit platform fields, while M1 adds semantic mechanisms extracted from negotiation records. Both models use the same decision-tree capacity within each domain.

| Domain | M0 MAE | M1 MAE | MAE reduction | M0 RMSE | M1 RMSE | RMSE change | M0 R² | M1 R² | R² change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Financial | 0.272831 | 0.260329 | 4.58% | 0.493929 | 0.498153 | +0.86% | 0.754050 | 0.749825 | -0.004225 |
| Telecom | 0.315839 | 0.248022 | 21.47% | 0.572854 | 0.511158 | -10.77% | 0.721561 | 0.778307 | +0.056746 |

Negotiation-derived semantic information reduces MAE in both domains. On the primary MAE measure, the model with implicit features predicts prices more accurately; therefore, M1 provides useful additional price-prediction information beyond M0.

## Financial causal pricing rules

The Financial experiments confirmed six causal pricing rules. They cover task fit, task mismatch, coverage fit, identifier fit, customization limits, and update-cadence fit. Three rules also received strict negotiated-price support: confirming task fit, confirming that coverage meets the buyer's task, and confirming that the product's update cadence meets the buyer's refresh requirement. The other three rules showed causal relationships with effective willingness to pay.

## Telecom causal pricing rules

To examine whether the proposed method can discover causal pricing rules across domains, we conducted an additional experiment on Telecom data products and identified three causal pricing rules. The rules concern documentation or verification gaps, coverage advantages, and coverage limitations. The specific rules discovered in Financial and Telecom need not be identical: domain-specific rules reflect differences in how data products are priced and demonstrate that the proposed method can be applied to different data markets.

