from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINANCIAL_PATH = ROOT / "data" / "financial.csv"
INPUT_PATH = ROOT / "results" / "work" / "negotiation_prep" / "train_negotiation_inputs.csv"
PAIRS_PATH = ROOT / "results" / "work" / "agents" / "train_fixed_agent_pairs.csv"
OUTPUT_DIR = ROOT / "results" / "work" / "quality_conditions"
SEED = 20260813
READINESS_VERSION = "observable-readiness-v1"
CONDITION_VERSION = "controlled-quality-v2-indirect-disclosure"

DELIVERY = ["S3Bucket", "Download", "RESTAPI", "UIExport", "Email", "FeedAPI", "EnrichApp"]
UPDATES = ["monthly", "weekly", "daily", "ondemand", "realtime"]
FORMATS = ["csv", "json"]
EXPLICIT = [
    "History", "people", "entities", "products", "records", "events", "symbols",
    "assets", "requests", "features", "locations", "USD", "sources", "units",
    "Limitations", "ProfServices", "IdIndividuals", "IdCompanies", "NCountries",
    "PercGDP", "DelMethod", *DELIVERY, *UPDATES, *FORMATS,
]

CONDITIONS: dict[str, dict[str, Any]] = {
    "none": {
        "display_name": "no additional controlled issue",
        "trigger_topics": [],
        "seller_private_fact": (
            "The sales record contains no additional recent internal quality observation beyond the documented "
            "evidence boundaries. This is not a certification and does not fill any source-record evidence gap."
        ),
        "indirect_disclosure_rule": (
            "If asked a neutral due-diligence question, state that no additional recent internal observation is "
            "recorded, while making clear that this is not an audit or certification."
        ),
    },
    "completeness_variation": {
        "display_name": "field completeness variation",
        "trigger_topics": ["missing values", "field completeness", "nulls", "usable fields", "sample records"],
        "seller_private_fact": (
            "A recent internal pre-sale check found that some optional fields were not populated "
            "consistently in one batch; the source record does not provide an audited missingness rate."
        ),
        "indirect_disclosure_rule": (
            "If asked about completeness or samples, state the batch observation and the absent rate. "
            "Do not label the dataset incomplete or low quality."
        ),
    },
    "temporal_lag": {
        "display_name": "refresh-timing uncertainty",
        "trigger_topics": ["freshness", "latency", "update timing", "refresh", "current data", "delivery deadline"],
        "seller_private_fact": (
            "The most recent refresh available to the sales record was completed later than the intended "
            "internal schedule; no formal maximum-latency commitment is supplied."
        ),
        "indirect_disclosure_rule": (
            "If timing matters, explain the observed delay and absence of a formal latency commitment. "
            "Do not call the data stale."
        ),
    },
    "cross_batch_consistency": {
        "display_name": "cross-batch coding variation",
        "trigger_topics": ["schema", "field definitions", "consistency", "coding", "versioning", "integration"],
        "seller_private_fact": (
            "Two recent internal batches used different labels for a small set of categorical values; "
            "a finalized cross-version mapping note is not supplied in the sales record."
        ),
        "indirect_disclosure_rule": (
            "If asked about schema stability or integration, describe the label difference and missing mapping note. "
            "Do not claim the whole dataset is inconsistent."
        ),
    },
    "coverage_imbalance": {
        "display_name": "segment coverage imbalance",
        "trigger_topics": ["representativeness", "coverage", "geography", "segments", "bias", "population fit"],
        "seller_private_fact": (
            "An internal spot check indicates that one part of the advertised coverage contributes fewer observations "
            "than the dominant segment; no population-benchmark report is supplied."
        ),
        "indirect_disclosure_rule": (
            "If asked about representativeness, describe the relative imbalance and missing benchmark. "
            "Do not infer that the product is unusable or globally biased."
        ),
    },
    "accuracy_validation_gap": {
        "display_name": "accuracy-validation gap",
        "trigger_topics": ["accuracy", "error rate", "matching", "validation", "benchmark", "reliability"],
        "seller_private_fact": (
            "The sales record does not include an independent benchmark for the product's matching or classification "
            "outputs, and a recent spot-check summary is not accompanied by a measured error rate."
        ),
        "indirect_disclosure_rule": (
            "If asked about accuracy, distinguish the absence of a benchmark and measured rate from proof of inaccuracy. "
            "Do not invent a percentage."
        ),
    },
    "governance_traceability_gap": {
        "display_name": "governance traceability gap",
        "trigger_topics": ["provenance", "authorization", "license", "permitted use", "privacy", "governance"],
        "seller_private_fact": (
            "The documentation package available to sales does not include a complete source-to-use authorization trail "
            "for one recent contribution; no legal conclusion is recorded."
        ),
        "indirect_disclosure_rule": (
            "If asked about provenance or authorization, describe the documentation boundary without saying the data is "
            "unauthorized, illegal, compliant, or noncompliant."
        ),
    },
    "delivery_reliability": {
        "display_name": "delivery reliability uncertainty",
        "trigger_topics": ["delivery reliability", "API uptime", "file availability", "retry", "SLA", "operational continuity"],
        "seller_private_fact": (
            "One recent internal delivery check required a retry before the package became available; "
            "the source record contains no formal uptime or delivery-reliability log."
        ),
        "indirect_disclosure_rule": (
            "If asked about reliable delivery, mention the retry and absent reliability record. "
            "Do not claim recurring outages or guaranteed uptime."
        ),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True).astype(float)


def readiness_scores(train: pd.DataFrame, financial: pd.DataFrame) -> pd.DataFrame:
    work = train[["Id", *EXPLICIT]].copy()
    source = financial.set_index("Id").loc[work["Id"]]
    word_columns = [column for column in financial.columns if column.lower().startswith("word")]
    work["delivery_count"] = work[DELIVERY].sum(axis=1)
    work["update_count"] = work[UPDATES].sum(axis=1)
    work["format_count"] = work[FORMATS].sum(axis=1)
    work["nonzero_explicit_count"] = work[EXPLICIT].ne(0).sum(axis=1)
    work["nonzero_word_signal_count"] = source[word_columns].ne(0).sum(axis=1).to_numpy()

    work["access_score"] = (
        0.45 * work["delivery_count"].gt(0).astype(float)
        + 0.35 * work["format_count"].gt(0).astype(float)
        + 0.20 * percentile(work["delivery_count"])
    )
    work["timeliness_score"] = (
        0.75 * work["update_count"].gt(0).astype(float)
        + 0.25 * percentile(work["update_count"])
    )
    work["coverage_score"] = pd.concat(
        [percentile(work["NCountries"]), percentile(work["PercGDP"]), percentile(work["History"])],
        axis=1,
    ).mean(axis=1)
    work["documentation_score"] = pd.concat(
        [
            percentile(work["nonzero_explicit_count"]),
            percentile(work["nonzero_word_signal_count"]),
            (work["delivery_count"].gt(0).astype(float)
             + work["update_count"].gt(0).astype(float)
             + work["format_count"].gt(0).astype(float)) / 3.0,
        ],
        axis=1,
    ).mean(axis=1)
    work["observable_readiness_score"] = work[
        ["access_score", "timeliness_score", "coverage_score", "documentation_score"]
    ].mean(axis=1)
    ordered_rank = work["observable_readiness_score"].rank(method="first")
    work["observable_readiness_level"] = pd.qcut(
        ordered_rank,
        q=[0.0, 0.30, 0.70, 1.0],
        labels=["low", "medium", "high"],
    ).astype(str)
    keep = [
        "Id", "access_score", "timeliness_score", "coverage_score", "documentation_score",
        "observable_readiness_score", "observable_readiness_level", "delivery_count",
        "update_count", "format_count", "nonzero_explicit_count", "nonzero_word_signal_count",
    ]
    return work[keep].sort_values("Id")


def observed_gaps(train: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    merged = train.merge(scores, on="Id", validate="one_to_one")
    rows: list[dict[str, Any]] = []

    def add(row: pd.Series, gap_id: str, statement: str, evidence: str) -> None:
        rows.append(
            {
                "product_id": int(row["Id"]),
                "gap_id": gap_id,
                "source_supported_statement": statement,
                "source_evidence": evidence,
                "interpretation_constraint": "An unspecified field is an information gap, not proof of product failure.",
            }
        )

    for _, row in merged.iterrows():
        if row["delivery_count"] == 0:
            add(row, "delivery_channel_unspecified", "No standard delivery channel is listed in the supplied fields.", "all registered delivery indicators are zero")
        if row["update_count"] == 0:
            add(row, "update_schedule_unspecified", "No update schedule is listed in the supplied fields.", "all registered update-frequency indicators are zero")
        if row["format_count"] == 0:
            add(row, "standard_format_unspecified", "Neither CSV nor JSON is listed in the supplied fields.", "csv=0 and json=0")
        if int(round(float(row["Limitations"]))) == 1:
            add(row, "use_limitations_indicated", "The source record indicates that use limitations exist, without supplying the full terms.", "Limitations=1")
        if int(round(float(row["IdIndividuals"]))) == 1:
            add(row, "individual_identifier_governance_evidence_unavailable", "Individual identifiers are indicated, while authorization and privacy documentation are not supplied in the experiment record.", "IdIndividuals=1 and no source authorization/privacy documents")
        if row["documentation_score"] <= merged["documentation_score"].quantile(1 / 3):
            add(row, "limited_platform_documentation", "The product has relatively sparse platform-field and description-term documentation within the training catalog.", "bottom training-catalog tertile of documentation score")
        if row["coverage_score"] <= merged["coverage_score"].quantile(1 / 3):
            add(row, "narrower_catalog_relative_coverage", "The product has lower catalog-relative coverage signals; the practical sufficiency depends on the buyer's task.", "bottom training-catalog tertile of coverage score")
    return pd.DataFrame(rows).sort_values(["product_id", "gap_id"])


def assign_conditions(scores: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    merged = scores.merge(
        pairs[["product_id", "buyer_profile_id", "seller_profile_id", "buyer_ceiling_usd", "seller_floor_usd"]],
        left_on="Id",
        right_on="product_id",
        validate="one_to_one",
    )
    rng = np.random.default_rng(SEED + 31)
    issue_ids = [condition for condition in CONDITIONS if condition != "none"]
    assigned: dict[int, str] = {}
    for level in ["low", "medium", "high"]:
        ids = merged.loc[merged["observable_readiness_level"].eq(level), "Id"].to_numpy(copy=True)
        rng.shuffle(ids)
        none_count = int(round(0.40 * len(ids)))
        issue_count = len(ids) - none_count
        labels = ["none"] * none_count
        labels.extend(issue_ids[index % len(issue_ids)] for index in range(issue_count))
        rng.shuffle(labels)
        assigned.update({int(product_id): label for product_id, label in zip(ids, labels)})

    rows = []
    for _, row in merged.sort_values("Id").iterrows():
        condition_id = assigned[int(row["Id"])]
        definition = CONDITIONS[condition_id]
        rows.append(
            {
                "product_id": int(row["Id"]),
                "observable_readiness_level": row["observable_readiness_level"],
                "controlled_condition_id": condition_id,
                "controlled_condition_present": condition_id != "none",
                "controlled_condition_display_name": definition["display_name"],
                "trigger_topics_json": json.dumps(definition["trigger_topics"], ensure_ascii=False),
                "seller_private_fact": definition["seller_private_fact"],
                "indirect_disclosure_rule": definition["indirect_disclosure_rule"],
                "assignment_seed": SEED + 31,
                "assignment_version": CONDITION_VERSION,
                "buyer_profile_id_audit_only": row["buyer_profile_id"],
                "seller_profile_id_audit_only": row["seller_profile_id"],
                "potential_price_overlap_audit_only": bool(row["buyer_ceiling_usd"] >= row["seller_floor_usd"]),
                "real_transaction_price_used": False,
            }
        )
    return pd.DataFrame(rows).sort_values("product_id")


def quality_ledger(
    condition: pd.Series,
    product_gaps: pd.DataFrame,
) -> dict[str, Any]:
    gap_items = [
        {
            "evidence_id": f"observed:{row.gap_id}",
            "evidence_class": "source-supported platform evidence gap",
            "topic": row.gap_id,
            "fact": row.source_supported_statement,
            "disclosure_rule": "State the field or documentation boundary without diagnosing product quality.",
        }
        for row in product_gaps.itertuples(index=False)
    ]
    controlled = {
        "evidence_id": f"controlled:{condition['controlled_condition_id']}",
        "evidence_class": "controlled synthetic quality observation",
        "topic": condition["controlled_condition_display_name"],
        "trigger_topics": json.loads(condition["trigger_topics_json"]),
        "fact": condition["seller_private_fact"],
        "disclosure_rule": condition["indirect_disclosure_rule"],
    }
    return {
        "product_id": int(condition["product_id"]),
        "ledger_version": CONDITION_VERSION,
        "observable_readiness_level": condition["observable_readiness_level"],
        "items": gap_items + [controlled],
        "seller_behavior_contract": {
            "opening": "Do not volunteer a controlled issue in the opening price message.",
            "when_relevant_question_is_asked": (
                "Answer truthfully with the concrete observation and its evidence boundary. "
                "Do not state the condition ID or a quality conclusion; let the buyer assess task implications."
            ),
            "neutral_due_diligence_question": (
                "Treat a buyer question about recent internal observations or evidence limitations as relevant to the "
                "registered quality-observation item."
            ),
            "anti_deception": (
                "Indirect disclosure may be tactful but cannot deny, contradict, or materially conceal the registered fact "
                "after a relevant direct question."
            ),
            "forbidden_labels": [
                "low quality", "high risk", "defective", "bad data", "hidden issue",
                "synthetic condition", "controlled treatment",
            ],
        },
        "interpretation_constraint": (
            "Controlled observations support mechanism-recovery experiments only. They are not claims about the real product."
        ),
    }


def run() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    financial = pd.read_csv(FINANCIAL_PATH, low_memory=False)
    train = pd.read_csv(INPUT_PATH, low_memory=False)
    pairs = pd.read_csv(PAIRS_PATH, low_memory=False)
    if len(train) != 1912 or len(pairs) != 1912 or set(train["Id"]) != set(pairs["product_id"]):
        raise RuntimeError("Quality preparation requires the fixed 1,912-product training set")
    forbidden = {"LogPriceMo", "TransactionPriceUSD"}
    if forbidden & set(train.columns) or forbidden & set(pairs.columns):
        raise RuntimeError("Observed transaction outcomes must not enter quality assignment")

    scores = readiness_scores(train, financial)
    gaps = observed_gaps(train, scores)
    conditions = assign_conditions(scores, pairs)
    scores.to_csv(OUTPUT_DIR / "product_readiness_scores.csv", index=False, encoding="utf-8-sig")
    gaps.to_csv(OUTPUT_DIR / "observed_evidence_gaps.csv", index=False, encoding="utf-8-sig")
    conditions.to_csv(OUTPUT_DIR / "controlled_quality_conditions.csv", index=False, encoding="utf-8-sig")

    gap_groups = {product_id: group for product_id, group in gaps.groupby("product_id")}
    ledger_path = OUTPUT_DIR / "seller_quality_ledgers.jsonl"
    with ledger_path.open("w", encoding="utf-8") as handle:
        for _, condition in conditions.iterrows():
            product_id = int(condition["product_id"])
            product_gaps = gap_groups.get(product_id, gaps.iloc[0:0])
            handle.write(json.dumps(quality_ledger(condition, product_gaps), ensure_ascii=False) + "\n")

    assignment_table = conditions.merge(
        scores[["Id", "observable_readiness_score"]],
        left_on="product_id",
        right_on="Id",
        validate="one_to_one",
    ).drop(columns=["Id"])
    condition_by_readiness = pd.crosstab(
        assignment_table["observable_readiness_level"], assignment_table["controlled_condition_id"]
    )
    condition_by_buyer = pd.crosstab(
        assignment_table["buyer_profile_id_audit_only"], assignment_table["controlled_condition_id"]
    )
    condition_by_seller = pd.crosstab(
        assignment_table["seller_profile_id_audit_only"], assignment_table["controlled_condition_id"]
    )
    audit = {
        "status": "quality_conditions_frozen_without_api_calls",
        "seed": SEED + 31,
        "readiness_version": READINESS_VERSION,
        "condition_version": CONDITION_VERSION,
        "training_products": len(scores),
        "readiness_counts": scores["observable_readiness_level"].value_counts().sort_index().to_dict(),
        "condition_counts": conditions["controlled_condition_id"].value_counts().sort_index().to_dict(),
        "controlled_issue_rate": float(conditions["controlled_condition_present"].mean()),
        "condition_by_readiness": condition_by_readiness.to_dict(orient="index"),
        "condition_by_buyer_profile_audit": condition_by_buyer.to_dict(orient="index"),
        "condition_by_seller_profile_audit": condition_by_seller.to_dict(orient="index"),
        "observed_gap_rows": len(gaps),
        "products_with_observed_gap": int(gaps["product_id"].nunique()),
        "real_transaction_price_used": False,
        "m0_reference_used_for_quality_assignment": False,
        "test_products_used": False,
        "one_controlled_condition_per_product": bool(not conditions["product_id"].duplicated().any()),
        "source_input_sha256": sha256_file(FINANCIAL_PATH),
        "train_input_sha256": sha256_file(INPUT_PATH),
        "pair_input_sha256": sha256_file(PAIRS_PATH),
    }
    (OUTPUT_DIR / "quality_assignment_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return audit


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
