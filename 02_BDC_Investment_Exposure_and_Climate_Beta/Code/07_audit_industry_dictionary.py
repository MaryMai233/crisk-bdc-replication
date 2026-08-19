from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "Processed"
AUDIT = PROCESSED / "Audit"


def load_module(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EXTRACT = load_module("02_extract_bdc_portfolio_exposure.py", "extraction_rules")
HELPER = load_module("01_industry_beta_helpers.py", "mapping_rules")


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    dictionary_rows = []
    for category, patterns in [
        ("narrow brown", EXTRACT.NARROW_PATTERNS),
        ("broad-only increment", EXTRACT.BROAD_EXTRA_PATTERNS),
    ]:
        for priority, pattern in enumerate(patterns, start=1):
            dictionary_rows.append(
                {
                    "classification": category,
                    "priority_within_class": priority,
                    "regular_expression": pattern,
                    "broad_brown": 1,
                    "narrow_brown": int(category == "narrow brown"),
                }
            )
    dictionary = pd.DataFrame(dictionary_rows)
    dictionary.to_csv(PROCESSED / "industry_classification_dictionary.csv", index=False)
    HELPER.export_dta(dictionary, PROCESSED / "industry_classification_dictionary.dta")

    mapping = pd.read_csv(PROCESSED / "dynamic_bdc_industry_ff12_mapping_2021_2025.csv")
    mapping["portfolio_fair_value_pct"] = pd.to_numeric(
        mapping["portfolio_fair_value_pct"], errors="coerce"
    )
    summary = (
        mapping.groupby(["mapping_confidence", "mapping_rule", "ff12_group"], as_index=False)
        .agg(
            industry_rows=("industry_reported", "size"),
            unique_reported_labels=("industry_reported", "nunique"),
            summed_portfolio_weight_pct=("portfolio_fair_value_pct", "sum"),
        )
        .sort_values(["mapping_confidence", "summed_portfolio_weight_pct"], ascending=[True, False])
    )
    summary.to_csv(PROCESSED / "industry_mapping_rule_audit.csv", index=False)
    HELPER.export_dta(summary, PROCESSED / "industry_mapping_rule_audit.dta")

    review = mapping[mapping["mapping_confidence"].eq("low")].copy()
    review = (
        review.groupby(["industry_reported", "mapping_rule", "ff12_group"], as_index=False)
        .agg(
            appearances=("industry_reported", "size"),
            summed_portfolio_weight_pct=("portfolio_fair_value_pct", "sum"),
            mean_portfolio_weight_pct=("portfolio_fair_value_pct", "mean"),
        )
        .sort_values("summed_portfolio_weight_pct", ascending=False)
    )
    review.to_csv(PROCESSED / "industry_low_confidence_review.csv", index=False)
    HELPER.export_dta(review, PROCESSED / "industry_low_confidence_review.dta")

    total_weight = mapping["portfolio_fair_value_pct"].sum()
    low_weight = mapping.loc[
        mapping["mapping_confidence"].eq("low"), "portfolio_fair_value_pct"
    ].sum()
    fallback_weight = mapping.loc[
        mapping["mapping_rule"].eq("UNMAPPED_FALLBACK_OTHER"),
        "portfolio_fair_value_pct",
    ].sum()
    audit = {
        "industry_rows": int(len(mapping)),
        "unique_reported_labels": int(mapping["industry_reported"].nunique()),
        "narrow_patterns": len(EXTRACT.NARROW_PATTERNS),
        "broad_increment_patterns": len(EXTRACT.BROAD_EXTRA_PATTERNS),
        "low_confidence_weight_share": float(low_weight / total_weight),
        "unmapped_fallback_weight_share": float(fallback_weight / total_weight),
        "mean_mapped_weight_by_firm_quarter": float(
            mapping.groupby(["ticker", "quarter"])["portfolio_fair_value_pct"].sum().mean()
        ),
        "classification_order": "narrow patterns first; broad measure equals narrow plus broad-only increment patterns",
        "normalization": "Tables summing to 97-110 percent are normalized to 100; incomplete subportfolio tables retain reported weights and leave the residual unclassified.",
    }
    (AUDIT / "industry_dictionary_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
