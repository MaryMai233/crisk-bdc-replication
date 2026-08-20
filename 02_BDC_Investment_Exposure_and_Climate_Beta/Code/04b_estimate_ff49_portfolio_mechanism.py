from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "Processed"
AUDIT = PROCESSED / "Audit"


def load_prior_module():
    path = Path(__file__).with_name("01_industry_beta_helpers.py")
    spec = importlib.util.spec_from_file_location("prior_h2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PRIOR = load_prior_module()


def clean_label(label: object) -> str:
    text = str(label).lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def label_to_ff49(label: object) -> tuple[list[int], str, str]:
    """Map a disclosed BDC industry label to one or more public FF49 groups.

    Combined and very broad disclosures deliberately map to a basket rather
    than to a fabricated single SIC code.  The basket is value weighted using
    CRSP industry market capitalization in the corresponding quarter.
    """
    text = clean_label(label)
    if not text or text in {"other", "others", "n a", "not applicable"}:
        return [], "unmapped_other", "unmapped"
    geography = {
        "united states", "canada", "europe", "australia", "asia",
        "north america", "latin america", "international", "global",
        "united kingdom", "ireland", "italy", "luxembourg", "netherlands",
        "norway", "france", "germany", "spain", "sweden", "midwest",
        "northeast", "south", "west",
    }
    if text in geography or any(text.startswith(x + " ") for x in geography):
        return [], "unmapped_geography", "unmapped"

    rules: list[tuple[str, list[int], str, str]] = [
        (r"coal", [29], "coal", "high"),
        (r"oil|gas|petroleum|energy equipment|energy service|energy$", [30], "oil_gas", "high"),
        (r"renewable|power generation|electric util|gas util|water util|utilities|utility", [31], "utilities", "high"),
        (r"aerospace.*defen|defen.*aerospace", [24, 26], "aerospace_defense_basket", "medium"),
        (r"aerospace|aircraft|aviation", [24], "aircraft", "high"),
        (r"defense|ordnance|weapon", [26], "defense", "high"),
        (r"software", [36], "software", "high"),
        (r"semiconductor|electronic equipment|electronic component|electronics$|communications equipment", [37], "electronic_equipment", "high"),
        (r"technology hardware|computer hardware|computers", [35], "computer_hardware", "high"),
        (r"measurement|measuring|test equipment|life sciences tools", [38], "measuring_equipment", "medium"),
        (r"it services|information technology|data processing|information service|internet service|research and consulting|professional service|business service|commercial and professional|marketing service|human resource|outsourc|consulting|engineering service|support service|payroll|staffing|advertising|facilities maintenance|field service|revenue cycle|government service|federal service|event service|corporate governance", [34], "business_services", "high"),
        (r"cyber security", [34, 36], "cybersecurity_basket", "medium"),
        (r"data storage", [35], "computer_storage", "high"),
        (r"high tech|technology$", [34, 35, 36, 37, 38], "technology_basket", "low"),
        (r"pharmaceutical|biotech|drug", [13], "pharmaceuticals", "high"),
        (r"health care equipment|healthcare equipment|medical equipment|healthcare product|health care supplies|healthcare supply|medical device|surgical device|diagnostic", [12], "medical_equipment", "high"),
        (r"health care provider|healthcare provider|health care service|healthcare service|veterinary|dental practice|health fitness|healthcare$|health care$", [11], "healthcare_services", "high"),
        (r"health care technology|healthcare technology", [11, 12, 36], "healthcare_technology_basket", "medium"),
        (r"insurance", [46], "insurance", "high"),
        (r"capital market|asset management|investment fund|investment vehicle|financial exchange|broker|trading", [48], "trading_investment", "high"),
        (r"structured finance|specialized finance|bank|consumer finance|lending|mortgage|credit service", [45, 48], "banking_credit", "medium"),
        (r"diversified financial|financial service|financials$", [45, 48], "financial_basket", "medium"),
        (r"real estate investment trust|reit", [47, 48], "real_estate_reit_basket", "medium"),
        (r"real estate", [47], "real_estate", "high"),
        (r"construction material|building product|building and infrastructure|forest product|lumber", [17], "construction_materials", "high"),
        (r"construction|engineering and construction", [18], "construction", "high"),
        (r"steel|iron", [19], "steel", "high"),
        (r"precious metal|gold|silver", [27], "precious_metals", "high"),
        (r"metal|mining", [19, 28], "metals_mining_basket", "medium"),
        (r"chemical", [14], "chemicals", "high"),
        (r"rubber|plastic", [15], "rubber_plastic", "high"),
        (r"paper", [39], "paper_business_supplies", "high"),
        (r"container|packaging", [40], "shipping_containers", "high"),
        (r"machinery|capital equipment", [21], "machinery", "high"),
        (r"electrical equipment|electrical component", [22], "electrical_equipment", "high"),
        (r"hvac monitoring", [38], "measuring_equipment", "medium"),
        (r"hvac", [18, 22, 34], "hvac_basket", "medium"),
        (r"fabricated product", [20], "fabricated_products", "high"),
        (r"industrial product|industrial service|manufacturing|capital goods", [20, 21, 22], "industrial_basket", "low"),
        (r"automobile|automotive|auto component|auto part", [23], "automobiles", "high"),
        (r"shipbuilding|railroad equipment", [25], "ship_rail_equipment", "high"),
        (r"air freight|airport service|logistics|transportation|road and rail|road rail|airline|marine|shipping", [41], "transportation", "high"),
        (r"telecommunication|communications$|wireless|cable|satellite|alternative carrier|broadcasting", [32], "telecommunications", "high"),
        (r"media|entertainment|movie|leisure service|leisure facilit|gaming|sport management|sports management", [7], "entertainment", "medium"),
        (r"printing|publishing", [8], "printing_publishing", "high"),
        (r"hotel|restaurant", [44], "hotels_restaurants", "high"),
        (r"leisure product|leisure equipment|recreation|sporting good|toy", [6], "recreation_products", "medium"),
        (r"education|consumer service|diversified consumer service|personal service|legal service|social service", [33], "personal_education_services", "medium"),
        (r"food and beverage|food beverage", [2, 3, 4], "food_beverage_basket", "medium"),
        (r"food product|food processing|packaged food", [2], "food_products", "high"),
        (r"soft drink", [3], "soft_drinks", "high"),
        (r"beverage", [3, 4], "beverage_basket", "medium"),
        (r"tobacco", [5], "tobacco", "high"),
        (r"agriculture|farming", [1], "agriculture", "high"),
        (r"textile.*apparel|apparel.*textile|consumer durables and apparel", [9, 10, 16], "consumer_apparel_basket", "medium"),
        (r"textile", [16], "textiles", "high"),
        (r"apparel|luxury good|footwear", [10], "apparel", "high"),
        (r"household|houseware|home furnishing|personal product|personal care|promotional product|consumer product|consumer goods|consumer durables|durable consumer|non durable consumer", [9], "consumer_goods", "medium"),
        (r"consumer related", [6, 7, 9, 10, 33, 43, 44], "consumer_basket", "low"),
        (r"wholesale|distributor|distribution|trading companies", [42], "wholesale", "medium"),
        (r"retail", [43], "retail", "high"),
        (r"office suppl|office product|business product", [39], "business_supplies", "medium"),
        (r"energy efficiency", [22, 31], "energy_efficiency_basket", "medium"),
        (r"space technolog", [24, 26], "space_aerospace_basket", "medium"),
        (r"net lease|property management", [47], "real_estate", "high"),
        (r"commercial service|commerical service|commercial supplies|environmental industr|industrial cleaning|waste", [34, 49], "commercial_environmental_services", "low"),
        (r"material$|materials$", [14, 17, 19, 20, 28, 39, 40], "materials_basket", "low"),
    ]
    for pattern, codes, rule, confidence in rules:
        if re.search(pattern, text):
            return codes, rule, confidence
    return [], "unmapped_no_rule", "unmapped"


def replace_geography_only_quarters(exposure: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Replace four parser-selected geography tables with the nearest valid industry table.

    The source extraction occasionally selected a 100-percent geography table
    even though the filing also contained an industry schedule.  Because the
    raw filing archive is not distributed in this module, the correction is a
    transparent nearest-quarter carry and is kept as a sensitivity flag.
    """
    exposure = exposure.copy()
    exposure["portfolio_exposure_imputation"] = "none"
    group_status: dict[tuple[str, str], bool] = {}
    for (ticker, quarter), group in exposure.groupby(["ticker", "calendar_quarter"]):
        group_status[(ticker, str(quarter))] = any(bool(label_to_ff49(x)[0]) for x in group["industry_reported"])
    valid_by_ticker: dict[str, list[str]] = {}
    for (ticker, quarter), valid in group_status.items():
        if valid:
            valid_by_ticker.setdefault(ticker, []).append(quarter)
    replacements: list[pd.DataFrame] = []
    audit: list[dict[str, str]] = []
    invalid_keys = [key for key, valid in group_status.items() if not valid]
    if not invalid_keys:
        return exposure, audit
    drop_mask = pd.Series(False, index=exposure.index)
    for ticker, target in invalid_keys:
        candidates = valid_by_ticker.get(ticker, [])
        if not candidates:
            continue
        target_period = pd.Period(target, freq="Q")
        source = min(candidates, key=lambda q: (abs(pd.Period(q, freq="Q").ordinal - target_period.ordinal), pd.Period(q, freq="Q").ordinal))
        source_rows = exposure.loc[
            exposure["ticker"].eq(ticker) & exposure["calendar_quarter"].astype(str).eq(source)
        ].copy()
        source_rows["calendar_quarter"] = target
        source_rows["portfolio_exposure_imputation"] = f"nearest_valid_quarter:{source}"
        replacements.append(source_rows)
        drop_mask |= exposure["ticker"].eq(ticker) & exposure["calendar_quarter"].astype(str).eq(target)
        audit.append({"ticker": ticker, "target_quarter": target, "source_quarter": source})
    corrected = pd.concat([exposure.loc[~drop_mask], *replacements], ignore_index=True)
    return corrected, audit


def add_standardized(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        std = frame[column].std(ddof=0)
        frame[f"z_{column}"] = (frame[column] - frame[column].mean()) / std if std > 0 else np.nan
    return frame


def build_portfolio_beta(
    exposure: pd.DataFrame, ff49: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped = exposure["industry_reported"].map(label_to_ff49)
    exposure = exposure.copy()
    exposure[["ff49_codes", "mapping_rule_ff49", "mapping_confidence_ff49"]] = pd.DataFrame(
        mapped.tolist(), index=exposure.index
    )
    exposure["portfolio_fair_value_pct"] = pd.to_numeric(
        exposure["portfolio_fair_value_pct"], errors="coerce"
    )
    exposure["quarter"] = exposure["calendar_quarter"].astype(str)
    exposure["mapped_ff49"] = exposure["ff49_codes"].map(bool)

    expanded = exposure.loc[
        exposure["mapped_ff49"] & exposure["portfolio_fair_value_pct"].notna()
    ].explode("ff49_codes").rename(columns={"ff49_codes": "ff49"})
    expanded["ff49"] = expanded["ff49"].astype(int)
    expanded = expanded.merge(ff49, on=["ff49", "quarter"], how="left", validate="many_to_one")
    expanded["code_market_cap"] = expanded["market_cap_sum"].clip(lower=0)
    group_keys = ["ticker", "quarter", "industry_reported"]
    expanded["row_market_cap_sum"] = expanded.groupby(group_keys)["code_market_cap"].transform("sum")
    expanded["within_label_weight"] = np.where(
        expanded["row_market_cap_sum"].gt(0),
        expanded["code_market_cap"] / expanded["row_market_cap_sum"],
        1.0 / expanded.groupby(group_keys)["ff49"].transform("size"),
    )
    expanded["label_beta_qend_piece"] = expanded["within_label_weight"] * expanded["cbeta_ff49_qend"]
    expanded["label_beta_qmean_piece"] = expanded["within_label_weight"] * expanded["cbeta_ff49_qmean"]
    label_beta = expanded.groupby(group_keys, as_index=False).agg(
        label_beta_qend=("label_beta_qend_piece", "sum"),
        label_beta_qmean=("label_beta_qmean_piece", "sum"),
        ff49_groups_for_label=("ff49", "nunique"),
    )
    mapped_rows = exposure.merge(label_beta, on=group_keys, how="left", validate="many_to_one")
    mapped_rows["mapped_weight"] = np.where(
        mapped_rows["label_beta_qend"].notna(), mapped_rows["portfolio_fair_value_pct"], 0.0
    )
    mapped_rows["weighted_beta_qend"] = mapped_rows["mapped_weight"] * mapped_rows["label_beta_qend"]
    mapped_rows["weighted_beta_qmean"] = mapped_rows["mapped_weight"] * mapped_rows["label_beta_qmean"]
    mapped_rows["high_conf_weight"] = np.where(
        mapped_rows["mapping_confidence_ff49"].eq("high"), mapped_rows["mapped_weight"], 0.0
    )
    mapped_rows["medium_conf_weight"] = np.where(
        mapped_rows["mapping_confidence_ff49"].eq("medium"), mapped_rows["mapped_weight"], 0.0
    )
    mapped_rows["low_conf_weight"] = np.where(
        mapped_rows["mapping_confidence_ff49"].eq("low"), mapped_rows["mapped_weight"], 0.0
    )
    portfolio = mapped_rows.groupby(["ticker", "quarter"], as_index=False).agg(
        mapped_weight_pct=("mapped_weight", "sum"),
        weighted_beta_qend_sum=("weighted_beta_qend", "sum"),
        weighted_beta_qmean_sum=("weighted_beta_qmean", "sum"),
        high_confidence_weight_pct=("high_conf_weight", "sum"),
        medium_confidence_weight_pct=("medium_conf_weight", "sum"),
        low_confidence_weight_pct=("low_conf_weight", "sum"),
        reported_industry_rows=("industry_reported", "size"),
    )
    portfolio["ff49_portfolio_climate_beta_qend"] = (
        portfolio["weighted_beta_qend_sum"] / portfolio["mapped_weight_pct"]
    )
    portfolio["ff49_portfolio_climate_beta_qmean"] = (
        portfolio["weighted_beta_qmean_sum"] / portfolio["mapped_weight_pct"]
    )
    return mapped_rows, portfolio


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    exposure = pd.read_csv(PROCESSED / "dynamic_bdc_industry_exposure_2021_2025.csv")
    exposure, imputation_audit = replace_geography_only_quarters(exposure)
    ff49 = pd.read_csv(
        PROCESSED / "ff49_industry_climate_beta_quarterly_2020_2025.csv",
        parse_dates=["quarter_end_observation_date"],
    )
    mapped, portfolio = build_portfolio_beta(exposure, ff49)
    mapped.to_csv(PROCESSED / "dynamic_bdc_industry_ff49_mapping_2021_2025.csv", index=False)
    PRIOR.export_dta(
        mapped.drop(columns=["ff49_codes"]),
        PROCESSED / "dynamic_bdc_industry_ff49_mapping_2021_2025.dta",
    )
    portfolio.to_csv(PROCESSED / "dynamic_bdc_ff49_portfolio_beta_2021_2025.csv", index=False)
    PRIOR.export_dta(portfolio, PROCESSED / "dynamic_bdc_ff49_portfolio_beta_2021_2025.dta")

    panel = pd.read_csv(
        PROCESSED / "bdc20_quarter_market_financial_panel_2021_2025_updated.csv",
        parse_dates=["datadate"],
    )
    panel["quarter"] = panel["datadate"].dt.to_period("Q").astype(str)
    panel = panel.merge(portfolio, on=["ticker", "quarter"], how="inner", validate="one_to_one")
    panel = panel.sort_values(["ticker", "quarter"]).reset_index(drop=True)
    panel["ff49_portfolio_climate_beta_qend_lag1"] = panel.groupby("ticker")[
        "ff49_portfolio_climate_beta_qend"
    ].shift(1)
    zcols = [
        "beta_climate_equity_report_month", "beta_climate_asset_report_month",
        "ff49_portfolio_climate_beta_qend", "ff49_portfolio_climate_beta_qmean",
        "ff49_portfolio_climate_beta_qend_lag1", "log_assets", "debt_to_assets",
        "roa_quarter", "book_to_market", "beta_market_report_month",
    ]
    panel = add_standardized(panel, zcols)
    panel.to_csv(PROCESSED / "bdc19_ff49_portfolio_mechanism_panel_2021_2025.csv", index=False)
    PRIOR.export_dta(panel, PROCESSED / "bdc19_ff49_portfolio_mechanism_panel_2021_2025.dta")

    controls = [
        "z_log_assets", "z_debt_to_assets", "z_roa_quarter",
        "z_book_to_market", "z_beta_market_report_month",
    ]
    specs = [
        ("FF49_1", "Asset beta, FF49 portfolio beta, quarter FE", "z_beta_climate_asset_report_month", "z_ff49_portfolio_climate_beta_qend", [], ["quarter"]),
        ("FF49_2", "Asset beta, FF49 portfolio beta plus controls", "z_beta_climate_asset_report_month", "z_ff49_portfolio_climate_beta_qend", controls, ["quarter"]),
        ("FF49_3", "Asset beta, FF49 portfolio beta, firm and quarter FE", "z_beta_climate_asset_report_month", "z_ff49_portfolio_climate_beta_qend", [], ["ticker", "quarter"]),
        ("FF49_4", "Equity beta, FF49 portfolio beta, quarter FE", "z_beta_climate_equity_report_month", "z_ff49_portfolio_climate_beta_qend", [], ["quarter"]),
        ("FF49_5", "Equity beta, FF49 portfolio beta plus controls", "z_beta_climate_equity_report_month", "z_ff49_portfolio_climate_beta_qend", controls, ["quarter"]),
        ("FF49_6", "Equity beta, FF49 portfolio beta, firm and quarter FE", "z_beta_climate_equity_report_month", "z_ff49_portfolio_climate_beta_qend", [], ["ticker", "quarter"]),
        ("FF49_7", "Asset beta, quarterly-mean portfolio beta, quarter FE", "z_beta_climate_asset_report_month", "z_ff49_portfolio_climate_beta_qmean", [], ["quarter"]),
        ("FF49_8", "Asset beta, lagged portfolio beta, quarter FE", "z_beta_climate_asset_report_month", "z_ff49_portfolio_climate_beta_qend_lag1", [], ["quarter"]),
    ]
    models = pd.DataFrame(
        [
            PRIOR.fit_ols(
                panel, outcome, exposure_name, model_id, label,
                controls=model_controls, fixed_effects=fixed_effects, cluster="ticker",
            )
            for model_id, label, outcome, exposure_name, model_controls, fixed_effects in specs
        ]
    )
    models["coefficient_unit"] = "Standard deviations of BDC climate beta per one-SD portfolio climate beta"
    models.to_csv(PROCESSED / "h2_ff49_portfolio_mechanism_models.csv", index=False)
    PRIOR.export_dta(models, PROCESSED / "h2_ff49_portfolio_mechanism_models.dta")

    unique_labels = mapped.drop_duplicates("industry_reported")[
        ["industry_reported", "mapping_rule_ff49", "mapping_confidence_ff49", "mapped_ff49"]
    ]
    label_weights = exposure.groupby("industry_reported", as_index=False)["portfolio_fair_value_pct"].sum()
    unique_labels = unique_labels.merge(label_weights, on="industry_reported", how="left")
    unique_labels.to_csv(PROCESSED / "ff49_bdc_label_mapping_dictionary.csv", index=False)
    PRIOR.export_dta(unique_labels, PROCESSED / "ff49_bdc_label_mapping_dictionary.dta")

    total_weight = float(exposure["portfolio_fair_value_pct"].sum())
    mapped_weight = float(mapped["mapped_weight"].sum())
    audit = {
        "status": "PASS",
        "reported_industry_labels": int(exposure["industry_reported"].nunique()),
        "mapped_industry_labels": int(unique_labels["mapped_ff49"].sum()),
        "total_reported_portfolio_weight": total_weight,
        "mapped_portfolio_weight": mapped_weight,
        "weighted_mapping_coverage": mapped_weight / total_weight,
        "company_quarters": int(len(portfolio)),
        "firms": int(portfolio["ticker"].nunique()),
        "median_company_quarter_coverage_pct": float(portfolio["mapped_weight_pct"].median()),
        "minimum_company_quarter_coverage_pct": float(portfolio["mapped_weight_pct"].min()),
        "model_rows": int(len(models)),
        "geography_table_quarters_replaced": imputation_audit,
    }
    (AUDIT / "ff49_bdc_portfolio_mechanism_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    print(models[["model_id", "coefficient_exposure", "standard_error", "p_two_sided", "n", "r_squared"]].to_string(index=False))


if __name__ == "__main__":
    main()
