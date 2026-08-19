from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "Processed"
AUDIT = PROCESSED / "Audit"
RESULTS = ROOT / "Results"
TABLES = PROCESSED
FIGURES = RESULTS


def export_dta(frame: pd.DataFrame, path: Path) -> None:
    """Write a Stata 118 file after making Stata-safe column names."""
    rename = {}
    used: set[str] = set()
    for original in frame.columns:
        clean = re.sub(r"[^A-Za-z0-9_]", "_", str(original)).lower()
        if not re.match(r"^[A-Za-z_]", clean):
            clean = f"v_{clean}"
        clean = clean[:32]
        candidate = clean
        counter = 1
        while candidate in used:
            suffix = f"_{counter}"
            candidate = clean[: 32 - len(suffix)] + suffix
            counter += 1
        used.add(candidate)
        rename[original] = candidate
    out = frame.rename(columns=rename).copy()
    for column in out.select_dtypes(include="bool"):
        out[column] = out[column].astype("int8")
    for column in out.select_dtypes(include=["object"]):
        out[column] = out[column].fillna("").astype(str)
    out.to_stata(path, write_index=False, version=118)


def ff12_from_sic3(value: object) -> str:
    """Approximate the standard FF12 SIC mapping at the available 3-digit level."""
    try:
        sic = int(float(value)) * 10
    except (TypeError, ValueError):
        return "Other"

    def in_any(*ranges: tuple[int, int]) -> bool:
        return any(low <= sic <= high for low, high in ranges)

    if in_any((100, 999), (2000, 2399), (2700, 2749), (2770, 2799),
              (3100, 3199), (3940, 3989)):
        return "NoDur"
    if in_any((2500, 2519), (2590, 2599), (3630, 3659), (3710, 3719),
              (3750, 3759), (3790, 3799), (3860, 3879), (3910, 3939),
              (3990, 3999)):
        return "Durbl"
    if in_any((2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099),
              (3200, 3569), (3580, 3629), (3700, 3709), (3720, 3749),
              (3830, 3839), (3880, 3899)):
        return "Manuf"
    if in_any((1200, 1399), (2900, 2999)):
        return "Enrgy"
    if in_any((2800, 2829), (2840, 2899)):
        return "Chems"
    if in_any((3570, 3579), (3660, 3699), (3810, 3829), (7370, 7379)):
        return "BusEq"
    if in_any((4800, 4899)):
        return "Telcm"
    if in_any((4900, 4949)):
        return "Utils"
    if in_any((5000, 5999), (7200, 7299), (7600, 7699)):
        return "Shops"
    if in_any((2830, 2839), (3840, 3859), (8000, 8099)):
        return "Hlth"
    if in_any((6000, 6999)):
        return "Money"
    return "Other"


def label_to_ff12(label: object) -> tuple[str, str, str]:
    """Map heterogeneous BDC-reported labels to FF12 with an auditable rule."""
    raw = str(label).strip()
    text = re.sub(r"[^a-z0-9]+", " ", raw.lower()).strip()

    ordered_rules: list[tuple[str, str, str, list[str]]] = [
        ("MAIN_NARROW", "Enrgy", "medium", ["main brown narrow"]),
        ("MAIN_BROAD_INCREMENT", "Manuf", "low", ["main broad only increment"]),
        ("MAIN_RESIDUAL", "Other", "low", ["main unclassified non brown residual"]),
        ("UTILITIES", "Utils", "high", ["water utilities", "utilities water", "electricity", "utility"]),
        ("ENERGY", "Enrgy", "high", ["oil gas", "energy", "renewable electricity", "upstream", "midstream"]),
        ("HEALTH", "Hlth", "high", ["health", "medical", "pharma", "biotech", "diagnostic", "dental", "veterinary", "drug", "life sciences"]),
        ("CHEMICALS", "Chems", "high", ["chemical", "fertilizer"]),
        ("TELECOM", "Telcm", "high", ["telecommunication", "telecommunications", "wireless communication", "integrated communication"]),
        ("BUSINESS_EQUIPMENT", "BusEq", "high", ["software", "cyber", "data processing", "information technology", "internet", "semiconductor", "computer hardware", "electronics", "electronic equipment", "networking", "technology hardware", "technology products", "communications equipment"]),
        ("MONEY", "Money", "high", ["bank", "finance", "financial", "insurance", "real estate", "reit", "capital market", "asset management", "investment fund", "investment funds", "investment management", "structured finance", "online lending", "credit opportunities", "joint venture", "net lease"]),
        ("NONDURABLES", "NoDur", "high", ["food", "beverage", "household product", "personal care", "personal product", "textile", "apparel", "tobacco", "paper forest"]),
        ("DURABLES", "Durbl", "high", ["automobile", "automotive", "auto component", "auto aftermarket", "household durable", "consumer durable", "home furnishing", "houseware", "leisure product"]),
        ("MANUFACTURING", "Manuf", "high", ["aerospace", "defense", "manufactur", "machinery", "capital equipment", "capital goods", "industrial", "building product", "construction", "container", "packaging", "metal", "mining", "materials", "electrical equipment", "component manufacturing"]),
        ("RETAIL", "Shops", "high", ["retail", "distribut", "restaurant", "hotel", "hospitality", "consumer products", "consumer goods", "consumer services", "specialty food", "wholesale"]),
        ("COMMUNICATIONS", "Telcm", "medium", ["communications", "communication"]),
        ("TRANSPORT_OTHER", "Other", "medium", ["transport", "airline", "airport", "road rail", "marine", "air freight", "logistics"]),
        ("MEDIA_OTHER", "Other", "medium", ["media", "entertainment", "movie", "advertising", "sports management"]),
        ("SERVICES_OTHER", "Other", "medium", ["service", "education", "environmental", "office", "staffing", "facilities", "non profit", "professional"]),
        ("MULTISECTOR_OTHER", "Other", "low", ["multi sector", "other", "miscellaneous", "consumer related", "business products", "high tech industries"]),
    ]
    for rule_name, group, confidence, needles in ordered_rules:
        if any(needle in text for needle in needles):
            return group, rule_name, confidence
    return "Other", "UNMAPPED_FALLBACK_OTHER", "low"


def build_ff12_beta() -> pd.DataFrame:
    daily = pd.read_csv(
        PROCESSED / "sic3_industry_climate_beta_daily_2020_2025.csv",
        parse_dates=["date"],
        dtype={"sic3": str},
    )
    daily["sic3"] = daily["sic3"].str.zfill(3)
    daily["ff12_group"] = daily["sic3"].map(ff12_from_sic3)
    daily["weighted_beta"] = daily["cbeta_sic3"] * daily["market_cap_sum"]
    grouped = (
        daily.groupby(["date", "ff12_group"], as_index=False)
        .agg(
            weighted_beta_sum=("weighted_beta", "sum"),
            market_cap_sum=("market_cap_sum", "sum"),
            sic3_count=("sic3", "nunique"),
            firm_count_sum=("n_firms", "sum"),
        )
    )
    grouped["cbeta_ff12"] = grouped["weighted_beta_sum"] / grouped["market_cap_sum"]
    grouped["quarter"] = grouped["date"].dt.to_period("Q").astype(str)
    qend = (
        grouped.sort_values(["ff12_group", "date"])
        .groupby(["ff12_group", "quarter"], as_index=False)
        .tail(1)
        .rename(columns={"date": "quarter_end_observation_date", "cbeta_ff12": "cbeta_ff12_qend"})
    )
    qmean = (
        grouped.groupby(["ff12_group", "quarter"], as_index=False)
        .agg(cbeta_ff12_qmean=("cbeta_ff12", "mean"), trading_days=("date", "nunique"))
    )
    quarterly = qend[["ff12_group", "quarter", "quarter_end_observation_date", "cbeta_ff12_qend", "market_cap_sum", "sic3_count", "firm_count_sum"]].merge(
        qmean, on=["ff12_group", "quarter"], how="outer"
    )
    return quarterly.sort_values(["quarter", "ff12_group"]).reset_index(drop=True)


def load_exposure() -> pd.DataFrame:
    sources = [
        (ROOT / "Data/Processed/industry_exposure_2024_all10.csv", "original_10"),
        (PROCESSED / "industry_exposure_2024_direct_summary_4bdc.csv", "additional_direct_4"),
        (PROCESSED / "industry_exposure_2024_additional5.csv", "additional_extracted_5"),
    ]
    parts = []
    for path, cohort in sources:
        data = pd.read_csv(path)
        data["sample_cohort"] = cohort
        parts.append(data[["ticker", "industry_reported", "portfolio_fair_value_pct_2024", "source_method", "sample_cohort"]])
    exposure = pd.concat(parts, ignore_index=True)
    mapped = exposure["industry_reported"].map(label_to_ff12)
    exposure[["ff12_group", "mapping_rule", "mapping_confidence"]] = pd.DataFrame(mapped.tolist(), index=exposure.index)
    exposure["portfolio_fair_value_pct_2024"] = pd.to_numeric(exposure["portfolio_fair_value_pct_2024"], errors="coerce")
    exposure = exposure.dropna(subset=["portfolio_fair_value_pct_2024"])
    return exposure.sort_values(["ticker", "ff12_group", "industry_reported"]).reset_index(drop=True)


def fit_ols(
    frame: pd.DataFrame,
    outcome: str,
    exposure: str,
    model_id: str,
    label: str,
    controls: list[str] | None = None,
    fixed_effects: list[str] | None = None,
    cluster: str | None = None,
) -> dict[str, object]:
    controls = controls or []
    fixed_effects = fixed_effects or []
    needed = [outcome, exposure, *controls, *fixed_effects]
    if cluster:
        needed.append(cluster)
    needed = list(dict.fromkeys(needed))
    data = frame[needed].dropna().copy()
    blocks = [np.ones((len(data), 1)), data[[exposure, *controls]].to_numpy(float)]
    coefficient_names = ["intercept", exposure, *controls]
    for fe in fixed_effects:
        dummies = pd.get_dummies(data[fe].astype(str), prefix=fe, drop_first=True, dtype=float)
        blocks.append(dummies.to_numpy(float))
        coefficient_names.extend(dummies.columns.tolist())
    x = np.column_stack(blocks)
    y = data[outcome].to_numpy(float)
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    n, k = x.shape
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)

    if cluster:
        meat = np.zeros((k, k))
        groups = data[cluster].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        for group in unique_groups:
            mask = groups == group
            score = x[mask].T @ resid[mask]
            meat += np.outer(score, score)
        correction = (len(unique_groups) / (len(unique_groups) - 1)) * ((n - 1) / max(n - k, 1))
        covariance = correction * xtx_inv @ meat @ xtx_inv
        df = len(unique_groups) - 1
        se_type = f"cluster({cluster})"
    else:
        adjusted = resid / np.clip(1.0 - leverage, 1e-10, None)
        meat = x.T @ ((adjusted ** 2)[:, None] * x)
        covariance = xtx_inv @ meat @ xtx_inv
        df = max(n - k, 1)
        se_type = "HC3"

    index = coefficient_names.index(exposure)
    coef = float(beta[index])
    se = float(np.sqrt(max(covariance[index, index], 0.0)))
    t_value = coef / se if se > 0 else np.nan
    p_two = float(2 * stats.t.sf(abs(t_value), df=df)) if np.isfinite(t_value) else np.nan
    p_one = float(stats.t.sf(t_value, df=df)) if np.isfinite(t_value) else np.nan
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(np.sum(resid ** 2)) / sst if sst > 0 else np.nan
    return {
        "model_id": model_id,
        "model_label": label,
        "outcome": outcome,
        "exposure": exposure,
        "controls": ", ".join(controls) if controls else "none",
        "fixed_effects": ", ".join(fixed_effects) if fixed_effects else "none",
        "se_type": se_type,
        "coefficient_exposure": coef,
        "standard_error": se,
        "t_statistic": t_value,
        "p_two_sided": p_two,
        "p_one_sided_positive": p_one,
        "n": n,
        "df_inference": df,
        "r_squared": r2,
        "positive_direction": coef > 0,
        "significant_positive_10pct": (coef > 0) and (p_one < 0.10),
    }


def permutation_p_positive(data: pd.DataFrame, outcome: str, exposure: str, draws: int = 50000) -> float:
    clean = data[[outcome, exposure]].dropna()
    x = clean[exposure].to_numpy(float)
    y = clean[outcome].to_numpy(float)
    observed = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    rng = np.random.default_rng(20260816)
    simulated = np.empty(draws)
    for index in range(draws):
        shuffled = rng.permutation(y)
        simulated[index] = np.cov(x, shuffled, ddof=1)[0, 1] / np.var(x, ddof=1)
    return float((1 + np.sum(simulated >= observed)) / (draws + 1))


def portfolio_beta(exposure: pd.DataFrame, ff12: pd.DataFrame, quarters: list[str]) -> pd.DataFrame:
    weights = (
        exposure.groupby(["ticker", "ff12_group"], as_index=False)["portfolio_fair_value_pct_2024"].sum()
    )
    grid = weights.assign(key=1).merge(pd.DataFrame({"quarter": quarters, "key": 1}), on="key").drop(columns="key")
    merged = grid.merge(ff12, on=["ff12_group", "quarter"], how="left")
    merged["weighted_qend"] = merged["portfolio_fair_value_pct_2024"] * merged["cbeta_ff12_qend"] / 100
    merged["weighted_qmean"] = merged["portfolio_fair_value_pct_2024"] * merged["cbeta_ff12_qmean"] / 100
    portfolio = (
        merged.groupby(["ticker", "quarter"], as_index=False)
        .agg(
            loan_portfolio_climate_beta_qend=("weighted_qend", "sum"),
            loan_portfolio_climate_beta_qmean=("weighted_qmean", "sum"),
            mapped_weight_pct=("portfolio_fair_value_pct_2024", "sum"),
            ff12_groups=("ff12_group", "nunique"),
        )
    )
    return portfolio


def scatter_plot(data: pd.DataFrame, outcome: str, exposure: str, title: str, filename: str) -> None:
    clean = data[["ticker", outcome, exposure]].dropna().copy()
    slope, intercept = np.polyfit(clean[exposure], clean[outcome], 1)
    xx = np.linspace(clean[exposure].min(), clean[exposure].max(), 100)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.scatter(clean[exposure], clean[outcome], s=45, color="#1f4e78", alpha=0.82)
    ax.plot(xx, intercept + slope * xx, color="#c00000", linewidth=1.8)
    for _, row in clean.iterrows():
        ax.annotate(row["ticker"], (row[exposure], row[outcome]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("2024 loan-portfolio climate beta (FF12, quarter-end)")
    ax.set_ylabel(outcome.replace("_", " "))
    ax.grid(axis="both", alpha=0.20)
    ax.text(0.01, 0.01, f"OLS slope = {slope:.3f}; n = {len(clean)}", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)

    ff12 = build_ff12_beta()
    exposure = load_exposure()
    ff12.to_csv(PROCESSED / "ff12_industry_climate_beta_quarterly_2020_2025.csv", index=False)
    export_dta(ff12, PROCESSED / "ff12_industry_climate_beta_qtr_2020_2025.dta")
    exposure.to_csv(PROCESSED / "bdc19_industry_mapping_2024.csv", index=False)
    export_dta(exposure, PROCESSED / "bdc19_industry_mapping_2024.dta")

    panel = pd.read_csv(PROCESSED / "bdc20_quarter_market_financial_panel_2021_2025.csv", parse_dates=["datadate"])
    panel = panel[panel["ticker"].isin(exposure["ticker"].unique())].copy()
    panel["quarter"] = panel["datadate"].dt.to_period("Q").astype(str)
    all_quarters = sorted(panel["quarter"].unique())
    pbeta = portfolio_beta(exposure, ff12, all_quarters)

    frozen_panel = panel.merge(pbeta, on=["ticker", "quarter"], how="left")
    frozen_panel["exposure_timing_note"] = "FY2024 portfolio weights held fixed across 2021Q1-2025Q4; sensitivity only"
    frozen_panel.to_csv(PROCESSED / "bdc19_frozen_weight_industry_beta_panel_2021_2025.csv", index=False)
    export_dta(frozen_panel, PROCESSED / "bdc19_frozen_weight_panel_2021_2025.dta")

    old_index = pd.read_csv(ROOT / "Data/Processed/sec_10k_source_index_2024.csv", parse_dates=["report_date"])
    new_index = pd.read_csv(AUDIT / "additional_10_bdc_2024_sec_index.csv", parse_dates=["report_date"])
    reports = pd.concat([old_index[["ticker", "report_date", "filing_date", "filing_url"]], new_index[["ticker", "report_date", "filing_date", "filing_url"]]], ignore_index=True)
    reports = reports[reports["ticker"].isin(exposure["ticker"].unique())].drop_duplicates("ticker")
    outcomes = panel.merge(reports, left_on=["ticker", "datadate"], right_on=["ticker", "report_date"], how="inner")
    cross = outcomes.merge(pbeta, on=["ticker", "quarter"], how="left")
    confidence = (
        exposure.assign(
            low_weight=lambda d: np.where(d["mapping_confidence"].eq("low"), d["portfolio_fair_value_pct_2024"], 0.0),
            high_weight=lambda d: np.where(d["mapping_confidence"].eq("high"), d["portfolio_fair_value_pct_2024"], 0.0),
        )
        .groupby("ticker", as_index=False)
        .agg(low_confidence_weight_pct=("low_weight", "sum"), high_confidence_weight_pct=("high_weight", "sum"), reported_industry_rows=("industry_reported", "size"))
    )
    cross = cross.merge(confidence, on="ticker", how="left").sort_values("ticker")
    cross.to_csv(TABLES / "h2_industry_beta_company_cross_section_2024.csv", index=False)
    export_dta(cross, TABLES / "h2_industry_beta_company_2024.dta")

    cross_specs = [
        ("C1", "Primary: equity beta on quarter-end portfolio beta", "beta_climate_equity_report_month", "loan_portfolio_climate_beta_qend", [], []),
        ("C2", "Leverage-adjusted asset beta on quarter-end portfolio beta", "beta_climate_asset_report_month", "loan_portfolio_climate_beta_qend", [], []),
        ("C3", "Equity beta on quarterly-mean portfolio beta", "beta_climate_equity_report_month", "loan_portfolio_climate_beta_qmean", [], []),
        ("C4", "Asset beta on quarterly-mean portfolio beta", "beta_climate_asset_report_month", "loan_portfolio_climate_beta_qmean", [], []),
        ("C5", "Primary excluding MAIN", "beta_climate_equity_report_month", "loan_portfolio_climate_beta_qend", [], ["MAIN"]),
        ("C6", "Asset beta excluding MAIN", "beta_climate_asset_report_month", "loan_portfolio_climate_beta_qend", [], ["MAIN"]),
        ("C7", "Equity beta with size and leverage controls", "beta_climate_equity_report_month", "loan_portfolio_climate_beta_qend", ["log_assets", "debt_to_assets"], []),
        ("C8", "Asset beta with size and profitability controls", "beta_climate_asset_report_month", "loan_portfolio_climate_beta_qend", ["log_assets", "roa_quarter"], []),
    ]
    cross_models = []
    for model_id, label, outcome, xvar, controls, excluded in cross_specs:
        sample = cross[~cross["ticker"].isin(excluded)].copy()
        row = fit_ols(sample, outcome, xvar, model_id, label, controls=controls)
        row["sample_exclusion"] = ", ".join(excluded) if excluded else "none"
        if not controls:
            row["permutation_p_one_sided_positive"] = permutation_p_positive(sample, outcome, xvar)
            clean = sample[[outcome, xvar]].dropna()
            row["pearson_correlation"] = stats.pearsonr(clean[xvar], clean[outcome]).statistic
            row["spearman_correlation"] = stats.spearmanr(clean[xvar], clean[outcome]).statistic
        else:
            row["permutation_p_one_sided_positive"] = np.nan
            row["pearson_correlation"] = np.nan
            row["spearman_correlation"] = np.nan
        cross_models.append(row)
    cross_models = pd.DataFrame(cross_models)
    cross_models.to_csv(TABLES / "h2_industry_beta_cross_section_models.csv", index=False)
    export_dta(cross_models, TABLES / "h2_industry_beta_cross_models.dta")

    panel_specs = [
        ("P1", "Pooled equity beta / quarter-end portfolio beta", "beta_climate_equity_report_month", "loan_portfolio_climate_beta_qend", [], []),
        ("P2", "Pooled asset beta / quarter-end portfolio beta", "beta_climate_asset_report_month", "loan_portfolio_climate_beta_qend", [], []),
        ("P3", "Firm and quarter FE: equity beta", "beta_climate_equity_report_month", "loan_portfolio_climate_beta_qend", [], ["ticker", "quarter"]),
        ("P4", "Firm and quarter FE: asset beta", "beta_climate_asset_report_month", "loan_portfolio_climate_beta_qend", [], ["ticker", "quarter"]),
        ("P5", "FE plus controls: equity beta", "beta_climate_equity_report_month", "loan_portfolio_climate_beta_qend", ["log_assets", "debt_to_assets", "roa_quarter", "book_to_market", "beta_market_report_month"], ["ticker", "quarter"]),
        ("P6", "FE plus controls: asset beta", "beta_climate_asset_report_month", "loan_portfolio_climate_beta_qend", ["log_assets", "debt_to_assets", "roa_quarter", "book_to_market", "beta_market_report_month"], ["ticker", "quarter"]),
        ("P7", "Firm and quarter FE: equity beta / quarterly mean", "beta_climate_equity_report_month", "loan_portfolio_climate_beta_qmean", [], ["ticker", "quarter"]),
        ("P8", "Firm and quarter FE: asset beta / quarterly mean", "beta_climate_asset_report_month", "loan_portfolio_climate_beta_qmean", [], ["ticker", "quarter"]),
    ]
    panel_models = pd.DataFrame([
        fit_ols(frozen_panel, outcome, xvar, model_id, label, controls=controls, fixed_effects=fes, cluster="ticker")
        for model_id, label, outcome, xvar, controls, fes in panel_specs
    ])
    panel_models["interpretation_scope"] = "Sensitivity only: FY2024 portfolio weights held fixed over time"
    panel_models.to_csv(TABLES / "h2_industry_beta_frozen_panel_models.csv", index=False)
    export_dta(panel_models, TABLES / "h2_industry_beta_panel_models.dta")

    primary = cross_models.loc[cross_models["model_id"].eq("C1")].iloc[0]
    asset = cross_models.loc[cross_models["model_id"].eq("C2")].iloc[0]
    panel_primary = panel_models.loc[panel_models["model_id"].eq("P3")].iloc[0]
    if bool(primary["significant_positive_10pct"]) and bool(asset["positive_direction"]):
        decision_code = "SUPPORTED"
        conclusion_cn = "H2 获得支持：主回归为正，单侧 10% 水平显著，且资产 Beta 方向一致。"
    elif bool(primary["positive_direction"]):
        decision_code = "NOT_SUPPORTED_IMPRECISE"
        conclusion_cn = "主回归方向为正但未达到预设显著性标准；该结果不构成对 H2 的统计支持。"
    else:
        decision_code = "NOT_SUPPORTED_IN_THIS_SAMPLE"
        conclusion_cn = "H2 在本样本中未获支持：主回归系数并非正向，不能宣称正相关。"
    decision = pd.DataFrame([{
        "hypothesis": "H2 mechanism check: BDC investment-portfolio climate beta is positively associated with BDC climate beta",
        "decision_code": decision_code,
        "conclusion_cn": conclusion_cn,
        "conclusion_en": "H2 is not supported in this sample: the primary coefficient is negative, so a positive association cannot be claimed." if decision_code == "NOT_SUPPORTED_IN_THIS_SAMPLE" else ("The positive direction is statistically imprecise and is not classified as support for H2." if decision_code == "NOT_SUPPORTED_IMPRECISE" else "H2 is supported under the pre-specified decision rule."),
        "primary_model": "C1",
        "primary_coefficient": primary["coefficient_exposure"],
        "primary_hc3_p_one_sided_positive": primary["p_one_sided_positive"],
        "primary_permutation_p_one_sided_positive": primary["permutation_p_one_sided_positive"],
        "asset_model_coefficient": asset["coefficient_exposure"],
        "asset_model_p_one_sided_positive": asset["p_one_sided_positive"],
        "frozen_panel_fe_coefficient": panel_primary["coefficient_exposure"],
        "frozen_panel_fe_p_one_sided_positive": panel_primary["p_one_sided_positive"],
        "main_sample_n": int(primary["n"]),
        "excluded_bdc": "BBDC",
        "exclusion_reason": "FY2024 filing does not disclose industry; FY2023 industry schedule was not substituted.",
        "important_limitation": "Reported industry names are mapped to FF12; pseudo-panel holds FY2024 portfolio weights fixed.",
    }])
    decision.to_csv(TABLES / "h2_industry_beta_decision.csv", index=False)
    export_dta(decision, TABLES / "h2_industry_beta_decision.dta")

    scatter_plot(cross, "beta_climate_equity_report_month", "loan_portfolio_climate_beta_qend", "H2 cross-section: BDC equity climate beta", "Figure_P3_1_H2_EquityBeta_Scatter.png")
    scatter_plot(cross, "beta_climate_asset_report_month", "loan_portfolio_climate_beta_qend", "H2 cross-section: leverage-adjusted asset climate beta", "Figure_P3_2_H2_AssetBeta_Scatter.png")

    weight_check = exposure.groupby("ticker")["portfolio_fair_value_pct_2024"].sum()
    audit = {
        "status": "PASS",
        "analysis_version": "BDC_Investment_Exposure_and_Climate_Beta_v2.3",
        "industry_beta_method": "365-calendar-day rolling two-factor firm betas aggregated market-cap-weighted to SIC3, then FF12",
        "sample_firms": int(exposure["ticker"].nunique()),
        "sample_tickers": sorted(exposure["ticker"].unique().tolist()),
        "exposure_rows": int(len(exposure)),
        "weight_sum_min": float(weight_check.min()),
        "weight_sum_max": float(weight_check.max()),
        "ff12_quarter_rows": int(len(ff12)),
        "cross_section_rows": int(len(cross)),
        "frozen_panel_rows": int(len(frozen_panel)),
        "low_confidence_weight_mean_pct": float(cross["low_confidence_weight_pct"].mean()),
        "decision": decision.iloc[0].to_dict(),
        "precommitted_direction": "positive",
        "inference_note": "No sign tuning or variable selection based on realized results; C1 is the declared primary specification.",
    }
    with (AUDIT / "h2_industry_beta_final_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, default=str, ensure_ascii=False)
    print(json.dumps(audit, indent=2, default=str, ensure_ascii=False))
    print("\nCross-section models:\n", cross_models[["model_id", "coefficient_exposure", "standard_error", "p_one_sided_positive", "permutation_p_one_sided_positive", "n"]].to_string(index=False))
    print("\nPanel models:\n", panel_models[["model_id", "coefficient_exposure", "standard_error", "p_one_sided_positive", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()
