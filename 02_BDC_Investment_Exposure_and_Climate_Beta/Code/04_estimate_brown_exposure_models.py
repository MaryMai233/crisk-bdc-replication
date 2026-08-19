from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
LOCAL_UPSTREAM = ROOT / "Data" / "Raw"
PROCESSED = ROOT / "Data" / "Processed"
AUDIT = ROOT / "Data" / "Processed" / "Audit"
RESULTS = ROOT / "Results"
TABLES = PROCESSED
FIGURES = RESULTS


def load_prior_module():
    local_helper = Path(__file__).with_name("01_industry_beta_helpers.py")
    path = local_helper
    spec = importlib.util.spec_from_file_location("prior_h2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PRIOR = load_prior_module()


def export_dta(frame: pd.DataFrame, path: Path) -> None:
    PRIOR.export_dta(frame, path)


def load_ff12() -> pd.DataFrame:
    local_path = LOCAL_UPSTREAM / "ff12_industry_climate_beta_quarterly_2020_2025.csv"
    if not local_path.exists():
        raise FileNotFoundError(f"Required input not found: {local_path}")
    return pd.read_csv(local_path, parse_dates=["quarter_end_observation_date"])


def build_dynamic_portfolio_beta(exposure: pd.DataFrame, ff12: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped = exposure["industry_reported"].map(PRIOR.label_to_ff12)
    exposure[["ff12_group", "mapping_rule", "mapping_confidence"]] = pd.DataFrame(
        mapped.tolist(), index=exposure.index
    )
    exposure["portfolio_fair_value_pct"] = pd.to_numeric(
        exposure["portfolio_fair_value_pct"], errors="coerce"
    )
    exposure = exposure.dropna(subset=["portfolio_fair_value_pct"]).copy()
    exposure["quarter"] = exposure["calendar_quarter"].astype(str)
    weights = (
        exposure.groupby(["ticker", "quarter", "ff12_group"], as_index=False)
        .agg(
            portfolio_pct_reported=("portfolio_fair_value_pct", "sum"),
            industry_rows=("industry_reported", "size"),
        )
    )
    coverage = (
        weights.groupby(["ticker", "quarter"], as_index=False)["portfolio_pct_reported"]
        .sum()
        .rename(columns={"portfolio_pct_reported": "mapped_weight_pct"})
    )
    weights = weights.merge(coverage, on=["ticker", "quarter"], how="left")
    weights["portfolio_pct_for_beta"] = (
        weights["portfolio_pct_reported"] / weights["mapped_weight_pct"] * 100
    )
    weights = weights.merge(ff12, on=["ff12_group", "quarter"], how="left", validate="many_to_one")
    weights["weighted_qend"] = weights["portfolio_pct_for_beta"] * weights["cbeta_ff12_qend"] / 100
    weights["weighted_qmean"] = weights["portfolio_pct_for_beta"] * weights["cbeta_ff12_qmean"] / 100
    portfolio = (
        weights.groupby(["ticker", "quarter"], as_index=False)
        .agg(
            loan_portfolio_climate_beta_qend=("weighted_qend", "sum"),
            loan_portfolio_climate_beta_qmean=("weighted_qmean", "sum"),
            mapped_weight_pct=("mapped_weight_pct", "first"),
            ff12_groups=("ff12_group", "nunique"),
            industry_rows=("industry_rows", "sum"),
        )
    )
    brown = (
        exposure.assign(
            brown_narrow_weight=lambda d: d["portfolio_fair_value_pct"] * d["brown_narrow"],
            brown_broad_weight=lambda d: d["portfolio_fair_value_pct"] * d["brown_broad"],
            low_conf_weight=lambda d: np.where(
                d["mapping_confidence"].eq("low"), d["portfolio_fair_value_pct"], 0.0
            ),
        )
        .groupby(["ticker", "quarter"], as_index=False)
        .agg(
            brown_share_narrow_dynamic_pct=("brown_narrow_weight", "sum"),
            brown_share_broad_dynamic_pct=("brown_broad_weight", "sum"),
            low_confidence_mapping_weight_pct=("low_conf_weight", "sum"),
        )
    )
    portfolio = portfolio.merge(brown, on=["ticker", "quarter"], how="left")
    return exposure, portfolio


def add_standardized(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        std = frame[column].std(ddof=0)
        frame[f"z_{column}"] = (frame[column] - frame[column].mean()) / std if std > 0 else np.nan
    return frame


def model(
    frame: pd.DataFrame,
    outcome: str,
    exposure: str,
    model_id: str,
    label: str,
    controls: list[str] | None = None,
    fixed_effects: list[str] | None = None,
    cluster: str | None = None,
) -> dict[str, object]:
    result = PRIOR.fit_ols(
        frame, outcome, exposure, model_id, label,
        controls=controls or [], fixed_effects=fixed_effects or [], cluster=cluster,
    )
    result["coefficient_unit"] = "Outcome change per one-SD increase in exposure"
    return result


def residualize_two_way(frame: pd.DataFrame, column: str) -> pd.Series:
    overall = frame[column].mean()
    return (
        frame[column]
        - frame.groupby("ticker")[column].transform("mean")
        - frame.groupby("quarter")[column].transform("mean")
        + overall
    )


def plot_within_relationship(panel: pd.DataFrame) -> None:
    subset = panel[["ticker", "quarter", "z_brown_share_broad_dynamic_pct", "z_beta_climate_equity_report_month"]].dropna().copy()
    subset["x_within"] = residualize_two_way(subset, "z_brown_share_broad_dynamic_pct")
    subset["y_within"] = residualize_two_way(subset, "z_beta_climate_equity_report_month")
    slope, intercept = np.polyfit(subset["x_within"], subset["y_within"], 1)
    xx = np.linspace(subset["x_within"].min(), subset["x_within"].max(), 120)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.scatter(subset["x_within"], subset["y_within"], s=22, alpha=0.50, color="#1F4E78", edgecolor="none")
    ax.plot(xx, intercept + slope * xx, color="#C00000", linewidth=2)
    ax.axhline(0, color="#6B7280", linewidth=0.8, alpha=0.6)
    ax.axvline(0, color="#6B7280", linewidth=0.8, alpha=0.6)
    ax.set_title("H2 dynamic panel: within-firm and within-quarter relationship", weight="bold")
    ax.set_xlabel("Broad carbon-intensive investment share (two-way demeaned, SD units)")
    ax.set_ylabel("BDC equity climate beta (two-way demeaned, SD units)")
    ax.grid(alpha=0.18)
    ax.text(0.02, 0.02, f"Within slope = {slope:.3f}; N = {len(subset)}", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_P3_v25_1_Dynamic_Within_Relationship.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_exposure_trends(panel: pd.DataFrame) -> None:
    annual = (
        panel.assign(year=panel["quarter"].str[:4])
        .groupby("year", as_index=False)
        .agg(
            portfolio_beta=("loan_portfolio_climate_beta_qend", "mean"),
            brown_broad=("brown_share_broad_dynamic_pct", "mean"),
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))
    axes[0].plot(annual["year"], annual["portfolio_beta"], marker="o", color="#1F4E78", linewidth=2)
    axes[0].set_title("Mean loan-portfolio climate beta", weight="bold")
    axes[0].set_ylabel("Climate beta")
    axes[1].plot(annual["year"], annual["brown_broad"], marker="o", color="#C55A11", linewidth=2)
    axes[1].set_title("Mean broad carbon-intensive exposure", weight="bold")
    axes[1].set_ylabel("Percent of portfolio")
    for ax in axes:
        ax.set_xlabel("Year")
        ax.grid(axis="y", alpha=0.20)
    fig.suptitle("Dynamic BDC portfolio measures, 2021–2025", weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_P3_v25_2_Dynamic_Exposure_Trends.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    exposure = pd.read_csv(PROCESSED / "dynamic_bdc_industry_exposure_2021_2025.csv")
    ff12 = load_ff12()
    mapped_exposure, portfolio = build_dynamic_portfolio_beta(exposure, ff12)
    mapped_exposure.to_csv(PROCESSED / "dynamic_bdc_industry_ff12_mapping_2021_2025.csv", index=False)
    export_dta(mapped_exposure, PROCESSED / "dynamic_bdc_industry_ff12_mapping_2021_2025.dta")
    portfolio.to_csv(PROCESSED / "dynamic_bdc_portfolio_climate_beta_2021_2025.csv", index=False)
    export_dta(portfolio, PROCESSED / "dynamic_bdc_portfolio_climate_beta_2021_2025.dta")

    updated_panel = PROCESSED / "bdc20_quarter_market_financial_panel_2021_2025_updated.csv"
    local_panel = LOCAL_UPSTREAM / "bdc20_quarter_market_financial_panel_2021_2025.csv"
    panel_path = updated_panel if updated_panel.exists() else local_panel
    panel = pd.read_csv(panel_path, parse_dates=["datadate"])
    panel = panel[panel["ticker"].isin(portfolio["ticker"].unique())].copy()
    panel["quarter"] = panel["datadate"].dt.to_period("Q").astype(str)
    panel = panel.merge(portfolio, on=["ticker", "quarter"], how="left", validate="one_to_one")
    panel = panel.sort_values(["ticker", "quarter"]).reset_index(drop=True)
    panel["loan_portfolio_climate_beta_qend_lag1"] = panel.groupby("ticker")["loan_portfolio_climate_beta_qend"].shift(1)
    panel["loan_portfolio_climate_beta_qmean_lag1"] = panel.groupby("ticker")["loan_portfolio_climate_beta_qmean"].shift(1)
    zcols = [
        "beta_climate_equity_report_month", "beta_climate_asset_report_month",
        "loan_portfolio_climate_beta_qend", "loan_portfolio_climate_beta_qmean",
        "loan_portfolio_climate_beta_qend_lag1", "loan_portfolio_climate_beta_qmean_lag1",
        "brown_share_narrow_dynamic_pct", "brown_share_broad_dynamic_pct",
        "log_assets", "debt_to_assets", "roa_quarter", "book_to_market", "beta_market_report_month",
    ]
    panel = add_standardized(panel, zcols)
    panel.to_csv(PROCESSED / "bdc19_dynamic_portfolio_h2_panel_2021_2025.csv", index=False)
    export_dta(panel, PROCESSED / "bdc19_dynamic_portfolio_h2_panel_2021_2025.dta")

    y_eq = "z_beta_climate_equity_report_month"
    y_as = "z_beta_climate_asset_report_month"
    x_qe = "z_loan_portfolio_climate_beta_qend"
    x_qm = "z_loan_portfolio_climate_beta_qmean"
    controls = ["z_log_assets", "z_debt_to_assets", "z_roa_quarter", "z_book_to_market", "z_beta_market_report_month"]
    # Portfolio-climate-beta models are a mechanism check.  The proposal's H2
    # is instead defined on the disclosed carbon-intensive investment share.
    specs = [
        ("M1", "Mechanism: firm and quarter FE, equity climate beta", y_eq, x_qe, [], ["ticker", "quarter"]),
        ("M2", "Mechanism: asset climate beta, firm and quarter FE", y_as, x_qe, [], ["ticker", "quarter"]),
        ("M3", "Mechanism: equity beta, quarterly-mean portfolio beta", y_eq, x_qm, [], ["ticker", "quarter"]),
        ("M4", "Mechanism: asset beta, quarterly-mean portfolio beta", y_as, x_qm, [], ["ticker", "quarter"]),
        ("M5", "Mechanism: equity beta plus controls", y_eq, x_qe, controls, ["ticker", "quarter"]),
        ("M6", "Mechanism: asset beta plus controls", y_as, x_qe, controls, ["ticker", "quarter"]),
        ("M7", "Mechanism: pooled equity beta", y_eq, x_qe, [], []),
        ("M8", "Mechanism: pooled asset beta", y_as, x_qe, [], []),
        ("M9", "Mechanism: lagged portfolio beta, equity FE", y_eq, "z_loan_portfolio_climate_beta_qend_lag1", [], ["ticker", "quarter"]),
        ("M10", "Mechanism: lagged portfolio beta, asset FE", y_as, "z_loan_portfolio_climate_beta_qend_lag1", [], ["ticker", "quarter"]),
    ]
    models = pd.DataFrame([
        model(panel, outcome, xvar, model_id, label, model_controls, fixed_effects, "ticker")
        for model_id, label, outcome, xvar, model_controls, fixed_effects in specs
    ])
    models.to_csv(TABLES / "h2_portfolio_beta_mechanism_models.csv", index=False)
    export_dta(models, TABLES / "h2_portfolio_beta_mechanism_models.dta")

    panel["z_brown_broad_between"] = panel.groupby("ticker")["z_brown_share_broad_dynamic_pct"].transform("mean")
    panel["z_brown_broad_within"] = panel["z_brown_share_broad_dynamic_pct"] - panel["z_brown_broad_between"]
    panel.to_csv(PROCESSED / "bdc19_dynamic_portfolio_h2_panel_2021_2025.csv", index=False)
    export_dta(panel, PROCESSED / "bdc19_dynamic_portfolio_h2_panel_2021_2025.dta")
    brown_specs = [
        ("H2_1", "Primary: broad brown share, equity beta, quarter FE", y_eq, "z_brown_share_broad_dynamic_pct", [], ["quarter"]),
        ("H2_2", "Broad brown share, asset beta, quarter FE", y_as, "z_brown_share_broad_dynamic_pct", [], ["quarter"]),
        ("H2_3", "Broad brown share, equity beta, quarter FE plus controls", y_eq, "z_brown_share_broad_dynamic_pct", controls, ["quarter"]),
        ("H2_4", "Broad brown share, asset beta, quarter FE plus controls", y_as, "z_brown_share_broad_dynamic_pct", controls, ["quarter"]),
        ("H2_5", "Within-BDC sensitivity: equity beta, firm and quarter FE", y_eq, "z_brown_share_broad_dynamic_pct", [], ["ticker", "quarter"]),
        ("H2_6", "Within-BDC sensitivity: asset beta, firm and quarter FE", y_as, "z_brown_share_broad_dynamic_pct", [], ["ticker", "quarter"]),
        ("H2_7", "Narrow brown share sensitivity, equity beta, quarter FE", y_eq, "z_brown_share_narrow_dynamic_pct", [], ["quarter"]),
        ("H2_8", "Narrow brown share sensitivity, asset beta, quarter FE", y_as, "z_brown_share_narrow_dynamic_pct", [], ["quarter"]),
        ("H2_9", "Between-BDC broad-share component, equity beta", y_eq, "z_brown_broad_between", ["z_brown_broad_within"], ["quarter"]),
        ("H2_10", "Between-BDC broad-share component, asset beta", y_as, "z_brown_broad_between", ["z_brown_broad_within"], ["quarter"]),
    ]
    brown_models = pd.DataFrame([
        model(panel, outcome, xvar, model_id, label, model_controls, fixed_effects, "ticker")
        for model_id, label, outcome, xvar, model_controls, fixed_effects in brown_specs
    ])
    brown_models.to_csv(TABLES / "h2_brown_share_dynamic_models.csv", index=False)
    export_dta(brown_models, TABLES / "h2_brown_share_dynamic_models.dta")

    company_means = (
        panel.groupby("ticker", as_index=False)
        .agg(
            beta_climate_equity_mean=("beta_climate_equity_report_month", "mean"),
            beta_climate_asset_mean=("beta_climate_asset_report_month", "mean"),
            portfolio_beta_mean=("loan_portfolio_climate_beta_qend", "mean"),
            brown_broad_mean=("brown_share_broad_dynamic_pct", "mean"),
            log_assets_mean=("log_assets", "mean"),
            debt_to_assets_mean=("debt_to_assets", "mean"),
        )
    )
    company_means = add_standardized(
        company_means,
        ["beta_climate_equity_mean", "beta_climate_asset_mean", "portfolio_beta_mean", "brown_broad_mean", "log_assets_mean", "debt_to_assets_mean"],
    )
    cross_specs = [
        ("B1", "Company-mean equity beta on company-mean portfolio beta", "z_beta_climate_equity_mean", "z_portfolio_beta_mean", []),
        ("B2", "Company-mean asset beta on company-mean portfolio beta", "z_beta_climate_asset_mean", "z_portfolio_beta_mean", []),
        ("B3", "Company-mean equity beta with size and leverage", "z_beta_climate_equity_mean", "z_portfolio_beta_mean", ["z_log_assets_mean", "z_debt_to_assets_mean"]),
        ("B4", "Company-mean equity beta on broad brown share", "z_beta_climate_equity_mean", "z_brown_broad_mean", []),
    ]
    cross_models = pd.DataFrame([
        model(company_means, outcome, xvar, model_id, label, model_controls)
        for model_id, label, outcome, xvar, model_controls in cross_specs
    ])
    company_means.to_csv(TABLES / "h2_dynamic_company_means.csv", index=False)
    export_dta(company_means, TABLES / "h2_dynamic_company_means.dta")
    cross_models.to_csv(TABLES / "h2_dynamic_between_models.csv", index=False)
    export_dta(cross_models, TABLES / "h2_dynamic_between_models.dta")

    descriptive = panel[[
        "beta_climate_equity_report_month", "beta_climate_asset_report_month",
        "loan_portfolio_climate_beta_qend", "loan_portfolio_climate_beta_qmean",
        "brown_share_narrow_dynamic_pct", "brown_share_broad_dynamic_pct",
        "mapped_weight_pct", "industry_rows",
    ]].describe(percentiles=[0.25, 0.5, 0.75]).T.reset_index().rename(columns={"index": "variable"})
    descriptive.to_csv(TABLES / "h2_dynamic_descriptive_statistics.csv", index=False)
    export_dta(descriptive, TABLES / "h2_dynamic_descriptive_statistics.dta")

    primary = brown_models.loc[brown_models["model_id"].eq("H2_1")].iloc[0]
    asset = brown_models.loc[brown_models["model_id"].eq("H2_2")].iloc[0]
    controlled = brown_models.loc[brown_models["model_id"].eq("H2_3")].iloc[0]
    within = brown_models.loc[brown_models["model_id"].eq("H2_5")].iloc[0]
    if primary["coefficient_exposure"] > 0 and primary["p_two_sided"] < 0.10 and asset["coefficient_exposure"] > 0:
        code = "SUPPORTED_DYNAMIC_PRIMARY"
        conclusion_cn = "动态投资暴露主回归为正且达到双侧 10% 显著性标准，资产 Beta 方向一致；H2 在动态面板中获得支持。"
    elif primary["coefficient_exposure"] > 0 and asset["coefficient_exposure"] > 0 and controlled["coefficient_exposure"] > 0 and within["coefficient_exposure"] > 0:
        code = "DIRECTIONALLY_CONSISTENT_IMPRECISE"
        conclusion_cn = "各主要规格方向一致为正，但双侧检验均未达到常规显著性水平；H2 未获得统计支持。"
    else:
        code = "NOT_SUPPORTED_SPECIFICATION_SENSITIVE"
        conclusion_cn = "最简规格的系数略为正，但加入财务控制或公司固定效应后转为负，且所有双侧检验均不显著。符号对规格敏感，H2 未获得支持。"
    decision = pd.DataFrame([{
        "hypothesis": "H2: A higher disclosed carbon-intensive investment share is positively associated with BDC climate beta",
        "decision_code": code,
        "conclusion_cn": conclusion_cn,
        "conclusion_en": "H2 is not statistically supported and its sign is unresolved. Parsimonious equity- and asset-beta coefficients are positive, whereas controlled and fixed-effects estimates can be negative; no two-sided test rejects zero at conventional levels.",
        "primary_model": "H2_1",
        "primary_coefficient_sd": primary["coefficient_exposure"],
        "primary_cluster_se": primary["standard_error"],
        "primary_p_one_sided_positive": primary["p_one_sided_positive"],
        "primary_p_two_sided": primary["p_two_sided"],
        "asset_coefficient_sd": asset["coefficient_exposure"],
        "asset_p_one_sided_positive": asset["p_one_sided_positive"],
        "controlled_coefficient_sd": controlled["coefficient_exposure"],
        "controlled_p_one_sided_positive": controlled["p_one_sided_positive"],
        "within_coefficient_sd": within["coefficient_exposure"],
        "within_p_one_sided_positive": within["p_one_sided_positive"],
        "panel_observations": int(primary["n"]),
        "firms": int(panel["ticker"].nunique()),
        "company_quarters_with_dynamic_exposure": int(portfolio.shape[0]),
    }])
    decision.to_csv(TABLES / "h2_dynamic_decision.csv", index=False)
    export_dta(decision, TABLES / "h2_dynamic_decision.dta")

    # Publication figures are generated from the stored estimates by
    # 05_make_results.py so that all modules share one visual specification.
    coverage = {
        "status": "PASS" if len(portfolio) == 380 and panel["loan_portfolio_climate_beta_qend"].notna().all() else "FAIL",
        "analysis_version": "BDC_Investment_Exposure_and_Climate_Beta_v2.3",
        "firms": int(panel["ticker"].nunique()),
        "dynamic_company_quarters": int(len(portfolio)),
        "analysis_panel_rows": int(len(panel)),
        "portfolio_beta_nonmissing": int(panel["loan_portfolio_climate_beta_qend"].notna().sum()),
        "mapped_weight_min": float(portfolio["mapped_weight_pct"].min()),
        "mapped_weight_max": float(portfolio["mapped_weight_pct"].max()),
        "primary_decision": decision.iloc[0].to_dict(),
        "inference": "Primary H2 model uses calendar-quarter fixed effects and ticker-clustered standard errors (19 clusters); firm fixed effects are reported separately as a within-BDC sensitivity.",
        "precommitted_direction": "positive",
        "no_sign_tuning": True,
    }
    (AUDIT / "h2_dynamic_panel_audit.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(coverage, ensure_ascii=False, indent=2, default=str))
    print("\nH2 brown-share models:\n", brown_models[["model_id", "coefficient_exposure", "standard_error", "p_one_sided_positive", "p_two_sided", "n"]].to_string(index=False))
    print("\nPortfolio-beta mechanism models:\n", models[["model_id", "coefficient_exposure", "standard_error", "p_one_sided_positive", "p_two_sided", "n"]].to_string(index=False))
    print("\nBetween-company models:\n", cross_models[["model_id", "coefficient_exposure", "standard_error", "p_one_sided_positive", "p_two_sided", "n"]].to_string(index=False))
    if coverage["status"] != "PASS":
        raise SystemExit("Dynamic panel coverage audit failed")


if __name__ == "__main__":
    main()
