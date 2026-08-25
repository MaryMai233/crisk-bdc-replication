from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
MODULE1 = PACKAGE / "01_Bank_CRISK_Replication"
PROCESSED = ROOT / "Data" / "Processed"
AUDIT = PROCESSED / "Audit"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DCC = load_module(MODULE1 / "Code" / "02_estimate_dcb_crisk.py", "top75_dcc")
FF = load_module(Path(__file__).with_name("04b_estimate_ff49_portfolio_mechanism.py"), "top75_ff")
BASE_DCC = load_module(Path(__file__).with_name("04c_estimate_ff49_dcc_portfolio_mechanism.py"), "top75_base_dcc")


def export_dta(frame: pd.DataFrame, path: Path) -> None:
    out = frame.copy()
    for column in out.select_dtypes(include="object"):
        out[column] = out[column].fillna("").astype(str)
    out.to_stata(path, write_index=False, version=118)


def estimate_institution_betas(factor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    input_panel = pd.read_csv(
        MODULE1 / "Data" / "Processed" / "dcb_input_panel_2010_2025.csv",
        parse_dates=["date", "datadate", "available_date"],
        low_memory=False,
    )
    panel = input_panel.drop(columns=["logret_spy", "ret_climate", "coal_leg_source"]).merge(
        factor[["date", "logret_spy", "ret_climate", "coal_leg_source"]],
        on="date",
        how="left",
        validate="many_to_one",
    )
    complete = panel.dropna(subset=["ret", "logret_spy", "ret_climate"]).copy()
    first_pass: dict[int, dict[str, object]] = {}
    parameter_rows: list[dict[str, object]] = []
    groups = list(complete.groupby("id", sort=True))
    for counter, (institution_id, group) in enumerate(groups, start=1):
        group = group.sort_values("date").copy()
        raw = group[["ret", "logret_spy", "ret_climate"]].to_numpy(float)
        centered = raw - raw.mean(axis=0, keepdims=True)
        fits = [DCC.fit_gjr(centered[:, index]) for index in range(3)]
        h = np.column_stack([item.variance for item in fits])
        z = np.column_stack([item.standardized for item in fits])
        parameters, success, objective = DCC.fit_dcc(z)
        first_pass[int(institution_id)] = {"group": group, "h": h, "z": z}
        parameter_rows.append(
            {
                "id": int(institution_id),
                "memo": str(group["memo"].iloc[0]),
                "ticker": str(group["current_ticker"].iloc[0]),
                "observations": int(len(group)),
                "dcc_alpha_first_pass": float(parameters[0]),
                "dcc_beta_first_pass": float(parameters[1]),
                "dcc_success": bool(success),
                "dcc_objective": float(objective),
                "gjr_all_success": bool(all(item.success for item in fits)),
            }
        )
        print(
            f"Institution [{counter:02d}/{len(groups):02d}] {group['current_ticker'].iloc[0]}: "
            f"T={len(group)}, DCC=({parameters[0]:.4f},{parameters[1]:.4f}), success={success}",
            flush=True,
        )
    parameters = pd.DataFrame(parameter_rows)
    median = parameters.loc[
        parameters["dcc_success"], ["dcc_alpha_first_pass", "dcc_beta_first_pass"]
    ].median().to_numpy(float)
    if median.sum() >= 0.999:
        raise RuntimeError("Top-75 median DCC parameter vector is outside the maintained region")
    pieces: list[pd.DataFrame] = []
    for institution_id, item in first_pass.items():
        _, correlations = DCC.dcc_path(item["z"], median, initial_innovation="ones")
        beta_market, beta_climate = DCC.conditional_betas(item["h"], correlations)
        out = item["group"].copy()
        out["beta_market_top75"] = beta_market
        out["beta_climate_top75"] = beta_climate
        out["dcc_alpha_top75_median"] = median[0]
        out["dcc_beta_top75_median"] = median[1]
        pieces.append(out)
    daily = pd.concat(pieces, ignore_index=True).sort_values(["id", "date"])
    parameters["dcc_alpha_median"] = median[0]
    parameters["dcc_beta_median"] = median[1]
    parameters["dcc_persistence_median"] = median.sum()
    return daily, parameters, median


def build_bdc_outcome_panel(daily: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.read_csv(
        PROCESSED / "bdc20_quarter_market_financial_panel_2021_2025_updated.csv",
        parse_dates=["datadate", "report_month_start", "report_month_end"],
    )
    tickers = set(baseline["ticker"])
    bdc = daily.loc[daily["current_ticker"].isin(tickers)].copy()
    bdc["report_month_start"] = bdc["date"].dt.to_period("M").dt.to_timestamp()
    monthly = bdc.groupby(["current_ticker", "report_month_start"], as_index=False).agg(
        beta_climate_equity_top75_report_month=("beta_climate_top75", "mean"),
        beta_market_top75_report_month=("beta_market_top75", "mean"),
        top75_beta_daily_observations=("date", "nunique"),
    ).rename(columns={"current_ticker": "ticker"})
    panel = baseline.merge(
        monthly, on=["ticker", "report_month_start"], how="left", validate="one_to_one"
    )
    panel["beta_climate_asset_top75_report_month"] = (
        panel["beta_climate_equity_top75_report_month"]
        * panel["market_equity_report_month_mn"]
        / (panel["market_equity_report_month_mn"] + panel["debt_total_mn"])
    )
    if panel["beta_climate_equity_top75_report_month"].isna().any():
        missing = panel.loc[
            panel["beta_climate_equity_top75_report_month"].isna(), ["ticker", "datadate"]
        ]
        raise ValueError(f"Missing top-75 BDC beta months:\n{missing.to_string(index=False)}")
    return panel


def estimate_industry_betas(factor: pd.DataFrame, median_dcc: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    industries = pd.read_csv(
        PROCESSED / "ff49_industry_climate_beta_daily_2020_2025.csv",
        parse_dates=["date"],
        usecols=[
            "date", "ff49", "ff49_abbreviation", "ff49_name", "industry_return",
            "market_cap_sum", "n_firms",
        ],
    )
    frame = industries.merge(
        factor[["date", "logret_spy", "ret_climate"]],
        on="date",
        how="left",
        validate="many_to_one",
    )
    frame["industry_log_return"] = np.log1p(frame["industry_return"].clip(lower=-0.999999))
    frame = frame.dropna(subset=["industry_log_return", "logret_spy", "ret_climate"]).copy()
    common = frame[["date", "logret_spy", "ret_climate"]].drop_duplicates("date").sort_values("date")
    common_raw = common[["logret_spy", "ret_climate"]].to_numpy(float)
    common_centered = common_raw - common_raw.mean(axis=0, keepdims=True)
    market_fit = DCC.fit_gjr(common_centered[:, 0])
    climate_fit = DCC.fit_gjr(common_centered[:, 1])
    pieces: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    for counter, (ff49, group) in enumerate(frame.groupby("ff49", sort=True), start=1):
        group = group.sort_values("date").merge(
            common[["date"]], on="date", how="right", validate="one_to_one"
        ).sort_values("date")
        y = group["industry_log_return"].to_numpy(float)
        if np.isnan(y).any():
            raise ValueError(f"FF49 industry {ff49} has missing common-date returns")
        fit = DCC.fit_gjr(y - y.mean())
        h = np.column_stack([fit.variance, market_fit.variance, climate_fit.variance])
        z = np.column_stack([fit.standardized, market_fit.standardized, climate_fit.standardized])
        _, correlations = DCC.dcc_path(z, median_dcc, initial_innovation="ones")
        beta_market, beta_climate = DCC.conditional_betas(h, correlations)
        group["mbeta_ff49_dcc_top75"] = beta_market
        group["cbeta_ff49_dcc_top75"] = beta_climate
        pieces.append(group)
        parameter_rows.append(
            {
                "ff49": int(ff49),
                "ff49_name": str(group["ff49_name"].dropna().iloc[0]),
                "observations": int(len(group)),
                "industry_gjr_alpha": float(fit.params[0]),
                "industry_gjr_gamma": float(fit.params[1]),
                "industry_gjr_beta": float(fit.params[2]),
                "industry_gjr_success": bool(fit.success),
            }
        )
        print(f"Industry [{counter:02d}/49] FF49 {int(ff49):02d}: success={fit.success}", flush=True)
    daily = pd.concat(pieces, ignore_index=True).sort_values(["date", "ff49"])
    daily["quarter"] = daily["date"].dt.to_period("Q").astype(str)
    qend = daily.groupby(["ff49", "quarter"], as_index=False).tail(1).rename(
        columns={
            "date": "quarter_end_observation_date",
            "cbeta_ff49_dcc_top75": "cbeta_ff49_dcc_top75_qend",
            "mbeta_ff49_dcc_top75": "mbeta_ff49_dcc_top75_qend",
        }
    )
    qmean = daily.groupby(["ff49", "quarter"], as_index=False).agg(
        cbeta_ff49_dcc_top75_qmean=("cbeta_ff49_dcc_top75", "mean"),
        mbeta_ff49_dcc_top75_qmean=("mbeta_ff49_dcc_top75", "mean"),
        trading_days=("date", "nunique"),
    )
    quarterly = qend[
        [
            "ff49", "ff49_abbreviation", "ff49_name", "quarter",
            "quarter_end_observation_date", "cbeta_ff49_dcc_top75_qend",
            "mbeta_ff49_dcc_top75_qend", "market_cap_sum", "n_firms",
        ]
    ].merge(qmean, on=["ff49", "quarter"], how="outer", validate="one_to_one")
    return daily, quarterly


def estimate_models(outcomes: pd.DataFrame, industries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exposure = pd.read_csv(PROCESSED / "dynamic_bdc_industry_exposure_2021_2025.csv")
    exposure, _ = FF.replace_geography_only_quarters(exposure)
    ff49 = industries.rename(
        columns={
            "cbeta_ff49_dcc_top75_qend": "cbeta_ff49_qend",
            "cbeta_ff49_dcc_top75_qmean": "cbeta_ff49_qmean",
            "mbeta_ff49_dcc_top75_qend": "mbeta_ff49_qend",
            "mbeta_ff49_dcc_top75_qmean": "mbeta_ff49_qmean",
        }
    )
    mapped, portfolio = FF.build_portfolio_beta(exposure, ff49)
    portfolio = portfolio.rename(
        columns={
            "ff49_portfolio_climate_beta_qend": "ff49_dcc_top75_portfolio_climate_beta_qend",
            "ff49_portfolio_climate_beta_qmean": "ff49_dcc_top75_portfolio_climate_beta_qmean",
        }
    )
    panel = outcomes.copy()
    panel["quarter"] = panel["datadate"].dt.to_period("Q").astype(str)
    panel = panel.merge(portfolio, on=["ticker", "quarter"], how="inner", validate="one_to_one")
    panel = panel.sort_values(["ticker", "quarter"]).reset_index(drop=True)
    panel["ff49_dcc_top75_portfolio_climate_beta_qend_lag1"] = panel.groupby("ticker")[
        "ff49_dcc_top75_portfolio_climate_beta_qend"
    ].shift(1)
    zcols = [
        "beta_climate_equity_top75_report_month",
        "beta_climate_asset_top75_report_month",
        "ff49_dcc_top75_portfolio_climate_beta_qend",
        "ff49_dcc_top75_portfolio_climate_beta_qmean",
        "ff49_dcc_top75_portfolio_climate_beta_qend_lag1",
        "log_assets", "debt_to_assets", "roa_quarter", "book_to_market",
        "beta_market_top75_report_month",
    ]
    panel = FF.add_standardized(panel, zcols)
    controls = [
        "z_log_assets", "z_debt_to_assets", "z_roa_quarter", "z_book_to_market",
        "z_beta_market_top75_report_month",
    ]
    specs = [
        ("T75_1", "Asset beta, quarter FE", "z_beta_climate_asset_top75_report_month", [], ["quarter"]),
        ("T75_2", "Asset beta plus controls", "z_beta_climate_asset_top75_report_month", controls, ["quarter"]),
        ("T75_3", "Asset beta, firm and quarter FE", "z_beta_climate_asset_top75_report_month", [], ["ticker", "quarter"]),
        ("T75_4", "Equity beta, quarter FE", "z_beta_climate_equity_top75_report_month", [], ["quarter"]),
        ("T75_5", "Equity beta plus controls", "z_beta_climate_equity_top75_report_month", controls, ["quarter"]),
        ("T75_6", "Equity beta, firm and quarter FE", "z_beta_climate_equity_top75_report_month", [], ["ticker", "quarter"]),
    ]
    models = pd.DataFrame(
        [
            FF.PRIOR.fit_ols(
                panel,
                outcome,
                "z_ff49_dcc_top75_portfolio_climate_beta_qend",
                model_id,
                label,
                controls=model_controls,
                fixed_effects=fixed_effects,
                cluster="ticker",
            )
            for model_id, label, outcome, model_controls, fixed_effects in specs
        ]
    )
    wild = pd.DataFrame(
        [
            BASE_DCC.wild_cluster_two_way_fe(
                panel,
                "z_beta_climate_equity_top75_report_month",
                "z_ff49_dcc_top75_portfolio_climate_beta_qend",
            ),
            BASE_DCC.wild_cluster_two_way_fe(
                panel,
                "z_beta_climate_asset_top75_report_month",
                "z_ff49_dcc_top75_portfolio_climate_beta_qend",
            ),
        ]
    )
    return mapped, panel, models, wild


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    factor = pd.read_csv(
        MODULE1 / "Data" / "Processed" / "climate_factor_daily_kol_top75_2010_2025.csv",
        parse_dates=["date"],
    )
    daily, parameters, median_dcc = estimate_institution_betas(factor)
    daily.to_csv(PROCESSED / "institution_dcc_beta_daily_kol_top75_2010_2025.csv", index=False)
    parameters.to_csv(PROCESSED / "institution_dcc_parameters_kol_top75.csv", index=False)
    outcomes = build_bdc_outcome_panel(daily)
    outcomes.to_csv(PROCESSED / "bdc20_market_financial_panel_kol_top75_2021_2025.csv", index=False)
    export_dta(outcomes, PROCESSED / "bdc20_market_financial_panel_kol_top75_2021_2025.dta")

    industry_daily, industry_quarterly = estimate_industry_betas(factor, median_dcc)
    industry_daily.to_csv(PROCESSED / "ff49_industry_dcc_beta_daily_kol_top75_2020_2025.csv", index=False)
    industry_quarterly.to_csv(PROCESSED / "ff49_industry_dcc_beta_quarterly_kol_top75_2020_2025.csv", index=False)
    export_dta(industry_quarterly, PROCESSED / "ff49_industry_dcc_beta_quarterly_kol_top75_2020_2025.dta")

    mapped, panel, models, wild = estimate_models(outcomes, industry_quarterly)
    mapped.to_csv(PROCESSED / "dynamic_bdc_industry_ff49_dcc_kol_top75_mapping_2021_2025.csv", index=False)
    panel.to_csv(PROCESSED / "bdc19_ff49_dcc_kol_top75_mechanism_panel_2021_2025.csv", index=False)
    export_dta(panel, PROCESSED / "bdc19_ff49_dcc_kol_top75_mechanism_panel_2021_2025.dta")
    models.to_csv(PROCESSED / "h2_ff49_dcc_kol_top75_models.csv", index=False)
    export_dta(models, PROCESSED / "h2_ff49_dcc_kol_top75_models.dta")
    wild.to_csv(PROCESSED / "h2_ff49_dcc_kol_top75_wild_cluster.csv", index=False)
    export_dta(wild, PROCESSED / "h2_ff49_dcc_kol_top75_wild_cluster.dta")

    tracking = pd.read_csv(
        MODULE1 / "Data" / "Processed" / "kol_top75_tracking_diagnostics.csv"
    )
    baseline = pd.read_csv(PROCESSED / "h2_ff49_dcc_portfolio_mechanism_models.csv")
    comparison = pd.concat(
        [
            baseline.assign(continuation="Published top-five").query("model_id in ['DCC49_1','DCC49_2','DCC49_3','DCC49_4','DCC49_5','DCC49_6']"),
            models.assign(continuation="SEC N-PORT cumulative 75 percent"),
        ],
        ignore_index=True,
        sort=False,
    )
    comparison.to_csv(PROCESSED / "h2_top5_vs_top75_dcc_model_comparison.csv", index=False)
    export_dta(comparison, PROCESSED / "h2_top5_vs_top75_dcc_model_comparison.dta")
    audit = {
        "status": "PASS",
        "tracking_daily": float(tracking.loc[tracking["frequency"].eq("Daily"), "correlation_with_kol"].iloc[0]),
        "tracking_weekly": float(tracking.loc[tracking["frequency"].eq("Weekly"), "correlation_with_kol"].iloc[0]),
        "median_dcc_alpha": float(median_dcc[0]),
        "median_dcc_beta": float(median_dcc[1]),
        "median_dcc_persistence": float(median_dcc.sum()),
        "institutions": int(parameters["ticker"].nunique()),
        "industry_count": int(industry_daily["ff49"].nunique()),
        "bdc_firms": int(panel["ticker"].nunique()),
        "bdc_quarters": int(len(panel)),
        "selection_rule": "Basket selected from SEC holdings before observing BDC regression outcomes",
        "interpretation": (
            "The top-five continuation remains the strict replication; the top-75 result tests whether "
            "a broader pre-specified basket reduces continuation measurement error."
        ),
    }
    (AUDIT / "h2_kol_top75_dcc_robustness_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print("\nTop-75 DCC models")
    print(
        models[
            ["model_id", "coefficient_exposure", "standard_error", "p_two_sided", "n", "r_squared"]
        ].to_string(index=False)
    )
    print("\nWild-cluster inference")
    print(wild.to_string(index=False))
    print("\nAudit")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
