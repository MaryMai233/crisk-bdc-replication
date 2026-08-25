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


DCC = load_module(MODULE1 / "Code" / "02_estimate_dcb_crisk.py", "top75_weekly_dcc")
FF = load_module(Path(__file__).with_name("04b_estimate_ff49_portfolio_mechanism.py"), "top75_weekly_ff")
BASE_DCC = load_module(Path(__file__).with_name("04c_estimate_ff49_dcc_portfolio_mechanism.py"), "top75_weekly_base")


def export_dta(frame: pd.DataFrame, path: Path) -> None:
    out = frame.copy()
    for column in out.select_dtypes(include="object"):
        out[column] = out[column].fillna("").astype(str)
    out.to_stata(path, write_index=False, version=118)


def weekly_frame(frame: pd.DataFrame, columns: list[str], min_days: int = 3) -> pd.DataFrame:
    weekly = frame.set_index("date")[columns].resample("W-FRI").agg(["sum", "count"])
    weekly.columns = ["_".join(item) for item in weekly.columns]
    keep = np.ones(len(weekly), dtype=bool)
    for column in columns:
        keep &= weekly[f"{column}_count"].ge(min_days).to_numpy()
    return weekly.loc[keep].copy()


def estimate_weekly_institutions(factor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    panel = pd.read_csv(
        MODULE1 / "Data" / "Processed" / "dcb_input_panel_2010_2025.csv",
        parse_dates=["date"],
        low_memory=False,
    )
    panel = panel.drop(columns=["logret_spy", "ret_climate", "coal_leg_source"]).merge(
        factor[["date", "logret_spy", "ret_climate", "coal_leg_source"]],
        on="date",
        how="left",
        validate="many_to_one",
    )
    panel["institution_log_return"] = np.log1p(panel["ret"].clip(lower=-0.999999))
    first: dict[int, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    groups = list(panel.dropna(subset=["institution_log_return", "logret_spy", "ret_climate"]).groupby("id"))
    for counter, (institution_id, group) in enumerate(groups, start=1):
        weekly = weekly_frame(
            group.sort_values("date"),
            ["institution_log_return", "logret_spy", "ret_climate"],
        )
        raw = weekly[
            ["institution_log_return_sum", "logret_spy_sum", "ret_climate_sum"]
        ].to_numpy(float)
        centered = raw - raw.mean(axis=0, keepdims=True)
        fits = [DCC.fit_gjr(centered[:, index]) for index in range(3)]
        h = np.column_stack([item.variance for item in fits])
        z = np.column_stack([item.standardized for item in fits])
        parameters, success, objective = DCC.fit_dcc(z)
        first[int(institution_id)] = {
            "weekly": weekly,
            "h": h,
            "z": z,
            "ticker": str(group["current_ticker"].iloc[0]),
        }
        rows.append(
            {
                "id": int(institution_id),
                "ticker": str(group["current_ticker"].iloc[0]),
                "weekly_observations": int(len(weekly)),
                "dcc_alpha_first_pass": float(parameters[0]),
                "dcc_beta_first_pass": float(parameters[1]),
                "dcc_success": bool(success),
                "dcc_objective": float(objective),
                "gjr_all_success": bool(all(item.success for item in fits)),
            }
        )
        print(f"Weekly institution [{counter:02d}/{len(groups):02d}] {group['current_ticker'].iloc[0]}: success={success}", flush=True)
    parameter_frame = pd.DataFrame(rows)
    median = parameter_frame.loc[
        parameter_frame["dcc_success"], ["dcc_alpha_first_pass", "dcc_beta_first_pass"]
    ].median().to_numpy(float)
    if median.sum() >= 0.999:
        raise RuntimeError("Weekly median DCC parameter vector is outside the maintained region")
    pieces: list[pd.DataFrame] = []
    for item in first.values():
        _, correlations = DCC.dcc_path(item["z"], median, initial_innovation="ones")
        beta_market, beta_climate = DCC.conditional_betas(item["h"], correlations)
        out = item["weekly"].reset_index()[["date"]].copy()
        out["ticker"] = item["ticker"]
        out["beta_market_top75_weekly"] = beta_market
        out["beta_climate_equity_top75_weekly"] = beta_climate
        pieces.append(out)
    beta = pd.concat(pieces, ignore_index=True).sort_values(["ticker", "date"])
    parameter_frame["dcc_alpha_median"] = median[0]
    parameter_frame["dcc_beta_median"] = median[1]
    parameter_frame["dcc_persistence_median"] = median.sum()
    return beta, parameter_frame, median


def bdc_quarter_outcomes(weekly: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.read_csv(
        PROCESSED / "bdc20_quarter_market_financial_panel_2021_2025_updated.csv",
        parse_dates=["datadate"],
    )
    weekly = weekly.loc[weekly["ticker"].isin(set(baseline["ticker"]))].copy()
    weekly["quarter"] = weekly["date"].dt.to_period("Q").astype(str)
    quarter = weekly.groupby(["ticker", "quarter"], as_index=False).agg(
        beta_climate_equity_top75_weekly_qmean=("beta_climate_equity_top75_weekly", "mean"),
        beta_market_top75_weekly_qmean=("beta_market_top75_weekly", "mean"),
        top75_weekly_observations=("date", "size"),
    )
    baseline["quarter"] = baseline["datadate"].dt.to_period("Q").astype(str)
    panel = baseline.merge(quarter, on=["ticker", "quarter"], how="left", validate="one_to_one")
    panel["beta_climate_asset_top75_weekly_qmean"] = (
        panel["beta_climate_equity_top75_weekly_qmean"]
        * panel["market_equity_report_month_mn"]
        / (panel["market_equity_report_month_mn"] + panel["debt_total_mn"])
    )
    return panel


def estimate_weekly_industries(factor: pd.DataFrame, median: np.ndarray) -> pd.DataFrame:
    daily = pd.read_csv(
        PROCESSED / "ff49_industry_climate_beta_daily_2020_2025.csv",
        parse_dates=["date"],
        usecols=[
            "date", "ff49", "ff49_abbreviation", "ff49_name", "industry_return",
            "market_cap_sum", "n_firms",
        ],
    ).merge(
        factor[["date", "logret_spy", "ret_climate"]],
        on="date",
        how="left",
        validate="many_to_one",
    )
    daily["industry_log_return"] = np.log1p(daily["industry_return"].clip(lower=-0.999999))
    first: dict[int, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for counter, (ff49, group) in enumerate(daily.groupby("ff49", sort=True), start=1):
        weekly = weekly_frame(
            group.sort_values("date"), ["industry_log_return", "logret_spy", "ret_climate"]
        )
        weekly_metadata = (
            group.sort_values("date")
            .set_index("date")[["market_cap_sum", "n_firms"]]
            .resample("W-FRI")
            .last()
            .reindex(weekly.index)
        )
        raw = weekly[["industry_log_return_sum", "logret_spy_sum", "ret_climate_sum"]].to_numpy(float)
        centered = raw - raw.mean(axis=0, keepdims=True)
        fits = [DCC.fit_gjr(centered[:, index]) for index in range(3)]
        h = np.column_stack([item.variance for item in fits])
        z = np.column_stack([item.standardized for item in fits])
        _, correlations = DCC.dcc_path(z, median, initial_innovation="ones")
        beta_market, beta_climate = DCC.conditional_betas(h, correlations)
        out = weekly.reset_index()[["date"]].copy()
        out["ff49"] = int(ff49)
        out["ff49_abbreviation"] = str(group["ff49_abbreviation"].iloc[0])
        out["ff49_name"] = str(group["ff49_name"].iloc[0])
        out["market_cap_sum"] = weekly_metadata["market_cap_sum"].to_numpy()
        out["n_firms"] = weekly_metadata["n_firms"].to_numpy()
        out["mbeta_ff49_dcc_top75_weekly"] = beta_market
        out["cbeta_ff49_dcc_top75_weekly"] = beta_climate
        rows.append(out)
        print(f"Weekly industry [{counter:02d}/49] FF49 {int(ff49):02d}", flush=True)
    weekly_beta = pd.concat(rows, ignore_index=True).sort_values(["ff49", "date"])
    weekly_beta["quarter"] = weekly_beta["date"].dt.to_period("Q").astype(str)
    qend = weekly_beta.groupby(["ff49", "quarter"], as_index=False).tail(1).rename(
        columns={
            "date": "quarter_end_observation_date",
            "cbeta_ff49_dcc_top75_weekly": "cbeta_ff49_dcc_top75_weekly_qend",
            "mbeta_ff49_dcc_top75_weekly": "mbeta_ff49_dcc_top75_weekly_qend",
        }
    )
    qmean = weekly_beta.groupby(["ff49", "quarter"], as_index=False).agg(
        cbeta_ff49_dcc_top75_weekly_qmean=("cbeta_ff49_dcc_top75_weekly", "mean"),
        mbeta_ff49_dcc_top75_weekly_qmean=("mbeta_ff49_dcc_top75_weekly", "mean"),
        weekly_observations=("date", "size"),
    )
    return qend[
        [
            "ff49", "ff49_abbreviation", "ff49_name", "quarter",
            "quarter_end_observation_date", "cbeta_ff49_dcc_top75_weekly_qend",
            "mbeta_ff49_dcc_top75_weekly_qend", "market_cap_sum", "n_firms",
        ]
    ].merge(qmean, on=["ff49", "quarter"], how="outer", validate="one_to_one")


def estimate_models(outcomes: pd.DataFrame, industries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exposure = pd.read_csv(PROCESSED / "dynamic_bdc_industry_exposure_2021_2025.csv")
    exposure, _ = FF.replace_geography_only_quarters(exposure)
    ff49 = industries.rename(
        columns={
            "cbeta_ff49_dcc_top75_weekly_qend": "cbeta_ff49_qend",
            "cbeta_ff49_dcc_top75_weekly_qmean": "cbeta_ff49_qmean",
            "mbeta_ff49_dcc_top75_weekly_qend": "mbeta_ff49_qend",
            "mbeta_ff49_dcc_top75_weekly_qmean": "mbeta_ff49_qmean",
        }
    )
    _, portfolio = FF.build_portfolio_beta(exposure, ff49)
    portfolio = portfolio.rename(
        columns={
            "ff49_portfolio_climate_beta_qend": "portfolio_beta_top75_weekly_qend",
            "ff49_portfolio_climate_beta_qmean": "portfolio_beta_top75_weekly_qmean",
        }
    )
    panel = outcomes.merge(portfolio, on=["ticker", "quarter"], how="inner", validate="one_to_one")
    panel = FF.add_standardized(
        panel,
        [
            "beta_climate_equity_top75_weekly_qmean",
            "beta_climate_asset_top75_weekly_qmean",
            "portfolio_beta_top75_weekly_qend",
            "portfolio_beta_top75_weekly_qmean",
            "log_assets", "debt_to_assets", "roa_quarter", "book_to_market",
            "beta_market_top75_weekly_qmean",
        ],
    )
    controls = [
        "z_log_assets", "z_debt_to_assets", "z_roa_quarter", "z_book_to_market",
        "z_beta_market_top75_weekly_qmean",
    ]
    specs = [
        ("T75W_1", "Asset beta, quarter FE", "z_beta_climate_asset_top75_weekly_qmean", [], ["quarter"]),
        ("T75W_2", "Asset beta plus controls", "z_beta_climate_asset_top75_weekly_qmean", controls, ["quarter"]),
        ("T75W_3", "Asset beta, firm and quarter FE", "z_beta_climate_asset_top75_weekly_qmean", [], ["ticker", "quarter"]),
        ("T75W_4", "Equity beta, quarter FE", "z_beta_climate_equity_top75_weekly_qmean", [], ["quarter"]),
        ("T75W_5", "Equity beta plus controls", "z_beta_climate_equity_top75_weekly_qmean", controls, ["quarter"]),
        ("T75W_6", "Equity beta, firm and quarter FE", "z_beta_climate_equity_top75_weekly_qmean", [], ["ticker", "quarter"]),
    ]
    models = pd.DataFrame(
        [
            FF.PRIOR.fit_ols(
                panel, outcome, "z_portfolio_beta_top75_weekly_qend", model_id, label,
                controls=model_controls, fixed_effects=fixed_effects, cluster="ticker",
            )
            for model_id, label, outcome, model_controls, fixed_effects in specs
        ]
    )
    wild = pd.DataFrame(
        [
            BASE_DCC.wild_cluster_two_way_fe(
                panel,
                "z_beta_climate_equity_top75_weekly_qmean",
                "z_portfolio_beta_top75_weekly_qend",
            ),
            BASE_DCC.wild_cluster_two_way_fe(
                panel,
                "z_beta_climate_asset_top75_weekly_qmean",
                "z_portfolio_beta_top75_weekly_qend",
            ),
        ]
    )
    return panel, models, wild


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    factor = pd.read_csv(
        MODULE1 / "Data" / "Processed" / "climate_factor_daily_kol_top75_2010_2025.csv",
        parse_dates=["date"],
    )
    weekly_path = PROCESSED / "institution_dcc_beta_weekly_kol_top75_2010_2025.csv"
    parameter_path = PROCESSED / "institution_dcc_parameters_weekly_kol_top75.csv"
    if weekly_path.exists() and parameter_path.exists():
        weekly = pd.read_csv(weekly_path, parse_dates=["date"])
        parameters = pd.read_csv(parameter_path)
        median = parameters[["dcc_alpha_median", "dcc_beta_median"]].iloc[0].to_numpy(float)
    else:
        weekly, parameters, median = estimate_weekly_institutions(factor)
        weekly.to_csv(weekly_path, index=False)
        parameters.to_csv(parameter_path, index=False)
    outcomes = bdc_quarter_outcomes(weekly)
    industry_path = PROCESSED / "ff49_industry_dcc_beta_quarterly_kol_top75_weekly_2020_2025.csv"
    if industry_path.exists() and "market_cap_sum" in pd.read_csv(industry_path, nrows=1).columns:
        industries = pd.read_csv(industry_path, parse_dates=["quarter_end_observation_date"])
    else:
        industries = estimate_weekly_industries(factor, median)
        industries.to_csv(industry_path, index=False)
    panel, models, wild = estimate_models(outcomes, industries)
    panel.to_csv(PROCESSED / "bdc19_ff49_dcc_kol_top75_weekly_panel_2021_2025.csv", index=False)
    export_dta(panel, PROCESSED / "bdc19_ff49_dcc_kol_top75_weekly_panel_2021_2025.dta")
    models.to_csv(PROCESSED / "h2_ff49_dcc_kol_top75_weekly_models.csv", index=False)
    export_dta(models, PROCESSED / "h2_ff49_dcc_kol_top75_weekly_models.dta")
    wild.to_csv(PROCESSED / "h2_ff49_dcc_kol_top75_weekly_wild_cluster.csv", index=False)
    export_dta(wild, PROCESSED / "h2_ff49_dcc_kol_top75_weekly_wild_cluster.dta")
    audit = {
        "status": "PASS",
        "median_dcc_alpha": float(median[0]),
        "median_dcc_beta": float(median[1]),
        "median_dcc_persistence": float(median.sum()),
        "institutions": int(parameters["ticker"].nunique()),
        "industry_count": int(industries["ff49"].nunique()),
        "bdc_firms": int(panel["ticker"].nunique()),
        "bdc_quarters": int(len(panel)),
        "frequency_role": "Same 15-name basket as daily robustness; weekly aggregation isolates non-synchronous trading",
    }
    (AUDIT / "h2_kol_top75_weekly_dcc_robustness_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print("\nTop-75 weekly DCC models")
    print(models[["model_id", "coefficient_exposure", "standard_error", "p_two_sided", "n", "r_squared"]].to_string(index=False))
    print("\nWild-cluster inference")
    print(wild.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
