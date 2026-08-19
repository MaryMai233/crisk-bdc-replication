from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
RAW = ROOT / "Data" / "Raw"
PROCESSED = ROOT / "Data" / "Processed"
AUDIT = PROCESSED / "Audit"
PUBLIC_MARKET_SERIES = (
    PACKAGE / "01_Bank_CRISK_Replication" / "Data" / "Raw"
    / "public_market_series_yahoo_2010_2025.csv"
)


def load_helper():
    path = Path(__file__).with_name("01_industry_beta_helpers.py")
    spec = importlib.util.spec_from_file_location("h2_helper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HELPER = load_helper()


@dataclass
class GjrFit:
    params: np.ndarray
    variance: np.ndarray
    standardized: np.ndarray
    success: bool


def gjr_variance(values: np.ndarray, params: np.ndarray) -> np.ndarray:
    alpha, gamma, beta = params
    target = np.var(values, ddof=1)
    variance = np.empty_like(values, dtype=float)
    variance[0] = np.var(values, ddof=0)
    intercept = (1.0 - alpha - 0.5 * gamma - beta) * target
    for index in range(1, len(values)):
        lag_squared = values[index - 1] ** 2
        variance[index] = (
            intercept
            + alpha * lag_squared
            + gamma * lag_squared * (values[index - 1] < 0)
            + beta * variance[index - 1]
        )
    return variance


def fit_gjr(values: np.ndarray) -> GjrFit:
    def objective(params: np.ndarray) -> float:
        variance = gjr_variance(values, params)
        if np.any(variance <= 1e-14) or not np.all(np.isfinite(variance)):
            return 1e12
        return float(
            0.5 * np.sum(np.log(variance) + values * values / variance + np.log(2 * np.pi))
        )

    result = minimize(
        objective,
        np.array([0.05, 0.05, 0.75]),
        method="SLSQP",
        bounds=[(1e-8, 0.999)] * 3,
        constraints=({"type": "ineq", "fun": lambda p: 0.999 - p[0] - 0.5 * p[1] - p[2]},),
        options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
    )
    variance = gjr_variance(values, result.x)
    return GjrFit(result.x, variance, values / np.sqrt(variance), bool(result.success))


def dcc_path(standardized: np.ndarray, params: np.ndarray) -> np.ndarray:
    alpha, beta = params
    unconditional = np.cov(standardized, rowvar=False, ddof=1)
    dimensions = standardized.shape[1]
    correlations = np.empty((len(standardized), dimensions, dimensions), dtype=float)
    previous_q = unconditional.copy()
    previous_z = np.ones(dimensions)
    for index in range(len(standardized)):
        q = (
            (1.0 - alpha - beta) * unconditional
            + alpha * np.outer(previous_z, previous_z)
            + beta * previous_q
        )
        scale = np.sqrt(np.clip(np.diag(q), 1e-14, None))
        correlations[index] = q / np.outer(scale, scale)
        previous_q = q
        previous_z = standardized[index]
    return correlations


def fit_dcc(standardized: np.ndarray) -> tuple[np.ndarray, bool]:
    def objective(params: np.ndarray) -> float:
        correlations = dcc_path(standardized, params)
        total = 0.0
        for index, correlation in enumerate(correlations):
            sign, logdet = np.linalg.slogdet(correlation)
            if sign <= 0 or not np.isfinite(logdet):
                return 1e12
            try:
                quadratic = float(
                    standardized[index] @ np.linalg.solve(correlation, standardized[index])
                )
            except np.linalg.LinAlgError:
                return 1e12
            total += 0.5 * (logdet + quadratic)
        return float(total)

    result = minimize(
        objective,
        np.array([0.02, 0.95]),
        method="SLSQP",
        bounds=[(1e-8, 0.999), (1e-8, 0.999)],
        constraints=({"type": "ineq", "fun": lambda p: 0.999 - p[0] - p[1]},),
        options={"ftol": 1e-8, "maxiter": 500, "disp": False},
    )
    return result.x, bool(result.success)


def load_credit_return_factor() -> pd.DataFrame:
    public = pd.read_csv(PUBLIC_MARKET_SERIES, parse_dates=["date"])
    prices = public.pivot(index="date", columns="symbol", values="adjusted_close").sort_index()
    returns = np.log(prices[["HYG", "JNK", "SHY"]]).diff()
    factor = pd.DataFrame(index=returns.index)
    factor["hyg_excess_logret"] = returns["HYG"] - returns["SHY"]
    factor["jnk_excess_logret"] = returns["JNK"] - returns["SHY"]
    factor["hy_credit_excess_logret"] = (
        0.5 * returns["HYG"] + 0.5 * returns["JNK"] - returns["SHY"]
    )
    return factor.reset_index().dropna(subset=["hy_credit_excess_logret"])


def estimate_dynamic_beta(daily: pd.DataFrame, factor_columns: list[str], label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_pass: dict[str, dict] = {}
    parameters = []
    for ticker, group in daily.groupby("current_ticker"):
        group = group.dropna(subset=["ret", *factor_columns]).sort_values("date").copy()
        raw = group[["ret", *factor_columns]].to_numpy(float)
        centered = raw - raw.mean(axis=0, keepdims=True)
        fits = [fit_gjr(centered[:, index]) for index in range(centered.shape[1])]
        variances = np.column_stack([fit.variance for fit in fits])
        standardized = np.column_stack([fit.standardized for fit in fits])
        dcc_params, success = fit_dcc(standardized)
        first_pass[ticker] = {
            "group": group,
            "variance": variances,
            "standardized": standardized,
        }
        parameters.append(
            {
                "model": label,
                "ticker": ticker,
                "observations": len(group),
                "dcc_alpha": dcc_params[0],
                "dcc_beta": dcc_params[1],
                "dcc_success": int(success),
                "gjr_all_success": int(all(fit.success for fit in fits)),
            }
        )
        print(f"{label}: {ticker}, T={len(group)}, DCC=({dcc_params[0]:.4f},{dcc_params[1]:.4f})")
    parameter_frame = pd.DataFrame(parameters)
    median_dcc = parameter_frame.loc[parameter_frame["dcc_success"].eq(1), ["dcc_alpha", "dcc_beta"]].median().to_numpy()
    if median_dcc.sum() >= 0.999:
        median_dcc = np.array([0.02, 0.95])
    outputs = []
    for ticker, item in first_pass.items():
        correlations = dcc_path(item["standardized"], median_dcc)
        volatility = np.sqrt(item["variance"])
        conditional_covariance = correlations * volatility[:, :, None] * volatility[:, None, :]
        factor_covariance = conditional_covariance[:, 1:, 1:]
        factor_institution_covariance = conditional_covariance[:, 1:, 0]
        betas = np.array(
            [
                np.linalg.solve(factor_covariance[index], factor_institution_covariance[index])
                for index in range(len(factor_covariance))
            ]
        )
        out = item["group"][["current_ticker", "date"]].copy()
        out[f"beta_market_{label}"] = betas[:, 0]
        out[f"beta_climate_{label}"] = betas[:, 1]
        if len(factor_columns) == 3:
            out[f"beta_credit_{label}"] = betas[:, 2]
        outputs.append(out)
    parameter_frame["median_dcc_alpha_used"] = median_dcc[0]
    parameter_frame["median_dcc_beta_used"] = median_dcc[1]
    return pd.concat(outputs, ignore_index=True), parameter_frame


def aggregate_to_report_month(panel: pd.DataFrame, daily_beta: pd.DataFrame, beta_column: str) -> pd.Series:
    values = []
    for row in panel.itertuples(index=False):
        sample = daily_beta[
            daily_beta["current_ticker"].eq(row.ticker)
            & daily_beta["date"].between(row.report_month_start, row.report_month_end)
        ]
        values.append(sample[beta_column].mean() if len(sample) >= 5 else np.nan)
    return pd.Series(values, index=panel.index)


def standardize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        deviation = frame[column].std(ddof=0)
        frame[f"z_{column}"] = (frame[column] - frame[column].mean()) / deviation
    return frame


def model(frame: pd.DataFrame, outcome: str, model_id: str, label: str, controls=None, fixed_effects=None) -> dict:
    return HELPER.fit_ols(
        frame,
        outcome,
        "z_brown_share_broad_dynamic_pct",
        model_id,
        label,
        controls=controls or [],
        fixed_effects=fixed_effects or ["quarter"],
        cluster="ticker",
    )


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    daily_path = RAW / "dcb_crisk_daily_2010_2025.csv"
    if not daily_path.exists():
        daily_path = (
            PACKAGE / "01_Bank_CRISK_Replication" / "Data" / "Processed"
            / "dcb_crisk_daily_2010_2025.csv"
        )
    daily = pd.read_csv(daily_path, parse_dates=["date"], low_memory=False)
    panel = pd.read_csv(
        PROCESSED / "bdc19_dynamic_portfolio_h2_panel_2021_2025.csv",
        parse_dates=["datadate", "report_month_start", "report_month_end"],
    )
    daily = daily[
        daily["group"].eq("BDC") & daily["current_ticker"].isin(panel["ticker"])
    ].copy()
    credit = load_credit_return_factor()
    daily = daily.drop(columns=["beta_market", "beta_climate"], errors="ignore").merge(
        credit, on="date", how="left", validate="many_to_one"
    )
    daily = daily.dropna(
        subset=["ret", "logret_spy", "ret_climate", "hy_credit_excess_logret"]
    )

    two_factor, parameters_two = estimate_dynamic_beta(
        daily, ["logret_spy", "ret_climate"], "full2f"
    )
    credit_adjusted, parameters_credit = estimate_dynamic_beta(
        daily, ["logret_spy", "ret_climate", "hy_credit_excess_logret"], "credit3f"
    )
    beta_daily = two_factor.merge(
        credit_adjusted, on=["current_ticker", "date"], how="inner", validate="one_to_one"
    )
    short_panel = panel.copy()
    short_panel["beta_climate_full2f"] = aggregate_to_report_month(
        short_panel, beta_daily, "beta_climate_full2f"
    )
    short_panel["beta_climate_credit3f"] = aggregate_to_report_month(
        short_panel, beta_daily, "beta_climate_credit3f"
    )
    leverage_scale = short_panel["market_equity_report_month_mn"] / (
        short_panel["market_equity_report_month_mn"] + short_panel["debt_total_mn"]
    )
    short_panel["beta_asset_full2f"] = short_panel["beta_climate_full2f"] * leverage_scale
    short_panel["beta_asset_credit3f"] = short_panel["beta_climate_credit3f"] * leverage_scale
    short_panel = short_panel.dropna(
        subset=[
            "beta_climate_equity_report_month", "beta_climate_asset_report_month",
            "beta_climate_full2f", "beta_climate_credit3f", "brown_share_broad_dynamic_pct",
        ]
    ).copy()
    columns = [
        "beta_climate_equity_report_month", "beta_climate_asset_report_month",
        "brown_share_broad_dynamic_pct", "beta_climate_full2f", "beta_climate_credit3f",
        "beta_asset_full2f", "beta_asset_credit3f", "log_assets", "debt_to_assets",
        "roa_quarter", "book_to_market", "beta_market_report_month",
    ]
    short_panel = standardize(short_panel, columns)
    controls = [
        "z_log_assets", "z_debt_to_assets", "z_roa_quarter", "z_book_to_market",
        "z_beta_market_report_month",
    ]
    specifications = [
        ("C1", "Archived primary two-factor DCC: equity beta", "z_beta_climate_equity_report_month", [], ["quarter"]),
        ("C2", "High-yield-return-adjusted DCC: equity beta", "z_beta_climate_credit3f", [], ["quarter"]),
        ("C3", "Archived primary two-factor DCC: asset beta", "z_beta_climate_asset_report_month", [], ["quarter"]),
        ("C4", "Credit-adjusted DCC: asset beta", "z_beta_asset_credit3f", [], ["quarter"]),
        ("C5", "Credit-adjusted DCC: equity beta plus controls", "z_beta_climate_credit3f", controls, ["quarter"]),
        ("C6", "Credit-adjusted DCC: within-BDC equity beta", "z_beta_climate_credit3f", [], ["ticker", "quarter"]),
    ]
    models = pd.DataFrame(
        [model(short_panel, outcome, model_id, label, model_controls, fixed_effects)
         for model_id, label, outcome, model_controls, fixed_effects in specifications]
    )
    beta_daily.to_csv(PROCESSED / "credit_adjusted_dcc_daily_2010_2025.csv", index=False)
    HELPER.export_dta(beta_daily, PROCESSED / "credit_adjusted_dcc_daily_2010_2025.dta")
    short_panel.to_csv(PROCESSED / "h2_credit_return_robustness_panel.csv", index=False)
    HELPER.export_dta(short_panel, PROCESSED / "h2_credit_return_robustness_panel.dta")
    models.to_csv(PROCESSED / "h2_credit_return_robustness_models.csv", index=False)
    HELPER.export_dta(models, PROCESSED / "h2_credit_return_robustness_models.dta")
    parameters = pd.concat([parameters_two, parameters_credit], ignore_index=True)
    parameters.to_csv(PROCESSED / "h2_credit_return_dcc_parameters.csv", index=False)
    HELPER.export_dta(parameters, PROCESSED / "h2_credit_return_dcc_parameters.dta")

    audit = {
        "source_file": str(PUBLIC_MARKET_SERIES.name),
        "credit_factor": "0.5*HYG total log return + 0.5*JNK total log return - SHY total log return",
        "available_sample": [str(daily["date"].min().date()), str(daily["date"].max().date())],
        "raw_series_redistributed": True,
        "daily_dates": int(beta_daily["date"].nunique()),
        "firms": int(short_panel["ticker"].nunique()),
        "firm_quarters": int(len(short_panel)),
        "baseline_definition": "C1 and C3 reuse the archived primary two-factor outcomes from Table 3; the separately re-estimated BDC-only two-factor series is retained as a diagnostic but is not labeled baseline.",
        "models": models[["model_id", "coefficient_exposure", "standard_error", "p_two_sided", "n"]].to_dict(orient="records"),
    }
    (AUDIT / "h2_credit_return_robustness_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (RAW / "credit_return_source_note.txt").write_text(
        "Series: HYG, JNK, and SHY adjusted total-return prices\n"
        "Source: Yahoo Finance chart endpoint; raw observations are stored in Part 1/Data/Raw.\n"
        "Transformation: 0.5*log return(HYG) + 0.5*log return(JNK) - log return(SHY).\n"
        "The factor is available throughout 2010-2025 and is used as a third DCC factor.\n",
        encoding="utf-8",
    )
    print(models[["model_id", "coefficient_exposure", "standard_error", "p_two_sided", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()
