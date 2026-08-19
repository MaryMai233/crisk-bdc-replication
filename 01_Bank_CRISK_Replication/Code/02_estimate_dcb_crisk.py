from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
INPUT = PACKAGE_ROOT / "Data" / "Processed"
OUTPUT_TABLES = PACKAGE_ROOT / "Data" / "Processed"
OUTPUT_DAILY = PACKAGE_ROOT / "Data" / "Processed"
OUTPUT_AUDIT = PACKAGE_ROOT / "Data" / "Processed" / "Audit"


@dataclass
class GjrFit:
    params: np.ndarray
    variance: np.ndarray
    standardized: np.ndarray
    success: bool
    objective: float


def gjr_variance(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    alpha, gamma, beta = params
    target = np.var(x, ddof=1)
    h = np.empty_like(x, dtype=float)
    h[0] = np.var(x, ddof=0)
    intercept = (1.0 - alpha - 0.5 * gamma - beta) * target
    for t in range(1, len(x)):
        lag_sq = x[t - 1] ** 2
        h[t] = (
            intercept
            + alpha * lag_sq
            + gamma * lag_sq * (x[t - 1] < 0)
            + beta * h[t - 1]
        )
    return h


def fit_gjr(x: np.ndarray) -> GjrFit:
    def objective(params: np.ndarray) -> float:
        h = gjr_variance(x, params)
        if not np.all(np.isfinite(h)) or np.any(h <= 1e-14):
            return 1e12
        return float(0.5 * np.sum(np.log(h) + x * x / h + np.log(2 * np.pi)))

    constraints = ({"type": "ineq", "fun": lambda p: 0.999 - p[0] - 0.5 * p[1] - p[2]},)
    result = minimize(
        objective,
        np.array([0.05, 0.05, 0.75]),
        method="SLSQP",
        bounds=[(1e-8, 0.999), (1e-8, 0.999), (1e-8, 0.999)],
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
    )
    params = result.x
    h = gjr_variance(x, params)
    return GjrFit(
        params=params,
        variance=h,
        standardized=x / np.sqrt(h),
        success=bool(result.success),
        objective=float(result.fun),
    )


def dcc_path(z: np.ndarray, params: np.ndarray, initial_innovation: str) -> tuple[np.ndarray, np.ndarray]:
    a, b = params
    qbar = np.cov(z, rowvar=False, ddof=1)
    k = z.shape[1]
    q_path = np.empty((len(z), k, k), dtype=float)
    r_path = np.empty_like(q_path)
    q_prev = qbar.copy()
    z_prev = np.zeros(k) if initial_innovation == "zeros" else np.ones(k)
    for t in range(len(z)):
        q_t = (1.0 - a - b) * qbar + a * np.outer(z_prev, z_prev) + b * q_prev
        diag = np.sqrt(np.clip(np.diag(q_t), 1e-14, None))
        r_t = q_t / np.outer(diag, diag)
        q_path[t] = q_t
        r_path[t] = r_t
        q_prev = q_t
        z_prev = z[t]
    return q_path, r_path


def fit_dcc(z: np.ndarray) -> tuple[np.ndarray, bool, float]:
    def objective(params: np.ndarray) -> float:
        _, r_path = dcc_path(z, params, initial_innovation="zeros")
        total = 0.0
        for t, r_t in enumerate(r_path):
            sign, logdet = np.linalg.slogdet(r_t)
            if sign <= 0 or not np.isfinite(logdet):
                return 1e12
            try:
                quad = float(z[t] @ np.linalg.solve(r_t, z[t]))
            except np.linalg.LinAlgError:
                return 1e12
            total += 0.5 * (logdet + quad)
        return float(total)

    constraints = ({"type": "ineq", "fun": lambda p: 0.999 - p[0] - p[1]},)
    result = minimize(
        objective,
        np.array([0.01, 0.97]),
        method="SLSQP",
        bounds=[(1e-8, 0.999), (1e-8, 0.999)],
        constraints=constraints,
        options={"ftol": 1e-8, "maxiter": 500, "disp": False},
    )
    return result.x, bool(result.success), float(result.fun)


def conditional_betas(h: np.ndarray, r_path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vol = np.sqrt(h)
    h_mkt = h[:, 1]
    h_climate = h[:, 2]
    cov_mkt_climate = r_path[:, 1, 2] * vol[:, 1] * vol[:, 2]
    cov_inst_mkt = r_path[:, 0, 1] * vol[:, 0] * vol[:, 1]
    cov_inst_climate = r_path[:, 0, 2] * vol[:, 0] * vol[:, 2]
    determinant = h_mkt * h_climate - cov_mkt_climate**2
    determinant = np.where(np.abs(determinant) < 1e-18, np.nan, determinant)
    beta_mkt = (
        h_climate * cov_inst_mkt - cov_mkt_climate * cov_inst_climate
    ) / determinant
    beta_climate = (
        h_mkt * cov_inst_climate - cov_mkt_climate * cov_inst_mkt
    ) / determinant
    return beta_mkt, beta_climate


def six_month_stress(factor: pd.DataFrame) -> dict:
    values = factor.sort_values("date")["ret_climate"].to_numpy(dtype=float)
    rolling_log = pd.Series(values).rolling(126, min_periods=126).sum()
    six_month_return = np.expm1(rolling_log.dropna())
    return {
        "p01": float(six_month_return.quantile(0.01)),
        "p05": float(six_month_return.quantile(0.05)),
        "median": float(six_month_return.quantile(0.50)),
        "stress_used": 0.50,
    }


def main() -> None:
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_DAILY.mkdir(parents=True, exist_ok=True)
    OUTPUT_AUDIT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(INPUT / "dcb_input_panel_2010_2025.csv", parse_dates=["date", "datadate", "available_date"])
    factor = pd.read_csv(INPUT / "climate_factor_daily_2010_2025.csv", parse_dates=["date"])
    complete = panel.dropna(subset=["ret", "logret_spy", "ret_climate"]).copy()

    first_pass: dict[int, dict] = {}
    parameter_rows = []
    for counter, (inst_id, group) in enumerate(complete.groupby("id"), start=1):
        group = group.sort_values("date").copy()
        raw = group[["ret", "logret_spy", "ret_climate"]].to_numpy(dtype=float)
        centered = raw - raw.mean(axis=0, keepdims=True)
        gjr_fits = [fit_gjr(centered[:, i]) for i in range(3)]
        h = np.column_stack([fit.variance for fit in gjr_fits])
        z = np.column_stack([fit.standardized for fit in gjr_fits])
        dcc_params, dcc_success, dcc_objective = fit_dcc(z)
        memo = str(group["memo"].iloc[0])
        first_pass[int(inst_id)] = {
            "group": group,
            "h": h,
            "z": z,
            "dcc": dcc_params,
        }
        parameter_rows.append(
            {
                "id": int(inst_id),
                "memo": memo,
                "observations": int(len(group)),
                "inst_alpha": gjr_fits[0].params[0],
                "inst_gamma": gjr_fits[0].params[1],
                "inst_beta": gjr_fits[0].params[2],
                "mkt_alpha": gjr_fits[1].params[0],
                "mkt_gamma": gjr_fits[1].params[1],
                "mkt_beta": gjr_fits[1].params[2],
                "climate_alpha": gjr_fits[2].params[0],
                "climate_gamma": gjr_fits[2].params[1],
                "climate_beta": gjr_fits[2].params[2],
                "gjr_all_success": all(fit.success for fit in gjr_fits),
                "dcc_alpha_first_pass": dcc_params[0],
                "dcc_beta_first_pass": dcc_params[1],
                "dcc_success": dcc_success,
                "dcc_objective": dcc_objective,
            }
        )
        print(
            f"[{counter:02d}/{complete['id'].nunique():02d}] {memo}: "
            f"T={len(group)}, DCC=({dcc_params[0]:.4f},{dcc_params[1]:.4f}), "
            f"success={dcc_success}",
            flush=True,
        )

    parameter_df = pd.DataFrame(parameter_rows).sort_values("id")
    successful_dcc = parameter_df[parameter_df["dcc_success"]]
    if successful_dcc.empty:
        raise RuntimeError("All DCC optimizations failed")
    median_dcc = successful_dcc[["dcc_alpha_first_pass", "dcc_beta_first_pass"]].median().to_numpy()
    if median_dcc.sum() >= 0.999:
        median_dcc = np.array([0.01, 0.97])

    beta_pieces = []
    for inst_id, item in first_pass.items():
        _, r_path = dcc_path(item["z"], median_dcc, initial_innovation="ones")
        beta_mkt, beta_climate = conditional_betas(item["h"], r_path)
        out = item["group"].copy()
        out["beta_market"] = beta_mkt
        out["beta_climate"] = beta_climate
        out["dcc_alpha_median"] = median_dcc[0]
        out["dcc_beta_median"] = median_dcc[1]
        beta_pieces.append(out)
    results = pd.concat(beta_pieces, ignore_index=True).sort_values(["id", "date"])
    bank_tickers = {"BAC", "BK", "C", "GS", "JPM", "MS", "PNC", "STT", "USB", "WFC"}
    results["group"] = np.where(
        results["current_ticker"].isin(bank_tickers), "Bank", "BDC"
    )

    stress = six_month_stress(factor)
    stress_cf = stress["stress_used"]
    k = 0.08
    results["lrmes_climate"] = 1.0 - np.exp(
        results["beta_climate"] * np.log(1.0 - stress_cf)
    )
    results["crisk_8pct_mn"] = (
        k * results["debt_mn"]
        - (1.0 - k)
        * results["mktcap_mn"]
        * (1.0 - results["lrmes_climate"])
    )
    results["crisk_8pct_positive_mn"] = results["crisk_8pct_mn"].clip(lower=0)

    annual = results.copy()
    annual["year"] = annual["date"].dt.year
    annual = (
        annual.groupby(
            ["group", "memo", "current_ticker", "company_name", "year"], as_index=False
        )
        .agg(
            beta_climate_mean=("beta_climate", "mean"),
            beta_climate_median=("beta_climate", "median"),
            beta_market_mean=("beta_market", "mean"),
            lrmes_climate_mean=("lrmes_climate", "mean"),
            crisk_8pct_mean_mn=("crisk_8pct_mn", "mean"),
            crisk_8pct_positive_mean_mn=("crisk_8pct_positive_mn", "mean"),
            observations=("date", "size"),
        )
        .sort_values(["memo", "year"])
    )

    result_cols = [
        "id",
        "group",
        "memo",
        "current_ticker",
        "company_name",
        "gvkey",
        "PERMNO",
        "historical_ticker",
        "date",
        "ret",
        "logret_spy",
        "ret_climate",
        "beta_market",
        "beta_climate",
        "mktcap_mn",
        "asset_mn",
        "book_equity_mn",
        "debt_mn",
        "lrmes_climate",
        "crisk_8pct_mn",
        "crisk_8pct_positive_mn",
        "coal_leg_source",
    ]
    results[result_cols].to_csv(OUTPUT_DAILY / "dcb_crisk_daily_2010_2025.csv", index=False)
    annual.to_csv(OUTPUT_TABLES / "dcb_crisk_annual_summary.csv", index=False)
    parameter_df.to_csv(OUTPUT_TABLES / "dcc_gjr_parameters.csv", index=False)

    audit = {
        "institutions": int(results["memo"].nunique()),
        "daily_rows": int(len(results)),
        "start": str(results["date"].min().date()),
        "end": str(results["date"].max().date()),
        "gjr_all_success_institutions": int(parameter_df["gjr_all_success"].sum()),
        "dcc_success_institutions": int(parameter_df["dcc_success"].sum()),
        "median_dcc_alpha": float(median_dcc[0]),
        "median_dcc_beta": float(median_dcc[1]),
        "median_dcc_persistence": float(median_dcc.sum()),
        "six_month_factor": stress,
        "beta_climate_p01": float(results["beta_climate"].quantile(0.01)),
        "beta_climate_median": float(results["beta_climate"].median()),
        "beta_climate_p99": float(results["beta_climate"].quantile(0.99)),
        "crisk_nonmissing": int(results["crisk_8pct_mn"].notna().sum()),
        "method": "Official CRISK two-step variance-targeted GJR-GARCH(1,1) and scalar DCC(1,1), with median DCC parameters across institutions",
    }
    (OUTPUT_AUDIT / "model_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
