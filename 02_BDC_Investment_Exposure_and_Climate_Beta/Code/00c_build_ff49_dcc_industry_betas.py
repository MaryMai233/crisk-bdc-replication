from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "Processed"
AUDIT = PROCESSED / "Audit"
BANK_ROOT = ROOT.parent / "01_Bank_CRISK_Replication"


def load_bank_dcc_module():
    path = BANK_ROOT / "Code" / "02_estimate_dcb_crisk.py"
    spec = importlib.util.spec_from_file_location("bank_dcc", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DCC = load_bank_dcc_module()


def export_dta(frame: pd.DataFrame, path: Path) -> None:
    out = frame.copy()
    for column in out.select_dtypes(include="object"):
        out[column] = out[column].fillna("").astype(str)
    out.to_stata(path, write_index=False, version=118)


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    industries = pd.read_csv(
        PROCESSED / "ff49_industry_climate_beta_daily_2020_2025.csv",
        parse_dates=["date"],
    )
    factors = pd.read_csv(
        BANK_ROOT / "Data" / "Processed" / "climate_factor_daily_2010_2025.csv",
        parse_dates=["date"],
        usecols=["date", "logret_spy", "ret_climate"],
    )
    frame = industries.merge(factors, on="date", how="left", validate="many_to_one")
    frame["industry_log_return"] = np.log1p(frame["industry_return"].clip(lower=-0.999999))
    frame = frame.dropna(subset=["industry_log_return", "logret_spy", "ret_climate"]).copy()

    bank_audit = json.loads(
        (BANK_ROOT / "Data" / "Processed" / "Audit" / "model_audit.json").read_text(encoding="utf-8")
    )
    median_dcc = np.array(
        [bank_audit["median_dcc_alpha"], bank_audit["median_dcc_beta"]], dtype=float
    )
    common = frame[["date", "logret_spy", "ret_climate"]].drop_duplicates("date").sort_values("date")
    common_raw = common[["logret_spy", "ret_climate"]].to_numpy(dtype=float)
    common_centered = common_raw - common_raw.mean(axis=0, keepdims=True)
    market_fit = DCC.fit_gjr(common_centered[:, 0])
    climate_fit = DCC.fit_gjr(common_centered[:, 1])
    pieces: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    for counter, (ff49, group) in enumerate(frame.groupby("ff49", sort=True), start=1):
        group = group.sort_values("date").copy()
        if not group["date"].equals(common["date"].reset_index(drop=True)):
            group = group.merge(
                common[["date"]], on="date", how="right", validate="one_to_one"
            ).sort_values("date")
        y = group["industry_log_return"].to_numpy(dtype=float)
        y_centered = y - np.nanmean(y)
        if np.isnan(y_centered).any():
            raise ValueError(f"FF49 industry {ff49} has missing returns on common trading dates")
        industry_fit = DCC.fit_gjr(y_centered)
        h = np.column_stack(
            [industry_fit.variance, market_fit.variance, climate_fit.variance]
        )
        z = np.column_stack(
            [industry_fit.standardized, market_fit.standardized, climate_fit.standardized]
        )
        _, r_path = DCC.dcc_path(z, median_dcc, initial_innovation="ones")
        beta_market, beta_climate = DCC.conditional_betas(h, r_path)
        group["mbeta_ff49_dcc"] = beta_market
        group["cbeta_ff49_dcc"] = beta_climate
        group["dcc_alpha_median"] = median_dcc[0]
        group["dcc_beta_median"] = median_dcc[1]
        pieces.append(group)
        parameter_rows.append(
            {
                "ff49": int(ff49),
                "ff49_name": str(group["ff49_name"].dropna().iloc[0]),
                "observations": int(len(group)),
                "industry_gjr_alpha": industry_fit.params[0],
                "industry_gjr_gamma": industry_fit.params[1],
                "industry_gjr_beta": industry_fit.params[2],
                "industry_gjr_success": industry_fit.success,
                "industry_gjr_objective": industry_fit.objective,
            }
        )
        print(f"[{counter:02d}/49] FF49 {int(ff49):02d}: GJR success={industry_fit.success}", flush=True)

    daily = pd.concat(pieces, ignore_index=True).sort_values(["date", "ff49"])
    keep = [
        "date", "ff49", "ff49_abbreviation", "ff49_name", "industry_return",
        "industry_log_return", "market_cap_sum", "n_firms", "mbeta_ff49_dcc",
        "cbeta_ff49_dcc", "dcc_alpha_median", "dcc_beta_median",
    ]
    daily = daily[keep].reset_index(drop=True)
    daily.to_csv(PROCESSED / "ff49_industry_dcc_beta_daily_2020_2025.csv", index=False)
    export_dta(daily, PROCESSED / "ff49_industry_dcc_beta_daily_2020_2025.dta")

    daily["quarter"] = daily["date"].dt.to_period("Q").astype(str)
    qend = (
        daily.sort_values(["ff49", "date"])
        .groupby(["ff49", "quarter"], as_index=False)
        .tail(1)
        .rename(
            columns={
                "date": "quarter_end_observation_date",
                "cbeta_ff49_dcc": "cbeta_ff49_dcc_qend",
                "mbeta_ff49_dcc": "mbeta_ff49_dcc_qend",
            }
        )
    )
    qmean = daily.groupby(["ff49", "quarter"], as_index=False).agg(
        cbeta_ff49_dcc_qmean=("cbeta_ff49_dcc", "mean"),
        mbeta_ff49_dcc_qmean=("mbeta_ff49_dcc", "mean"),
        trading_days=("date", "nunique"),
        mean_firms=("n_firms", "mean"),
    )
    quarterly = qend[
        [
            "ff49", "ff49_abbreviation", "ff49_name", "quarter",
            "quarter_end_observation_date", "cbeta_ff49_dcc_qend",
            "mbeta_ff49_dcc_qend", "market_cap_sum", "n_firms",
        ]
    ].merge(qmean, on=["ff49", "quarter"], how="outer", validate="one_to_one")
    quarterly = quarterly.sort_values(["quarter", "ff49"]).reset_index(drop=True)
    quarterly.to_csv(PROCESSED / "ff49_industry_dcc_beta_quarterly_2020_2025.csv", index=False)
    export_dta(quarterly, PROCESSED / "ff49_industry_dcc_beta_quarterly_2020_2025.dta")

    parameters = pd.DataFrame(parameter_rows)
    parameters.to_csv(PROCESSED / "ff49_industry_gjr_parameters.csv", index=False)
    export_dta(parameters, PROCESSED / "ff49_industry_gjr_parameters.dta")
    audit = {
        "status": "PASS" if bool(parameters["industry_gjr_success"].all()) else "REVIEW",
        "method": "GJR-GARCH variances and scalar DCC using the same median DCC parameters as the bank-BDC estimator",
        "median_dcc_alpha": float(median_dcc[0]),
        "median_dcc_beta": float(median_dcc[1]),
        "median_dcc_persistence": float(median_dcc.sum()),
        "ff49_industries": int(daily["ff49"].nunique()),
        "daily_rows": int(len(daily)),
        "quarterly_rows": int(len(quarterly)),
        "start": str(daily["date"].min().date()),
        "end": str(daily["date"].max().date()),
        "industry_gjr_success_count": int(parameters["industry_gjr_success"].sum()),
        "market_gjr_success": bool(market_fit.success),
        "climate_gjr_success": bool(climate_fit.success),
    }
    (AUDIT / "ff49_industry_dcc_beta_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
