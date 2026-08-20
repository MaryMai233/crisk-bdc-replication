from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data" / "Raw" / "CRSP_All_US_Daily_2020_2025"
PROCESSED = ROOT / "Data" / "Processed"
AUDIT = PROCESSED / "Audit"
BANK_PROCESSED = ROOT.parent / "01_Bank_CRISK_Replication" / "Data" / "Processed"

WINDOW = 252
MIN_OBSERVATIONS = 126


def export_dta(frame: pd.DataFrame, path: Path) -> None:
    out = frame.copy()
    for column in out.select_dtypes(include="object"):
        out[column] = out[column].fillna("").astype(str)
    out.to_stata(path, write_index=False, version=118)


def read_and_aggregate_year(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    pieces: list[pd.DataFrame] = []
    raw_rows = 0
    retained_rows = 0
    reader = pd.read_sas(
        path,
        format="sas7bdat",
        encoding="utf-8",
        iterator=True,
        chunksize=250_000,
    )
    required = {
        "PERMNO", "DLYCALDT", "PRIMARYEXCH", "CONDITIONALTYPE",
        "TRADINGSTATUSFLG", "USINCFLG", "ISSUERTYPE", "SECURITYTYPE",
        "SECURITYSUBTYPE", "SHARETYPE", "SICCD", "DLYCAP",
        "DLYPREVCAP", "DLYRET",
    }
    missing = sorted(required.difference(reader.column_names))
    if missing:
        raise ValueError(f"{path.name} is missing required variables: {missing}")

    for chunk in reader:
        raw_rows += len(chunk)
        for column in (
            "PRIMARYEXCH", "CONDITIONALTYPE", "TRADINGSTATUSFLG",
            "USINCFLG", "ISSUERTYPE", "SECURITYTYPE",
            "SECURITYSUBTYPE", "SHARETYPE",
        ):
            chunk[column] = chunk[column].astype(str).str.upper().str.strip()
        for column in ("PERMNO", "SICCD", "DLYCAP", "DLYPREVCAP", "DLYRET"):
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")

        mask = (
            chunk["SHARETYPE"].eq("NS")
            & chunk["SECURITYTYPE"].eq("EQTY")
            & chunk["SECURITYSUBTYPE"].eq("COM")
            & chunk["USINCFLG"].eq("Y")
            & chunk["ISSUERTYPE"].isin(["ACOR", "CORP"])
            & chunk["PRIMARYEXCH"].isin(["N", "A", "Q"])
            & chunk["CONDITIONALTYPE"].eq("RW")
            & chunk["TRADINGSTATUSFLG"].eq("A")
            & chunk["SICCD"].between(100, 9998)
            & chunk["DLYRET"].notna()
            & chunk["DLYRET"].gt(-1.0)
            & chunk["DLYPREVCAP"].gt(0)
        )
        data = chunk.loc[
            mask,
            ["PERMNO", "DLYCALDT", "SICCD", "DLYCAP", "DLYPREVCAP", "DLYRET"],
        ].copy()
        retained_rows += len(data)
        data["sic3"] = np.floor(data["SICCD"] / 10).astype("int16")
        data["weighted_return"] = data["DLYRET"] * data["DLYPREVCAP"]
        grouped = (
            data.groupby(["DLYCALDT", "sic3"], as_index=False)
            .agg(
                weighted_return_sum=("weighted_return", "sum"),
                previous_market_cap_sum=("DLYPREVCAP", "sum"),
                market_cap_sum=("DLYCAP", "sum"),
                n_firms=("PERMNO", "size"),
            )
        )
        pieces.append(grouped)
    reader.close()

    aggregated = pd.concat(pieces, ignore_index=True)
    aggregated = (
        aggregated.groupby(["DLYCALDT", "sic3"], as_index=False)
        .agg(
            weighted_return_sum=("weighted_return_sum", "sum"),
            previous_market_cap_sum=("previous_market_cap_sum", "sum"),
            market_cap_sum=("market_cap_sum", "sum"),
            n_firms=("n_firms", "sum"),
        )
    )
    aggregated["industry_return"] = (
        aggregated["weighted_return_sum"] / aggregated["previous_market_cap_sum"]
    )
    aggregated = aggregated.rename(columns={"DLYCALDT": "date"})
    audit = {
        "file": path.name,
        "raw_rows": raw_rows,
        "retained_common_stock_rows": retained_rows,
        "retention_rate": retained_rows / raw_rows,
        "trading_days": int(aggregated["date"].nunique()),
        "sic3_groups": int(aggregated["sic3"].nunique()),
        "industry_day_rows": int(len(aggregated)),
    }
    return aggregated, audit


def rolling_two_factor_beta(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").copy()
    y = np.log1p(group["industry_return"].clip(lower=-0.999999))
    m = group["logret_spy"]
    c = group["ret_climate"]
    valid = y.notna() & m.notna() & c.notna()
    y = y.where(valid)
    m = m.where(valid)
    c = c.where(valid)
    rolling = lambda values: values.rolling(WINDOW, min_periods=MIN_OBSERVATIONS).mean()

    ey, em, ec = rolling(y), rolling(m), rolling(c)
    syy = rolling(y * y) - ey * ey
    smm = rolling(m * m) - em * em
    scc = rolling(c * c) - ec * ec
    sym = rolling(y * m) - ey * em
    syc = rolling(y * c) - ey * ec
    smc = rolling(m * c) - em * ec
    denominator = smm * scc - smc * smc
    group["mbeta_sic3"] = (sym * scc - syc * smc) / denominator
    group["cbeta_sic3"] = (syc * smm - sym * smc) / denominator
    fitted_variance = (
        group["mbeta_sic3"] ** 2 * smm
        + group["cbeta_sic3"] ** 2 * scc
        + 2 * group["mbeta_sic3"] * group["cbeta_sic3"] * smc
    )
    group["rolling_r_squared"] = fitted_variance / syy
    group["rolling_observations"] = valid.rolling(WINDOW, min_periods=1).sum()
    unstable = denominator.abs().lt(1e-14)
    group.loc[unstable, ["mbeta_sic3", "cbeta_sic3", "rolling_r_squared"]] = np.nan
    return group


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    annual_files = sorted(RAW.glob("crsp_all_us_daily_202*.gz"))
    if len(annual_files) != 6:
        raise FileNotFoundError(f"Expected six annual CRSP files in {RAW}; found {len(annual_files)}")

    industry_returns: list[pd.DataFrame] = []
    annual_audits: list[dict[str, object]] = []
    for path in annual_files:
        print(f"Reading and aggregating {path.name} ...", flush=True)
        annual, audit = read_and_aggregate_year(path)
        industry_returns.append(annual)
        annual_audits.append(audit)
        print(json.dumps(audit, indent=2), flush=True)

    returns = pd.concat(industry_returns, ignore_index=True).sort_values(["sic3", "date"])
    factors = pd.read_csv(
        BANK_PROCESSED / "climate_factor_daily_2010_2025.csv",
        parse_dates=["date"],
        usecols=["date", "logret_spy", "ret_climate"],
    )
    returns = returns.merge(factors, on="date", how="left", validate="many_to_one")
    if returns[["logret_spy", "ret_climate"]].isna().any().any():
        missing_dates = returns.loc[
            returns[["logret_spy", "ret_climate"]].isna().any(axis=1), "date"
        ].drop_duplicates()
        raise ValueError(f"Climate factors are missing on {len(missing_dates)} CRSP dates")

    print("Estimating trailing 252-trading-day SIC3 climate betas ...", flush=True)
    beta_parts: list[pd.DataFrame] = []
    for sic3, group in returns.groupby("sic3", sort=True):
        estimated = rolling_two_factor_beta(group)
        estimated["sic3"] = sic3
        beta_parts.append(estimated)
    beta = pd.concat(beta_parts, ignore_index=True)
    beta["sic3"] = beta["sic3"].astype("int16")
    keep = [
        "date", "sic3", "industry_return", "market_cap_sum",
        "previous_market_cap_sum", "n_firms", "mbeta_sic3", "cbeta_sic3",
        "rolling_r_squared", "rolling_observations",
    ]
    beta = beta[keep].sort_values(["date", "sic3"]).reset_index(drop=True)
    beta.to_csv(PROCESSED / "sic3_industry_climate_beta_daily_2020_2025.csv", index=False)
    export_dta(beta, PROCESSED / "sic3_industry_climate_beta_daily_2020_2025.dta")

    beta["quarter"] = beta["date"].dt.to_period("Q").astype(str)
    available = beta.dropna(subset=["cbeta_sic3"]).copy()
    qend = (
        available.sort_values(["sic3", "date"])
        .groupby(["sic3", "quarter"], as_index=False)
        .tail(1)
        .rename(
            columns={
                "date": "quarter_end_observation_date",
                "cbeta_sic3": "cbeta_sic3_qend",
                "mbeta_sic3": "mbeta_sic3_qend",
            }
        )
    )
    qmean = (
        available.groupby(["sic3", "quarter"], as_index=False)
        .agg(
            cbeta_sic3_qmean=("cbeta_sic3", "mean"),
            mbeta_sic3_qmean=("mbeta_sic3", "mean"),
            trading_days=("date", "nunique"),
            mean_firms=("n_firms", "mean"),
        )
    )
    quarterly = qend[
        [
            "sic3", "quarter", "quarter_end_observation_date",
            "cbeta_sic3_qend", "mbeta_sic3_qend", "market_cap_sum", "n_firms",
        ]
    ].merge(qmean, on=["sic3", "quarter"], how="outer")
    quarterly = quarterly.sort_values(["quarter", "sic3"]).reset_index(drop=True)
    quarterly.to_csv(PROCESSED / "sic3_industry_climate_beta_quarterly_2020_2025.csv", index=False)
    export_dta(quarterly, PROCESSED / "sic3_industry_climate_beta_quarterly_2020_2025.dta")

    audit = {
        "status": "PASS",
        "source_format": "CRSP Stock Version 2 (CIZ), annual gzip-compressed SAS7BDAT",
        "common_stock_filter": {
            "ShareType": "NS",
            "SecurityType": "EQTY",
            "SecuritySubType": "COM",
            "USIncFlg": "Y",
            "IssuerType": ["ACOR", "CORP"],
            "PrimaryExch": ["N", "A", "Q"],
            "ConditionalType": "RW",
            "TradingStatusFlg": "A",
        },
        "industry_return": "DlyPrevCap-weighted DlyRet within historical three-digit SIC",
        "beta_model": "Trailing 252-trading-day OLS with intercept, market, and published stranded-asset factor",
        "minimum_observations": MIN_OBSERVATIONS,
        "annual_files": annual_audits,
        "daily_rows": int(len(beta)),
        "quarterly_rows": int(len(quarterly)),
        "sic3_groups": int(beta["sic3"].nunique()),
        "nonmissing_daily_climate_betas": int(beta["cbeta_sic3"].notna().sum()),
        "first_beta_date": str(beta.loc[beta["cbeta_sic3"].notna(), "date"].min().date()),
        "last_beta_date": str(beta.loc[beta["cbeta_sic3"].notna(), "date"].max().date()),
    }
    (AUDIT / "sic3_industry_climate_beta_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
