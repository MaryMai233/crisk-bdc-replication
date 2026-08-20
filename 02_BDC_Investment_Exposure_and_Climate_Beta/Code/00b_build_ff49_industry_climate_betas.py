from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data" / "Raw" / "CRSP_All_US_Daily_2020_2025"
DEFINITIONS = ROOT / "Data" / "Raw" / "Siccodes49.txt"
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


def parse_ff49(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    lookup = np.zeros(10_000, dtype=np.int16)
    records: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    header = re.compile(r"^\s*(\d+)\s+([A-Za-z]+)\s+(.+?)\s*$")
    sic_range = re.compile(r"^\s*(\d{3,4})-(\d{3,4})\s+(.+?)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match_header = header.match(line)
        if match_header:
            current = {
                "ff49": int(match_header.group(1)),
                "ff49_abbreviation": match_header.group(2),
                "ff49_name": match_header.group(3).strip(),
            }
            continue
        match_range = sic_range.match(line)
        if current and match_range:
            start, end = int(match_range.group(1)), int(match_range.group(2))
            lookup[start : end + 1] = int(current["ff49"])
            records.append(
                {
                    **current,
                    "sic_start": start,
                    "sic_end": end,
                    "sic_description": match_range.group(3).strip(),
                }
            )
    definitions = pd.DataFrame(records)
    if definitions["ff49"].nunique() != 49:
        raise ValueError("The parsed definition file does not contain all 49 industries")
    return lookup, definitions


def read_and_aggregate_year(
    path: Path, lookup: np.ndarray
) -> tuple[pd.DataFrame, dict[str, object]]:
    pieces: list[pd.DataFrame] = []
    raw_rows = 0
    retained_rows = 0
    classified_rows = 0
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
        sic4 = data["SICCD"].round().astype(int).clip(0, 9999)
        data["ff49"] = lookup[sic4.to_numpy()]
        data = data.loc[data["ff49"].gt(0)].copy()
        classified_rows += len(data)
        data["weighted_return"] = data["DLYRET"] * data["DLYPREVCAP"]
        pieces.append(
            data.groupby(["DLYCALDT", "ff49"], as_index=False).agg(
                weighted_return_sum=("weighted_return", "sum"),
                previous_market_cap_sum=("DLYPREVCAP", "sum"),
                market_cap_sum=("DLYCAP", "sum"),
                n_firms=("PERMNO", "size"),
            )
        )
    reader.close()

    aggregated = pd.concat(pieces, ignore_index=True)
    aggregated = aggregated.groupby(["DLYCALDT", "ff49"], as_index=False).agg(
        weighted_return_sum=("weighted_return_sum", "sum"),
        previous_market_cap_sum=("previous_market_cap_sum", "sum"),
        market_cap_sum=("market_cap_sum", "sum"),
        n_firms=("n_firms", "sum"),
    )
    aggregated["industry_return"] = (
        aggregated["weighted_return_sum"] / aggregated["previous_market_cap_sum"]
    )
    aggregated = aggregated.rename(columns={"DLYCALDT": "date"})
    audit = {
        "file": path.name,
        "raw_rows": raw_rows,
        "retained_common_stock_rows": retained_rows,
        "ff49_classified_rows": classified_rows,
        "classification_rate_after_common_stock_filter": classified_rows / retained_rows,
        "trading_days": int(aggregated["date"].nunique()),
        "ff49_groups": int(aggregated["ff49"].nunique()),
        "industry_day_rows": int(len(aggregated)),
    }
    return aggregated, audit


def rolling_two_factor_beta(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").copy()
    y = np.log1p(group["industry_return"].clip(lower=-0.999999))
    m = group["logret_spy"]
    c = group["ret_climate"]
    valid = y.notna() & m.notna() & c.notna()
    y, m, c = y.where(valid), m.where(valid), c.where(valid)
    rolling = lambda values: values.rolling(WINDOW, min_periods=MIN_OBSERVATIONS).mean()
    ey, em, ec = rolling(y), rolling(m), rolling(c)
    syy = rolling(y * y) - ey * ey
    smm = rolling(m * m) - em * em
    scc = rolling(c * c) - ec * ec
    sym = rolling(y * m) - ey * em
    syc = rolling(y * c) - ey * ec
    smc = rolling(m * c) - em * ec
    denominator = smm * scc - smc * smc
    group["mbeta_ff49"] = (sym * scc - syc * smc) / denominator
    group["cbeta_ff49"] = (syc * smm - sym * smc) / denominator
    fitted_variance = (
        group["mbeta_ff49"] ** 2 * smm
        + group["cbeta_ff49"] ** 2 * scc
        + 2 * group["mbeta_ff49"] * group["cbeta_ff49"] * smc
    )
    group["rolling_r_squared"] = fitted_variance / syy
    group["rolling_observations"] = valid.rolling(WINDOW, min_periods=1).sum()
    unstable = denominator.abs().lt(1e-14)
    group.loc[unstable, ["mbeta_ff49", "cbeta_ff49", "rolling_r_squared"]] = np.nan
    return group


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    lookup, definitions = parse_ff49(DEFINITIONS)
    definitions.to_csv(PROCESSED / "ff49_sic_definition_ranges.csv", index=False)
    annual_files = sorted(RAW.glob("crsp_all_us_daily_202*.gz"))
    if len(annual_files) != 6:
        raise FileNotFoundError(f"Expected six annual CRSP files; found {len(annual_files)}")

    industry_returns: list[pd.DataFrame] = []
    annual_audits: list[dict[str, object]] = []
    for path in annual_files:
        print(f"Reading and aggregating {path.name} ...", flush=True)
        annual, audit = read_and_aggregate_year(path, lookup)
        industry_returns.append(annual)
        annual_audits.append(audit)
        print(json.dumps(audit, indent=2), flush=True)

    returns = pd.concat(industry_returns, ignore_index=True).sort_values(["ff49", "date"])
    factors = pd.read_csv(
        BANK_PROCESSED / "climate_factor_daily_2010_2025.csv",
        parse_dates=["date"],
        usecols=["date", "logret_spy", "ret_climate"],
    )
    returns = returns.merge(factors, on="date", how="left", validate="many_to_one")
    if returns[["logret_spy", "ret_climate"]].isna().any().any():
        raise ValueError("Climate factors are missing on one or more CRSP dates")

    print("Estimating trailing 252-trading-day FF49 climate betas ...", flush=True)
    beta_parts: list[pd.DataFrame] = []
    for ff49, group in returns.groupby("ff49", sort=True):
        estimated = rolling_two_factor_beta(group)
        estimated["ff49"] = ff49
        beta_parts.append(estimated)
    beta = pd.concat(beta_parts, ignore_index=True)
    metadata = definitions[["ff49", "ff49_abbreviation", "ff49_name"]].drop_duplicates()
    beta = beta.merge(metadata, on="ff49", how="left", validate="many_to_one")
    keep = [
        "date", "ff49", "ff49_abbreviation", "ff49_name", "industry_return",
        "market_cap_sum", "previous_market_cap_sum", "n_firms", "mbeta_ff49",
        "cbeta_ff49", "rolling_r_squared", "rolling_observations",
    ]
    beta = beta[keep].sort_values(["date", "ff49"]).reset_index(drop=True)
    beta.to_csv(PROCESSED / "ff49_industry_climate_beta_daily_2020_2025.csv", index=False)
    export_dta(beta, PROCESSED / "ff49_industry_climate_beta_daily_2020_2025.dta")

    beta["quarter"] = beta["date"].dt.to_period("Q").astype(str)
    available = beta.dropna(subset=["cbeta_ff49"]).copy()
    qend = (
        available.sort_values(["ff49", "date"])
        .groupby(["ff49", "quarter"], as_index=False)
        .tail(1)
        .rename(
            columns={
                "date": "quarter_end_observation_date",
                "cbeta_ff49": "cbeta_ff49_qend",
                "mbeta_ff49": "mbeta_ff49_qend",
            }
        )
    )
    qmean = available.groupby(["ff49", "quarter"], as_index=False).agg(
        cbeta_ff49_qmean=("cbeta_ff49", "mean"),
        mbeta_ff49_qmean=("mbeta_ff49", "mean"),
        trading_days=("date", "nunique"),
        mean_firms=("n_firms", "mean"),
    )
    quarterly = qend[
        [
            "ff49", "ff49_abbreviation", "ff49_name", "quarter",
            "quarter_end_observation_date", "cbeta_ff49_qend", "mbeta_ff49_qend",
            "market_cap_sum", "n_firms",
        ]
    ].merge(qmean, on=["ff49", "quarter"], how="outer")
    quarterly = quarterly.sort_values(["quarter", "ff49"]).reset_index(drop=True)
    quarterly.to_csv(PROCESSED / "ff49_industry_climate_beta_quarterly_2020_2025.csv", index=False)
    export_dta(quarterly, PROCESSED / "ff49_industry_climate_beta_quarterly_2020_2025.dta")

    audit = {
        "status": "PASS",
        "industry_definition": "Fama-French 49 industries, parsed from Siccodes49.txt",
        "industry_definition_source": "Kenneth R. French Data Library",
        "industry_return": "DlyPrevCap-weighted DlyRet within FF49 industries",
        "beta_model": "Trailing 252-trading-day OLS with intercept, market, and stranded-asset factor",
        "minimum_observations": MIN_OBSERVATIONS,
        "annual_files": annual_audits,
        "daily_rows": int(len(beta)),
        "quarterly_rows": int(len(quarterly)),
        "ff49_groups": int(beta["ff49"].nunique()),
        "nonmissing_daily_climate_betas": int(beta["cbeta_ff49"].notna().sum()),
        "first_beta_date": str(beta.loc[beta["cbeta_ff49"].notna(), "date"].min().date()),
        "last_beta_date": str(beta.loc[beta["cbeta_ff49"].notna(), "date"].max().date()),
    }
    (AUDIT / "ff49_industry_climate_beta_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
