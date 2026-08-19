from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
UPLOAD = PACKAGE_ROOT / "Data" / "Raw"
OUTPUT = PACKAGE_ROOT / "Data" / "Processed"


def read_xlsx(name: str) -> pd.DataFrame:
    return pd.read_excel(UPLOAD / name)


def to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def excel_link_date(series: pd.Series, end: bool = False) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid_serial = numeric.between(1, 100000)
    parsed = pd.to_datetime(series.where(~valid_serial), errors="coerce")
    serial_dates = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    serial_dates.loc[valid_serial] = (
        pd.Timestamp("1899-12-30")
        + pd.to_timedelta(numeric.loc[valid_serial], unit="D")
    )
    dates = parsed.fillna(serial_dates)
    if end:
        dates = dates.fillna(pd.Timestamp("2099-12-31"))
    return dates.dt.normalize()


def normalize_crsp(df: pd.DataFrame, source: str, priority: int) -> pd.DataFrame:
    out = df.drop_duplicates().copy()
    out = out.rename(
        columns={
            "Daily Calendar Date": "date",
            "Daily Total Return": "ret",
            "Daily Price Return": "price_ret",
            "Daily Price": "price",
            "Daily Capitalization": "mktcap_thousand",
            "Security Name": "security_name",
            "Ticker": "historical_ticker",
            "Trading Status Flag": "trading_status",
        }
    )
    out["date"] = to_date(out["date"])
    for col in ["ret", "price_ret", "price", "mktcap_thousand"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["mktcap_thousand"] = out["mktcap_thousand"].abs()
    out["source_file"] = source
    out["source_priority"] = priority
    return out


def build_climate_factor() -> tuple[pd.DataFrame, dict]:
    etf = normalize_crsp(
        read_xlsx("crsp_factor_etfs_daily_2010_2025.xlsx"), "crsp_factor_etfs", 1
    )
    etf = etf.dropna(subset=["date", "historical_ticker"])
    etf = etf.sort_values(["historical_ticker", "date", "source_priority"])
    etf = etf.drop_duplicates(["historical_ticker", "date"], keep="last")
    returns = etf.pivot(index="date", columns="historical_ticker", values="ret")

    coal_raw = read_xlsx("crsp_us_coal_daily_2010_2025.xlsx").drop_duplicates().copy()
    coal_raw = coal_raw.rename(
        columns={
            "Daily Calendar Date": "date",
            "Daily Total Return": "ret",
            "Daily Capitalization": "mktcap_thousand",
            "Security Name": "security_name",
            "Ticker": "ticker",
            "Sic Code": "sic",
        }
    )
    coal_raw["date"] = to_date(coal_raw["date"])
    coal_raw["ret"] = pd.to_numeric(coal_raw["ret"], errors="coerce")
    coal_raw["mktcap_thousand"] = pd.to_numeric(
        coal_raw["mktcap_thousand"], errors="coerce"
    ).abs()
    coal_raw["sic"] = pd.to_numeric(coal_raw["sic"], errors="coerce")
    coal_raw = coal_raw.sort_values(["PERMNO", "date"])
    coal_raw["lag_mktcap"] = coal_raw.groupby("PERMNO")["mktcap_thousand"].shift(1)
    coal_raw["lag_date"] = coal_raw.groupby("PERMNO")["date"].shift(1)
    stale_lag = (coal_raw["date"] - coal_raw["lag_date"]).dt.days > 10
    coal_raw.loc[stale_lag, "lag_mktcap"] = np.nan

    coal_raw["is_adr"] = coal_raw["security_name"].astype(str).str.contains(
        "ADR|ADS|AMERICAN DEPOSIT", case=False, regex=True
    )

    def proxy_series(mask: pd.Series, weighting: str) -> tuple[pd.Series, int]:
        candidate = coal_raw[
            mask & coal_raw["PERMNO"].ne(13411) & coal_raw["ret"].notna()
        ].copy()
        if weighting == "value weighted":
            candidate = candidate[candidate["lag_mktcap"].gt(0)].copy()
            candidate["weighted_ret"] = candidate["ret"] * candidate["lag_mktcap"]
            series = candidate.groupby("date").apply(
                lambda x: x["weighted_ret"].sum() / x["lag_mktcap"].sum(),
                include_groups=False,
            )
        else:
            series = candidate.groupby("date")["ret"].mean()
        return series.rename("proxy"), int(candidate["PERMNO"].nunique())

    def proxy_diagnostic(label: str, mask: pd.Series, weighting: str) -> dict:
        series, permnos = proxy_series(mask, weighting)
        overlap_frame = returns[["KOL"]].merge(series, left_index=True, right_index=True, how="inner").dropna()
        overlap_frame = overlap_frame.loc[
            overlap_frame.index.to_series().between("2019-01-01", "2020-12-14")
        ]
        return {
            "definition": label,
            "weighting": weighting,
            "overlap_days": int(len(overlap_frame)),
            "correlation_with_kol": float(overlap_frame["KOL"].corr(overlap_frame["proxy"])),
            "proxy_to_kol_volatility_ratio": float(
                overlap_frame["proxy"].std() / overlap_frame["KOL"].std()
            ),
            "securities_full_period": permnos,
            "maintained_factor": 0,
        }

    strict_sic = {1220, 1221, 1222, 1231}
    strict_us, strict_us_n = proxy_series(
        coal_raw["sic"].isin(strict_sic) & ~coal_raw["is_adr"], "value weighted"
    )

    public_path = UPLOAD / "public_market_series_yahoo_2010_2025.csv"
    if not public_path.exists():
        raise FileNotFoundError(
            "Run Code/00_download_public_market_series.py before constructing the factor"
        )
    public = pd.read_csv(public_path, parse_dates=["date"])
    public["adjusted_close"] = pd.to_numeric(public["adjusted_close"], errors="coerce")
    prices = public.pivot(index="date", columns="symbol", values="adjusted_close").sort_index()
    master_dates = pd.DatetimeIndex(returns.index).sort_values()

    top_five_weights = {
        "SOL.AX": 6.85,
        "AZJ.AX": 6.19,
        "UNTR.JK": 6.00,
        "1088.HK": 5.97,
        "ADRO.JK": 5.63,
    }
    currency_fx = {
        "SOL.AX": "AUDUSD=X",
        "AZJ.AX": "AUDUSD=X",
        "UNTR.JK": "IDRUSD=X",
        "1088.HK": "HKDUSD=X",
        "ADRO.JK": "IDRUSD=X",
    }

    local_return = pd.DataFrame(index=master_dates)
    usd_return = pd.DataFrame(index=master_dates)
    for symbol, fx_symbol in currency_fx.items():
        local_price = prices[symbol].reindex(prices.index.union(master_dates)).sort_index().ffill().reindex(master_dates)
        fx_price = prices[fx_symbol].reindex(prices.index.union(master_dates)).sort_index().ffill().reindex(master_dates)
        local_return[symbol] = np.log(local_price).diff()
        usd_return[symbol] = np.log(local_price * fx_price).diff()

    normalized_weights = pd.Series(top_five_weights, dtype=float)
    normalized_weights /= normalized_weights.sum()

    def weighted_available(frame: pd.DataFrame) -> pd.Series:
        available_weights = frame.notna().mul(normalized_weights, axis=1)
        denominator = available_weights.sum(axis=1).replace(0, np.nan)
        return frame.mul(normalized_weights, axis=1).sum(axis=1, min_count=1) / denominator

    top5_local = weighted_available(local_return).rename("top5_local_logret")
    top5_usd = weighted_available(usd_return).rename("top5_usd_logret")
    factor = returns.reset_index().merge(
        pd.concat([top5_local, top5_usd], axis=1).reset_index(names="date"),
        on="date",
        how="left",
    )
    factor = factor.merge(
        strict_us.rename("coal_us_proxy_ret").reset_index(), on="date", how="left"
    )
    factor["coal_us_proxy_logret"] = np.log1p(factor["coal_us_proxy_ret"])
    for ticker in ["SPY", "XLE", "KOL"]:
        factor[f"ret_{ticker.lower()}"] = pd.to_numeric(
            factor.get(ticker), errors="coerce"
        )
        factor[f"logret_{ticker.lower()}"] = np.log1p(factor[f"ret_{ticker.lower()}"])
    cutoff = pd.Timestamp("2020-12-14")
    factor["coal_leg_logret"] = np.where(
        factor["date"].le(cutoff), factor["logret_kol"], factor["top5_usd_logret"]
    )
    factor["coal_leg_source"] = np.where(
        factor["date"].le(cutoff), "KOL ETF", "KOL pre-liquidation top-five constituents"
    )
    factor["ret_climate"] = (
        0.3 * factor["logret_xle"]
        + 0.7 * factor["coal_leg_logret"]
        - factor["logret_spy"]
    )
    factor = factor[
        [
            "date",
            "ret_spy",
            "ret_xle",
            "ret_kol",
            "logret_spy",
            "logret_xle",
            "logret_kol",
            "top5_local_logret",
            "top5_usd_logret",
            "coal_us_proxy_ret",
            "coal_us_proxy_logret",
            "coal_leg_logret",
            "coal_leg_source",
            "ret_climate",
        ]
    ].sort_values("date")

    overlap = factor.loc[
        factor["date"].between("2019-01-01", cutoff),
        ["date", "logret_kol", "top5_local_logret", "top5_usd_logret", "coal_us_proxy_logret"],
    ].dropna().set_index("date")

    def horizon_correlation(left: str, right: str, frequency: str | None = None) -> float:
        pair = overlap[[left, right]].copy()
        if frequency is not None:
            pair = pair.resample(frequency).sum(min_count=1).dropna()
        return float(pair[left].corr(pair[right]))

    top5_daily_corr = horizon_correlation("logret_kol", "top5_usd_logret")
    top5_weekly_corr = horizon_correlation("logret_kol", "top5_usd_logret", "W-FRI")
    top5_monthly_corr = horizon_correlation("logret_kol", "top5_usd_logret", "ME")
    diagnostics = pd.DataFrame(
        [
            {
                "definition": "KOL top five, fixed pre-liquidation weights, USD total returns",
                "weighting": "6.85/6.19/6.00/5.97/5.63 percent, normalized",
                "overlap_days": len(overlap),
                "correlation_with_kol": top5_daily_corr,
                "weekly_correlation_with_kol": top5_weekly_corr,
                "monthly_correlation_with_kol": top5_monthly_corr,
                "proxy_to_kol_volatility_ratio": overlap["top5_usd_logret"].std() / overlap["logret_kol"].std(),
                "securities_full_period": 5,
                "maintained_factor": 1,
            },
            {
                "definition": "KOL top five, fixed pre-liquidation weights, local-currency total returns",
                "weighting": "6.85/6.19/6.00/5.97/5.63 percent, normalized",
                "overlap_days": len(overlap),
                "correlation_with_kol": overlap["logret_kol"].corr(overlap["top5_local_logret"]),
                "weekly_correlation_with_kol": horizon_correlation("logret_kol", "top5_local_logret", "W-FRI"),
                "monthly_correlation_with_kol": horizon_correlation("logret_kol", "top5_local_logret", "ME"),
                "proxy_to_kol_volatility_ratio": overlap["top5_local_logret"].std() / overlap["logret_kol"].std(),
                "securities_full_period": 5,
                "maintained_factor": 0,
            },
            proxy_diagnostic(
                "Strict coal SIC, U.S. listings only",
                coal_raw["sic"].isin(strict_sic) & ~coal_raw["is_adr"],
                "value weighted",
            ),
            proxy_diagnostic(
                "Strict coal SIC, including U.S.-traded ADRs",
                coal_raw["sic"].isin(strict_sic),
                "value weighted",
            ),
            proxy_diagnostic(
                "Broad mining SIC 1000-1499, including ADRs",
                coal_raw["sic"].between(1000, 1499),
                "value weighted",
            ),
        ]
    )
    diagnostics.to_csv(OUTPUT / "coal_proxy_diagnostics.csv", index=False)
    factor_audit = {
        "factor_start": str(factor["date"].min().date()),
        "factor_end": str(factor["date"].max().date()),
        "factor_rows": int(len(factor)),
        "factor_nonmissing": int(factor["ret_climate"].notna().sum()),
        "kol_top5_usd_logret_corr_daily_2019_2020": top5_daily_corr,
        "kol_top5_usd_logret_corr_weekly_2019_2020": top5_weekly_corr,
        "kol_top5_usd_logret_corr_monthly_2019_2020": top5_monthly_corr,
        "kol_top5_local_logret_corr_2019_2020": float(overlap["logret_kol"].corr(overlap["top5_local_logret"])),
        "kol_us_coal_proxy_corr_2019_2020": float(overlap["logret_kol"].corr(overlap["coal_us_proxy_logret"])),
        "proxy_diagnostic_file": "coal_proxy_diagnostics.csv",
        "top_five_weights_percent": top_five_weights,
        "coal_rule": "KOL through 2020-12-14; thereafter fixed weighted average of KOL's five largest pre-liquidation constituents, using USD-converted adjusted total returns",
        "alternative_us_coal_proxy_securities": strict_us_n,
    }
    return factor, factor_audit


def build_link_table() -> pd.DataFrame:
    links = read_xlsx("ccm_linktable_banks_bdc.xlsx").rename(
        columns={
            "Global Company Key": "gvkey",
            "Company Name": "company_name",
            "Ticker Symbol": "current_ticker",
            "Primary Link Marker": "linkprim",
            "Link Type Codd": "linktype",
            "Historical CRSP PERMNO Link to COMPUSTAT Record": "PERMNO",
            "First Effective Date of Link": "link_start_raw",
            "Last Effective Date of Link": "link_end_raw",
        }
    )
    links["gvkey"] = links["gvkey"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
    links["PERMNO"] = pd.to_numeric(links["PERMNO"], errors="coerce").astype("Int64")
    links["link_start"] = excel_link_date(links["link_start_raw"])
    links["link_end"] = excel_link_date(links["link_end_raw"], end=True)
    links = links[
        [
            "gvkey",
            "company_name",
            "current_ticker",
            "PERMNO",
            "linkprim",
            "linktype",
            "link_start",
            "link_end",
        ]
    ].dropna(subset=["PERMNO", "link_start"])
    manual_path = UPLOAD / "bk_manual_identifier_link.csv"
    if manual_path.exists():
        manual = pd.read_csv(manual_path, dtype={"gvkey": str})
        manual["PERMNO"] = pd.to_numeric(manual["PERMNO"], errors="coerce").astype("Int64")
        manual["link_start"] = to_date(manual["link_start"])
        manual["link_end"] = to_date(manual["link_end"])
        manual = manual[
            ["gvkey", "company_name", "current_ticker", "PERMNO", "linkprim", "linktype", "link_start", "link_end"]
        ]
        links = pd.concat([links, manual], ignore_index=True)
    return links


def map_links(crsp: pd.DataFrame, links: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = crsp.merge(links, on="PERMNO", how="left", indicator=True)
    valid = (
        merged["date"].ge(merged["link_start"])
        & merged["date"].le(merged["link_end"])
        & merged["linktype"].isin(["LC", "LU", "MANUAL_SEC"])
    )
    mapped = merged[valid].copy()
    mapped = mapped.sort_values(
        ["PERMNO", "date", "source_priority", "linkprim"],
        ascending=[True, True, False, True],
    ).drop_duplicates(["PERMNO", "date"], keep="first")
    matched_keys = mapped[["PERMNO", "date"]].drop_duplicates().assign(linked=True)
    unmatched = crsp.merge(matched_keys, on=["PERMNO", "date"], how="left")
    unmatched = unmatched[unmatched["linked"].isna()].drop(columns="linked")
    return mapped, unmatched


def build_fundamentals() -> pd.DataFrame:
    comp = read_xlsx("compustat_quarterly_banks_bdc_2010_2025.xlsx").copy()
    rename = {
        "(tic) Ticker Symbol": "comp_ticker",
        "(datadate) Data Date": "datadate",
        "(gvkey) Global Company Key": "gvkey",
        "(conm) Company Name": "comp_company_name",
        "(rdq) Report Date of Quarterly Earnings": "rdq",
        "(atq) Assets - Total": "asset_mn",
        "(ceqq) Common/Ordinary Equity - Total": "book_equity_mn",
        "(ltq) Liabilities - Total": "liabilities_mn",
        "(dlcq) Debt in Current Liabilities": "short_debt_mn",
        "(dlttq) Long-Term Debt - Total": "long_debt_mn",
        "(pstkq) Preferred/Preference Stock (Capital) - Total": "preferred_mn",
    }
    comp = comp.rename(columns=rename)
    comp["gvkey"] = comp["gvkey"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
    comp["datadate"] = to_date(comp["datadate"])
    comp["rdq"] = to_date(comp["rdq"])
    comp["available_date"] = comp["rdq"].fillna(comp["datadate"] + pd.Timedelta(days=90))
    for col in [
        "asset_mn",
        "book_equity_mn",
        "liabilities_mn",
        "short_debt_mn",
        "long_debt_mn",
        "preferred_mn",
    ]:
        comp[col] = pd.to_numeric(comp[col], errors="coerce")
    fallback_equity = comp["asset_mn"] - comp["liabilities_mn"]
    comp["book_equity_mn"] = comp["book_equity_mn"].fillna(fallback_equity)
    comp["debt_mn"] = comp["asset_mn"] - comp["book_equity_mn"]
    comp = comp.sort_values(["gvkey", "available_date", "datadate"]).drop_duplicates(
        ["gvkey", "available_date"], keep="last"
    )
    comp = comp[
        [
            "gvkey",
            "comp_ticker",
            "comp_company_name",
            "datadate",
            "rdq",
            "available_date",
            "asset_mn",
            "book_equity_mn",
            "liabilities_mn",
            "debt_mn",
            "short_debt_mn",
            "long_debt_mn",
            "preferred_mn",
        ]
    ]
    bk_path = UPLOAD / "bk_sec_quarterly_fundamentals_2010_2025.csv"
    if bk_path.exists():
        bk = pd.read_csv(
            bk_path,
            dtype={"gvkey": str},
            parse_dates=["datadate", "rdq", "available_date"],
        )
        for column in ["short_debt_mn", "long_debt_mn", "preferred_mn"]:
            bk[column] = np.nan
        bk = bk[comp.columns]
        comp = pd.concat([comp, bk], ignore_index=True)
    return comp


def merge_fundamentals(panel: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for gvkey, daily in panel.groupby("gvkey", sort=False):
        q = fundamentals[fundamentals["gvkey"].eq(gvkey)].sort_values("available_date")
        daily = daily.sort_values("date")
        if q.empty:
            out = daily.copy()
            for col in fundamentals.columns:
                if col not in out.columns:
                    out[col] = np.nan
        else:
            out = pd.merge_asof(
                daily,
                q.drop(columns="gvkey"),
                left_on="date",
                right_on="available_date",
                direction="backward",
                allow_exact_matches=True,
            )
            out["gvkey"] = gvkey
        pieces.append(out)
    out = pd.concat(pieces, ignore_index=True)
    out["fundamental_age_days"] = (out["date"] - out["available_date"]).dt.days
    stale = out["fundamental_age_days"].gt(180)
    fundamental_cols = [
        "asset_mn",
        "book_equity_mn",
        "liabilities_mn",
        "debt_mn",
        "short_debt_mn",
        "long_debt_mn",
        "preferred_mn",
    ]
    out.loc[stale, fundamental_cols] = np.nan
    return out


def build_institution_panel(factor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    files = [
        ("crsp_us_banks_daily_2010_2025.xlsx", "crsp_us_banks", 3),
        ("crsp_bdc_daily_2010_2025.xlsx", "crsp_bdc_current", 3),
        ("crsp_bdc_old_tickers_daily_2010_2025.xlsx", "crsp_bdc_old", 2),
    ]
    raw = pd.concat(
        [normalize_crsp(read_xlsx(name), source, priority) for name, source, priority in files],
        ignore_index=True,
    )
    raw = raw.dropna(subset=["PERMNO", "date"])
    raw["PERMNO"] = pd.to_numeric(raw["PERMNO"], errors="coerce").astype("Int64")
    before_key_dedup = len(raw)
    raw = raw.sort_values(
        ["PERMNO", "date", "source_priority"], ascending=[True, True, False]
    ).drop_duplicates(["PERMNO", "date"], keep="first")

    links = build_link_table()
    mapped, unmatched = map_links(raw, links)
    panel = mapped.merge(
        factor[["date", "logret_spy", "ret_climate", "coal_leg_source"]],
        on="date",
        how="left",
    )
    panel["mktcap_mn"] = panel["mktcap_thousand"] / 1000.0
    panel["memo"] = panel["current_ticker"].astype(str) + ":US"
    panel["id"] = pd.factorize(panel["memo"], sort=True)[0] + 1
    panel = merge_fundamentals(panel, build_fundamentals())
    panel = panel.sort_values(["id", "date"])
    panel["ret_spy_lag"] = panel.groupby("id")["logret_spy"].shift(1)
    panel["ret_climate_lag"] = panel.groupby("id")["ret_climate"].shift(1)

    output_cols = [
        "id",
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
        "ret_spy_lag",
        "ret_climate_lag",
        "mktcap_mn",
        "asset_mn",
        "book_equity_mn",
        "debt_mn",
        "datadate",
        "available_date",
        "fundamental_age_days",
        "coal_leg_source",
        "source_file",
    ]
    panel = panel[output_cols]

    qc = (
        panel.groupby(
            ["id", "memo", "current_ticker", "company_name", "gvkey"], as_index=False
        )
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            rows=("date", "size"),
            return_rows=("ret", "count"),
            factor_rows=("ret_climate", "count"),
            fundamental_rows=("asset_mn", "count"),
            permnos=("PERMNO", "nunique"),
            historical_tickers=(
                "historical_ticker",
                lambda x: ", ".join(sorted(set(x.dropna().astype(str)))),
            ),
        )
        .sort_values("id")
    )
    qc["fundamental_coverage"] = qc["fundamental_rows"] / qc["rows"]

    audit = {
        "institution_raw_rows_after_exact_dedup": int(before_key_dedup),
        "institution_key_dedup_rows": int(len(raw)),
        "institution_duplicate_permno_date_removed": int(before_key_dedup - len(raw)),
        "linked_panel_rows": int(len(panel)),
        "linked_institutions": int(panel["memo"].nunique()),
        "unmatched_rows": int(len(unmatched)),
        "unmatched_permnos": [int(x) for x in sorted(unmatched["PERMNO"].dropna().unique())],
        "panel_start": str(panel["date"].min().date()),
        "panel_end": str(panel["date"].max().date()),
        "return_nonmissing": int(panel["ret"].notna().sum()),
        "factor_nonmissing": int(panel["ret_climate"].notna().sum()),
        "fundamental_nonmissing": int(panel["asset_mn"].notna().sum()),
    }
    return panel, qc, audit


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    factor, factor_audit = build_climate_factor()
    panel, institution_qc, panel_audit = build_institution_panel(factor)

    factor.to_csv(OUTPUT / "climate_factor_daily_2010_2025.csv", index=False)
    panel.to_csv(OUTPUT / "dcb_input_panel_2010_2025.csv", index=False)
    institution_qc.to_csv(OUTPUT / "institution_qc.csv", index=False)
    audit = {"factor": factor_audit, "panel": panel_audit}
    (OUTPUT / "processing_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("\nInstitution QC")
    print(institution_qc.to_string(index=False))


if __name__ == "__main__":
    main()
