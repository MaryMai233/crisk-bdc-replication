from __future__ import annotations

import datetime as dt
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data" / "Raw"
PROCESSED = ROOT / "Data" / "Processed"
AUDIT = PROCESSED / "Audit"
PUBLIC = RAW / "public_market_series_kol_top75_yahoo_2019_2025.csv"
HOLDINGS_OUT = RAW / "kol_sec_nport_2020q3_cumulative_75_holdings.csv"
FACTOR_OUT = PROCESSED / "climate_factor_daily_kol_top75_2010_2025.csv"
DIAGNOSTICS_OUT = PROCESSED / "kol_top75_tracking_diagnostics.csv"

START = dt.datetime(2019, 1, 1, tzinfo=dt.timezone.utc)
END = dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc)
CUTOFF = pd.Timestamp("2020-12-14")
SEC_SOURCE = (
    "https://www.sec.gov/Archives/edgar/data/1137360/"
    "000175272420245726/NPORT_247470289678245.htm"
)

# Official KOL common-stock market values in VanEck's N-PORT schedule dated
# 30 September 2020.  Securities are included in descending market-value order
# until cumulative coverage first exceeds 75 percent of the common-stock total.
HOLDINGS = [
    ("TECK", "Teck Resources", 2_623_182, None),
    ("AZJ.AX", "Aurizon Holdings", 2_193_239, "AUDUSD=X"),
    ("1088.HK", "China Shenhua Energy H", 2_034_296, "HKDUSD=X"),
    ("SOL.AX", "Washington H. Soul Pattinson", 1_858_279, "AUDUSD=X"),
    ("UNTR.JK", "United Tractors", 1_802_851, "IDRUSD=X"),
    ("ADRO.JK", "Adaro Energy / Alamtri Resources", 1_439_350, "IDRUSD=X"),
    ("HCC", "Warrior Met Coal", 1_304_809, None),
    ("EXX.JO", "Exxaro Resources", 1_272_404, "ZARUSD=X"),
    ("BANPU.BK", "Banpu", 1_252_630, "THBUSD=X"),
    ("1171.HK", "Yankuang Energy / Yanzhou Coal", 1_230_055, "HKDUSD=X"),
    ("1898.HK", "China Coal Energy H", 1_219_063, "HKDUSD=X"),
    ("PTBA.JK", "Bukit Asam", 1_203_067, "IDRUSD=X"),
    ("0639.HK", "Shougang Fushan Resources", 948_170, "HKDUSD=X"),
    ("WTE.TO", "Westshore Terminals", 934_474, "CADUSD=X"),
    ("NHC.AX", "New Hope Corporation", 922_313, "AUDUSD=X"),
]
COMMON_STOCK_TOTAL = 28_859_195


def chart_url(symbol: str) -> str:
    query = urllib.parse.urlencode(
        {
            "period1": int(START.timestamp()),
            "period2": int(END.timestamp()),
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
    )
    return (
        "https://query2.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(symbol, safe='')}?{query}"
    )


def download_symbol(symbol: str, security_name: str) -> pd.DataFrame:
    url = chart_url(symbol)
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 academic CRISK replication"}
    )
    last_error: Exception | None = None
    for attempt in range(7):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read())
            result = payload["chart"]["result"][0]
            quote = result["indicators"]["quote"][0]
            adjusted = result["indicators"].get("adjclose", [{}])[0].get(
                "adjclose", quote.get("close")
            )
            frame = pd.DataFrame(
                {
                    "date": pd.to_datetime(result["timestamp"], unit="s", utc=True)
                    .tz_convert(None)
                    .normalize(),
                    "symbol": symbol,
                    "security_name": security_name,
                    "adjusted_close": adjusted,
                    "currency": result["meta"].get("currency"),
                    "exchange": result["meta"].get("exchangeName"),
                    "source_url": url,
                }
            )
            return frame.dropna(subset=["date", "adjusted_close"])
        except Exception as error:  # pragma: no cover - network retry path
            last_error = error
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Unable to download {symbol}: {last_error}")


def download_prices(refresh: bool = False) -> pd.DataFrame:
    names = {symbol: name for symbol, name, _, _ in HOLDINGS}
    for _, _, _, fx in HOLDINGS:
        if fx is not None:
            names[fx] = f"Foreign-exchange conversion series {fx}"
    if PUBLIC.exists() and not refresh:
        cached = pd.read_csv(PUBLIC, parse_dates=["date"])
        if set(names).issubset(set(cached["symbol"].unique())):
            return cached
    frames: list[pd.DataFrame] = []
    for counter, (symbol, name) in enumerate(names.items(), start=1):
        frame = download_symbol(symbol, name)
        frames.append(frame)
        print(f"[{counter:02d}/{len(names):02d}] {symbol}: {len(frame):,} observations", flush=True)
        time.sleep(0.4)
    output = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    output.to_csv(PUBLIC, index=False)
    return output


def weighted_available(frame: pd.DataFrame, weights: pd.Series) -> tuple[pd.Series, pd.Series]:
    availability = frame.notna().mul(weights, axis=1)
    denominator = availability.sum(axis=1).replace(0, np.nan)
    weighted = frame.mul(weights, axis=1).sum(axis=1, min_count=1) / denominator
    return weighted, denominator


def tracking_rows(pair: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, rule in [("Daily", None), ("Weekly", "W-FRI"), ("Monthly", "ME")]:
        sample = pair if rule is None else pair.resample(rule).sum(min_count=1).dropna()
        error = sample["top75_usd_logret"] - sample["logret_kol"]
        rows.append(
            {
                "frequency": label,
                "observations": int(len(sample)),
                "correlation_with_kol": float(sample["top75_usd_logret"].corr(sample["logret_kol"])),
                "tracking_error_std": float(error.std(ddof=1)),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "proxy_to_kol_volatility_ratio": float(
                    sample["top75_usd_logret"].std(ddof=1) / sample["logret_kol"].std(ddof=1)
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)

    holdings = pd.DataFrame(
        HOLDINGS, columns=["symbol", "security_name", "market_value_usd", "fx_symbol"]
    )
    holdings["weight_in_kol_common_stocks"] = holdings["market_value_usd"] / COMMON_STOCK_TOTAL
    holdings["cumulative_weight_in_kol_common_stocks"] = holdings[
        "weight_in_kol_common_stocks"
    ].cumsum()
    holdings["normalized_basket_weight"] = holdings["market_value_usd"] / holdings[
        "market_value_usd"
    ].sum()
    holdings["schedule_date"] = "2020-09-30"
    holdings["source_url"] = SEC_SOURCE
    holdings.to_csv(HOLDINGS_OUT, index=False)

    public = download_prices()
    prices = public.pivot(index="date", columns="symbol", values="adjusted_close").sort_index()
    baseline = pd.read_csv(
        PROCESSED / "climate_factor_daily_2010_2025.csv", parse_dates=["date"]
    ).sort_values("date")
    master_dates = pd.DatetimeIndex(baseline["date"])
    union_index = prices.index.union(master_dates).sort_values()

    local_returns = pd.DataFrame(index=master_dates)
    usd_returns = pd.DataFrame(index=master_dates)
    for item in holdings.itertuples(index=False):
        local_price = prices[item.symbol].reindex(union_index).ffill().reindex(master_dates)
        local_returns[item.symbol] = np.log(local_price).diff()
        if pd.isna(item.fx_symbol) or item.fx_symbol in (None, ""):
            usd_price = local_price
        else:
            fx_price = prices[item.fx_symbol].reindex(union_index).ffill().reindex(master_dates)
            usd_price = local_price * fx_price
        usd_returns[item.symbol] = np.log(usd_price).diff()

    weights = holdings.set_index("symbol")["normalized_basket_weight"]
    top75_local, local_weight = weighted_available(local_returns, weights)
    top75_usd, usd_weight = weighted_available(usd_returns, weights)
    proxy = pd.DataFrame(
        {
            "date": master_dates,
            "top75_local_logret": top75_local.to_numpy(),
            "top75_usd_logret": top75_usd.to_numpy(),
            "top75_available_weight": usd_weight.to_numpy(),
            "top75_available_names": usd_returns.notna().sum(axis=1).to_numpy(),
        }
    )
    factor = baseline.drop(columns=["ret_climate", "coal_leg_logret", "coal_leg_source"]).merge(
        proxy, on="date", how="left", validate="one_to_one"
    )
    factor["coal_leg_logret"] = np.where(
        factor["date"].le(CUTOFF), factor["logret_kol"], factor["top75_usd_logret"]
    )
    factor["coal_leg_source"] = np.where(
        factor["date"].le(CUTOFF),
        "KOL ETF",
        "SEC N-PORT 2020Q3 cumulative-75-percent basket",
    )
    factor["ret_climate"] = (
        0.3 * factor["logret_xle"]
        + 0.7 * factor["coal_leg_logret"]
        - factor["logret_spy"]
    )
    factor.to_csv(FACTOR_OUT, index=False)

    pair = factor.loc[
        factor["date"].between("2019-01-01", CUTOFF),
        ["date", "logret_kol", "top75_usd_logret"],
    ].dropna().set_index("date")
    diagnostics = tracking_rows(pair)
    diagnostics["overlap_start"] = str(pair.index.min().date())
    diagnostics["overlap_end"] = str(pair.index.max().date())
    diagnostics["basket_names"] = int(len(holdings))
    diagnostics["portfolio_coverage"] = float(
        holdings["weight_in_kol_common_stocks"].sum()
    )
    diagnostics.to_csv(DIAGNOSTICS_OUT, index=False)

    audit = {
        "status": "PASS" if factor.loc[factor["date"].gt(CUTOFF), "ret_climate"].notna().all() else "REVIEW",
        "source": SEC_SOURCE,
        "source_schedule_date": "2020-09-30",
        "selection_rule": "Descending N-PORT market value until cumulative common-stock coverage first exceeds 75 percent",
        "basket_names": int(len(holdings)),
        "coverage_of_kol_common_stocks": float(holdings["weight_in_kol_common_stocks"].sum()),
        "first_post_liquidation_date": str(factor.loc[factor["date"].gt(CUTOFF), "date"].min().date()),
        "minimum_post_liquidation_available_weight": float(
            factor.loc[factor["date"].gt(CUTOFF), "top75_available_weight"].min()
        ),
        "minimum_post_liquidation_available_names": int(
            factor.loc[factor["date"].gt(CUTOFF), "top75_available_names"].min()
        ),
        "tracking": diagnostics.to_dict(orient="records"),
        "interpretation_rule": (
            "The published top-five continuation remains the strict replication. "
            "This cumulative-75-percent basket is a pre-specified breadth robustness and is not selected from BDC outcomes."
        ),
    }
    (AUDIT / "kol_top75_continuation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(diagnostics.to_string(index=False), flush=True)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
