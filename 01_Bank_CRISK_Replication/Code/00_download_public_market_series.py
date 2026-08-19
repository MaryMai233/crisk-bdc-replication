from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data" / "Raw"
OUTPUT = RAW / "public_market_series_yahoo_2010_2025.csv"
START = dt.datetime(2010, 1, 1, tzinfo=dt.timezone.utc)
END = dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc)

# Five largest KOL constituents immediately before liquidation.  The weights
# are the last portfolio weights reported by ETFdb and are normalized in the
# factor-construction script because the five securities sum to 30.64%.
SYMBOLS = {
    "SOL.AX": "Washington H. Soul Pattinson",
    "AZJ.AX": "Aurizon Holdings",
    "UNTR.JK": "United Tractors",
    "1088.HK": "China Shenhua Energy H",
    "ADRO.JK": "Adaro Energy / Alamtri Resources",
    "AUDUSD=X": "AUD per USD conversion quote (USD per AUD)",
    "IDRUSD=X": "IDR per USD conversion quote (USD per IDR)",
    "HKDUSD=X": "HKD per USD conversion quote (USD per HKD)",
    "HYG": "iShares iBoxx High Yield Corporate Bond ETF",
    "JNK": "SPDR Bloomberg High Yield Bond ETF",
    "SHY": "iShares 1-3 Year Treasury Bond ETF",
}


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
    safe_symbol = urllib.parse.quote(symbol, safe="")
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{safe_symbol}?{query}"


def download_symbol(symbol: str) -> pd.DataFrame:
    url = chart_url(symbol)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 CRISK academic replication"},
    )
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
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
                    "security_name": SYMBOLS[symbol],
                    "close": quote.get("close"),
                    "adjusted_close": adjusted,
                    "currency": result["meta"].get("currency"),
                    "exchange": result["meta"].get("exchangeName"),
                }
            )
            frame["source_url"] = url
            return frame.dropna(subset=["date", "adjusted_close"])
        except Exception as error:  # pragma: no cover - network retry path
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(f"Unable to download {symbol}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if OUTPUT.exists() and not args.refresh:
        print(f"Using cached public market series: {OUTPUT}")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    frames = []
    for symbol in SYMBOLS:
        frame = download_symbol(symbol)
        frames.append(frame)
        print(f"{symbol}: {len(frame):,} daily observations", flush=True)
    output = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    output.to_csv(OUTPUT, index=False)
    print(f"Saved {len(output):,} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
