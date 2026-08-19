from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data" / "Raw"
CIK = "0001390777"
COMPANY_FACTS_URL = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json"


def download_company_facts() -> dict:
    request = urllib.request.Request(
        COMPANY_FACTS_URL,
        headers={"User-Agent": "CRISK academic replication"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def expected_fiscal_period(end: pd.Timestamp) -> str:
    return {3: "Q1", 6: "Q2", 9: "Q3", 12: "FY"}[end.month]


def select_quarterly_fact(company_facts: dict, tag: str) -> pd.DataFrame:
    observations = company_facts["facts"]["us-gaap"][tag]["units"]["USD"]
    frame = pd.DataFrame(observations)
    frame["end"] = pd.to_datetime(frame["end"], errors="coerce")
    frame["filed"] = pd.to_datetime(frame["filed"], errors="coerce")
    frame["fy"] = pd.to_numeric(frame["fy"], errors="coerce")
    frame = frame[
        frame["form"].isin(["10-Q", "10-K"])
        & frame["end"].between("2010-01-01", "2025-12-31")
        & frame["end"].dt.month.isin([3, 6, 9, 12])
    ].copy()
    frame["expected_fp"] = frame["end"].map(expected_fiscal_period)
    preferred = frame[
        frame["fy"].eq(frame["end"].dt.year)
        & frame["fp"].eq(frame["expected_fp"])
    ].copy()
    preferred = preferred.sort_values(["end", "filed", "accn"]).drop_duplicates(
        "end", keep="first"
    )
    return preferred[["end", "filed", "val", "accn", "form", "fy", "fp"]].rename(
        columns={
            "filed": f"{tag}_filed",
            "val": tag,
            "accn": f"{tag}_accession",
            "form": f"{tag}_form",
            "fy": f"{tag}_fy",
            "fp": f"{tag}_fp",
        }
    )


def build_bk_quarterly(company_facts: dict) -> pd.DataFrame:
    assets = select_quarterly_fact(company_facts, "Assets")
    equity = select_quarterly_fact(company_facts, "StockholdersEquity")
    data = assets.merge(equity, on="end", how="outer", validate="one_to_one")
    data["available_date"] = data[
        ["Assets_filed", "StockholdersEquity_filed"]
    ].max(axis=1)
    data["asset_mn"] = pd.to_numeric(data["Assets"], errors="coerce") / 1_000_000
    data["book_equity_mn"] = (
        pd.to_numeric(data["StockholdersEquity"], errors="coerce") / 1_000_000
    )
    data["liabilities_mn"] = data["asset_mn"] - data["book_equity_mn"]
    data["debt_mn"] = data["liabilities_mn"]
    data["gvkey"] = "SEC1390777"
    data["comp_ticker"] = "BK"
    data["comp_company_name"] = "THE BANK OF NEW YORK MELLON CORPORATION"
    data["datadate"] = data["end"]
    data["rdq"] = data["available_date"]
    data["source"] = "SEC Company Facts; CIK 0001390777"
    data["source_url"] = COMPANY_FACTS_URL
    data = data.dropna(subset=["asset_mn", "book_equity_mn", "available_date"])
    columns = [
        "gvkey", "comp_ticker", "comp_company_name", "datadate", "rdq",
        "available_date", "asset_mn", "book_equity_mn", "liabilities_mn",
        "debt_mn", "source", "source_url", "Assets_accession",
        "StockholdersEquity_accession",
    ]
    return data[columns].sort_values("datadate").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover BK fundamentals from SEC Company Facts and document the manual CRSP link."
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    output = RAW / "bk_sec_quarterly_fundamentals_2010_2025.csv"
    link_output = RAW / "bk_manual_identifier_link.csv"
    if args.refresh or not output.exists():
        facts = download_company_facts()
        quarterly = build_bk_quarterly(facts)
        quarterly.to_csv(output, index=False)
    else:
        quarterly = pd.read_csv(output)
    link = pd.DataFrame(
        [{
            "gvkey": "SEC1390777",
            "company_name": "THE BANK OF NEW YORK MELLON CORPORATION",
            "current_ticker": "BK",
            "PERMNO": 49656,
            "PERMCO": 20265,
            "header_cusip8": "06405810",
            "cik": "0001390777",
            "linkprim": "P",
            "linktype": "MANUAL_SEC",
            "link_start": "2010-01-01",
            "link_end": "2025-12-31",
            "verification": "CRSP ticker/name/CUSIP and SEC registrant ticker/CIK",
        }]
    )
    link.to_csv(link_output, index=False)
    if len(quarterly) < 60:
        raise RuntimeError(f"Expected approximately 64 BK quarters, found {len(quarterly)}")
    print(f"BK quarterly fundamentals: {len(quarterly)} rows")
    print(link.to_string(index=False))


if __name__ == "__main__":
    main()
