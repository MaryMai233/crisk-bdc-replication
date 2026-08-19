from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
RAW = ROOT / "Data" / "Raw"


def export_dta(frame: pd.DataFrame, path: Path) -> None:
    out = frame.copy()
    for column in out.select_dtypes(include="bool"):
        out[column] = out[column].astype("int8")
    for column in out.select_dtypes(include=["object"]):
        out[column] = out[column].fillna("").astype(str)
    out.to_stata(path, write_index=False, version=118)


def main() -> None:
    updated_panel_path = (
        PACKAGE / "02_BDC_Investment_Exposure_and_Climate_Beta/Data/Processed/"
        "bdc20_quarter_market_financial_panel_2021_2025_updated.csv"
    )
    daily_path = (
        PACKAGE / "01_Bank_CRISK_Replication/Data/Processed/"
        "dcb_crisk_daily_2010_2025.csv"
    )
    panel = pd.read_csv(updated_panel_path)
    daily = pd.read_csv(daily_path, low_memory=False)
    tickers = sorted(panel["ticker"].dropna().unique())
    daily = daily[
        daily["group"].eq("BDC")
        & daily["current_ticker"].isin(tickers)
        & pd.to_datetime(daily["date"]).between("2021-01-01", "2025-12-31")
    ].copy()
    panel.to_csv(RAW / "bdc20_quarter_market_financial_panel_2021_2025.csv", index=False)
    daily.to_csv(RAW / "dcb_crisk_bdc19_daily_2021_2025.csv", index=False)
    export_dta(daily, RAW / "dcb_crisk_bdc19_daily_2021_2025.dta")
    print(f"Updated market panel: {len(panel)} rows; daily DCB input: {len(daily)} rows")


if __name__ == "__main__":
    main()
