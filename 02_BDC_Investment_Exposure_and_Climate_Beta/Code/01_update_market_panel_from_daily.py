from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
RAW = ROOT / "Data" / "Raw"
PROCESSED = ROOT / "Data" / "Processed"


def export_dta(frame: pd.DataFrame, path: Path) -> None:
    out = frame.copy()
    for column in out.select_dtypes(include="bool"):
        out[column] = out[column].astype("int8")
    for column in out.select_dtypes(include=["object"]):
        out[column] = out[column].fillna("").astype(str)
    out.to_stata(path, write_index=False, version=118)


def main() -> None:
    panel = pd.read_csv(
        RAW / "bdc20_quarter_market_financial_panel_2021_2025.csv",
        parse_dates=["datadate", "report_month_start", "report_month_end"],
    )
    daily_path = RAW / "dcb_crisk_daily_2010_2025.csv"
    if not daily_path.exists():
        daily_path = (
            PACKAGE / "01_Bank_CRISK_Replication" / "Data" / "Processed"
            / "dcb_crisk_daily_2010_2025.csv"
        )
    daily = pd.read_csv(
        daily_path,
        parse_dates=["date"],
        low_memory=False,
    )
    daily = daily[daily["group"].eq("BDC") & daily["current_ticker"].isin(panel["ticker"])].copy()
    records = []
    for row in panel.itertuples(index=False):
        sample = daily[
            daily["current_ticker"].eq(row.ticker)
            & daily["date"].between(row.report_month_start, row.report_month_end)
        ]
        records.append(
            {
                "ticker": row.ticker,
                "datadate": row.datadate,
                "beta_climate_equity_report_month_new": sample["beta_climate"].mean(),
                "beta_market_report_month_new": sample["beta_market"].mean(),
                "market_equity_report_month_mn_new": sample["mktcap_mn"].mean(),
                "beta_daily_observations_new": sample["beta_climate"].count(),
                "beta_month_first_date_new": sample["date"].min(),
                "beta_month_last_date_new": sample["date"].max(),
            }
        )
    refreshed = pd.DataFrame(records)
    panel = panel.merge(refreshed, on=["ticker", "datadate"], how="left", validate="one_to_one")
    replacement = {
        "beta_climate_equity_report_month": "beta_climate_equity_report_month_new",
        "beta_market_report_month": "beta_market_report_month_new",
        "market_equity_report_month_mn": "market_equity_report_month_mn_new",
        "beta_daily_observations": "beta_daily_observations_new",
        "beta_month_first_date": "beta_month_first_date_new",
        "beta_month_last_date": "beta_month_last_date_new",
    }
    for target, source in replacement.items():
        panel[target] = panel[source]
        panel = panel.drop(columns=source)
    panel["beta_climate_asset_report_month"] = (
        panel["beta_climate_equity_report_month"]
        * panel["market_equity_report_month_mn"]
        / (panel["market_equity_report_month_mn"] + panel["debt_total_mn"])
    )
    panel["market_panel_version"] = "complete-10-bank median-DCC re-estimation"
    panel["market_panel_beta_nonmissing"] = panel["beta_climate_equity_report_month"].notna().astype(int)
    if panel["market_panel_beta_nonmissing"].sum() < 395:
        raise RuntimeError("Updated BDC beta coverage is unexpectedly incomplete")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    output = PROCESSED / "bdc20_quarter_market_financial_panel_2021_2025_updated.csv"
    panel.to_csv(output, index=False)
    export_dta(panel, output.with_suffix(".dta"))
    comparison = pd.DataFrame(
        [{
            "rows": len(panel),
            "firms": panel["ticker"].nunique(),
            "new_beta_nonmissing": panel["beta_climate_equity_report_month"].notna().sum(),
            "mean_equity_beta": panel["beta_climate_equity_report_month"].mean(),
            "mean_asset_beta": panel["beta_climate_asset_report_month"].mean(),
        }]
    )
    comparison.to_csv(PROCESSED / "updated_market_panel_audit.csv", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
