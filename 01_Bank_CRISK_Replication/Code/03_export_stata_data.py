from __future__ import annotations

from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

FILES = [
    (PACKAGE_ROOT / "Data/Processed/climate_factor_daily_2010_2025.csv", ["date"]),
    (PACKAGE_ROOT / "Data/Processed/coal_proxy_diagnostics.csv", []),
    (PACKAGE_ROOT / "Data/Processed/dcb_input_panel_2010_2025.csv", ["date", "datadate", "available_date"]),
    (PACKAGE_ROOT / "Data/Processed/institution_qc.csv", ["start_date", "end_date"]),
    (PACKAGE_ROOT / "Data/Processed/dcb_crisk_annual_summary.csv", []),
    (PACKAGE_ROOT / "Data/Processed/dcc_gjr_parameters.csv", []),
    (PACKAGE_ROOT / "Data/Processed/dcb_crisk_daily_2010_2025.csv", ["date"]),
]


def export_one(csv_path: Path, date_cols: list[str]) -> None:
    dtype = {"gvkey": "string"} if "gvkey" in pd.read_csv(csv_path, nrows=0).columns else None
    df = pd.read_csv(csv_path, dtype=dtype, low_memory=False)
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    dta_path = csv_path.with_suffix(".dta")
    df.to_stata(dta_path, write_index=False, version=118)
    check = pd.read_stata(dta_path)
    if len(check) != len(df) or list(check.columns) != list(df.columns):
        raise RuntimeError(f"DTA round-trip check failed: {dta_path}")
    print(f"Wrote {dta_path.relative_to(PACKAGE_ROOT)} ({len(df):,} rows)")


def main() -> None:
    for csv_path, date_cols in FILES:
        export_one(csv_path, date_cols)


if __name__ == "__main__":
    main()
