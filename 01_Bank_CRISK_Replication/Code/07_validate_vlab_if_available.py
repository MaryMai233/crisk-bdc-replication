"""Optional validation against a user-downloaded V-Lab daily CRISK file.

V-Lab requires an authenticated download. This script never fabricates a
comparison: it exits with instructions when the optional file is absent.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data" / "Raw"
PROCESSED = ROOT / "Data" / "Processed"
INPUT = RAW / "vlab_crisk_daily.csv"
OUTPUT = PROCESSED / "vlab_daily_validation.csv"
AUDIT = PROCESSED / "Audit" / "vlab_validation.json"


def main() -> None:
    if not INPUT.exists():
        message = {
            "status": "NOT_RUN",
            "reason": "V-Lab historical downloads require authentication.",
            "expected_file": str(INPUT.relative_to(ROOT)),
            "required_columns": ["date", "ticker", "crisk"],
            "download_page": "https://vlab.stern.nyu.edu/climate/CLIM.WORLDFIN-MR.CMES",
        }
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        AUDIT.write_text(json.dumps(message, indent=2), encoding="utf-8")
        print(json.dumps(message, indent=2))
        return

    vlab = pd.read_csv(INPUT)
    required = {"date", "ticker", "crisk"}
    missing = required.difference(vlab.columns)
    if missing:
        raise ValueError(f"V-Lab file is missing columns: {sorted(missing)}")
    vlab["date"] = pd.to_datetime(vlab["date"])
    vlab["ticker"] = vlab["ticker"].astype(str).str.upper().str.strip()
    vlab["crisk"] = pd.to_numeric(vlab["crisk"], errors="coerce")

    ours = pd.read_csv(PROCESSED / "bank_h1_daily_2010_2025.csv", parse_dates=["date"])
    ours = ours.rename(columns={"current_ticker": "ticker", "crisk_8pct_mn": "crisk_replication"})
    joined = ours[["date", "ticker", "crisk_replication"]].merge(
        vlab[["date", "ticker", "crisk"]], on=["date", "ticker"], how="inner"
    ).dropna()
    if joined.empty:
        raise ValueError("No overlapping date-ticker observations were found.")

    rows = []
    for ticker, group in joined.groupby("ticker"):
        error = group["crisk_replication"] - group["crisk"]
        rows.append({
            "ticker": ticker,
            "n": len(group),
            "date_start": group["date"].min(),
            "date_end": group["date"].max(),
            "correlation": group["crisk_replication"].corr(group["crisk"]),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mean_error": float(error.mean()),
        })
    result = pd.DataFrame(rows).sort_values("ticker")
    result.to_csv(OUTPUT, index=False)
    AUDIT.write_text(json.dumps({
        "status": "COMPLETED",
        "matched_rows": int(len(joined)),
        "institutions": int(result["ticker"].nunique()),
        "output": str(OUTPUT.relative_to(ROOT)),
    }, indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
