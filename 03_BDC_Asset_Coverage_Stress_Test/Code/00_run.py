from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "Code"
RAW = ROOT / "Data" / "Raw"
PROCESSED = ROOT / "Data" / "Processed"


def run(script: str, *arguments: str) -> None:
    subprocess.run([sys.executable, str(CODE / script), *arguments], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the BDC asset-coverage stress module.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Re-extract SEC asset coverage and re-estimate all stress scenarios.",
    )
    args = parser.parse_args()

    if args.full:
        run("00_refresh_market_inputs.py")
        run(
            "01_extract_sec_coverage.py",
            "--zip", str(RAW / "SEC_BDC_Filings_2021_2025.zip"),
            "--outdir", str(PROCESSED),
        )
        run(
            "02_estimate_coverage_stress.py",
            "--market-panel", str(RAW / "bdc20_quarter_market_financial_panel_2021_2025.csv"),
            "--daily-crisk", str(RAW / "dcb_crisk_bdc19_daily_2021_2025.csv"),
            "--sec-coverage", str(PROCESSED / "sec_asset_coverage_extracted.csv"),
            "--outdir", str(PROCESSED),
        )
    run("03_make_results.py")


if __name__ == "__main__":
    main()
