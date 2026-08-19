from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CODE = Path(__file__).resolve().parent


def run(script: str) -> None:
    subprocess.run([sys.executable, str(CODE / script)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the bank CRISK module.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Rebuild processed data and estimates from the licensed raw inputs.",
    )
    args = parser.parse_args()

    if args.full:
        for script in (
            "00_download_public_market_series.py",
            "01a_download_bk_sec_fundamentals.py",
            "01_build_crisk_inputs.py",
            "02_estimate_dcb_crisk.py",
            "03_export_stata_data.py",
            "04_validate_bank_replication.py",
        ):
            run(script)
    run("05_make_results.py")


if __name__ == "__main__":
    main()
