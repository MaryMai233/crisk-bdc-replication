from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CODE = Path(__file__).resolve().parent


def run(script: str, *arguments: str) -> None:
    subprocess.run([sys.executable, str(CODE / script), *arguments], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the BDC investment-exposure and climate-beta module.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Re-extract SEC portfolio exposures and re-estimate all models.",
    )
    args = parser.parse_args()

    if args.full:
        run("00_build_sic3_industry_climate_betas.py")
        run("00b_build_ff49_industry_climate_betas.py")
        run("00c_build_ff49_dcc_industry_betas.py")
        run("01_update_market_panel_from_daily.py")
        run("02_extract_bdc_portfolio_exposure.py")
        run("03_complete_portfolio_fallbacks.py")
        run("04_estimate_brown_exposure_models.py")
        run("04b_estimate_ff49_portfolio_mechanism.py")
        run("04c_estimate_ff49_dcc_portfolio_mechanism.py")
        run("06_high_yield_credit_factor.py")
        run("07_audit_industry_dictionary.py")
        run("08_inference_and_classification_robustness.py")
        run("09_factor_continuation_frequency_robustness.py", "--reestimate-weekly")
    else:
        run("04c_estimate_ff49_dcc_portfolio_mechanism.py")
    run("05_make_results.py")
    run("11_make_ff49_dcc_results.py")
    if not args.full:
        run("09_factor_continuation_frequency_robustness.py")
    run("10_descriptive_statistics.py")


if __name__ == "__main__":
    main()
