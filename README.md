# Climate Risk and Business Development Companies

[![Repository audit](https://github.com/MaryMai233/crisk-bdc-replication/actions/workflows/ci.yml/badge.svg)](https://github.com/MaryMai233/crisk-bdc-replication/actions/workflows/ci.yml)

Public replication code, paper, publication-style tables, figures, and audit documentation for *Climate Risk and Business Development Companies: Replication, Factor Maintenance, and Regulatory Capacity* (Version 3.0).

## Main results

- **Bank replication.** Mean climate beta rises from 0.193 in 2019 to 0.424 in 2020. The paired increase is 0.230*** (SE 0.029), and all ten banks move in the predicted direction. End-2020 top-four mCRISK is USD 221.7 billion, 85.3% of the published benchmark; the aggregate CRISK increase recovers 86.5%.
- **BDC portfolio mechanism.** A coarse brown-industry share is uninformative, while estimator-aligned DCC-FF49 portfolio beta has a firm- and quarter-fixed-effects equity coefficient of 0.152* (SE 0.085). The 19-cluster wild-bootstrap p-value is 0.153, so this conventional 10% result is reported with a small-cluster qualification.
- **Basket breadth.** The published top-five KOL continuation tracks KOL at 0.454 daily and 0.832 weekly. A pre-specified 15-security basket covering 77.1% of the official September 2020 N-PORT schedule raises tracking to 0.790 and 0.945. Its BDC equity coefficients are 0.184 daily and 0.173 weekly, but neither is conventionally significant.
- **Why banks differ from BDCs.** The published Y-14 validation uses continuous loan-size-weighted climate beta on more than five million loans and 666 bank-quarters; it does not test a binary brown-loan share. Public BDC filings supply 380 aggregated institution-quarters, with material label aggregation and post-2020 sample truncation.
- **BDC statutory stress.** Under the applicable BDC asset-coverage rule, the maintained climate scenario reduces the mean buffer by 5.85 percentage points without a primary-scenario breach. The bank `k=8%` parameter is not used to characterize BDC capital adequacy.

## Repository structure

```text
01_Bank_CRISK_Replication/
02_BDC_Investment_Exposure_and_Climate_Beta/
03_BDC_Asset_Coverage_Stress_Test/
    Code/       Python and Stata reproduction code
    Data/       data-availability instructions only
    Results/    publication-style RTF tables and PNG figures
Paper/          PDF, editable Word file, LaTeX source, and Word build script
```

The paper uses conventional significance stars (`*` 10%, `**` 5%, `***` 1%) and standard errors in parentheses. Result workbooks are not used.

## Reproduction

Licensed CRSP and Compustat inputs are not redistributed. After placing the required source files in the module-specific `Data/Raw/` directories, run:

```bash
python run_all.py --full
```

The cumulative-75-percent factor is built by `01_Bank_CRISK_Replication/Code/08_build_kol_top75_continuation.py`; daily and weekly DCC robustness are estimated by Module 2 scripts `12` and `13`. Each module also contains a Stata do-file for the displayed regressions.

Run the public release guard with:

```bash
python validate_public_repository.py
```

See [REPLICATION.md](REPLICATION.md) and [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for file placement and data-use restrictions.

## Citation and data use

Please cite the archive using `CITATION.cff`. CRSP and Compustat inputs remain subject to their original license terms; no licensed observation-level data are published here.
