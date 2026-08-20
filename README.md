# Climate Risk and Business Development Companies

Replication code and public results for *Climate Risk and Business Development Companies: Replication, Factor Maintenance, and Regulatory Capacity* (Version 2.7).

## Main results

- **Bank replication.** Mean climate beta rises from 0.193 in 2019 to 0.424 in 2020. The paired increase is 0.230*** (SE 0.029), and all ten banks move in the predicted direction. The replicated top-four marginal CRISK is USD 221.7 billion, or 85.3% of the published benchmark.
- **BDC portfolio mechanism.** Replacing a coarse brown-industry share with value-weighted FF49 industry climate betas and an estimator-aligned DCC model yields a firm- and quarter-fixed-effects equity coefficient of 0.152* (clustered SE 0.085). The 19-cluster wild-bootstrap p-value is 0.153, so the conventional 10% result is reported with a small-cluster qualification.
- **Factor maintenance.** The fixed international top-five continuation basket tracks KOL at 0.454 daily and 0.832 weekly. The original article specifies the basket after KOL's liquidation but does not state a periodic refresh rule.
- **BDC statutory stress.** Under the BDC asset-coverage rule, the maintained climate scenario reduces the mean coverage buffer by 5.85 percentage points without a primary-scenario breach. Matched-tail climate compression is 0.343 of the market benchmark.

## Repository structure

Each empirical module follows the same structure:

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

The public repository contains the complete code, paper, tables, figures, and data instructions. Licensed CRSP and Compustat inputs are not redistributed.

After placing the required source files in the module-specific `Data/Raw/` directories, run from the repository root:

```bash
python run_all.py --full
```

For a single module:

```bash
cd 02_BDC_Investment_Exposure_and_Climate_Beta
python Code/00_run.py --full
```

The Stata do-file in each module reproduces the displayed regressions from the corresponding analysis-ready `.dta` file in the private archive. `LICENSED_DATA_MANIFEST.csv` records the six CRSP annual filenames, row counts, file sizes, and SHA-256 checksums used for Version 2.7.

Run the public release guard with:

```bash
python validate_public_repository.py
```

## Required licensed CRSP files for the FF49 extension

Place these gzip-compressed SAS files in `02_BDC_Investment_Exposure_and_Climate_Beta/Data/Raw/CRSP_All_US_Daily_2020_2025/`:

```text
crsp_all_us_daily_2020.gz
crsp_all_us_daily_2021.gz
crsp_all_us_daily_2022.gz
crsp_all_us_daily_2023.gz
crsp_all_us_daily_2024.gz
crsp_all_us_daily_2025.gz
```

The full SEC extraction also requires `SEC_BDC_Filings_2021_2025.zip`. See `DATA_AVAILABILITY.md` and the `Data/README.md` files for the remaining inputs.

## Citation and data use

Please cite the archive using `CITATION.cff`. CRSP and Compustat inputs remain subject to their original license terms; no licensed observation-level data are published in this repository.
