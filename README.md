# Climate Risk and Business Development Companies

## Replication Archive, Version 4.2

This archive accompanies *Climate Risk and Business Development Companies: Replication, Factor Maintenance, and Regulatory Capacity*. It contains one paper and three empirical modules. Every module uses the same `Code` / `Data` / `Results` structure and includes both Python and Stata code.

Repository: https://github.com/MaryMai233/crisk-bdc-replication

`LICENSED_DATA_MANIFEST.csv` records the six CRSP annual filenames, row counts, byte sizes, and SHA-256 checksums needed for a licensed full rebuild without redistributing those files publicly.

## Contents

- `Paper/`
  - `Climate_Risk_and_BDCs.pdf`: complete academic draft.
  - `Climate_Risk_and_BDCs_Word.docx`: editable Word version with all tables and figures.
  - `Climate_Risk_and_BDCs.tex`: LaTeX source. All body references to tables and figures are clickable.
  - `build_word_report.py`: rebuilds the formatted Word version from the LaTeX narrative and archived results.
- `01_Bank_CRISK_Replication/`: ten-bank CRISK replication, identifier recovery, paired-event statistics, and factor-input audits.
- `02_BDC_Investment_Exposure_and_Climate_Beta/`: BDC portfolio exposure measured from reported industry weights and value-weighted FF49 industry climate betas; estimator-aligned DCC models, coarse-share diagnostics, factor-continuation tests, and small-cluster inference.
- `03_BDC_Asset_Coverage_Stress_Test/`: BDC statutory asset-coverage calibration, matched-tail market comparison, NAV sensitivity, and the exposure-to-capacity mapping.

Each empirical module contains:

- `Code/`: Python scripts, a one-command runner, a Stata reproduction do-file, and dependencies.
- `Data/Raw/`: source files distributed with this archive.
- `Data/Processed/`: analysis-ready CSV and Stata `.dta` files plus audit records.
- `Results/`: publication-style RTF tables and PNG figures. No spreadsheet is used as a result file.

## Reproduction

Run these commands from the archive root:

```bash
cd 01_Bank_CRISK_Replication
python Code/00_run.py

cd ../02_BDC_Investment_Exposure_and_Climate_Beta
python Code/00_run.py

cd ../03_BDC_Asset_Coverage_Stress_Test
python Code/00_run.py
```

These commands regenerate the result tables and figures from included processed data. To rebuild estimates from licensed CRSP inputs, place the six annual gzip-compressed SAS files in `02_BDC_Investment_Exposure_and_Climate_Beta/Data/Raw/CRSP_All_US_Daily_2020_2025/` and add `--full`:

```bash
python Code/00_run.py --full
```

The annual filenames must be `crsp_all_us_daily_2020.gz` through `crsp_all_us_daily_2025.gz`. The full SEC text extraction additionally requires `SEC_BDC_Filings_2021_2025.zip`, which is omitted from the public archive because of size. Its extracted rows, source accessions, confidence flags, analysis panels, and audits are included. Place the SEC ZIP in the relevant `Data/Raw/` directories before using `--full` for Modules 2 and 3.

Python dependencies appear in each module's `Code/requirements.txt`. The DCC and SEC extraction stages can take several minutes. The final numbered Stata do-file in each `Code/` directory reproduces the regression tables from the included `.dta` files. In Stata, change the working directory to the module and run the do-file.

After reproducing the modules, run `python validate_release.py` from the archive root. It checks the daily marginal-CRISK identity, the book-to-market reciprocal identity, the BDC paired validation, all 49 industry volatility fits, FF49 coverage, equality of repeated estimates, the conventional and wild-cluster inference records, weekly convergence, and all required outputs.

## Principal additions in Version 4.2

1. All paper tables use sparse academic formatting with coefficients followed by t statistics in parentheses and no dense multi-panel mega-table. **Every table number/title is placed above the table, with notes below, and every main table/figure is embedded in the relevant narrative section near its first substantive discussion rather than collected in an end-of-paper display section.** Long Word tables may begin on a fresh page to avoid splitting, but they remain inside the corresponding section. `TABLE_CROSSWALK.md` states which BDC table is a direct analogue of each original table and where no valid analogue exists.
2. The BDC main regression now has a literal four-column analogue of the original Table 1: exposure only; + controls; + BDC fixed effects; + year fixed effects. The portfolio-beta coefficients are 0.223*, 0.126, 0.421***, and 0.694***.
3. A deliberately stricter specification adds financial controls together with BDC and quarter fixed effects. The top-five equity coefficient is 0.402** (t=2.34; conventional p=0.019), with wild-cluster p=0.092. This is reported separately from the direct published-specification analogue rather than mixing the two designs in one table.
4. The published top-five KOL continuation remains the strict replication. A pre-specified 15-security basket covering 77.1% of the official 30 September 2020 N-PORT common-stock schedule remains a breadth robustness. It raises daily KOL tracking from 0.454 to 0.790, but the stricter equity wild-cluster p-value moves from 0.092 to 0.162.
5. The paper and user-facing outputs use the original terminology CRISK and mCRISK. No additional CRISK variant is presented as a new measure.
6. The original Y-14 robustness results are separated into simple reference tables: all continuous-exposure variants in Tables F.1-F.3 remain significant at 1%; the actual-weight placebo remains significant while shuffled weights do not.
7. **Factor coverage is now stated explicitly.** The original paper constructs four climate transition factors (stranded asset, emission, BMG, and CEP). This archive currently replicates only the stranded-asset factor, which is the paper's headline CRISK baseline. The other three climate-factor constructions are not silently treated as reproduced; the archive does not contain the required emissions/CEP inputs.

## Main findings

- **Replication fidelity:** Mean ten-bank climate beta rises from 0.193 in 2019 to 0.424 in 2020; the paired change is 0.230*** (SE 0.029), and all ten changes are positive. End-2020 top-four mCRISK is USD 221.7 billion (85.3% of the published benchmark), while the CRISK increase recovers 86.5%.
- **Direct BDC analogue of original Table 1:** The portfolio-beta coefficient is 0.223* without controls, 0.126 with controls, 0.421*** after BDC fixed effects, and 0.694*** after adding year fixed effects.
- **Stricter BDC identification:** With financial controls, BDC fixed effects, and quarter fixed effects, the top-five equity coefficient is 0.402** under conventional clustered inference and has wild-cluster p=0.092. The corresponding asset-beta coefficient is 0.176** with wild-cluster p=0.124.
- **Measurement resolution:** Under one common controlled two-way-FE specification, the standardized equity coefficient progresses from -0.032 for a broad brown share to 0.100 for FF12, 0.167* for rolling-OLS FF49, and 0.222** for estimator-aligned DCC-FF49.
- **Continuation-rule breadth:** The 15-security cumulative-75% basket materially improves factor tracking but not small-cluster inference. This separates factor-maintenance quality from portfolio-exposure identification.
- **BDC statutory capacity:** Under the applicable asset-coverage rule, the maintained 50% climate scenario reduces the mean buffer by 5.85 percentage points without a primary-scenario breach. The bank `k=8%` calculation is not interpreted as a BDC capital requirement.

## Audit and interpretation notes

- The bank sample contains all ten intended benchmark institutions. BK is recovered manually because the user-supplied restricted ticker/code extracts lack a usable sequence; this is not a claim that BK is absent from Compustat.
- The balanced BDC sample contains 19 firms and 380 firm-quarters from 2021Q1 through 2025Q4. BBDC is excluded because a reconciled portfolio-industry distribution is unavailable under the maintained parser. The panel is selected for continuous disclosure and may exhibit survivor/disclosure-consistency bias.
- The original coarse carbon-intensity dictionary maps 99.76% of reported portfolio weight. The more granular FF49 crosswalk maps 95.2% overall; its row-level rules, confidence levels, four geography-table imputations, and exclusion checks are included. The published Y-14 validation instead uses continuous loan-size-weighted borrower-industry beta on more than five million loans; it does not validate CRISK with a binary brown-loan share.
- The archived DCC systems converge. Full-sample OLS and mean DCC climate betas have a cross-institution correlation of 0.795; the archive also includes an LRMES boundary check and exact CRISK accounting-identity tests.
- The optional `01_Bank_CRISK_Replication/Code/07_validate_vlab_if_available.py` computes daily V-Lab correlations and RMSE if an authorized `vlab_crisk_daily.csv` is supplied. Without that file it records `NOT_RUN`; no V-Lab statistic is fabricated.
- Public code, results, and the paper are maintained at https://github.com/MaryMai233/crisk-bdc-replication. Licensed CRSP and Compustat files remain only in the private archive. A Zenodo DOI will be added after the first tagged release is deposited.

## Data-use note

CRSP and Compustat inputs remain subject to their original license and access terms. SEC filings and public market downloads remain subject to their source terms. The archive is intended for research replication; users are responsible for confirming redistribution rights before public release.
