# Climate Risk and Business Development Companies

## Replication Archive, Version 4.3

This archive accompanies *Climate Risk and Business Development Companies: A CRISK Replication and Extension*.

Repository: https://github.com/MaryMai233/crisk-bdc-replication

## Paper design in v4.3

The paper was deliberately shortened. It now keeps only **3 tables and 2 figures**:

- Table 1: bank replication benchmarks.
- Figure 1: annual climate beta for banks and BDCs.
- Table 2: direct BDC analogue of the original Table 1.
- Figure 2: measurement-resolution diagnostic.
- Table 3: BDC statutory asset-coverage stress.

Descriptive statistics, asset-beta robustness, detailed KOL continuation tables, Y-14 transcription tables, placebo tables, and supplementary diagnostic figures remain in the replication modules but are no longer repeated in the paper. The main text reports only the key numbers needed for the argument.

The original paper constructs four climate transition factors. This archive currently replicates the **Stranded Asset factor** only; the cumulative-75% KOL basket is a robustness check for continuation of that same factor, not a second climate factor.

## Main findings

- Mean ten-bank climate beta rises from 0.193 in 2019 to 0.424 in 2020; the paired change is 0.230***.
- End-2020 top-four mCRISK is USD 221.7 billion, 85.3% of the published benchmark; the 2020 CRISK increase recovers 86.5%.
- The direct BDC Table-1 analogue gives portfolio-beta coefficients of 0.223*, 0.126, 0.421***, and 0.694*** as controls and fixed effects are added.
- Under a stricter BDC-and-quarter-FE specification, measurement improves from a coarse brown share to DCC-FF49; the strict top-five DCC-FF49 coefficient is 0.402** with wild-cluster p=0.092.
- Expanding KOL to a 15-security basket covering 77.1% raises daily tracking from 0.454 to 0.790 but does not improve small-cluster inference.
- A 50% climate-factor stress reduces mean BDC statutory asset-coverage buffer by 5.85 percentage points with no breach.

## Reproduction

Each empirical module contains `Code/`, `Data/`, and `Results/` directories. Run the module-level `Code/00_run.py` scripts to regenerate results. Licensed CRSP/Compustat inputs are not redistributed. `LICENSED_DATA_MANIFEST.csv` records required filenames and checksums.

After reproducing the modules, run:

```bash
python validate_release.py
```

The release validator checks the key numerical identities and required files. Version 4.3 passes the validator.

## Data-use note

CRSP and Compustat inputs remain subject to their original license and access terms. SEC filings and public market downloads remain subject to their source terms.
