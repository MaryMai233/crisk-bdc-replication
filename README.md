# Climate Risk and Business Development Companies

[![Repository audit](https://github.com/MaryMai233/crisk-bdc-replication/actions/workflows/ci.yml/badge.svg)](https://github.com/MaryMai233/crisk-bdc-replication/actions/workflows/ci.yml)

Public replication code, paper, publication-style tables, figures, and audit documentation for *Climate Risk and Business Development Companies: Replication, Factor Maintenance, and Regulatory Capacity*.

## Main findings

- **Replication:** Ten-bank climate beta rises by 0.230*** in 2020; marginal and signed CRISK increases recover 85.3% and 86.5% of published magnitudes.
- **Factor maintenance:** The post-KOL basket tracks KOL at 0.454 daily and 0.832 weekly. Its BDC exposure estimate remains below the design's detectable threshold.
- **BDC stress:** Bank `k=8%` is vacuous for BDCs. The applicable asset-coverage mapping reduces the mean buffer by 5.85 percentage points without a primary-scenario breach.

The accompanying paper is a 12-page research note. Detailed robustness tables remain in module `Results/` rather than the paper.

## Structure

Each module contains `Code/`, `Data/Raw/`, `Data/Processed/`, and `Results/`. Licensed CRSP and Compustat data are not redistributed; the Data folders explain placement.

## Quick audit

```bash
python validate_public_repository.py
```

For full reproduction, see [REPLICATION.md](REPLICATION.md) and [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). The repository is prepared for Zenodo archiving; the DOI field will be added after the first tagged release is deposited.
