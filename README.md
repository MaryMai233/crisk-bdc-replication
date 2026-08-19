# Climate Risk and Business Development Companies

[![Repository audit](https://github.com/MaryMai233/crisk-bdc-replication/actions/workflows/ci.yml/badge.svg)](https://github.com/MaryMai233/crisk-bdc-replication/actions/workflows/ci.yml)

Public replication code, paper, publication-style tables, figures, and audit documentation for *Climate Risk and Business Development Companies: Replication, Factor Maintenance, and Regulatory Capacity*.

## Main findings

- **H1:** Mean climate beta rises from 0.193 in 2019 to 0.424 in 2020 for all ten benchmark banks. This is cross-institution consistency under one common factor realization, not ten independent event replications. Replicated top-four marginal CRISK and signed CRISK growth recover 85.3% and 86.5% of published magnitudes.
- **H2:** H2 is not statistically supported. Parsimonious estimates are positive and rise with factor tracking; controlled and fixed-effects estimates can be negative and are too imprecise to interpret by sign. Holding the top-five basket fixed, weekly aggregation raises KOL tracking from 0.454 to 0.832 and the equity-beta coefficient from 0.025 to 0.154.
- **H3:** A maintained 50% climate shock reduces the mean statutory asset-coverage buffer by 5.85 percentage points without a primary-scenario breach. At matched empirical tail probabilities, climate compression is 0.343 of the market benchmark.

## Structure

Each module contains `Code/`, `Data/Raw/`, `Data/Processed/`, and `Results/`. Licensed CRSP and Compustat data are not redistributed; the Data folders explain placement.

## Quick audit

```bash
python validate_public_repository.py
```

For full reproduction, see [REPLICATION.md](REPLICATION.md) and [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). The repository is prepared for Zenodo archiving; the DOI field will be added after the first tagged release is deposited.
