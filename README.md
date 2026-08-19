# Climate Risk and Business Development Companies

[![Repository audit](https://github.com/MaryMai233/crisk-bdc-replication/actions/workflows/ci.yml/badge.svg)](https://github.com/MaryMai233/crisk-bdc-replication/actions/workflows/ci.yml)

Public replication code, paper, publication-style tables, figures, and audit documentation for *Climate Risk and Business Development Companies: Replication, Factor Maintenance, and Regulatory Capacity*.

## Main findings

- **Replication fidelity:** Mean climate beta rises from 0.193 in 2019 to 0.424 in 2020 for all ten benchmark banks; the paired change is 0.230 (SE 0.029). Replicated top-four marginal CRISK and signed CRISK increases recover 85.3% and 86.5% of the published magnitudes. Because all banks share one factor realization, the paired statistic measures cross-bank consistency rather than ten independent event replications.
- **Continuation-rule shelf life:** After KOL closes, the prescribed international top-five basket tracks KOL at 0.454 daily and 0.832 weekly. Holding the basket fixed, the downstream BDC equity-beta coefficient moves from 0.025 to 0.154. The full exposure grid remains statistically underpowered: the attenuation extrapolation, 0.215, is below the pooled 80% MDE of 0.273.
- **Capital-parameter portability:** Applying the bank capital ratio mechanically makes positive CRISK zero throughout the BDC sample. A BDC-specific 50% climate scenario instead reduces the mean statutory asset-coverage buffer by 5.85 percentage points without a primary-scenario breach; at matched empirical tail probabilities, climate compression is 0.343 of the market benchmark.

## Structure

Each module contains `Code/`, `Data/Raw/`, `Data/Processed/`, and `Results/`. Licensed CRSP and Compustat data are not redistributed; the Data folders explain placement.

## Quick audit

```bash
python validate_public_repository.py
```

For full reproduction, see [REPLICATION.md](REPLICATION.md) and [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). The repository is prepared for Zenodo archiving; the DOI field will be added after the first tagged release is deposited.
