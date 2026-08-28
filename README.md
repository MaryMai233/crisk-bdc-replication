# Climate Risk and Business Development Companies

## Replication Archive, Version 4.4

This archive accompanies *Climate Risk and Business Development Companies: A CRISK Replication and Extension*.

Repository: https://github.com/MaryMai233/crisk-bdc-replication

## Paper design in v4.4

The paper is intentionally concise but no longer stripped down as aggressively as v4.3. It keeps **4 tables and 3 figures**, all embedded in the main text:

- Table 1: bank replication benchmarks.
- Figure 1: annual climate beta for banks and BDCs.
- Table 2: direct BDC analogue of the original Table 1.
- Figure 2: measurement-resolution diagnostic.
- Table 3: KOL continuation breadth and BDC inference.
- Figure 3: KOL basket breadth and the BDC portfolio mechanism.
- Table 4: BDC statutory asset-coverage stress.

There is no separate abstract section. Its content is folded into the opening of the Introduction. The main text reports descriptive conclusions rather than repeating detailed coefficients, significance stars, and p-values already visible in the tables. Appendix-style diagnostics remain in the replication modules but are not repeated in the paper.

The original paper constructs four climate transition factors. This archive currently replicates the **Stranded Asset factor** only; the cumulative-75% KOL basket is a robustness check for continuation of that same factor, not a second climate factor.

## Main findings

- The bank replication closely matches the published CRISK benchmark.
- BDC portfolio climate exposure is positively related to traded BDC climate beta, with a clearer signal under finer exposure measurement.
- A broader KOL continuation improves tracking of the discontinued ETF but does not materially improve small-cluster inference.
- The BDC-specific stress test compresses statutory asset-coverage buffers without producing a baseline-scenario breach.

## Reproduction

Each empirical module contains `Code/`, `Data/`, and `Results/` directories. Run the module-level `Code/00_run.py` scripts to regenerate results. Licensed CRSP/Compustat inputs are not redistributed. `LICENSED_DATA_MANIFEST.csv` records required filenames and checksums.

After reproducing the modules, run:

```bash
python validate_release.py
```

## Data-use note

CRSP and Compustat inputs remain subject to their original license and access terms. SEC filings and public market downloads remain subject to their source terms.
