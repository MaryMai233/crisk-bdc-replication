# Replication instructions

Create the environment:

```bash
conda env create -f environment.yml
conda activate crisk-bdc
```

Run from archived processed data in the private package:

```bash
python run_all.py
```

After placing all licensed and public raw inputs in the documented locations, rebuild every stage:

```bash
python run_all.py --full
```

Module 1 script `08_build_kol_top75_continuation.py` downloads public price and FX series for the 15 securities selected from the official 30 September 2020 N-PORT schedule. Module 2 scripts `12_estimate_kol_top75_dcc_robustness.py` and `13_estimate_kol_top75_weekly_dcc_robustness.py` re-estimate the daily and weekly institution/industry DCC systems. Script `14_make_top75_results.py` creates the comparison table and figure.

Each module contains a Stata do-file that reproduces the displayed regressions from the analysis-ready `.dta` inputs in the private archive.
