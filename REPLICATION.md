# Replication instructions

Create the environment:

```bash
conda env create -f environment.yml
conda activate crisk-bdc
```

Run from existing processed data:

```bash
python run_all.py
```

After placing all licensed and public raw inputs in the documented locations, rebuild every stage:

```bash
python run_all.py --full
```

Each module also contains a Stata do-file that reproduces stored regression tables from the analysis-ready `.dta` inputs in the private archive.
