from __future__ import annotations

import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULES = [ROOT / "01_Bank_CRISK_Replication", ROOT / "02_BDC_Investment_Exposure_and_Climate_Beta", ROOT / "03_BDC_Asset_Coverage_Stress_Test"]
RESTRICTED = {".csv", ".dta", ".xlsx", ".xls", ".sas7bdat", ".parquet", ".zip"}

required = [ROOT / "README.md", ROOT / "DATA_AVAILABILITY.md", ROOT / "REPLICATION.md", ROOT / "CITATION.cff", ROOT / "Paper" / "Climate_Risk_and_BDCs.tex", ROOT / "Paper" / "Climate_Risk_and_BDCs.pdf"]
missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
assert not missing, f"Missing required files: {missing}"
for module in MODULES:
    assert (module / "Code" / "00_run.py").exists()
    assert list((module / "Results").glob("*.rtf"))
    assert list((module / "Results").glob("*.png"))
    for path in (module / "Data").rglob("*"):
        if path.is_file() and path.suffix.lower() in RESTRICTED:
            raise AssertionError(f"Restricted data included: {path.relative_to(ROOT)}")
assert not list(ROOT.rglob("__pycache__"))
assert not list(ROOT.glob("**/Results/*.xlsx"))
for source in ROOT.rglob("*.py"):
    py_compile.compile(source, doraise=True)
print("PASS: public repository structure, source compilation, and data-license guard")
