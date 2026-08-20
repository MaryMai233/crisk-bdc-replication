from __future__ import annotations

import py_compile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULES = [ROOT / "01_Bank_CRISK_Replication", ROOT / "02_BDC_Investment_Exposure_and_Climate_Beta", ROOT / "03_BDC_Asset_Coverage_Stress_Test"]
RESTRICTED = {".csv", ".dta", ".xlsx", ".xls", ".sas7bdat", ".parquet", ".zip"}

required = [
    ROOT / "README.md",
    ROOT / "DATA_AVAILABILITY.md",
    ROOT / "REPLICATION.md",
    ROOT / "CITATION.cff",
    ROOT / "LICENSED_DATA_MANIFEST.csv",
    ROOT / "Paper" / "Climate_Risk_and_BDCs.tex",
    ROOT / "Paper" / "Climate_Risk_and_BDCs.pdf",
    ROOT / "Paper" / "Climate_Risk_and_BDCs_Word.docx",
    ROOT / "Paper" / "build_word_report.py",
]
required += [
    ROOT / "01_Bank_CRISK_Replication" / "Results" / "Table_1_Bank_CRISK_Replication.rtf",
    ROOT / "01_Bank_CRISK_Replication" / "Results" / "Figure03_Bank_Level_Beta_Changes.png",
    ROOT / "01_Bank_CRISK_Replication" / "Results" / "Figure04_Published_vs_Replicated.png",
    ROOT / "02_BDC_Investment_Exposure_and_Climate_Beta" / "Results" / "Table_3_BDC_Investment_Exposure.rtf",
    ROOT / "02_BDC_Investment_Exposure_and_Climate_Beta" / "Results" / "Table_4_Factor_Continuation_and_H2_Robustness.rtf",
    ROOT / "02_BDC_Investment_Exposure_and_Climate_Beta" / "Results" / "Table_3_BDC_FF49_DCC_Portfolio_Mechanism.rtf",
    ROOT / "02_BDC_Investment_Exposure_and_Climate_Beta" / "Results" / "Figure_3_FF49_DCC_Portfolio_Mechanism.png",
    ROOT / "02_BDC_Investment_Exposure_and_Climate_Beta" / "Results" / "Figure_5_Factor_Continuation_Sensitivity.png",
    ROOT / "03_BDC_Asset_Coverage_Stress_Test" / "Results" / "Table_3_BDC_Asset_Coverage_Stress.rtf",
    ROOT / "03_BDC_Asset_Coverage_Stress_Test" / "Results" / "Figure_6_Asset_Coverage_Before_After.png",
    ROOT / "03_BDC_Asset_Coverage_Stress_Test" / "Results" / "Figure_7_Threshold_Proximity.png",
]
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
with tempfile.TemporaryDirectory(prefix="crisk_public_compile_") as cache_dir:
    cache_root = Path(cache_dir)
    for source in ROOT.rglob("*.py"):
        relative = source.relative_to(ROOT)
        compiled = cache_root / relative.with_suffix(".pyc")
        compiled.parent.mkdir(parents=True, exist_ok=True)
        py_compile.compile(source, cfile=str(compiled), doraise=True)
print("PASS: public repository structure, source compilation, and data-license guard")
