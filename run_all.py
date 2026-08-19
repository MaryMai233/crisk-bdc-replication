from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULES = [
    "01_Bank_CRISK_Replication",
    "02_BDC_Investment_Exposure_and_Climate_Beta",
    "03_BDC_Asset_Coverage_Stress_Test",
]

parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true")
args = parser.parse_args()
for module in MODULES:
    command = [sys.executable, str(ROOT / module / "Code" / "00_run.py")]
    if args.full:
        command.append("--full")
    subprocess.run(command, cwd=ROOT / module, check=True)
