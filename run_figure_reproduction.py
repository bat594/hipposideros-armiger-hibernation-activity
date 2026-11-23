#!/usr/bin/env python3
"""Run the current figure-generation scripts for Figures 2-7."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ROOT / "figure_reproduction" / "fig02_validation" / "Fig2_make.py",
    ROOT / "figure_reproduction" / "fig03_activity_heatmap" / "Fig3_make.py",
    ROOT / "figure_reproduction" / "fig04_periodicity_sensitivity" / "Fig4_make.py",
    ROOT / "figure_reproduction" / "fig05_daily_activity_patterns" / "Fig5_make.py",
    ROOT / "figure_reproduction" / "fig06_temporal_progression" / "Fig6_make.py",
    ROOT / "figure_reproduction" / "fig07_behavioural_composition" / "Fig7_make.py",
]

for script in SCRIPTS:
    print(f"\n=== Running {script.relative_to(ROOT)} ===", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=script.parent, check=True)
print("\nAll figure scripts completed successfully.")
