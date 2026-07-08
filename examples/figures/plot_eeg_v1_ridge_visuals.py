"""
Render EEG v1 ridge visual diagnostics
======================================

This executable example regenerates the synthetic EEG v1 ridge visual packet. It is intentionally a diagnostic refit, not a public EEG benchmark.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def find_repo_root() -> Path:
    """Find the repo root under normal execution and Sphinx-Gallery execution."""
    for base in (Path.cwd(), *Path.cwd().parents):
        if (base / "pyproject.toml").exists() and (base / "scripts" / "render_eeg_v1_ridge_visuals.py").exists():
            return base
    raise RuntimeError("could not find Kahlus repo root")


ROOT = find_repo_root()
OUT = ROOT / "docs" / "research" / "eeg_v1_ridge_visuals"

cmd = [
    sys.executable,
    str(ROOT / "scripts" / "render_eeg_v1_ridge_visuals.py"),
    "--dataset",
    "synthetic_fixture",
    "--out-dir",
    str(OUT),
]
result = subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True)
print(result.stdout.strip())
print("Generated files:")
for path in sorted(OUT.glob("eeg_v1_ridge_*")):
    print(f"- {path.relative_to(ROOT)}")
