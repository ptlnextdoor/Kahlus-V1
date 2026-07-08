"""
Render EEG/ridge versions evidence figures
==========================================

This executable example regenerates the evidence-driven EEG/ridge figure packet.
It scans the local `/Users/aayu/Downloads/versions` evidence bundles, normalizes
saved CSV/JSON artifacts, and refuses to render waveform or prediction overlays
unless real tensor/prediction arrays exist.
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
VERSIONS_ROOT = Path("/Users/aayu/Downloads/versions")

cmd = [
    sys.executable,
    str(ROOT / "scripts" / "render_eeg_v1_ridge_visuals.py"),
    "--versions-root",
    str(VERSIONS_ROOT),
    "--out-dir",
    str(OUT),
]
result = subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True)
print(result.stdout.strip())
print("Generated files:")
for path in sorted(OUT.glob("eeg_v1_ridge_*")):
    print(f"- {path.relative_to(ROOT)}")
for path in sorted((ROOT / "docs" / "research" / "eeg_v1_figure_source" / "figures").glob("Figure*.png")):
    print(f"- {path.relative_to(ROOT)}")
