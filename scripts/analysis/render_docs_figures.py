#!/usr/bin/env python3
"""Regenerate documentation figures for docs/figures.

Refreshes the HNPH protocol figure set from the frozen v0.4 protocol renderer.
docs/figures/hnph_protocol is the single canonical location; the preprint reads
it directly via \\graphicspath rather than keeping a second copy.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HNPH_SCRIPT = ROOT / "scripts" / "analysis" / "plot_hnph_preprint_figures.py"
PROTOCOL = ROOT / "configs" / "protocol" / "hnph_phase0_v0.4.yaml"
ARCHIVE_FIGURES = ROOT / "docs" / "figures" / "hnph_protocol"


def _run_hnph_renderer(out_dir: Path) -> None:
    command = [
        sys.executable,
        str(HNPH_SCRIPT),
        "--protocol",
        str(PROTOCOL),
        "--out-dir",
        str(out_dir),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "HNPH figure render failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-hnph",
        action="store_true",
        help="Do not regenerate HNPH protocol figures.",
    )
    args = parser.parse_args()

    if not args.skip_hnph:
        print(f"rendering HNPH figures -> {ARCHIVE_FIGURES}")
        _run_hnph_renderer(ARCHIVE_FIGURES)

    print("docs figure render complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
