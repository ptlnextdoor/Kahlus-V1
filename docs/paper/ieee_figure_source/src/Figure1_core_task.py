"""Figure 1 — Kahlus M4 horizon sweep via kahlus-sweep-figure helper.

From ``data/m4_horizon_sweep.csv``. One row of panels, one takeaway sentence.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_figure import sweep_figure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "m4_horizon_sweep.csv"
OUT = ROOT / "figures" / "fig1_core_task"


def load_rows() -> list[dict]:
    rows: list[dict] = []
    with DATA.open(newline="") as f:
        for raw in csv.DictReader(f):
            rows.append(
                {
                    "horizon": int(raw["horizon"]),
                    "method": raw["method"],
                    "rfs_bits": float(raw["rfs_bits"]),
                    "truth": float(raw["truth"]),
                    "detected": float(raw["detected"]),
                    "false_certify": float(raw["false_certify"]),
                }
            )
    return rows


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    panels = [
        dict(
            y="rfs_bits",
            label="residual forecastability RFS (bits)",
            title="RFS vs horizon vs analytic zero",
            truth="truth",
            ylim=(-0.08, 0.28),
        ),
        dict(
            y="detected",
            label="detection of residual signal",
            title="detection power vs horizon",
            ylim=(-0.05, 1.05),
        ),
        dict(
            y="false_certify",
            label="false-certify rate on null world",
            title="false-certify vs horizon  (lower is better)",
            ylim=(-0.05, 1.05),
        ),
    ]
    for ext in ("png", "pdf"):
        sweep_figure(
            rows,
            x="horizon",
            x_label="forecast horizon (steps)",
            panels=panels,
            suptitle=(
                "Kahlus M4 synthetic sweep: residual signal detected across horizons, "
                "zero false-certify on the null"
            ),
            out=str(OUT.with_suffix(f".{ext}")),
            figsize=(16.5, 4.5),
            dpi=130,
        )
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
