from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from tueplots import bundles

BLUE = "#0072B2"
TEAL = "#009E73"
ORANGE = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#6B7280"
LIGHT = "#E5E7EB"
INK = "#111827"


def apply_style() -> None:
    plt.rcParams.update(bundles.neurips2024(usetex=False))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "kahlus-eeg-v1",
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#C7CDD4",
            "grid.color": LIGHT,
            "grid.linewidth": 0.65,
        }
    )


def save(fig: plt.Figure, stem: Path) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
