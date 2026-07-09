from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from tueplots import bundles

BLUE = "#0072B2"
TEAL = "#009E73"
ORANGE = "#D55E00"
GRAY = "#6B7280"
LIGHT = "#E5E7EB"
INK = "#111827"
LIGHT_BLUE = "#DBEAFE"
LIGHT_ORANGE = "#FFEDD5"


FIGURE_TIMESTAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)
SVG_METADATA = {"Creator": "Kahlus EEG v1 figure renderer", "Date": FIGURE_TIMESTAMP.isoformat()}
PDF_METADATA = {
    "Creator": "Kahlus EEG v1 figure renderer",
    "CreationDate": FIGURE_TIMESTAMP,
    "ModDate": FIGURE_TIMESTAMP,
}


def apply_style() -> None:
    plt.rcParams.update(bundles.neurips2024(usetex=False))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "kahlus-ridge-sanity",
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
        metadata = PDF_METADATA if ext == "pdf" else SVG_METADATA if ext == "svg" else None
        fig.savefig(
            stem.with_suffix(f".{ext}"),
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
            metadata=metadata,
        )
    plt.close(fig)
