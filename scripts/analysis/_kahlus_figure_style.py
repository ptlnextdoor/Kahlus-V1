"""Shared matplotlib styling for Kahlus documentation figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#059669"
ORANGE = "#ea580c"
PURPLE = "#7c3aed"
GRAY = "#6b7280"
INK = "#374151"
WHITE = "#ffffff"
LIGHT = "#f3f4f6"
LIGHT_BLUE = "#eff6ff"
LIGHT_RED = "#fef2f2"
LIGHT_GREEN = "#ecfdf5"
LIGHT_ORANGE = "#fff7ed"
LIGHT_PURPLE = "#f5f3ff"


def apply_kahlus_style(*, dpi: int = 160) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.transparent": False,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": dpi,
            "savefig.dpi": max(dpi, 170),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": "#d1d5db",
            "axes.edgecolor": INK,
            "text.color": INK,
        }
    )


def provenance_footer(fig: plt.Figure, text: str, *, color: str = GRAY) -> None:
    fig.text(0.5, 0.012, text, ha="center", va="bottom", fontsize=8.5, color=color, wrap=True)


def stamp_schematic(fig: plt.Figure, text: str) -> None:
    provenance_footer(fig, text, color=RED)
