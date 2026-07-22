"""Figure 4 — What Kahlus is (success frame) + honest claim boundary.

Left: RFS definition as the product thesis.
Right: allowed vs blocked claims after evidence gates (includes INVALID forecast sidecar).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "fig4_mse_bar"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)

    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Kahlus product thesis (Layer 1 courtroom)")
    boxes = [
        (0.06, 0.62, 0.40, 0.28, "Neural-CASP arena", "M0-M5 gates\nmanifests + controls", "#0f766e"),
        (0.54, 0.62, 0.40, 0.28, "RFS bits", r"$(NLL_B - NLL_{BZ})/\ln 2$", "#2563eb"),
        (0.06, 0.18, 0.40, 0.28, "Passive PCI track", "synthetic instrument\npublic data next", "#b45309"),
        (0.54, 0.18, 0.40, 0.28, "Evidence gate", "beat best baseline\ncontrols + scope", "#7c3aed"),
    ]
    for x, y, w, h, title, body, c in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.015,rounding_size=0.03",
                fc="white",
                ec=c,
                lw=1.4,
            )
        )
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=10, fontweight="bold", color=c)
        ax.text(x + w / 2, y + h * 0.32, body, ha="center", va="center", fontsize=8, color="#334155")

    ax = axes[1]
    claims = [
        ("Neural-CASP harness / gates", 1),
        ("M1 synthetic RFS recovery", 1),
        ("M5 synthetic Passive PIC instrument", 1),
        ("overlap audit / invalidation", 1),
        ("GRU forecast skill (3.116 / 0.972)", 0),
        ("clinical / foundation-model claim", 0),
    ]
    y = np.arange(len(claims))
    vals = [c[1] for c in claims]
    colors = ["#0f766e" if v else "#b91c1c" for v in vals]
    ax.barh(y, vals, color=colors, height=0.55, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([c[0] for c in claims], fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("allowed (1) / blocked (0)")
    ax.set_title("claim scope: show the courtroom wins, not the invalid sidecar")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle(
        "Kahlus v1 success is leakage-controlled residual forecastability evaluation, not the invalidated 3.116 MSE headline",
        fontsize=11,
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=130 if ext == "png" else None)
    plt.close(fig)
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
