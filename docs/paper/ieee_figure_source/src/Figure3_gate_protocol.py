"""Figure 3 — M5 Passive PIC synthetic instrument success."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "m5_pic_worlds.csv"
OUT = ROOT / "figures" / "fig3_gate_protocol"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with DATA.open(newline="") as f:
        rows = [
            {
                **r,
                "pic_bits": float(r["pic_bits"]),
                "pic_ci_low": float(r["pic_ci_low"]),
                "pic_ci_high": float(r["pic_ci_high"]),
                "rfs_bits": float(r["rfs_bits"]),
            }
            for r in csv.DictReader(f)
        ]

    # preferred display order
    order = ["integrated_predictive", "independent_predictable", "nuisance_only", "white_noise"]
    label_map = {
        "integrated_predictive": "integrated\n(predictive)",
        "independent_predictable": "independent\n(predictable)",
        "nuisance_only": "nuisance\nonly",
        "white_noise": "white\nnoise",
    }
    rows = sorted(rows, key=lambda r: order.index(r["world"]))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)

    ax = axes[0]
    xs = np.arange(len(rows))
    vals = [r["pic_bits"] for r in rows]
    yerr = [
        [r["pic_bits"] - r["pic_ci_low"] for r in rows],
        [r["pic_ci_high"] - r["pic_bits"] for r in rows],
    ]
    colors = ["#0f766e" if r["world"] == "integrated_predictive" else "#94a3b8" for r in rows]
    ax.bar(xs, vals, yerr=yerr, color=colors, width=0.6, edgecolor="white", capsize=3, error_kw=dict(lw=1.0))
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([label_map[r["world"]] for r in rows], fontsize=8)
    ax.set_ylabel("PIC (bits)")
    ax.set_title("Passive PIC recovers integrated world only")
    for i, r in enumerate(rows):
        ax.text(i, r["pic_ci_high"] + 0.08, f"{r['pic_bits']:.2f}", ha="center", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    vals = [r["rfs_bits"] for r in rows]
    colors = ["#2563eb" if r["world"] == "integrated_predictive" else "#cbd5e1" for r in rows]
    ax.bar(xs, vals, color=colors, width=0.6, edgecolor="white")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([label_map[r["world"]] for r in rows], fontsize=8)
    ax.set_ylabel("integration-feature residual RFS (bits)")
    ax.set_title("residual RFS stays near zero except integrated world")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "Kahlus M5 success: synthetic Passive PIC instrument fires on integrated worlds and stays near zero on controls",
        fontsize=11,
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=130 if ext == "png" else None)
    plt.close(fig)
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
