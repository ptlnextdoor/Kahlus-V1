"""Figure 2 — Neural-CASP gate ladder (Kahlus courtroom success + honest fails)."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "neural_casp_gates.csv"
OUT = ROOT / "figures" / "fig2_nfc_schematic"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with DATA.open(newline="") as f:
        rows = list(csv.DictReader(f))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)

    ax = axes[0]
    milestones = [r["milestone"] for r in rows]
    passed = [int(r["gate_passed"]) for r in rows]
    colors = ["#0f766e" if p else "#b91c1c" for p in passed]
    y = np.arange(len(milestones))
    ax.barh(y, [1] * len(milestones), color=colors, height=0.62, edgecolor="white")
    for i, (m, p, r) in enumerate(zip(milestones, passed, rows)):
        ax.text(0.5, i, f"{m}  {'PASS' if p else 'FAIL'}  —  {r['note']}", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_title("Neural-CASP Forecastability Trial 0 gate ladder")
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax = axes[1]
    # summary counts
    n_pass = sum(passed)
    n_fail = len(passed) - n_pass
    ax.bar([0, 1], [n_pass, n_fail], color=["#0f766e", "#b91c1c"], width=0.55, edgecolor="white")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"passed ({n_pass})", f"failed honestly ({n_fail})"])
    ax.set_ylabel("gates")
    ax.set_ylim(0, 6.5)
    ax.set_title("courtroom result: pass when valid, fail when underpowered")
    for i, v in enumerate([n_pass, n_fail]):
        ax.text(i, v + 0.15, str(v), ha="center", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "Kahlus succeeds as a Neural-CASP courtroom: M0/M1/M2/M5 pass; M3/M4 fail honestly",
        fontsize=12,
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=130 if ext == "png" else None)
    plt.close(fig)
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
