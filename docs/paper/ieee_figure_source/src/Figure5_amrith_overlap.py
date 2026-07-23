"""Figure 5 — OLD overlapping metric vs NEW isolated forecast (Amrith audit).

Left: the historical 127-length target illusion (ridge looks near-perfect).
Right: isolated single-sample forecast at h=1 (subject-held-out Sleep-EDF).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "figures" / "fig5_amrith_overlap"


def _load_headline() -> dict:
    path = DATA / "amrith_overlap_headline.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run scripts/amrith_isolated_forecast_check.py first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_sweep_rows() -> list[dict[str, float | int | str]]:
    path = DATA / "amrith_horizon_sweep.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    headline = _load_headline()
    old = headline["old_overlapping_metric_full_127_target"]
    new = headline["new_isolated_metric_h1"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)

    ax = axes[0]
    labels = ["mean of trace", "persistence", "ridge (full window)"]
    vals = [
        old["mean_of_trace_full_window"],
        old["persistence_full_window"],
        old["ridge_full_window"],
    ]
    colors = ["#64748b", "#2563eb", "#b91c1c"]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, height=0.55, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("MSE (lower is better)")
    ax.set_title("OLD: 127-length overlapping target\n(126/127 samples shared with input)")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")
    ax.axvline(0.05, color="#0f766e", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(0.06, 0.05, "illusion floor", transform=ax.transAxes, fontsize=8, color="#0f766e")

    ax = axes[1]
    labels = ["mean of trace", "random", "persistence", "ridge-AR", "Kahlus GRU"]
    vals = [
        new["mean_of_trace"],
        new["random"],
        new["persistence"],
        new["ridge_ar"],
        new["kahlus_gru"],
    ]
    colors = ["#64748b", "#94a3b8", "#2563eb", "#7c3aed", "#0f766e"]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, height=0.55, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("MSE (lower is better)")
    ax.set_title("NEW: isolated h=1 forecast\n(predict only the 128th sample)")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle(
        "Overlap metric makes ridge look unbeatable; isolated future-sample scoring removes the trap "
        "(subject-held-out Sleep-EDF cassette)",
        fontsize=11,
    )

    # Optional inset: horizon sweep for GRU vs persistence
    rows = _load_sweep_rows()
    if rows:
        methods = {"kahlus_gru": "#0f766e", "persistence": "#2563eb", "ridge_ar": "#7c3aed"}
        inset = fig.add_axes([0.58, 0.12, 0.36, 0.28])
        for method, color in methods.items():
            sub = sorted(
                [r for r in rows if r["method"] == method],
                key=lambda r: int(r["horizon"]),
            )
            if not sub:
                continue
            xs = [int(r["horizon"]) for r in sub]
            ys = [float(r["mse"]) for r in sub]
            inset.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, label=method, color=color)
        inset.set_xlabel("horizon (samples)", fontsize=7)
        inset.set_ylabel("isolated MSE", fontsize=7)
        inset.set_title("horizon sweep", fontsize=8)
        inset.tick_params(labelsize=7)
        inset.grid(alpha=0.25)
        inset.legend(fontsize=6, loc="upper left")

    for ext in ("png", "pdf"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=130 if ext == "png" else None)
    plt.close(fig)
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
