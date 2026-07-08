# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
# ---
"""Figure 2: EEG v1 audit and artifact matrix.

Reference-derived design: checklist/status data should be rendered as compact
heatmaps, not hand-drawn boxes and arrows.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def main() -> None:
    apply_style()
    audits = read_audits()
    inventory = read_json(DATA / "inventory.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), layout="constrained", width_ratios=[1.2, 1.0])
    plot_audit_status(axes[0], audits)
    plot_inventory_matrix(axes[1], inventory)
    fig.suptitle("EEG v1 evidence checks from cached audits and inventory", x=0.02, ha="left", fontsize=11, fontweight="bold")
    save(fig, FIGURES / "Figure2_eeg_v1_audit_matrix")


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "kahlus-eeg-v1",
            "savefig.dpi": 300,
        }
    )
    sns.set_theme(context="paper", style="white", rc={"font.family": "DejaVu Sans"})


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_audits() -> pd.DataFrame:
    path = DATA / "audits.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["audit_label"] = df["audit_type"].map(lambda x: str(x).replace("_", " "))
    df["status"] = df["passed"].map({True: "passed", False: "failed", "true": "passed", "false": "failed"}).fillna("unknown")
    return df


def plot_audit_status(ax: plt.Axes, audits: pd.DataFrame) -> None:
    ax.set_title("A. audit status counts", loc="left", fontweight="bold")
    labels = ["leakage", "eval", "paper mode gate"]
    columns = ["passed", "failed", "unknown"]
    if audits.empty:
        matrix = pd.DataFrame(0, index=labels, columns=columns)
    else:
        matrix = (
            pd.crosstab(audits["audit_label"], audits["status"])
            .reindex(index=labels, columns=columns, fill_value=0)
            .astype(int)
        )
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        linewidths=0.75,
        linecolor="white",
        square=False,
        ax=ax,
    )
    ax.set_xlabel("status")
    ax.set_ylabel("audit type")


def plot_inventory_matrix(ax: plt.Axes, inventory: dict) -> None:
    ax.set_title("B. artifact inventory", loc="left", fontweight="bold")
    rows = {
        "evidence bundles": inventory.get("bundle_count", 0),
        "task results CSV": inventory.get("task_results_csv", 0),
        "baseline ranking CSV": inventory.get("baseline_ranking_csv", 0),
        "metrics CSV": inventory.get("metrics_csv", 0),
        "metric summary JSON": inventory.get("metric_summary_json", 0),
        "leakage reports": inventory.get("leakage_report_json", 0),
        "eval audits": inventory.get("eval_audit_json", 0),
        "paper gates": inventory.get("paper_mode_gate_json", 0),
        "tensor arrays found": len(inventory.get("array_like_artifacts") or []),
    }
    matrix = pd.DataFrame({"count": rows}).astype(float)
    vmax = max(float(matrix["count"].max()), 1.0)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        cbar=False,
        linewidths=0.75,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")


def save(fig: plt.Figure, stem: Path) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
