# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
# ---
"""Figure 3: EEG v1 baseline ranking.

Reference-derived design: long model names and benchmark rankings should use
horizontal dot/summary plots, not crowded vertical bars or diagrams.
"""
from __future__ import annotations

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
    ranking = read_baselines()
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    plot_baseline_ranking(ax, ranking)
    fig.suptitle("EEG v1 baseline ranking from cached baseline_ranking.csv", x=0.02, ha="left", fontsize=11, fontweight="bold")
    save(fig, FIGURES / "Figure3_eeg_v1_baseline_ranking")


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#C7CDD4",
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "kahlus-eeg-v1",
            "savefig.dpi": 300,
        }
    )
    sns.set_theme(context="paper", style="whitegrid", rc={"font.family": "DejaVu Sans"})


def read_baselines() -> pd.DataFrame:
    path = DATA / "baseline_ranking.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["value"] = pd.to_numeric(df.get("value"), errors="coerce")
    df["rank"] = pd.to_numeric(df.get("rank"), errors="coerce")
    df = df[df["metric"].astype(str).str.lower().eq("mse") & df["value"].notna()].copy()
    if df.empty:
        return df
    df["model_label"] = df["model_id"].map(lambda x: str(x).replace("_", " "))
    df["task_label"] = df["task_id"].map(short_task)
    median_order = df.groupby("model_label")["value"].median().sort_values().index.tolist()[:10]
    return df[df["model_label"].isin(median_order)].copy()


def plot_baseline_ranking(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.set_title("A. MSE by model, sorted by median", loc="left", fontweight="bold")
    if df.empty:
        ax.text(0.5, 0.5, "no baseline-ranking rows", ha="center", va="center", transform=ax.transAxes, color="#6B7280")
        ax.set_axis_off()
        return
    order = df.groupby("model_label")["value"].median().sort_values().index.tolist()
    sns.stripplot(
        data=df,
        x="value",
        y="model_label",
        hue="task_label",
        order=order,
        dodge=False,
        jitter=0.18,
        size=4.4,
        linewidth=0.35,
        edgecolor="white",
        alpha=0.82,
        palette="colorblind",
        ax=ax,
    )
    sns.pointplot(
        data=df,
        x="value",
        y="model_label",
        order=order,
        estimator=np.median,
        errorbar=None,
        color="#111827",
        markers="D",
        linestyles="none",
        markersize=3.8,
        ax=ax,
    )
    ax.set_xlabel("MSE, lower is better")
    ax.set_ylabel("")
    ax.legend(title="task", frameon=False, loc="lower right")
    ax.grid(axis="x", color="#E5E7EB")


def short_task(value: str) -> str:
    return {
        "future_state_forecasting": "future forecast",
        "masked_neural_reconstruction": "masked recon",
        "stimulus_to_fmri_response": "stimulus → fMRI",
    }.get(str(value), str(value).replace("_", " ")[:24])


def save(fig: plt.Figure, stem: Path) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
