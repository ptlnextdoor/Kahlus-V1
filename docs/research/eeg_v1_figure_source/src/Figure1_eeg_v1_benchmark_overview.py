# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
# ---
"""Figure 1: EEG v1 benchmark overview.

Reference-derived design: use the standard scientific Python stack
(pandas + seaborn + matplotlib constrained layout) for statistical plots.
No custom box/arrow layout is used.
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

PALETTE = {
    "future forecast": "#1F77B4",
    "masked recon": "#2CA02C",
    "stimulus → fMRI": "#9467BD",
    "other": "#7F7F7F",
}


def main() -> None:
    apply_style()
    tasks = read_task_results()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), layout="constrained")
    metrics = [
        ("eval_pearsonr", "A. Pearson r", "higher is better", None),
        ("eval_r2", "B. R2", "higher is better", None),
        ("test_mse", "C. test MSE", "lower is better", None),
        ("best_val_mse", "D. best validation MSE", "lower is better", None),
    ]
    for ax, (metric, title, ylabel, ylim) in zip(axes.flat, metrics, strict=True):
        plot_metric(ax, tasks, metric, title, ylabel, ylim)
    fig.suptitle("EEG v1 benchmark overview from cached task_results.csv", x=0.02, ha="left", fontsize=11, fontweight="bold")
    save(fig, FIGURES / "Figure1_eeg_v1_benchmark_overview")


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
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


def read_task_results() -> pd.DataFrame:
    path = DATA / "task_results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    for col in ("eval_pearsonr", "eval_r2", "test_mse", "best_val_mse"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df = df[(df.get("source_modality") == "eeg") & (df.get("target_modality") == "eeg")].copy()
    if df.empty:
        return df
    df["task_label"] = df["task_id"].map(short_task).fillna("other")
    return df


def plot_metric(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str, ylabel: str, ylim: tuple[float, float] | None) -> None:
    ax.set_title(title, loc="left", fontweight="bold")
    if df.empty or metric not in df or df[metric].dropna().empty:
        ax.text(0.5, 0.5, "no EEG→EEG rows", ha="center", va="center", transform=ax.transAxes, color="#6B7280")
        ax.set_axis_off()
        return
    order = sorted(df["task_label"].dropna().unique())
    sns.stripplot(
        data=df,
        x="task_label",
        y=metric,
        order=order,
        hue="task_label",
        palette=PALETTE,
        jitter=0.16,
        size=4.0,
        linewidth=0.35,
        edgecolor="white",
        alpha=0.82,
        legend=False,
        ax=ax,
    )
    sns.pointplot(
        data=df,
        x="task_label",
        y=metric,
        order=order,
        estimator=np.median,
        errorbar=None,
        color="#111827",
        markers="_",
        linestyles="none",
        markersize=16,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if metric in {"eval_pearsonr", "eval_r2"}:
        ax.axhline(0, color="#9CA3AF", lw=0.8, zorder=0)
    ax.tick_params(axis="x", rotation=18)


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
