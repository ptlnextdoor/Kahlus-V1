# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
# ---
"""Figure 1: EEG v1 benchmark trajectory.

Publication-style data plot using pandas, seaborn, matplotlib, and tueplots
when installed. The figure is driven only by cached task_results.csv rows.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

TASK_ORDER = ["future_state_forecasting", "masked_neural_reconstruction"]
TASK_LABELS = {
    "future_state_forecasting": "future forecasting",
    "masked_neural_reconstruction": "masked reconstruction",
    "stimulus_to_fmri_response": "stimulus → fMRI",
}
TASK_PALETTE = {
    "future forecasting": "#0072B2",
    "masked reconstruction": "#009E73",
}
METRIC_PALETTE = {
    "Pearson r": "#0072B2",
    "R²": "#D55E00",
}


def main() -> None:
    apply_style(width="full")
    tasks = read_task_results()
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.45), layout="constrained", width_ratios=[1.25, 1.0])
    plot_mse_trajectory(axes[0], tasks)
    plot_quality_trajectory(axes[1], tasks)
    fig.suptitle("EEG→EEG benchmark trajectory from cached task_results.csv", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.text(0.01, -0.015, "Points are saved evidence rows. Lines connect rows in bundle chronology. No synthetic overlays or hand-drawn diagram elements are used.", fontsize=7.4, color="#4B5563")
    save(fig, FIGURES / "Figure1_eeg_v1_benchmark_overview")


def apply_style(width: str = "full") -> None:
    rc: dict[str, object] = {}
    try:
        from tueplots import bundles  # type: ignore

        for name in ("neurips2024", "neurips2023", "icml2022"):
            factory = getattr(bundles, name, None)
            if factory is None:
                continue
            for kwargs in ({"usetex": False}, {}):
                try:
                    rc = factory(**kwargs)
                    break
                except TypeError:
                    continue
            if rc:
                break
    except Exception:
        rc = {}
    mpl.rcParams.update(rc)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#C7CDD4",
            "axes.linewidth": 0.8,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "kahlus-eeg-v1-paper",
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
    df = df[df["task_id"].isin(TASK_ORDER)].copy()
    if df.empty:
        return df
    df["task_label"] = pd.Categorical(df["task_id"].map(TASK_LABELS), categories=[TASK_LABELS[t] for t in TASK_ORDER], ordered=True)
    df["bundle_date"] = df["bundle"].map(extract_date)
    df["run_id"] = df["bundle"].map(extract_run_id)
    df = df.sort_values(["bundle", "task_id"]).reset_index(drop=True)
    run_order = {bundle: i for i, bundle in enumerate(df["bundle"].drop_duplicates())}
    df["run_index"] = df["bundle"].map(run_order)
    df["run_label"] = df["bundle_date"] + "\n" + df["run_id"]
    return df


def plot_mse_trajectory(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.set_title("A. Held-out test MSE by evidence bundle", loc="left", fontweight="bold")
    if df.empty or df["test_mse"].dropna().empty:
        empty_axis(ax, "no EEG→EEG MSE rows")
        return
    sns.lineplot(data=df, x="run_index", y="test_mse", hue="task_label", marker="o", markersize=4.8, linewidth=1.6, palette=TASK_PALETTE, ax=ax)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.set_ylabel("test MSE, log scale\nlower is better")
    ax.set_xlabel("evidence bundle chronology")
    set_run_ticks(ax, df)
    best = df.loc[df["test_mse"].idxmin()]
    ax.annotate(
        f"best saved row\nMSE {best['test_mse']:.3g}",
        xy=(best["run_index"], best["test_mse"]),
        xytext=(12, -28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#374151"},
        fontsize=7.2,
        color="#111827",
    )
    ax.legend(title="task", frameon=False, loc="upper right")


def plot_quality_trajectory(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.set_title("B. Correlation quality for saved EEG runs", loc="left", fontweight="bold")
    if df.empty or df[["eval_pearsonr", "eval_r2"]].dropna(how="all").empty:
        empty_axis(ax, "no EEG→EEG quality rows")
        return
    future = df[df["task_id"].eq("future_state_forecasting")].copy()
    if future.empty:
        future = df.copy()
    long = future.melt(
        id_vars=["run_index", "run_label"],
        value_vars=["eval_pearsonr", "eval_r2"],
        var_name="metric",
        value_name="value",
    ).dropna()
    long["metric"] = long["metric"].map({"eval_pearsonr": "Pearson r", "eval_r2": "R²"})
    sns.lineplot(data=long, x="run_index", y="value", hue="metric", marker="o", markersize=4.8, linewidth=1.6, palette=METRIC_PALETTE, ax=ax)
    ax.axhline(0, color="#9CA3AF", linewidth=0.8, zorder=0)
    ax.set_ylim(-0.08, 1.03)
    ax.set_ylabel("future forecasting quality\nhigher is better")
    ax.set_xlabel("evidence bundle chronology")
    set_run_ticks(ax, df)
    best = long[long["metric"].eq("Pearson r")].sort_values("value", ascending=False).head(1)
    if not best.empty:
        row = best.iloc[0]
        ax.annotate(
            f"Pearson {row['value']:.3f}",
            xy=(row["run_index"], row["value"]),
            xytext=(-48, -22),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#374151"},
            fontsize=7.2,
            color="#111827",
        )
    ax.legend(title="metric", frameon=False, loc="lower right")


def set_run_ticks(ax: plt.Axes, df: pd.DataFrame) -> None:
    ticks = df[["run_index", "run_label"]].drop_duplicates().sort_values("run_index")
    ax.set_xticks(ticks["run_index"].to_numpy(), ticks["run_label"].to_list(), rotation=32, ha="right")


def extract_date(bundle: str) -> str:
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", str(bundle))
    return match.group(1)[5:] if match else "saved"


def extract_run_id(bundle: str) -> str:
    text = str(bundle)
    for pattern in (r"results-([0-9a-f]{6,8})", r"neurotwin-([0-9a-f]{6,8})", r"([0-9a-f]{7})"):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return text.replace(".zip", "")[-8:]


def empty_axis(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#6B7280")
    ax.set_axis_off()


def save(fig: plt.Figure, stem: Path) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
