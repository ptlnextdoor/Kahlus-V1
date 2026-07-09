# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
# ---
"""Figure 1: EEG v1 benchmark trajectory.

Benchmark evidence figure. Data source is cached task_results.csv only.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figure_style as style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

TASKS = ("future_state_forecasting", "masked_neural_reconstruction")
TASK_LABELS = {
    "future_state_forecasting": "future forecasting",
    "masked_neural_reconstruction": "masked reconstruction",
}


def main() -> None:
    style.apply_style()
    df = add_bundle_order(eeg_to_eeg(load_task_results(DATA / "task_results.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.45), constrained_layout=True, width_ratios=[1.25, 1.0])
    plot_mse_trajectory(axes[0], df)
    plot_quality_trajectory(axes[1], to_quality_long(df))
    fig.suptitle("EEG→EEG benchmark trajectory from cached task_results.csv", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.text(0.01, -0.015, "Points are saved evidence rows. Lines connect rows in bundle chronology. No synthetic overlays or hand-drawn diagrams.", fontsize=7.2, color=style.GRAY)
    style.save(fig, FIGURES / "Figure1_eeg_v1_benchmark_overview")


def load_task_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in ("eval_pearsonr", "eval_r2", "test_mse", "best_val_mse"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def eeg_to_eeg(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[(df.source_modality == "eeg") & (df.target_modality == "eeg") & df.task_id.isin(TASKS)].copy()
    rows["task_label"] = rows.task_id.map(TASK_LABELS)
    return rows


def add_bundle_order(df: pd.DataFrame) -> pd.DataFrame:
    rows = df.sort_values(["bundle", "task_id"]).reset_index(drop=True).copy()
    run_order = {bundle: idx for idx, bundle in enumerate(rows.bundle.drop_duplicates())}
    rows["run_index"] = rows.bundle.map(run_order)
    rows["run_label"] = rows.bundle.map(short_bundle_label)
    return rows


def to_quality_long(df: pd.DataFrame) -> pd.DataFrame:
    future = df[df.task_id == "future_state_forecasting"]
    return future.melt(
        id_vars=("run_index", "run_label"),
        value_vars=("eval_pearsonr", "eval_r2"),
        var_name="metric",
        value_name="value",
    ).replace({"eval_pearsonr": "Pearson r", "eval_r2": "R²"})


def plot_mse_trajectory(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.set_title("A. Held-out test MSE by evidence bundle", loc="left", fontweight="bold")
    if df.empty:
        empty_axis(ax, "no EEG→EEG task rows")
        return
    sns.lineplot(data=df, x="run_index", y="test_mse", hue="task_label", marker="o", markersize=4.8, linewidth=1.6, palette="colorblind", ax=ax)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.set_ylabel("test MSE, log scale\nlower is better")
    ax.set_xlabel("evidence bundle chronology")
    set_run_ticks(ax, df)
    best = df.loc[df.test_mse.idxmin()]
    ax.annotate(
        f"best saved row\nMSE {best.test_mse:.3g}",
        xy=(best.run_index, best.test_mse),
        xytext=(12, -28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": style.INK},
        fontsize=7.2,
        color=style.INK,
    )
    ax.legend(title="task", frameon=False, loc="upper right")


def plot_quality_trajectory(ax: plt.Axes, long_df: pd.DataFrame) -> None:
    ax.set_title("B. Future-forecasting correlation quality", loc="left", fontweight="bold")
    if long_df.empty:
        empty_axis(ax, "no quality rows")
        return
    sns.lineplot(data=long_df, x="run_index", y="value", hue="metric", marker="o", markersize=4.8, linewidth=1.6, palette=(style.BLUE, style.ORANGE), ax=ax)
    ax.axhline(0, color=style.GRAY, linewidth=0.8, zorder=0)
    ax.set_ylim(-0.08, 1.03)
    ax.set_ylabel("quality\nhigher is better")
    ax.set_xlabel("evidence bundle chronology")
    set_run_ticks(ax, long_df)
    best = long_df[long_df.metric == "Pearson r"].sort_values("value", ascending=False).head(1)
    if not best.empty:
        row = best.iloc[0]
        ax.annotate(
            f"Pearson {row.value:.3f}",
            xy=(row.run_index, row.value),
            xytext=(-48, -22),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": style.INK},
            fontsize=7.2,
            color=style.INK,
        )
    ax.legend(title="metric", frameon=False, loc="lower right")


def set_run_ticks(ax: plt.Axes, df: pd.DataFrame) -> None:
    ticks = df[["run_index", "run_label"]].drop_duplicates().sort_values("run_index")
    ax.set_xticks(ticks.run_index.to_numpy(), ticks.run_label.to_list(), rotation=32, ha="right")


def short_bundle_label(bundle: str) -> str:
    date_match = re.search(r"(20\d{2})-(\d{2})-(\d{2})", str(bundle))
    run_match = re.search(r"results-([0-9a-f]{6,8})", str(bundle))
    date = f"{date_match.group(2)}-{date_match.group(3)}" if date_match else "saved"
    run = run_match.group(1) if run_match else str(bundle).replace(".zip", "")[-8:]
    return f"{date}\n{run}"


def empty_axis(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color=style.GRAY)
    ax.set_axis_off()


if __name__ == "__main__":
    main()
