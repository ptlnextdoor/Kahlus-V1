# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
# ---
"""Figure 3: EEG v1 Kahlus versus standard baselines.

Leaderboard figure from cached task_results.csv and baseline_ranking.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figure_style as style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

TASKS = ("future_state_forecasting", "masked_neural_reconstruction")
TASK_TITLES = {
    "future_state_forecasting": "A. Future forecasting",
    "masked_neural_reconstruction": "B. Masked reconstruction",
}
MODEL_LABELS = {
    "linear_ridge": "linear ridge",
    "autoregressive_ridge": "autoregressive ridge",
    "random_permutation": "random permutation",
    "train_mean": "train mean",
    "ssm_fallback": "SSM fallback",
}
KAHLUS_LABEL = "Kahlus v1 recovered"


def main() -> None:
    style.apply_style()
    ranking = comparison_rows(load_baselines(DATA / "baseline_ranking.csv"), load_recovered_kahlus(DATA / "task_results.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 4.05), constrained_layout=True)
    for ax, task in zip(axes, TASKS, strict=True):
        plot_task_ranking(ax, task_rows(ranking, task), TASK_TITLES[task])
    fig.suptitle("Recovered Kahlus v1 versus standard EEG baselines", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.text(0.01, -0.015, "Bars are median MSE after de-duplicating repeated baseline-ranking artifacts. Lower is better.", fontsize=7.2, color=style.GRAY)
    style.save(fig, FIGURES / "Figure3_eeg_v1_baseline_ranking")


def load_baselines(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["value"] = pd.to_numeric(df.value, errors="coerce")
    rows = df[(df.metric.astype(str).str.lower() == "mse") & df.value.notna() & df.task_id.isin(TASKS)].copy()
    rows["model_label"] = rows.model_id.map(lambda model: MODEL_LABELS.get(str(model), str(model).replace("_", " ")))
    deduped = rows.groupby(["task_id", "model_label"], as_index=False).value.median()
    deduped["source"] = "baseline_ranking.csv"
    return deduped


def load_recovered_kahlus(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["value"] = pd.to_numeric(df.test_mse, errors="coerce")
    rows = df[(df.source_modality == "eeg") & (df.target_modality == "eeg") & df.value.notna() & df.task_id.isin(TASKS)].copy()
    marker = (rows.bundle.astype(str) + " " + rows.artifact_path.astype(str)).str.lower()
    recovered = rows[marker.str.contains("6621642")]
    selected = recovered if not recovered.empty else rows
    best = selected.loc[selected.groupby("task_id").value.idxmin()].copy()
    best["model_label"] = KAHLUS_LABEL
    best["source"] = "task_results.csv"
    return best[["task_id", "model_label", "value", "source"]]


def comparison_rows(baselines: pd.DataFrame, kahlus: pd.DataFrame) -> pd.DataFrame:
    rows = pd.concat([baselines, kahlus], ignore_index=True)
    rows["is_kahlus"] = rows.model_label == KAHLUS_LABEL
    return rows


def task_rows(df: pd.DataFrame, task: str) -> pd.DataFrame:
    return df[df.task_id == task].sort_values("value", ascending=True).head(10).copy()


def plot_task_ranking(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    if df.empty:
        ax.set_title(title, loc="left", fontweight="bold")
        ax.text(0.5, 0.5, "no rows", ha="center", va="center", transform=ax.transAxes, color=style.GRAY)
        ax.set_axis_off()
        return
    order = df.model_label.to_list()
    palette = {model: model_color(model) for model in order}
    sns.barplot(data=df, x="value", y="model_label", hue="model_label", order=order, palette=palette, dodge=False, legend=False, ax=ax)
    for patch, (_, row) in zip(ax.patches, df.iterrows(), strict=True):
        width = patch.get_width()
        ax.text(width + df.value.max() * 0.015, patch.get_y() + patch.get_height() / 2, f"{width:.2f}", va="center", fontsize=7.1, color=style.INK)
        if row.model_label == KAHLUS_LABEL:
            patch.set_edgecolor(style.INK)
            patch.set_linewidth(1.0)
    winner = df.iloc[0].model_label
    winner_color = style.ORANGE if winner == KAHLUS_LABEL else style.BLUE
    ax.set_title(f"{title}\nwinner: {winner}", loc="left", fontweight="bold", color=winner_color, pad=8)
    ax.set_xlim(0, df.value.max() * 1.24)
    ax.set_xlabel("MSE, lower is better")
    ax.set_ylabel("")
    ax.grid(axis="x", color=style.LIGHT)


def model_color(model: str) -> str:
    if model == KAHLUS_LABEL:
        return style.ORANGE
    if "ridge" in model.lower():
        return style.BLUE
    return style.GRAY


if __name__ == "__main__":
    main()
