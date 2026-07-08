# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
# ---
"""Figure 3: EEG v1 Kahlus versus standard baselines.

Task-wise MSE comparison using horizontal seaborn barplots. The recovered
Kahlus v1 row is joined with saved baseline_ranking.csv rows from the archive.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

TASKS = ["future_state_forecasting", "masked_neural_reconstruction"]
TASK_TITLES = {
    "future_state_forecasting": "A. Future forecasting",
    "masked_neural_reconstruction": "B. Masked reconstruction",
}
TASK_LABELS = {
    "future_state_forecasting": "future forecasting",
    "masked_neural_reconstruction": "masked reconstruction",
}
MODEL_RENAME = {
    "linear_ridge": "linear ridge",
    "autoregressive_ridge": "autoregressive ridge",
    "random_permutation": "random permutation",
    "train_mean": "train mean",
    "ssm_fallback": "SSM fallback",
    "Kahlus v1 recovered": "Kahlus v1 recovered",
}
HIGHLIGHT = "Kahlus v1 recovered"


def main() -> None:
    apply_style()
    ranking = read_comparison_rows()
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 4.05), layout="constrained", sharex=False)
    for ax, task in zip(axes, TASKS, strict=True):
        plot_task_ranking(ax, ranking, task)
    fig.suptitle("Recovered Kahlus v1 versus standard EEG baselines", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.text(0.01, -0.015, "Bars are median MSE after de-duplicating repeated baseline-ranking artifacts. Lower is better.", fontsize=7.4, color="#4B5563")
    save(fig, FIGURES / "Figure3_eeg_v1_baseline_ranking")


def apply_style() -> None:
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


def read_comparison_rows() -> pd.DataFrame:
    baselines = read_baselines()
    kahlus = read_recovered_kahlus_results()
    df = pd.concat([baselines, kahlus], ignore_index=True)
    if df.empty:
        return df
    df = df[df["task_id"].isin(TASKS)].copy()
    df["task_label"] = df["task_id"].map(TASK_LABELS)
    df["is_kahlus"] = df["model_label"].eq(HIGHLIGHT)
    return df


def read_baselines() -> pd.DataFrame:
    path = DATA / "baseline_ranking.csv"
    if not path.exists():
        return pd.DataFrame(columns=["task_id", "model_label", "value", "source"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["task_id", "model_label", "value", "source"])
    df["value"] = pd.to_numeric(df.get("value"), errors="coerce")
    df = df[df["metric"].astype(str).str.lower().eq("mse") & df["value"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["task_id", "model_label", "value", "source"])
    df["model_label"] = df["model_id"].map(format_model)
    # The archive contains duplicate BASELINE_RANKING.csv and tables/baseline_ranking.csv files.
    # Median aggregation prevents repeated copies from visually overweighting a model.
    deduped = df.groupby(["task_id", "model_label"], as_index=False)["value"].median()
    deduped["source"] = "baseline_ranking.csv"
    return deduped


def read_recovered_kahlus_results() -> pd.DataFrame:
    path = DATA / "task_results.csv"
    if not path.exists():
        return pd.DataFrame(columns=["task_id", "model_label", "value", "source"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["task_id", "model_label", "value", "source"])
    df["value"] = pd.to_numeric(df.get("test_mse"), errors="coerce")
    df = df[(df.get("source_modality") == "eeg") & (df.get("target_modality") == "eeg") & df["value"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["task_id", "model_label", "value", "source"])
    marker_text = (df.get("bundle", "").astype(str) + " " + df.get("artifact_path", "").astype(str)).str.lower()
    recovered = df[marker_text.str.contains("6621642")].copy()
    if recovered.empty:
        idx = df.groupby("task_id")["value"].idxmin()
        recovered = df.loc[idx].copy()
    else:
        idx = recovered.groupby("task_id")["value"].idxmin()
        recovered = recovered.loc[idx].copy()
    recovered["model_label"] = HIGHLIGHT
    recovered["source"] = "task_results.csv"
    return recovered[["task_id", "model_label", "value", "source"]]


def plot_task_ranking(ax: plt.Axes, df: pd.DataFrame, task: str) -> None:
    task_df = df[df["task_id"].eq(task)].copy()
    if task_df.empty:
        ax.set_title(TASK_TITLES[task], loc="left", fontweight="bold")
        ax.text(0.5, 0.5, "no rows", ha="center", va="center", transform=ax.transAxes, color="#6B7280")
        ax.set_axis_off()
        return
    task_df = task_df.sort_values("value", ascending=True).head(10).copy()
    order = task_df["model_label"].tolist()
    colors = {model: ("#D55E00" if model == HIGHLIGHT else "#0072B2" if "ridge" in model.lower() else "#6B7280") for model in order}
    sns.barplot(data=task_df, x="value", y="model_label", hue="model_label", order=order, palette=colors, dodge=False, legend=False, ax=ax)
    for patch, (_, row) in zip(ax.patches, task_df.iterrows(), strict=True):
        width = patch.get_width()
        ax.text(width + task_df["value"].max() * 0.015, patch.get_y() + patch.get_height() / 2, f"{width:.2f}", va="center", fontsize=7.1, color="#111827")
        if row["model_label"] == HIGHLIGHT:
            patch.set_edgecolor("#111827")
            patch.set_linewidth(1.0)
    winner = task_df.iloc[0]
    if winner["model_label"] == HIGHLIGHT:
        note = "winner: Kahlus v1 recovered"
        color = "#D55E00"
    else:
        note = f"winner: {winner['model_label']}"
        color = "#0072B2"
    ax.set_title(f"{TASK_TITLES[task]}\n{note}", loc="left", fontweight="bold", color=color, pad=8)
    ax.set_xlim(0, task_df["value"].max() * 1.24)
    ax.set_xlabel("MSE, lower is better")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E5E7EB")


def format_model(model: str) -> str:
    text = str(model)
    return MODEL_RENAME.get(text, text.replace("_", " "))


def save(fig: plt.Figure, stem: Path) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
