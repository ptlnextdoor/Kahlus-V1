# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
# ---
"""Figure 2: EEG v1 audit and artifact coverage.

Compact evidence-coverage plots using seaborn/matplotlib. This figure shows
what can and cannot support claims from the cached archive.
"""
from __future__ import annotations

import json
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

AUDIT_ORDER = ["leakage", "eval", "paper mode gate"]
STATUS_ORDER = ["passed", "failed", "unknown"]
STATUS_COLORS = {"passed": "#009E73", "failed": "#D55E00", "unknown": "#9CA3AF"}


def main() -> None:
    apply_style()
    audits = read_audits()
    inventory = read_json(DATA / "inventory.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.25), layout="constrained", width_ratios=[1.08, 1.22])
    plot_audit_status(axes[0], audits)
    plot_inventory_counts(axes[1], inventory)
    fig.suptitle("Evidence coverage and audit status from cached CSV/JSON artifacts", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.text(0.01, -0.015, "This is an evidence-availability figure. It does not imply physiological validity or clinical utility.", fontsize=7.4, color="#4B5563")
    save(fig, FIGURES / "Figure2_eeg_v1_audit_matrix")


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
    ax.set_title("A. Audit artifacts by status", loc="left", fontweight="bold")
    if audits.empty:
        matrix = pd.DataFrame(0, index=AUDIT_ORDER, columns=STATUS_ORDER)
    else:
        matrix = pd.crosstab(audits["audit_label"], audits["status"]).reindex(index=AUDIT_ORDER, columns=STATUS_ORDER, fill_value=0).astype(int)
    left = pd.Series(0, index=matrix.index, dtype=float)
    y = range(len(matrix.index))
    for status in STATUS_ORDER:
        values = matrix[status]
        ax.barh(y, values, left=left, color=STATUS_COLORS[status], label=status, height=0.62)
        for yi, value, x0 in zip(y, values, left, strict=True):
            if value > 0:
                ax.text(x0 + value / 2, yi, str(int(value)), ha="center", va="center", fontsize=7.2, color="white", fontweight="bold")
        left += values
    ax.set_yticks(list(y), matrix.index)
    ax.invert_yaxis()
    ax.set_xlabel("audit files")
    ax.set_ylabel("")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#E5E7EB")


def plot_inventory_counts(ax: plt.Axes, inventory: dict) -> None:
    ax.set_title("B. Cached artifact families", loc="left", fontweight="bold")
    rows = pd.DataFrame(
        [
            ("evidence bundles", inventory.get("bundle_count", 0)),
            ("task_results.csv", inventory.get("task_results_csv", 0)),
            ("baseline_ranking.csv", inventory.get("baseline_ranking_csv", 0)),
            ("metrics.csv", inventory.get("metrics_csv", 0)),
            ("metric_summary.json", inventory.get("metric_summary_json", 0)),
            ("leakage reports", inventory.get("leakage_report_json", 0)),
            ("eval audits", inventory.get("eval_audit_json", 0)),
            ("paper gates", inventory.get("paper_mode_gate_json", 0)),
            ("tensor/prediction arrays", len(inventory.get("array_like_artifacts") or [])),
        ],
        columns=["artifact", "count"],
    )
    rows["count"] = pd.to_numeric(rows["count"], errors="coerce").fillna(0)
    rows = rows.sort_values("count", ascending=True)
    colors = ["#D55E00" if name == "tensor/prediction arrays" and count == 0 else "#0072B2" for name, count in zip(rows["artifact"], rows["count"], strict=True)]
    sns.barplot(data=rows, x="count", y="artifact", hue="artifact", palette=dict(zip(rows["artifact"], colors, strict=True)), dodge=False, legend=False, ax=ax)
    for patch in ax.patches:
        width = patch.get_width()
        ax.text(width + max(rows["count"].max() * 0.015, 0.15), patch.get_y() + patch.get_height() / 2, f"{int(width)}", va="center", fontsize=7.2, color="#111827")
    ax.set_xlabel("files found in versions evidence")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E5E7EB")


def save(fig: plt.Figure, stem: Path) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
