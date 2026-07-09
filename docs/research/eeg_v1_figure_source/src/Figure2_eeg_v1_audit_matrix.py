# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
# ---
"""Figure 2: EEG v1 audit and artifact coverage.

Benchmark-evidence availability figure from cached audits.csv and inventory.json.
"""
from __future__ import annotations

import json
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

AUDIT_ORDER = ("leakage", "eval", "paper mode gate")
STATUS_ORDER = ("passed", "failed", "unknown")


def main() -> None:
    style.apply_style()
    audits = load_audits(DATA / "audits.csv")
    inventory = load_inventory(DATA / "inventory.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.25), constrained_layout=True, width_ratios=[1.0, 1.25])
    plot_audit_status(axes[0], audit_status_matrix(audits))
    plot_inventory_counts(axes[1], inventory_counts(inventory))
    fig.suptitle("Evidence coverage and audit status from cached CSV/JSON artifacts", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.text(0.01, -0.015, "Evidence-availability figure only. It does not imply physiological validity or clinical utility.", fontsize=7.2, color=style.GRAY)
    style.save(fig, FIGURES / "Figure2_eeg_v1_audit_matrix")


def load_audits(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["audit_label"] = df.audit_type.map(lambda value: str(value).replace("_", " "))
    df["status"] = df.passed.map({True: "passed", False: "failed", "true": "passed", "false": "failed"}).fillna("unknown")
    return df


def load_inventory(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_status_matrix(audits: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(audits.audit_label, audits.status).reindex(index=AUDIT_ORDER, columns=STATUS_ORDER, fill_value=0).astype(int)


def inventory_counts(inventory: dict[str, object]) -> pd.DataFrame:
    rows = [
        ("evidence bundles", inventory.get("bundle_count", 0)),
        ("task_results.csv", inventory.get("task_results_csv", 0)),
        ("baseline_ranking.csv", inventory.get("baseline_ranking_csv", 0)),
        ("metrics.csv", inventory.get("metrics_csv", 0)),
        ("metric_summary.json", inventory.get("metric_summary_json", 0)),
        ("leakage reports", inventory.get("leakage_report_json", 0)),
        ("eval audits", inventory.get("eval_audit_json", 0)),
        ("paper gates", inventory.get("paper_mode_gate_json", 0)),
        ("tensor/prediction arrays", len(inventory.get("array_like_artifacts") or [])),
    ]
    df = pd.DataFrame(rows, columns=("artifact", "count"))
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)
    return df.sort_values("count", ascending=True)


def plot_audit_status(ax: plt.Axes, matrix: pd.DataFrame) -> None:
    sns.heatmap(matrix, annot=True, fmt="d", cmap="cividis", cbar=False, linewidths=0.8, linecolor="white", ax=ax)
    ax.set_title("A. Audit artifacts by status", loc="left", fontweight="bold")
    ax.set_xlabel("status")
    ax.set_ylabel("audit type")


def plot_inventory_counts(ax: plt.Axes, counts: pd.DataFrame) -> None:
    sns.barplot(data=counts, x="count", y="artifact", hue="count", palette="viridis", dodge=False, legend=False, ax=ax)
    offset = max(float(counts["count"].max()) * 0.015, 0.15)
    for patch in ax.patches:
        width = patch.get_width()
        ax.text(width + offset, patch.get_y() + patch.get_height() / 2, f"{int(width)}", va="center", fontsize=7.2, color=style.INK)
    ax.set_title("B. Cached artifact families", loc="left", fontweight="bold")
    ax.set_xlabel("files found in versions evidence")
    ax.set_ylabel("")
    ax.grid(axis="x", color=style.LIGHT)


if __name__ == "__main__":
    main()
