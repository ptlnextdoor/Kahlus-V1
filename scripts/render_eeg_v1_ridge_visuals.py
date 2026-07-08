#!/usr/bin/env python3
"""Render real artifact-driven EEG/ridge evidence figures from versions bundles.

This script intentionally avoids synthetic traces and hand-drawn prediction overlays.
It inventories evidence zips under /Users/aayu/Downloads/versions, normalizes the
CSV/JSON artifacts that are actually present, and renders figures only from those
artifacts. If raw tensor or prediction arrays are absent, the generated report says
so instead of inventing a waveform.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import runpy
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

KBLUE = "#0B3D91"
KTEAL = "#0F766E"
KGOLD = "#B7791F"
KRED = "#B42318"
KGREEN = "#146C43"
KGRAY = "#F5F7FA"
KINK = "#1F2937"
KMID = "#6B7280"
KLINE = "#D1D5DB"

DEFAULT_VERSIONS_ROOT = Path("/Users/aayu/Downloads/versions")
CANONICAL_FIGURE_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "docs" / "research" / "eeg_v1_figure_source" / "src"
ARRAY_ARTIFACT_RE = re.compile(
    r"(\.npz$|\.npy$|\.parquet$|\.pkl$|pred|prediction|y_true|y_pred|forecast|tensor|epoch|\.fif$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TaskResult:
    bundle: str
    artifact_path: str
    task_id: str
    source_modality: str
    target_modality: str
    eval_mse: float
    eval_mae: float
    eval_pearsonr: float
    eval_r2: float
    test_mse: float
    best_val_mse: float


@dataclass(frozen=True)
class BaselineResult:
    bundle: str
    artifact_path: str
    task_id: str
    model_id: str
    metric: str
    value: float
    rank: float


@dataclass(frozen=True)
class AuditResult:
    bundle: str
    artifact_path: str
    audit_type: str
    passed: bool | None
    violations: int
    warnings: int
    checked: int
    observed_seeds: int | None = None
    window_count: int | None = None


@dataclass(frozen=True)
class ArtifactInventory:
    bundle_count: int
    task_results_csv: int
    baseline_ranking_csv: int
    metrics_csv: int
    metric_summary_json: int
    leakage_report_json: int
    eval_audit_json: int
    paper_mode_gate_json: int
    array_like_artifacts: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions-root", type=Path, default=DEFAULT_VERSIONS_ROOT)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    evidence = load_versions_evidence(args.versions_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    figure_source_root = args.out_dir.parent / "eeg_v1_figure_source"
    write_figure_source_packet(evidence, args.versions_root, figure_source_root)
    run_figure_source_scripts(figure_source_root)

    figure_stems = {
        "benchmark_overview": "../eeg_v1_figure_source/figures/Figure1_eeg_v1_benchmark_overview",
        "audit_matrix": "../eeg_v1_figure_source/figures/Figure2_eeg_v1_audit_matrix",
        "baseline_ranking": "../eeg_v1_figure_source/figures/Figure3_eeg_v1_baseline_ranking",
    }

    summary = build_summary(evidence, args.versions_root, figure_stems)
    write_json(args.out_dir / "eeg_v1_ridge_visual_summary.json", summary)
    (args.out_dir / "eeg_v1_ridge_visual_analysis.md").write_text(analysis_md(summary), encoding="utf-8")
    print(args.out_dir / "eeg_v1_ridge_visual_analysis.md")
    return 0


def load_versions_evidence(root: Path) -> dict[str, Any]:
    zip_paths = evidence_zip_paths(root)
    inventory_counts = {
        "task_results_csv": 0,
        "baseline_ranking_csv": 0,
        "metrics_csv": 0,
        "metric_summary_json": 0,
        "leakage_report_json": 0,
        "eval_audit_json": 0,
        "paper_mode_gate_json": 0,
    }
    array_like: list[str] = []
    task_results: list[TaskResult] = []
    baseline_results: list[BaselineResult] = []
    audits: list[AuditResult] = []

    for zip_path in zip_paths:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                array_like.extend(f"{zip_path.name}:{name}" for name in names if ARRAY_ARTIFACT_RE.search(name))
                for name in names:
                    lower = name.lower()
                    if lower.endswith("task_results.csv"):
                        inventory_counts["task_results_csv"] += 1
                        task_results.extend(read_task_results(zf, name, zip_path.name))
                    elif lower.endswith("baseline_ranking.csv"):
                        inventory_counts["baseline_ranking_csv"] += 1
                        baseline_results.extend(read_baseline_ranking(zf, name, zip_path.name))
                    elif lower.endswith("metrics.csv"):
                        inventory_counts["metrics_csv"] += 1
                    elif lower.endswith("metric_summary.json"):
                        inventory_counts["metric_summary_json"] += 1
                    elif lower.endswith("leakage_report.json"):
                        inventory_counts["leakage_report_json"] += 1
                        audits.append(read_audit(zf, name, zip_path.name, "leakage"))
                    elif lower.endswith("eval_audit.json"):
                        inventory_counts["eval_audit_json"] += 1
                        audits.append(read_audit(zf, name, zip_path.name, "eval"))
                    elif lower.endswith("paper_mode_gate.json"):
                        inventory_counts["paper_mode_gate_json"] += 1
                        audits.append(read_audit(zf, name, zip_path.name, "paper_mode_gate"))
        except zipfile.BadZipFile:
            continue

    inventory = ArtifactInventory(bundle_count=len(zip_paths), array_like_artifacts=sorted(array_like), **inventory_counts)
    return {
        "inventory": inventory,
        "task_results": task_results,
        "baseline_results": baseline_results,
        "audits": audits,
    }


def write_figure_source_packet(evidence: dict[str, Any], versions_root: Path, root: Path) -> None:
    """Write CEBRA-style cached data files consumed by figure source scripts."""

    data_dir = root / "data"
    figures_dir = root / "figures"
    src_dir = root / "src"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)
    copy_canonical_figure_sources(src_dir)

    inv: ArtifactInventory = evidence["inventory"]
    write_json(data_dir / "inventory.json", asdict(inv))
    write_json(
        data_dir / "provenance.json",
        {
            "versions_root": str(versions_root),
            "source_mode": "versions_evidence",
            "renderer": "scripts/render_eeg_v1_ridge_visuals.py",
            "figure_source_root": str(root),
            "evidence_rule": "Only cached CSV/JSON/NPZ artifacts may drive public figures.",
        },
    )
    write_rows(
        data_dir / "task_results.csv",
        evidence["task_results"],
        [
            "bundle",
            "artifact_path",
            "task_id",
            "source_modality",
            "target_modality",
            "eval_mse",
            "eval_mae",
            "eval_pearsonr",
            "eval_r2",
            "test_mse",
            "best_val_mse",
        ],
    )
    write_rows(
        data_dir / "baseline_ranking.csv",
        evidence["baseline_results"],
        ["bundle", "artifact_path", "task_id", "model_id", "metric", "value", "rank"],
    )
    write_rows(
        data_dir / "audits.csv",
        evidence["audits"],
        [
            "bundle",
            "artifact_path",
            "audit_type",
            "passed",
            "violations",
            "warnings",
            "checked",
            "observed_seeds",
            "window_count",
        ],
    )
    (root / "README.md").write_text(figure_source_readme(), encoding="utf-8")


def write_rows(path: Path, rows: list[Any], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            normalized = {key: normalize_cell(data.get(key)) for key in fieldnames}
            writer.writerow(normalized)


def normalize_cell(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return value


def run_figure_source_scripts(root: Path) -> None:
    for script in (
        root / "src" / "Figure1_eeg_v1_benchmark_overview.py",
        root / "src" / "Figure2_eeg_v1_audit_matrix.py",
        root / "src" / "Figure3_eeg_v1_baseline_ranking.py",
    ):
        if not script.exists():
            raise FileNotFoundError(f"missing figure source script: {script}")
        runpy.run_path(str(script), run_name="__main__")


def copy_canonical_figure_sources(target_src_dir: Path) -> None:
    for name in (
        "Figure1_eeg_v1_benchmark_overview.py",
        "Figure2_eeg_v1_audit_matrix.py",
        "Figure3_eeg_v1_baseline_ranking.py",
    ):
        source = CANONICAL_FIGURE_SOURCE_ROOT / name
        target = target_src_dir / name
        if source.resolve() == target.resolve():
            continue
        if not source.exists():
            raise FileNotFoundError(f"missing canonical figure source script: {source}")
        shutil.copy2(source, target)


def figure_source_readme() -> str:
    return """# EEG v1 figure-source packet

This directory follows the CEBRA paper-figure pattern: cached data artifacts live in `data/`, figure source scripts live in `src/`, and rendered figures live in `figures/`.

## Files

- `data/task_results.csv`: normalized rows parsed from versions evidence `task_results.csv` artifacts.
- `data/baseline_ranking.csv`: normalized rows parsed from `baseline_ranking.csv` artifacts.
- `data/audits.csv`: leakage, eval, and paper-mode gate rows parsed from JSON artifacts.
- `data/inventory.json`: counts of evidence artifacts and whether raw tensor/prediction arrays exist.
- `data/provenance.json`: source root and renderer provenance.
- `src/Figure1_eeg_v1_benchmark_overview.py`: standard seaborn task metric panels.
- `src/Figure2_eeg_v1_audit_matrix.py`: compact audit and inventory heatmaps.
- `src/Figure3_eeg_v1_baseline_ranking.py`: horizontal baseline-ranking dot plot.

## Regenerate

```bash
PYTHONPATH=src python scripts/render_eeg_v1_ridge_visuals.py \
  --versions-root /Users/aayu/Downloads/versions \
  --out-dir docs/research/eeg_v1_ridge_visuals
```

Public rule: no raw tensor or prediction-array artifact means no waveform overlay, no residual trace, and no clinical/physiology claim figure. Public evidence figures use standard matplotlib/seaborn axes with constrained layout, not hand-drawn box diagrams.
"""


def evidence_zip_paths(root: Path) -> list[Path]:
    candidates = [root / "_organized_index" / "artifact_views" / "evidence_zips", root]
    paths: list[Path] = []
    for base in candidates:
        if base.exists():
            paths.extend(p for p in base.rglob("*.zip") if p.is_file())
    dedup = {p.resolve(): p for p in paths}
    return sorted(dedup.values())


def read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    text = zf.read(name).decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def read_json_from_zip(zf: zipfile.ZipFile, name: str) -> dict[str, Any]:
    data = json.loads(zf.read(name).decode("utf-8", errors="replace"))
    return data if isinstance(data, dict) else {}


def read_task_results(zf: zipfile.ZipFile, name: str, bundle: str) -> list[TaskResult]:
    rows: list[TaskResult] = []
    for row in read_csv_from_zip(zf, name):
        try:
            rows.append(
                TaskResult(
                    bundle=bundle,
                    artifact_path=name,
                    task_id=row.get("task_id", "unknown"),
                    source_modality=row.get("source_modality", "unknown"),
                    target_modality=row.get("target_modality", "unknown"),
                    eval_mse=to_float(row.get("eval_mse")),
                    eval_mae=to_float(row.get("eval_mae")),
                    eval_pearsonr=to_float(row.get("eval_pearsonr")),
                    eval_r2=to_float(row.get("eval_r2")),
                    test_mse=to_float(row.get("test_mse")),
                    best_val_mse=to_float(row.get("best_val_mse")),
                )
            )
        except ValueError:
            continue
    return rows


def read_baseline_ranking(zf: zipfile.ZipFile, name: str, bundle: str) -> list[BaselineResult]:
    rows: list[BaselineResult] = []
    for row in read_csv_from_zip(zf, name):
        if not row.get("model_id"):
            continue
        try:
            rows.append(
                BaselineResult(
                    bundle=bundle,
                    artifact_path=name,
                    task_id=row.get("task_id", "unknown"),
                    model_id=row.get("model_id", "unknown"),
                    metric=row.get("metric", "unknown"),
                    value=to_float(row.get("value")),
                    rank=to_float(row.get("rank")),
                )
            )
        except ValueError:
            continue
    return rows


def read_audit(zf: zipfile.ZipFile, name: str, bundle: str, audit_type: str) -> AuditResult:
    data = read_json_from_zip(zf, name)
    return AuditResult(
        bundle=bundle,
        artifact_path=name,
        audit_type=audit_type,
        passed=data.get("passed") if isinstance(data.get("passed"), bool) else None,
        violations=count_items(data.get("violations")),
        warnings=count_items(data.get("warnings")),
        checked=count_items(data.get("checked") or data.get("checked_keys")),
        observed_seeds=count_items(data.get("observed_seeds")) if "observed_seeds" in data else None,
        window_count=int(data["window_count"]) if isinstance(data.get("window_count"), int) else None,
    )


def to_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def count_items(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return 1


def finite(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.06, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=12, fontweight="bold", color=KINK, va="top")


def card(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, *, fc: str, ec: str, color: str = KINK, weight: str = "normal") -> None:
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec=ec, lw=1.0)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6, color=color, weight=weight, linespacing=1.18)


def render_inventory_figure(evidence: dict[str, Any], versions_root: Path, stem: Path) -> None:
    inv: ArtifactInventory = evidence["inventory"]
    counts = {
        "task\nresults": inv.task_results_csv,
        "baseline\nrankings": inv.baseline_ranking_csv,
        "metrics\nCSV": inv.metrics_csv,
        "metric\nsummary": inv.metric_summary_json,
        "leakage\nreports": inv.leakage_report_json,
        "eval\naudits": inv.eval_audit_json,
        "paper\ngates": inv.paper_mode_gate_json,
    }
    fig = plt.figure(figsize=(7.4, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A")
    labels = list(counts)
    vals = list(counts.values())
    colors = [KBLUE, KTEAL, KBLUE, KTEAL, KGOLD, KGREEN, KRED]
    ax.bar(labels, vals, color=colors, alpha=0.86)
    ax.set_title("Evidence artifact inventory", color=KBLUE, weight="bold")
    ax.set_ylabel("files found in evidence zips")
    ax.grid(axis="y", alpha=0.8)
    ax.tick_params(axis="x", rotation=0)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals + [1]) * 0.02, str(v), ha="center", va="bottom", fontsize=8, color=KINK)

    ax2 = fig.add_subplot(gs[0, 1])
    panel_label(ax2, "B")
    ax2.set_axis_off()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    card(ax2, 0.05, 0.72, 0.90, 0.18, f"{inv.bundle_count} evidence bundles\nscanned", fc="#DBEAFE", ec=KBLUE, weight="bold")
    card(ax2, 0.05, 0.49, 0.90, 0.18, f"{len(evidence['task_results'])} task rows\n{len(evidence['baseline_results'])} baseline rows", fc="#ECFDF5", ec=KTEAL, weight="bold")
    tensor_text = "raw tensors found" if inv.array_like_artifacts else "no raw tensor arrays\nfound in bundles"
    tensor_fc = "#FEE2E2" if not inv.array_like_artifacts else "#ECFDF5"
    tensor_ec = KRED if not inv.array_like_artifacts else KGREEN
    card(ax2, 0.05, 0.26, 0.90, 0.18, tensor_text, fc=tensor_fc, ec=tensor_ec, weight="bold")
    ax2.text(0.05, 0.09, f"source root:\n{versions_root}", fontsize=7.2, color=KMID, va="top", family="monospace")
    save_figure(fig, stem)


def render_task_metrics_figure(task_results: list[TaskResult], stem: Path) -> None:
    eeg = [r for r in task_results if r.source_modality == "eeg" and r.target_modality == "eeg"]
    fig = plt.figure(figsize=(7.4, 4.8))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.33)
    metrics = [
        ("eval_pearsonr", "Pearson r", KBLUE),
        ("eval_r2", "$R^2$", KTEAL),
        ("test_mse", "test MSE", KRED),
        ("best_val_mse", "best val MSE", KGOLD),
    ]
    tasks = sorted({r.task_id for r in eeg}) or ["no_eeg_rows"]
    for ax_i, (field, label, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[ax_i // 2, ax_i % 2])
        panel_label(ax, chr(ord("A") + ax_i))
        for x, task in enumerate(tasks):
            vals = finite(getattr(r, field) for r in eeg if r.task_id == task)
            if vals:
                jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else [0.0]
                ax.scatter(np.full(len(vals), x) + jitter, vals, s=30, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
                ax.plot([x - 0.18, x + 0.18], [np.median(vals), np.median(vals)], color=KINK, lw=1.2)
            else:
                ax.text(x, 0.5, "no data", ha="center", va="center", color=KMID, transform=ax.get_xaxis_transform())
        ax.set_title(label, color=KINK)
        ax.set_xticks(range(len(tasks)), [short_task(t) for t in tasks], rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.8)
        if "mse" in field:
            ax.set_ylabel("lower is better")
        else:
            ax.set_ylabel("higher is better")
    fig.suptitle("[Benchmark evidence] EEG task metrics from versions artifacts", color=KBLUE, weight="bold", y=1.02)
    save_figure(fig, stem)


def render_baseline_ranking_figure(rows: list[BaselineResult], stem: Path) -> None:
    mse_rows = [r for r in rows if r.metric.lower() == "mse" and math.isfinite(r.value)]
    by_model: dict[str, list[float]] = {}
    for row in mse_rows:
        by_model.setdefault(row.model_id, []).append(row.value)
    models = sorted(by_model, key=lambda m: np.median(by_model[m]))[:8]

    fig = plt.figure(figsize=(7.4, 4.25))
    ax = fig.add_subplot(111)
    panel_label(ax, "A")
    if models:
        for x, model in enumerate(models):
            vals = by_model[model]
            jitter = np.linspace(-0.10, 0.10, len(vals)) if len(vals) > 1 else [0.0]
            ax.scatter(np.full(len(vals), x) + jitter, vals, s=36, color=KBLUE if "ridge" in model else KTEAL, alpha=0.78, edgecolor="white", linewidth=0.5)
            ax.plot([x - 0.22, x + 0.22], [np.median(vals), np.median(vals)], color=KINK, lw=1.3)
        ax.set_xticks(range(len(models)), [m.replace("_", "\n") for m in models])
        ax.set_ylabel("MSE across saved baseline-ranking artifacts")
        ax.grid(axis="y", alpha=0.85)
    else:
        ax.text(0.5, 0.5, "No non-empty baseline_ranking.csv rows found", ha="center", va="center", transform=ax.transAxes, color=KMID)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("[Benchmark evidence] Real baseline ranking rows", color=KBLUE, weight="bold")
    ax.text(0.01, -0.20, "Dots are individual evidence bundles. Black bars are medians. No synthetic model overlays are plotted.", transform=ax.transAxes, fontsize=7.6, color=KMID)
    save_figure(fig, stem)


def render_audit_figure(audits: list[AuditResult], inv: ArtifactInventory, stem: Path) -> None:
    fig = plt.figure(figsize=(7.4, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A")
    audit_types = ["leakage", "eval", "paper_mode_gate"]
    passed = [sum(1 for a in audits if a.audit_type == t and a.passed is True) for t in audit_types]
    failed = [sum(1 for a in audits if a.audit_type == t and a.passed is False) for t in audit_types]
    unknown = [sum(1 for a in audits if a.audit_type == t and a.passed is None) for t in audit_types]
    x = np.arange(len(audit_types))
    ax.bar(x, passed, color=KGREEN, label="passed")
    ax.bar(x, failed, bottom=passed, color=KRED, label="failed")
    ax.bar(x, unknown, bottom=np.array(passed) + np.array(failed), color=KMID, label="unknown")
    ax.set_xticks(x, [t.replace("_", "\n") for t in audit_types])
    ax.set_ylabel("audit artifacts")
    ax.set_title("Leakage and paper-mode audit status", color=KBLUE, weight="bold")
    ax.grid(axis="y", alpha=0.8)
    ax.legend(frameon=False, loc="upper left")

    ax2 = fig.add_subplot(gs[0, 1])
    panel_label(ax2, "B")
    ax2.set_axis_off()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    violations = sum(a.violations for a in audits)
    warnings = sum(a.warnings for a in audits)
    seeds = [a.observed_seeds for a in audits if a.observed_seeds is not None]
    windows = finite(a.window_count for a in audits if a.window_count is not None)
    card(ax2, 0.05, 0.72, 0.90, 0.18, f"violations logged\n{violations}", fc="#ECFDF5" if violations == 0 else "#FEE2E2", ec=KGREEN if violations == 0 else KRED, weight="bold")
    card(ax2, 0.05, 0.49, 0.90, 0.18, f"warnings logged\n{warnings}", fc="#FEF3C7" if warnings else "#ECFDF5", ec=KGOLD if warnings else KGREEN, weight="bold")
    seed_text = f"paper-mode seeds\nmedian n={int(np.median(seeds))}" if seeds else "paper-mode seeds\nnot found"
    card(ax2, 0.05, 0.26, 0.90, 0.18, seed_text, fc="#DBEAFE", ec=KBLUE, weight="bold")
    tensor_text = "No raw tensor or prediction-array artifact was found; waveform overlays are intentionally omitted." if not inv.array_like_artifacts else f"{len(inv.array_like_artifacts)} tensor-like artifacts found."
    ax2.text(0.05, 0.11, tensor_text, fontsize=8.1, color=KRED if not inv.array_like_artifacts else KGREEN, va="top", wrap=True)
    if windows:
        ax2.text(0.05, 0.02, f"median audited windows: {int(np.median(windows)):,}", fontsize=7.4, color=KMID)
    save_figure(fig, stem)


def short_task(task: str) -> str:
    mapping = {
        "future_state_forecasting": "future\nforecast",
        "masked_neural_reconstruction": "masked\nrecon",
        "stimulus_to_fmri_response": "stimulus\n→ fMRI",
    }
    return mapping.get(task, task.replace("_", "\n"))


def build_summary(evidence: dict[str, Any], versions_root: Path, figure_stems: dict[str, str]) -> dict[str, Any]:
    inv: ArtifactInventory = evidence["inventory"]
    task_results: list[TaskResult] = evidence["task_results"]
    baseline_results: list[BaselineResult] = evidence["baseline_results"]
    audits: list[AuditResult] = evidence["audits"]
    eeg_rows = [r for r in task_results if r.source_modality == "eeg" and r.target_modality == "eeg"]
    summary: dict[str, Any] = {
        "source_mode": "versions_evidence",
        "versions_root": str(versions_root),
        "bundle_count": inv.bundle_count,
        "task_result_rows": len(task_results),
        "eeg_task_result_rows": len(eeg_rows),
        "baseline_rows": len(baseline_results),
        "audit_rows": len(audits),
        "raw_tensor_artifacts_found": bool(inv.array_like_artifacts),
        "array_like_artifact_examples": inv.array_like_artifacts[:20],
        "inventory": asdict(inv),
        "eeg_metric_summary": summarize_task_metrics(eeg_rows),
        "baseline_metric_summary": summarize_baselines(baseline_results),
        "audit_summary": summarize_audits(audits),
        "figure_files": {f"{key}_{ext}": f"{stem}.{ext}" for key, stem in figure_stems.items() for ext in ("png", "pdf", "svg")},
    }
    return summary


def summarize_task_metrics(rows: list[TaskResult]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for task in sorted({r.task_id for r in rows}):
        task_rows = [r for r in rows if r.task_id == task]
        out[task] = {
            "n": len(task_rows),
            "median_eval_pearsonr": median_or_nan(r.eval_pearsonr for r in task_rows),
            "median_eval_r2": median_or_nan(r.eval_r2 for r in task_rows),
            "median_test_mse": median_or_nan(r.test_mse for r in task_rows),
            "median_best_val_mse": median_or_nan(r.best_val_mse for r in task_rows),
        }
    return out


def summarize_baselines(rows: list[BaselineResult]) -> dict[str, dict[str, float | int]]:
    mse_rows = [r for r in rows if r.metric.lower() == "mse" and math.isfinite(r.value)]
    out: dict[str, dict[str, float | int]] = {}
    for model in sorted({r.model_id for r in mse_rows}):
        model_rows = [r for r in mse_rows if r.model_id == model]
        out[model] = {"n": len(model_rows), "median_mse": median_or_nan(r.value for r in model_rows), "median_rank": median_or_nan(r.rank for r in model_rows)}
    return out


def summarize_audits(rows: list[AuditResult]) -> dict[str, Any]:
    out: dict[str, Any] = {"violations": sum(r.violations for r in rows), "warnings": sum(r.warnings for r in rows), "by_type": {}}
    for audit_type in sorted({r.audit_type for r in rows}):
        typed = [r for r in rows if r.audit_type == audit_type]
        out["by_type"][audit_type] = {
            "n": len(typed),
            "passed": sum(1 for r in typed if r.passed is True),
            "failed": sum(1 for r in typed if r.passed is False),
            "unknown": sum(1 for r in typed if r.passed is None),
        }
    return out


def median_or_nan(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.median(vals)) if vals else float("nan")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def analysis_md(summary: dict[str, Any]) -> str:
    tensors = "Yes" if summary["raw_tensor_artifacts_found"] else "No"
    metric_lines = []
    for task, vals in summary["eeg_metric_summary"].items():
        metric_lines.append(
            f"- `{task}`: n={vals['n']}, median Pearson r={fmt(vals['median_eval_pearsonr'])}, "
            f"median $R^2$={fmt(vals['median_eval_r2'])}, median test MSE={fmt(vals['median_test_mse'])}"
        )
    if not metric_lines:
        metric_lines.append("- No EEG→EEG task-result rows were found in the scanned evidence bundles.")

    baseline_lines = []
    for model, vals in sorted(summary["baseline_metric_summary"].items(), key=lambda kv: kv[1]["median_mse"]):
        baseline_lines.append(f"- `{model}`: n={vals['n']}, median MSE={fmt(vals['median_mse'])}, median rank={fmt(vals['median_rank'])}")
    if not baseline_lines:
        baseline_lines.append("- No non-empty baseline-ranking rows were found.")

    return f"""# EEG/ridge versions evidence figures

## Real evidence artifacts

Generated from `{summary['versions_root']}` by `scripts/render_eeg_v1_ridge_visuals.py --versions-root ...`.

The renderer now writes a CEBRA-style figure-source packet at `docs/research/eeg_v1_figure_source`: cached `data/*.csv` and `data/*.json`, standard matplotlib/seaborn source scripts in `src/`, and rendered PNG/PDF/SVG panels in `figures/`.

- Evidence bundles scanned: **{summary['bundle_count']}**
- Task-result rows: **{summary['task_result_rows']}**
- EEG→EEG task rows: **{summary['eeg_task_result_rows']}**
- Baseline-ranking rows: **{summary['baseline_rows']}**
- Audit rows: **{summary['audit_rows']}**
- Raw tensor or prediction arrays found: **{tensors}**

No raw tensor or prediction-array artifact was found, so this renderer intentionally does **not** generate a prediction overlay, waveform trace, or synthetic EEG window figure.

## EEG task metrics

{chr(10).join(metric_lines)}

## Baseline rankings

{chr(10).join(baseline_lines)}

## Audit summary

- Total violations recorded in parsed audits: **{summary['audit_summary']['violations']}**
- Total warnings recorded in parsed audits: **{summary['audit_summary']['warnings']}**

## Figure files

- `docs/research/eeg_v1_figure_source/figures/Figure1_eeg_v1_benchmark_overview.png/.pdf/.svg`
- `docs/research/eeg_v1_figure_source/figures/Figure2_eeg_v1_audit_matrix.png/.pdf/.svg`
- `docs/research/eeg_v1_figure_source/figures/Figure3_eeg_v1_baseline_ranking.png/.pdf/.svg`
"""


def fmt(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(v):
        return "nan"
    return f"{v:.3g}"


if __name__ == "__main__":
    raise SystemExit(main())
