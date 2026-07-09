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
        "_figure_style.py",
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
- `src/_figure_style.py`: shared matplotlib/tueplots style and PNG/PDF/SVG save helper.
- `src/Figure1_eeg_v1_benchmark_overview.py`: matplotlib/seaborn/tueplots-style EEG metric trajectory plots.
- `src/Figure2_eeg_v1_audit_matrix.py`: compact audit-status and artifact-coverage plots.
- `src/Figure3_eeg_v1_baseline_ranking.py`: task-wise recovered-Kahlus-versus-baseline MSE bar plots.

## Regenerate

```bash
PYTHONPATH=src python scripts/render_eeg_v1_ridge_visuals.py \
  --versions-root /Users/aayu/Downloads/versions \
  --out-dir docs/research/eeg_v1_ridge_visuals
```

Public rule: no raw tensor or prediction-array artifact means no waveform overlay, no residual trace, and no clinical/physiology claim figure. Public evidence figures use standard matplotlib/seaborn/tueplots axes with constrained layout, CEBRA-style cached data, and built-in perceptual colormaps such as `viridis`/`cividis`, not hand-drawn box diagrams.
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
        "recovered_kahlus_vs_ridge": summarize_recovered_kahlus_vs_ridge(eeg_rows, baseline_results),
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


def summarize_recovered_kahlus_vs_ridge(task_rows: list[TaskResult], baseline_rows: list[BaselineResult]) -> dict[str, dict[str, float | str | None]]:
    """Compare best recovered Kahlus task-result MSE against linear ridge.

    The recovered 6621642 run is stored in task_results.csv, while the ridge rows
    live in baseline_ranking.csv. Keeping these separate made Figure 3 misleading,
    so this summary joins them at the task level.
    """

    out: dict[str, dict[str, float | str | None]] = {}
    task_ids = sorted({r.task_id for r in task_rows if r.source_modality == "eeg" and r.target_modality == "eeg"})
    for task_id in task_ids:
        candidate_rows = [r for r in task_rows if r.task_id == task_id and r.source_modality == "eeg" and r.target_modality == "eeg" and math.isfinite(r.test_mse)]
        recovered_rows = [r for r in candidate_rows if "6621642" in f"{r.bundle} {r.artifact_path}".lower()]
        comparison_rows = recovered_rows or candidate_rows
        kahlus_vals = finite(r.test_mse for r in comparison_rows)
        ridge_vals = finite(r.value for r in baseline_rows if r.task_id == task_id and r.model_id == "linear_ridge" and r.metric.lower() == "mse")
        kahlus_best = min(kahlus_vals) if kahlus_vals else None
        ridge_best = min(ridge_vals) if ridge_vals else None
        winner: str | None
        if kahlus_best is None or ridge_best is None:
            winner = None
        elif kahlus_best < ridge_best:
            winner = "kahlus_v1_recovered"
        else:
            winner = "linear_ridge"
        out[task_id] = {
            "kahlus_v1_recovered_mse": kahlus_best,
            "linear_ridge_mse": ridge_best,
            "winner": winner,
        }
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

    comparison_lines = []
    for task, vals in summary.get("recovered_kahlus_vs_ridge", {}).items():
        kahlus = vals.get("kahlus_v1_recovered_mse")
        ridge = vals.get("linear_ridge_mse")
        winner = vals.get("winner")
        if kahlus is None or ridge is None or winner is None:
            comparison_lines.append(f"- `{task}`: incomplete Kahlus/ridge comparison in cached artifacts.")
        elif winner == "kahlus_v1_recovered":
            comparison_lines.append(f"- Kahlus v1 recovered beats linear ridge on {task}: MSE {fmt(kahlus)} vs {fmt(ridge)}.")
        else:
            comparison_lines.append(f"- linear ridge beats Kahlus v1 recovered on {task}: MSE {fmt(ridge)} vs {fmt(kahlus)}.")
    if not comparison_lines:
        comparison_lines.append("- No recovered Kahlus versus linear ridge comparison could be computed.")

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

## Recovered Kahlus v1 versus ridge

{chr(10).join(comparison_lines)}

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
