from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neurotwin.benchmarks.nfc_suite import format_nfc_synthetic_report, run_nfc_synthetic_suite
from neurotwin.benchmarks.registry import competitor_registry
from neurotwin.benchmarks.suite import run_neural_translation_v1_synthetic
from neurotwin.benchmarks.task_specs import default_translation_tasks
from neurotwin.eval.paper_gate import load_paper_mode_gate, load_run_summary, paper_mode_gate_allows_claim
from neurotwin.reports.model_card import generate_model_card_report

COMPARE_FIELDS = (
    "run",
    "status",
    "synthetic_only",
    "real_data_smoke",
    "scientific_claim_allowed",
    "paper_mode_gate_allows_claim",
    "best_task_id",
    "best_val_mse",
    "final_val_mse",
    "best_step",
    "test_mse",
    "test_mae",
    "test_pearsonr",
    "test_r2",
    "test_spearmanr",
)

COMPARE_METRIC_FALLBACKS = {
    "test_mse": ("test_mse", "eval_mse"),
    "test_mae": ("test_mae", "eval_mae"),
    "test_pearsonr": ("test_pearsonr", "eval_pearsonr"),
    "test_r2": ("test_r2", "eval_r2"),
    "test_spearmanr": ("test_spearmanr", "eval_spearmanr"),
}

TASK_RESULT_FIELDS = (
    "task_id",
    "source_modality",
    "target_modality",
    "eval_mse",
    "eval_mae",
    "eval_pearsonr",
    "eval_r2",
    "test_mse",
    "best_val_mse",
    "final_val_mse",
    "best_step",
)


def generate_suite_report(suite: str) -> str:
    if suite == "nfc_synthetic":
        return format_nfc_synthetic_report(run_nfc_synthetic_suite(seed=0, train_steps=1))
    if suite == "neural_translation_v1":
        payload = run_neural_translation_v1_synthetic(seed=0)
        lines = [
            "# NeuroTwin Neural Translation V1 Suite",
            "",
            "Scope: synthetic-only plumbing report. This is not scientific evidence.",
            "",
            "## Local Baseline Ranking",
            "",
            "| Model | Mean Rank | Tasks Ranked |",
            "| --- | ---: | ---: |",
        ]
        for row in _baseline_aggregate_rank_rows(payload):
            lines.append(f"| {row['model_id']} | {float(row['mean_rank']):.3f} | {row['tasks_ranked']} |")
        lines.extend(
            [
                "",
                "Required real-data acceptance still requires held-out subject/site/dataset splits, bootstrap CIs, and exact baseline protocols.",
            ]
        )
        return "\n".join(lines)
    if suite != "translation_smoke":
        raise ValueError("Supported suite reports: translation_smoke, neural_translation_v1")

    tasks = default_translation_tasks()
    competitors = competitor_registry()
    lines = [
        "# NeuroTwin Translation Smoke Suite",
        "",
        "Claim under test: leakage-proof Neural Translation, not first multimodal brain foundation model.",
        "Required split discipline: held-out subject/site/dataset before preprocessing, windowing, or augmentation.",
        "",
        "## Required Tasks",
        "",
        "| Task | Inputs | Targets | Metrics |",
        "| --- | --- | --- | --- |",
    ]
    for task in tasks:
        lines.append(
            f"| {task.name} | {', '.join(task.inputs)} | {', '.join(task.targets)} | {', '.join(task.metrics)} |"
        )

    lines.extend(
        [
            "",
            "## Primary Competitors",
            "",
            "| Competitor | Role | Implementation | License Status |",
            "| --- | --- | --- | --- |",
        ]
    )
    for competitor in competitors:
        lines.append(
            f"| {competitor.display_name} | {competitor.role} | {competitor.implementation_status} | {competitor.license_status} |"
        )

    lines.extend(
        [
            "",
            "## Leakage Checks",
            "",
            "- Record IDs cannot appear in more than one split.",
            "- Held-out subject/site/dataset keys cannot overlap when a policy requires them.",
            "- Splits are generated from recording-level manifests before preprocessing/windowing.",
            "- Clinical label prediction is secondary and cannot be the headline claim.",
            "",
            "## Current Acceptance Bar",
            "",
            "NeuroTwin must beat the best strong baseline on aggregate rank and at least two modality groups, with bootstrap confidence intervals.",
        ]
    )
    return "\n".join(lines)


def _baseline_aggregate_rank_rows(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    baseline_suite = payload.get("baseline_suite")
    if not isinstance(baseline_suite, dict):
        return []
    aggregate = baseline_suite.get("aggregate")
    if not isinstance(aggregate, dict):
        return []
    rows = aggregate.get("aggregate_rank")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def generate_run_report(run_dir: str | Path) -> str:
    path = Path(run_dir)
    summary = load_run_summary(path)
    summary_claim = _summary_scientific_claim_allowed(summary)
    gate_allows_claim = paper_mode_gate_allows_claim(load_paper_mode_gate(path))
    artifact_paths = _write_run_table_artifacts(path, summary=summary, scientific_claim_allowed=summary_claim, gate_allows_claim=gate_allows_claim)
    lines = [
        "# NeuroTwin Run Report",
        "",
        f"run_dir={path}",
        f"scientific_claim_allowed={summary_claim}",
        f"paper_mode_gate_allows_claim={gate_allows_claim}",
    ]
    for filename in ("config.yaml", "environment.json", "metrics.json", "summary.json"):
        file_path = path / filename
        if not file_path.exists():
            continue
        lines.extend(["", f"## {filename}", ""])
        if file_path.suffix == ".json":
            payload = _read_json(file_path)
            if _is_artifact_error(payload):
                lines.append(f"artifact_error={payload.get('error')} path={payload.get('path')} message={payload.get('message', '')}")
                raw_text = _read_text_best_effort(file_path)
                if raw_text is not None:
                    lines.append("```")
                    lines.append(raw_text.rstrip())
                    lines.append("```")
            else:
                lines.append("```json")
                lines.append(json.dumps(payload, indent=2, sort_keys=True))
                lines.append("```")
        else:
            lines.append("```yaml")
            lines.append(file_path.read_text(encoding="utf-8").rstrip())
            lines.append("```")
    if len(lines) == 5:
        lines.append("")
        lines.append("No run artifacts found.")
    elif artifact_paths:
        lines.extend(["", "## paper_ready_artifacts", ""])
        for artifact in artifact_paths:
            lines.append(f"- {artifact}")
    return "\n".join(lines)


def generate_compare_report(run_dirs: list[str] | tuple[str, ...], out_dir: str | Path | None = None) -> str:
    rows = []
    for run_dir in run_dirs:
        path = Path(run_dir)
        metrics_payload = _read_json(path / "metrics.json")
        summary_payload = _read_json(path / "summary.json")
        metrics = metrics_payload if isinstance(metrics_payload, dict) else {}
        summary = summary_payload if isinstance(summary_payload, dict) else {}
        if not metrics and not summary:
            continue
        row = {"run": path.name}
        artifact_error = _artifact_error_text(metrics_payload, summary_payload)
        for field in COMPARE_FIELDS:
            if field == "run":
                continue
            row[field] = _summary_or_metrics(summary, metrics, field, *COMPARE_METRIC_FALLBACKS.get(field, (field,)))
        if artifact_error and not row.get("status"):
            row["status"] = "artifact_error"
        if artifact_error:
            row["artifact_error"] = artifact_error
        row["scientific_claim_allowed"] = _summary_scientific_claim_allowed(summary)
        row["paper_mode_gate_allows_claim"] = paper_mode_gate_allows_claim(load_paper_mode_gate(path))
        rows.append(row)
    destination = Path(out_dir) if out_dir else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        destination.joinpath("compare_runs.csv").write_text(
            _csv_rows(COMPARE_FIELDS, [tuple(row.get(key, "") for key in COMPARE_FIELDS) for row in rows]),
            encoding="utf-8",
        )
        destination.joinpath("compare_runs.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# NeuroTwin Run Comparison",
        "",
        "| Run | Status | Claim Allowed | Gate Allows | Best Task | Best Val MSE | Test MSE |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['status']} | {row['scientific_claim_allowed']} | {row['paper_mode_gate_allows_claim']} | {row['best_task_id']} | {row['best_val_mse']} | {row['test_mse']} |"
        )
    artifact_errors = [row for row in rows if row.get("artifact_error")]
    if artifact_errors:
        lines.extend(["", "## artifact_errors", ""])
        for row in artifact_errors:
            lines.append(f"{row['run']}: {row['artifact_error']}")
    if destination is not None:
        lines.extend(["", f"artifacts={destination}"])
    return "\n".join(lines)


def _write_run_table_artifacts(
    path: Path,
    summary: dict[str, Any] | None = None,
    scientific_claim_allowed: bool | None = None,
    gate_allows_claim: bool | None = None,
) -> list[str]:
    if not path.exists():
        return []
    metrics_path = path / "metrics.json"
    summary_path = path / "summary.json"
    if not metrics_path.exists() and not summary_path.exists():
        return []
    tables_dir = path / "tables"
    figures_dir = path / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = _read_json(metrics_path)
    summary_payload = summary if isinstance(summary, dict) else _read_json(summary_path)
    summary_claim = _summary_scientific_claim_allowed(summary_payload) if scientific_claim_allowed is None else bool(scientific_claim_allowed)
    gate_allows = paper_mode_gate_allows_claim(load_paper_mode_gate(path)) if gate_allows_claim is None else bool(gate_allows_claim)
    artifacts: list[str] = []
    flat_rows = list(_flatten_metrics({"metrics": metrics, "summary": summary_payload}))
    flat_csv = tables_dir / "metrics_flat.csv"
    flat_csv.write_text(_csv_rows(("metric", "value"), flat_rows), encoding="utf-8")
    artifacts.append(str(flat_csv))

    task_results = _task_results(metrics, summary_payload)
    artifacts.extend(_write_task_results_table(tables_dir, task_results))
    artifacts.extend(_write_baseline_tables(path, tables_dir, metrics))
    artifacts.append(str(_write_metric_summary_figure(figures_dir, summary_payload, scientific_claim_allowed=summary_claim, paper_mode_gate_allows_claim=gate_allows)))
    adaptation_rows = _adaptation_rows(task_results)
    if adaptation_rows:
        adaptation_json = figures_dir / "adaptation_curve.json"
        adaptation_json.write_text(json.dumps(adaptation_rows, indent=2, sort_keys=True), encoding="utf-8")
        artifacts.append(str(adaptation_json))
    return artifacts


def _summary_or_metrics(summary: dict[str, Any], metrics: dict[str, Any], summary_key: str, *metric_keys: str) -> Any:
    if summary_key in summary:
        return summary.get(summary_key)
    for key in metric_keys:
        if key in metrics:
            return metrics.get(key)
    return ""


def _task_results(metrics: Any, summary: Any) -> Any:
    task_results = metrics.get("task_results") if isinstance(metrics, dict) else None
    if not isinstance(task_results, list) and isinstance(summary, dict):
        task_results = summary.get("task_results")
    return task_results


def _write_task_results_table(tables_dir: Path, task_results: Any) -> list[str]:
    if not isinstance(task_results, list):
        return []
    rows = []
    for row in task_results:
        if isinstance(row, dict):
            rows.append(_task_result_row(row))
    task_csv = tables_dir / "task_results.csv"
    task_csv.write_text(_csv_rows(TASK_RESULT_FIELDS, rows), encoding="utf-8")
    return [str(task_csv)]


def _task_result_row(row: dict[str, Any]) -> tuple[Any, ...]:
    values = []
    for field in TASK_RESULT_FIELDS:
        if field == "test_mse":
            values.append(row.get("test_mse", row.get("eval_mse", "")))
        else:
            values.append(row.get(field, ""))
    return tuple(values)


def _write_baseline_tables(run_dir: Path, tables_dir: Path, metrics: Any) -> list[str]:
    baseline_suite = metrics.get("baseline_suite") if isinstance(metrics, dict) else None
    if not isinstance(baseline_suite, dict):
        baseline_suite = metrics if isinstance(metrics, dict) and "baseline_failures" in metrics else {}
    if not _baseline_ranking_rows_from_suite(baseline_suite):
        prepared_suite = _read_json(run_dir / "prepared_baseline_suite.json")
        if isinstance(prepared_suite, dict) and _baseline_ranking_rows_from_suite(prepared_suite):
            baseline_suite = prepared_suite
    ranking_rows = []
    failures = []
    tasks = baseline_suite.get("tasks", {}) if isinstance(baseline_suite, dict) else {}
    if isinstance(tasks, dict):
        for task_id, task_payload in tasks.items():
            if not isinstance(task_payload, dict):
                continue
            for row in task_payload.get("ranking", []):
                if isinstance(row, dict):
                    ranking_rows.append(
                        (task_id, row.get("model_id", ""), row.get("metric", ""), row.get("value", ""), row.get("rank", ""))
                    )
            for row in task_payload.get("failures", []):
                if isinstance(row, dict):
                    failures.append((task_id, row.get("model_id", ""), row.get("reason", "")))

    raw_failures = metrics.get("baseline_failures") if isinstance(metrics, dict) else None
    if isinstance(raw_failures, list):
        for row in raw_failures:
            if isinstance(row, dict):
                failures.append((row.get("task_id", ""), row.get("model_id", ""), row.get("reason", "")))

    baseline_ranking_csv = tables_dir / "baseline_ranking.csv"
    if not ranking_rows:
        ranking_rows.append(("baseline_ranking_unavailable", "", "status", "no colocated baseline rankings found", ""))
    baseline_ranking_csv.write_text(_csv_rows(("task_id", "model_id", "metric", "value", "rank"), ranking_rows), encoding="utf-8")
    baseline_failures_csv = tables_dir / "baseline_failures.csv"
    baseline_failures_csv.write_text(_csv_rows(("task_id", "model_id", "reason"), failures), encoding="utf-8")
    return [str(baseline_ranking_csv), str(baseline_failures_csv)]


def _write_metric_summary_figure(
    figures_dir: Path,
    summary: Any,
    *,
    scientific_claim_allowed: bool,
    paper_mode_gate_allows_claim: bool,
) -> Path:
    payload = {
        "synthetic_only": bool(summary.get("synthetic_only")) if isinstance(summary, dict) else None,
        "status": summary.get("status") if isinstance(summary, dict) else None,
        "best_task_id": summary.get("best_task_id") if isinstance(summary, dict) else None,
        "best_eval_mse": summary.get("best_eval_mse") if isinstance(summary, dict) else None,
        "best_val_mse": summary.get("best_val_mse") if isinstance(summary, dict) else None,
        "final_val_mse": summary.get("final_val_mse") if isinstance(summary, dict) else None,
        "best_step": summary.get("best_step") if isinstance(summary, dict) else None,
        "best_checkpoint_path": summary.get("best_checkpoint_path") if isinstance(summary, dict) else None,
        "final_checkpoint_path": summary.get("final_checkpoint_path") if isinstance(summary, dict) else None,
        "checkpoint_selection_metric": summary.get("checkpoint_selection_metric") if isinstance(summary, dict) else None,
        "checkpoint_selection_mode": summary.get("checkpoint_selection_mode") if isinstance(summary, dict) else None,
        "test_mse": summary.get("test_mse") if isinstance(summary, dict) else None,
        "real_data_smoke": bool(summary.get("real_data_smoke")) if isinstance(summary, dict) else False,
        "scientific_claim_allowed": bool(scientific_claim_allowed),
        "paper_mode_gate_allows_claim": bool(paper_mode_gate_allows_claim),
    }
    figure_json = figures_dir / "metric_summary.json"
    figure_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return figure_json


def _summary_scientific_claim_allowed(summary: Any) -> bool:
    return bool(summary.get("scientific_claim_allowed")) if isinstance(summary, dict) else False


def _baseline_ranking_rows_from_suite(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    tasks = payload.get("tasks", {})
    rows = []
    if isinstance(tasks, dict):
        for task_payload in tasks.values():
            if not isinstance(task_payload, dict):
                continue
            ranking = task_payload.get("ranking", [])
            if isinstance(ranking, list):
                rows.extend(row for row in ranking if isinstance(row, dict))
    return rows


def _adaptation_rows(task_results: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(task_results, list):
        return rows
    for task in task_results:
        if not isinstance(task, dict) or task.get("task_id") != "few_shot_subject_adaptation":
            continue
        metrics = task.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if key.startswith("k") and key.endswith("_adaptation_gain"):
                support = key.split("_", 1)[0][1:]
                rows.append({"support_windows": int(support), "adaptation_gain": value})
    return rows


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"error": "invalid_json", "path": str(path), "message": str(exc)}
    except OSError as exc:
        return {"error": "read_failed", "path": str(path), "message": str(exc)}


def _read_text_best_effort(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _is_artifact_error(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("error"), str) and isinstance(payload.get("path"), str)


def _artifact_error_text(*payloads: Any) -> str:
    errors = []
    for payload in payloads:
        if _is_artifact_error(payload):
            errors.append(f"{payload.get('error')}:{payload.get('path')}")
    return "; ".join(errors)


def _flatten_metrics(payload: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in sorted(payload.items()):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_metrics(value, next_prefix))
    elif isinstance(payload, list):
        if not payload:
            rows.append((prefix, "[]"))
        elif all(isinstance(item, dict) and "task_id" in item for item in payload):
            for item in payload:
                label = str(item.get("task_id"))
                rows.extend(_flatten_metrics(item, f"{prefix}.{label}"))
    elif isinstance(payload, (int, float, str, bool)) or payload is None:
        rows.append((prefix, payload))
    return rows


def _csv_rows(header: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(_csv_cell(value) for value in row))
    return "\n".join(lines) + "\n"


def _csv_cell(value: Any) -> str:
    text = str(value)
    if any(char in text for char in [",", "\"", "\n"]):
        return "\"" + text.replace("\"", "\"\"") + "\""
    return text
