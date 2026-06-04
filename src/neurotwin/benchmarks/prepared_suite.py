from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np

from neurotwin.benchmarks.baseline_suite import (
    PreparedBaselineSuitePayload,
    PreparedPaperModePayload,
    PreparedTaskPayload,
    run_supervised_window_tasks,
)
from neurotwin.benchmarks.tasks import (
    TaskResult,
    run_dataset_site_generalization_task,
    run_subject_adaptation_task,
)
from neurotwin.contracts.paper_mode import CANONICAL_REQUIRED_SEEDS
from neurotwin.data.event_io import event_manifest_summary, load_event_batches
from neurotwin.data.manifest_io import load_split_manifest
from neurotwin.data.prepared_tasks import (
    PreparedSuiteConfig,
    build_prepared_window_tasks,
    first_prepared_modality_with_splits,
    prepared_windows_by_split,
)
from neurotwin.data.schemas import NeuralEventBatch
from neurotwin.data.split_manifest import SplitManifest
from neurotwin.repro import write_json


def run_prepared_baseline_suite(
    config: PreparedSuiteConfig,
    out_dir: str | Path | None = None,
) -> PreparedBaselineSuitePayload:
    batches = load_event_batches(config.event_manifest)
    split = load_split_manifest(config.split_manifest)
    tasks, skipped = build_prepared_window_tasks(
        batches,
        split,
        window_length=config.window_length,
        stride=config.stride,
        seed=config.seed,
    )
    payload: PreparedBaselineSuitePayload
    if tasks:
        payload = cast(
            PreparedBaselineSuitePayload,
            run_supervised_window_tasks(
                tasks,
                seed=config.seed,
                train_steps=config.train_steps,
                scope_status=_scope_status(batches),
                scope_notes=(
                    "Uses prepared event batches and a recording-level split manifest.",
                    "Claims still require real public data, preregistered protocols, and confidence intervals.",
                ),
            ),
        )
    else:
        payload = {
            "scope": {
                "status": _scope_status(batches),
                "notes": ["No runnable supervised window tasks were available for this prepared manifest."],
            },
            "tasks": {},
            "aggregate": {"selection_metric": "mse", "higher_is_better": False, "aggregate_rank": []},
            "seed": int(config.seed),
            "seeds": [int(config.seed)],
            "benchmark_contract": {
                "required_seeds": list(CANONICAL_REQUIRED_SEEDS),
                "require_ci": True,
                "notes": [
                    "Paper mode requires a passed prepared eval audit, nonempty rankings, all required seeds, and CI summaries.",
                    "Task 3 is expected to replace the single-seed payload with aggregated seed results.",
                ],
            },
            "baseline_catalog": [],
            "baseline_failures": [],
        }
    auxiliary_tasks, auxiliary_skipped = run_prepared_auxiliary_tasks(
        batches,
        split,
        window_length=config.window_length,
        stride=config.stride,
    )
    task_payloads = payload.get("tasks")
    if isinstance(task_payloads, dict):
        task_payloads.update(auxiliary_tasks)
    else:
        payload["tasks"] = dict(auxiliary_tasks)
    skipped.extend(auxiliary_skipped)
    payload["prepared_data"] = {
        "event_manifest": str(config.event_manifest),
        "split_manifest": str(config.split_manifest),
        "event_summary": event_manifest_summary(config.event_manifest),
        "window_length": config.window_length,
        "stride": config.stride,
        "skipped_tasks": skipped,
        "stimulus_evidence": _stimulus_evidence_from_tasks(tasks),
    }
    payload["paper_mode_contract"] = {
        "required_seeds": list(CANONICAL_REQUIRED_SEEDS),
        "observed_seeds": payload["seeds"],
        "require_ci": bool(config.require_ci),
        "gate_status": "not_run",
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "prepared_baseline_suite.json", payload)
        write_json(out / "baseline_failures.json", payload.get("baseline_failures", []))
    return payload


def format_prepared_baseline_report(payload: PreparedBaselineSuitePayload | PreparedPaperModePayload) -> str:
    prepared = payload.get("prepared_data", {})
    scope = payload.get("scope", {})
    aggregate = payload.get("aggregate", {})
    lines = [
        "# NeuroTwin Prepared Baseline Suite",
        "",
        f"scope={scope.get('status') if isinstance(scope, dict) else 'unknown'}",
    ]
    if isinstance(prepared, dict):
        lines.extend(
            [
                f"event_manifest={prepared.get('event_manifest')}",
                f"split_manifest={prepared.get('split_manifest')}",
                f"window_length={prepared.get('window_length')}",
                f"stride={prepared.get('stride')}",
                f"stimulus_evidence={_format_stimulus_evidence(prepared.get('stimulus_evidence'))}",
                "",
            ]
        )
    if isinstance(aggregate, dict):
        lines.append("## aggregate_rank")
        for row in aggregate.get("aggregate_rank", []):
            if isinstance(row, dict):
                lines.append(
                    f"ranked_model={row.get('model_id')} mean_rank={row.get('mean_rank')} tasks={row.get('tasks_ranked')}"
                )
        lines.append("")
    seed_aggregate = payload.get("seed_aggregate", [])
    if isinstance(seed_aggregate, list) and seed_aggregate:
        lines.append("## seed_aggregate")
        for row in seed_aggregate[:20]:
            if isinstance(row, dict):
                lines.append(
                    "seed_metric="
                    f"{row.get('task_id')}:{row.get('model_id')}:{row.get('metric')} "
                    f"mean={row.get('mean')} std={row.get('std')} "
                    f"ci95=[{row.get('ci_low')},{row.get('ci_high')}] n={row.get('n_seeds')}"
                )
        if len(seed_aggregate) > 20:
            lines.append(f"seed_metric_rows_omitted={len(seed_aggregate) - 20}")
        lines.append("")
    catalog = payload.get("baseline_catalog", [])
    if isinstance(catalog, list):
        lines.append("## baseline_catalog")
        for row in catalog:
            if isinstance(row, dict):
                lines.append(f"{row.get('model_id')}: {row.get('status')} - {row.get('notes')}")
        lines.append("")
    tasks = payload.get("tasks", {})
    if isinstance(tasks, dict):
        for task_id, result in tasks.items():
            if not isinstance(result, dict):
                continue
            lines.append(f"## {task_id}")
            lines.append(f"status={result.get('status')}")
            metrics = result.get("metrics", {})
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    lines.append(f"{key}={value}")
            for row in result.get("ranking", []):
                if isinstance(row, dict):
                    model_id = row.get("model_id")
                    metric = row.get("metric")
                    suffix = ""
                    metrics_by_model = result.get("metrics_by_model", {})
                    if isinstance(metrics_by_model, dict) and model_id in metrics_by_model and metric == "mse":
                        model_metrics = metrics_by_model.get(model_id, {})
                        if isinstance(model_metrics, dict):
                            suffix = f" ci95=[{model_metrics.get('mse_ci_low')},{model_metrics.get('mse_ci_high')}]"
                    lines.append(f"{model_id}_rank={row.get('rank')} {metric}={row.get('value')}{suffix}")
            for row in result.get("failures", []):
                if isinstance(row, dict):
                    lines.append(f"failed_model={row.get('model_id')} reason={row.get('reason')}")
            lines.append("")
    failures = payload.get("baseline_failures", [])
    if isinstance(failures, list) and failures:
        lines.append("## baseline_failures")
        for row in failures:
            if isinstance(row, dict):
                lines.append(f"{row.get('task_id')}:{row.get('model_id')}: {row.get('reason')}")
        lines.append("")
    if isinstance(prepared, dict) and prepared.get("skipped_tasks"):
        lines.append("## skipped_tasks")
        for row in prepared.get("skipped_tasks", []):
            if isinstance(row, dict):
                lines.append(f"{row.get('task_id')}: {row.get('reason')}")
        lines.append("")
    lines.append("Prepared-data rankings are benchmark plumbing unless run on real held-out public data with locked protocols.")
    return "\n".join(lines)


def run_prepared_auxiliary_tasks(
    batches: list[NeuralEventBatch],
    split: SplitManifest,
    window_length: int,
    stride: int,
) -> tuple[dict[str, PreparedTaskPayload], list[dict[str, str]]]:
    windows_by_split = prepared_windows_by_split(batches, split, window_length=window_length, stride=stride)
    results: dict[str, PreparedTaskPayload] = {}
    skipped: list[dict[str, str]] = []
    adaptation = _subject_adaptation_from_windows(windows_by_split)
    if adaptation is None:
        skipped.append({"task_id": "few_shot_subject_adaptation", "reason": "need at least one held-out subject with support/query windows"})
    else:
        results[adaptation.task_id] = _task_result_to_dict(adaptation)

    generalization = _dataset_site_generalization_from_windows(windows_by_split)
    if generalization is None:
        skipped.append({"task_id": "dataset_site_generalization", "reason": "need train/test windows from different datasets or sites"})
    else:
        results[generalization.task_id] = _task_result_to_dict(generalization)
    return results, skipped


def _subject_adaptation_from_windows(windows_by_split: dict[str, list[NeuralEventBatch]]) -> TaskResult | None:
    modality = first_prepared_modality_with_splits(windows_by_split)
    if modality is None:
        return None
    by_subject: dict[str, list[NeuralEventBatch]] = {}
    for window in windows_by_split["test"]:
        if window.modality == modality:
            by_subject.setdefault(window.subject_id, []).append(window)
    for subject_id, subject_windows in sorted(by_subject.items()):
        if len(subject_windows) >= 2:
            signals = [window.signal for window in subject_windows]
            metrics: dict[str, float] = {}
            notes = [f"prepared held-out subject={subject_id} modality={modality}"]
            result: TaskResult | None = None
            for support_size in (1, 5, 20):
                if len(signals) <= support_size:
                    continue
                support = np.asarray(signals[:support_size], dtype=np.float32)
                query = np.asarray(signals[support_size:], dtype=np.float32)
                if query.size == 0:
                    continue
                current = run_subject_adaptation_task(support, query)
                result = current
                for key, value in current.metrics.items():
                    metrics[f"k{support_size}_{key}"] = float(value)
                sampling_rate = subject_windows[0].sampling_rate
                if sampling_rate:
                    try:
                        minutes = (support_size * support.shape[1]) / float(sampling_rate) / 60.0
                        metrics[f"k{support_size}_support_minutes"] = float(minutes)
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
            if result is None:
                continue
            return TaskResult(
                result.task_id,
                result.status,
                metrics,
                tuple(notes),
            )
    return None


def _dataset_site_generalization_from_windows(windows_by_split: dict[str, list[NeuralEventBatch]]) -> TaskResult | None:
    modality = first_prepared_modality_with_splits(windows_by_split)
    if modality is None:
        return None
    source_groups = _group_windows(windows_by_split["train"], modality)
    target_groups = _group_windows(windows_by_split["test"], modality)
    for source_key, source_signals in sorted(source_groups.items()):
        for target_key, target_signals in sorted(target_groups.items()):
            if source_key == target_key:
                continue
            source = np.asarray(source_signals, dtype=np.float32)
            target = np.asarray(target_signals, dtype=np.float32)
            if source.size == 0 or target.size == 0:
                continue
            n_time = min(source.shape[1], target.shape[1])
            n_space = min(source.shape[2], target.shape[2])
            result = run_dataset_site_generalization_task(
                source[:, :n_time, :n_space],
                target[:, :n_time, :n_space],
                source_name="/".join(source_key),
                target_name="/".join(target_key),
            )
            return result
    return None


def _group_windows(windows: list[NeuralEventBatch], modality: str) -> dict[tuple[str, str], list[np.ndarray]]:
    groups: dict[tuple[str, str], list[np.ndarray]] = {}
    for window in windows:
        if window.modality != modality:
            continue
        groups.setdefault((window.dataset, window.site_id), []).append(window.signal)
    return groups


def _scope_status(batches: list[NeuralEventBatch]) -> str:
    return "prepared-synthetic" if all(batch.metadata.get("synthetic") for batch in batches) else "prepared-data"


def _task_result_to_dict(result: TaskResult) -> PreparedTaskPayload:
    return {
        "status": result.status,
        "metrics": result.metrics,
        "notes": result.notes,
    }


def _stimulus_evidence_from_tasks(tasks: tuple[object, ...]) -> dict[str, object] | None:
    for task in tasks:
        metadata = getattr(task, "metadata", {})
        if isinstance(metadata, dict) and isinstance(metadata.get("stimulus_evidence"), dict):
            return dict(metadata["stimulus_evidence"])
    return None


def _format_stimulus_evidence(value: object) -> str:
    if not isinstance(value, dict):
        return "none"
    return f"{value.get('status', 'unknown')} claim_eligible={value.get('claim_eligible', False)}"
