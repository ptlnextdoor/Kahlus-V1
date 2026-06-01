from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from neurotwin.benchmarks.baseline_suite import (
    PreparedAggregateRankPayload,
    PreparedBaselineSuitePayload,
    PreparedPaperModePayload,
    SeedAggregatedTaskPayload,
    PreparedTaskPayload,
    SeedAggregatePayload,
)
from neurotwin.benchmarks.prepared_suite import run_prepared_baseline_suite
from neurotwin.contracts.paper_mode import CANONICAL_REQUIRED_SEEDS
from neurotwin.data.event_io import event_manifest_summary
from neurotwin.data.prepared_tasks import PreparedSuiteConfig
from neurotwin.eval.paper_gate import PaperModeGateReport
from neurotwin.repro import write_json
from neurotwin.scoring.metrics import bootstrap_ci, rank_models


def run_prepared_baseline_suite_multi_seed(
    config: PreparedSuiteConfig,
    seeds: tuple[int, ...] | list[int] = CANONICAL_REQUIRED_SEEDS,
    out_dir: str | Path | None = None,
) -> PreparedPaperModePayload:
    """Run prepared baselines for multiple seeds and assemble paper-mode artifacts."""

    seed_values = tuple(int(seed) for seed in seeds)
    seed_results: list[PreparedBaselineSuitePayload] = []
    for seed in seed_values:
        seed_payload = run_prepared_baseline_suite(
            PreparedSuiteConfig(
                event_manifest=config.event_manifest,
                split_manifest=config.split_manifest,
                window_length=config.window_length,
                stride=config.stride,
                seed=seed,
                train_steps=config.train_steps,
                require_ci=config.require_ci,
            ),
            out_dir=None,
        )
        seed_payload["seed"] = seed
        seed_payload["seeds"] = [seed]
        seed_results.append(seed_payload)

    payload = _merge_seed_payloads(config, seed_values, seed_results)
    if out_dir is not None:
        write_prepared_paper_mode_artifacts(payload, out_dir)
    return payload


def write_prepared_paper_mode_artifacts(
    payload: PreparedPaperModePayload,
    out_dir: str | Path,
    gate: PaperModeGateReport | None = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "prepared_baseline_suite.json", payload)
    write_json(out / "seed_aggregate.json", payload.get("seed_aggregate", []))
    (out / "seed_aggregate.csv").write_text(
        _seed_aggregate_csv(payload.get("seed_aggregate", [])),
        encoding="utf-8",
    )
    write_json(out / "baseline_failures.json", payload.get("baseline_failures", []))
    if gate is not None:
        write_json(out / "paper_mode_gate.json", gate.to_dict())


def _merge_seed_payloads(
    config: PreparedSuiteConfig,
    seeds: tuple[int, ...],
    seed_results: list[PreparedBaselineSuitePayload],
) -> PreparedPaperModePayload:
    first: PreparedBaselineSuitePayload | dict[str, object] = seed_results[0] if seed_results else {}
    aggregate_rank = _aggregate_seed_ranks(seed_results)
    seed_aggregate = _aggregate_seed_metrics(seed_results)
    tasks = _aggregate_seed_tasks(first, seed_aggregate)
    failures: list[dict[str, str]] = []
    for result in seed_results:
        raw_failures = result.get("baseline_failures", [])
        if isinstance(raw_failures, list):
            for failure in raw_failures:
                if isinstance(failure, dict):
                    failures.append({str(key): str(value) for key, value in failure.items()})
    return {
        "scope": first.get("scope", {"status": "prepared-data", "notes": []}),
        "tasks": tasks,
        "aggregate": {
            "selection_metric": "mse",
            "higher_is_better": False,
            "aggregate_rank": aggregate_rank,
        },
        "seed": int(seeds[0]) if seeds else int(config.seed),
        "seeds": [int(seed) for seed in seeds],
        "seed_results": seed_results,
        "seed_aggregate": seed_aggregate,
        "representative_seed_tasks": first.get("tasks", {}),
        "prepared_data": first.get(
            "prepared_data",
            {
                "event_manifest": str(config.event_manifest),
                "split_manifest": str(config.split_manifest),
                "event_summary": event_manifest_summary(config.event_manifest),
                "window_length": config.window_length,
                "stride": config.stride,
                "skipped_tasks": [],
            },
        ),
        "paper_mode_contract": {
            "required_seeds": list(CANONICAL_REQUIRED_SEEDS),
            "observed_seeds": [int(seed) for seed in seeds],
            "require_ci": bool(config.require_ci),
            "gate_status": "not_run",
        },
        "baseline_catalog": first.get("baseline_catalog", []),
        "baseline_failures": failures,
    }


def _aggregate_seed_tasks(
    representative_payload: PreparedBaselineSuitePayload | dict[str, object],
    seed_aggregate: list[SeedAggregatePayload],
) -> dict[str, SeedAggregatedTaskPayload]:
    representative_tasks = representative_payload.get("tasks", {})
    if not isinstance(representative_tasks, dict):
        representative_tasks = {}

    rows_by_task: dict[str, list[SeedAggregatePayload]] = {}
    for row in seed_aggregate:
        task_id = row.get("task_id")
        if task_id is not None:
            rows_by_task.setdefault(str(task_id), []).append(row)

    tasks: dict[str, SeedAggregatedTaskPayload] = {}
    for task_id, rows in sorted(rows_by_task.items()):
        representative = representative_tasks.get(task_id, {})
        if not isinstance(representative, dict):
            representative = {}
        metrics: dict[str, float] = {}
        metrics_by_model: dict[str, dict[str, float]] = {}
        for row in rows:
            model_id = row.get("model_id")
            metric = row.get("metric")
            if model_id is None or metric is None:
                continue
            metric_name = str(metric)
            mean = _finite_row_float(row, "mean")
            ci_low = _finite_row_float(row, "ci_low")
            ci_high = _finite_row_float(row, "ci_high")
            if mean is None or ci_low is None or ci_high is None:
                continue
            target = metrics if str(model_id) == "task_metric" else metrics_by_model.setdefault(str(model_id), {})
            target[metric_name] = mean
            target[f"{metric_name}_ci_low"] = ci_low
            target[f"{metric_name}_ci_high"] = ci_high

        ranking = [
            {"model_id": row.model_id, "metric": row.metric, "value": row.value, "rank": row.rank}
            for row in rank_models(metrics_by_model, metric="mse", higher_is_better=False)
        ] if metrics_by_model and all("mse" in model_metrics for model_metrics in metrics_by_model.values()) else []
        tasks[task_id] = {
            "status": "seed_aggregated",
            "source_modality": representative.get("source_modality"),
            "target_modality": representative.get("target_modality"),
            "metrics": metrics,
            "metrics_by_model": metrics_by_model,
            "ranking": ranking,
            "failures": [],
            "notes": [
                "aggregate across seed_results; per-seed concrete results are stored in seed_results",
                *_representative_notes(representative),
            ],
        }
    return tasks


def _representative_notes(task_payload: PreparedTaskPayload | dict[str, object]) -> list[str]:
    notes = task_payload.get("notes", [])
    return [str(note) for note in notes] if isinstance(notes, list) else []


def _finite_row_float(row: SeedAggregatePayload | dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool) and np.isfinite(float(value)):
        return float(value)
    return None


def _aggregate_seed_ranks(seed_results: list[PreparedBaselineSuitePayload]) -> list[PreparedAggregateRankPayload]:
    ranks_by_model = _collect_concrete_seed_ranks(seed_results)
    return sorted(
        (
            {
                "model_id": model_id,
                "mean_rank": float(np.mean(ranks)),
                "std_rank": float(np.std(ranks)),
                "tasks_ranked": len(ranks),
                "n_seeds": len(seed_keys),
            }
            for model_id, values in ranks_by_model.items()
            for ranks, seed_keys in [([rank for _, rank in values], {seed_key for seed_key, _ in values})]
            if ranks
        ),
        key=lambda row: (float(row["mean_rank"]), str(row["model_id"])),
    )


def _collect_concrete_seed_ranks(seed_results: list[PreparedBaselineSuitePayload]) -> dict[str, list[tuple[str, float]]]:
    ranks_by_model: dict[str, list[tuple[str, float]]] = {}
    for result_idx, result in enumerate(seed_results):
        seed_key = _seed_key(result, result_idx)
        tasks = result.get("tasks", {})
        if not isinstance(tasks, dict):
            continue
        for task_payload in tasks.values():
            if not isinstance(task_payload, dict):
                continue
            ranking = task_payload.get("ranking", [])
            if not isinstance(ranking, list):
                continue
            for row in ranking:
                if not isinstance(row, dict):
                    continue
                model_id = row.get("model_id")
                if model_id is None:
                    continue
                try:
                    rank = float(row.get("rank", row.get("mean_rank")))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(rank):
                    ranks_by_model.setdefault(str(model_id), []).append((seed_key, rank))
    return ranks_by_model


def _seed_key(result: PreparedBaselineSuitePayload, fallback_index: int) -> str:
    value = result.get("seed", fallback_index)
    return str(value)


def _aggregate_seed_metrics(seed_results: list[PreparedBaselineSuitePayload]) -> list[SeedAggregatePayload]:
    values: dict[tuple[str, str, str], list[float]] = {}
    for result in seed_results:
        tasks = result.get("tasks", {})
        if not isinstance(tasks, dict):
            continue
        for task_id, task_payload in tasks.items():
            if not isinstance(task_payload, dict):
                continue
            metrics_by_model = task_payload.get("metrics_by_model", {})
            if isinstance(metrics_by_model, dict):
                for model_id, metrics in metrics_by_model.items():
                    if isinstance(metrics, dict):
                        _collect_metric_values(values, str(task_id), str(model_id), metrics)
            task_metrics = task_payload.get("metrics", {})
            if isinstance(task_metrics, dict):
                _collect_metric_values(values, str(task_id), "task_metric", task_metrics)
    rows: list[SeedAggregatePayload] = []
    for (task_id, model_id, metric), metric_values in sorted(values.items()):
        arr = np.asarray(metric_values, dtype=float)
        if arr.size == 0 or not np.isfinite(arr).all():
            continue
        if arr.size == 1:
            ci_low = ci_high = float(arr[0])
        else:
            ci_low, ci_high = bootstrap_ci(arr, seed=_stable_metric_seed(task_id, model_id, metric), n_boot=200)
        rows.append(
            {
                "task_id": task_id,
                "model_id": model_id,
                "metric": metric,
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n_seeds": int(arr.size),
            }
        )
    return rows


def _collect_metric_values(
    values: dict[tuple[str, str, str], list[float]],
    task_id: str,
    model_id: str,
    metrics: dict[str, Any],
) -> None:
    for metric, value in metrics.items():
        if metric.endswith(("_ci_low", "_ci_high")):
            continue
        if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool) and np.isfinite(float(value)):
            values.setdefault((task_id, model_id, metric), []).append(float(value))


def _seed_aggregate_csv(rows: object) -> str:
    header = ("task_id", "model_id", "metric", "mean", "std", "ci_low", "ci_high", "n_seeds")
    lines = [",".join(header)]
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                lines.append(",".join(_csv_cell(row.get(key, "")) for key in header))
    return "\n".join(lines) + "\n"


def _csv_cell(value: object) -> str:
    text = str(value)
    if any(char in text for char in (",", "\"", "\n")):
        return "\"" + text.replace("\"", "\"\"") + "\""
    return text


def _stable_metric_seed(task_id: str, model_id: str, metric: str) -> int:
    return sum(ord(char) for char in f"{task_id}:{model_id}:{metric}") % 1_000_003
