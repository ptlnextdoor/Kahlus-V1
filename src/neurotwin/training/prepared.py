from __future__ import annotations

from pathlib import Path
import math
from typing import Any

import torch

from neurotwin.config_types import PreparedTrainingConfigInput
from neurotwin.data.event_io import event_manifest_summary, load_event_batches
from neurotwin.data.manifest_io import load_split_manifest
from neurotwin.data.prepared_tasks import build_prepared_window_tasks
from neurotwin.runtime.distributed import (
    cleanup_process_group,
    get_distributed_info,
    maybe_init_process_group,
)
from neurotwin.training.prepared_checkpoints import resolve_run_paths, save_task_checkpoint
from neurotwin.training.prepared_loop import train_single_task
from neurotwin.training.prepared_metrics import aggregate_task_results, write_metrics_csv
from neurotwin.training.prepared_runtime import build_prepared_runtime_context
from neurotwin.training.prepared_types import (
    PreparedBatchSampler,
    PreparedRuntimeContext,
    PreparedTaskSelection,
    PreparedTaskTrainingState,
    PreparedTrainingConfig,
    PreparedTrainingResult,
    PreparedTrainingRunPaths,
)


def run_prepared_training(
    config: PreparedTrainingConfigInput | PreparedTrainingConfig,
    checkpoint_path: str | Path | None = None,
    resume_path: str | Path | None = None,
    metrics_csv_path: str | Path | None = None,
    metrics_jsonl_path: str | Path | None = None,
    best_checkpoint_path: str | Path | None = None,
    paths: PreparedTrainingRunPaths | None = None,
) -> PreparedTrainingResult:
    resolved = config if isinstance(config, PreparedTrainingConfig) else PreparedTrainingConfig.from_mapping(config)
    run_paths = resolve_run_paths(
        paths,
        checkpoint_path=checkpoint_path,
        resume_path=resume_path,
        metrics_csv_path=metrics_csv_path,
        metrics_jsonl_path=metrics_jsonl_path,
        best_checkpoint_path=best_checkpoint_path,
    )

    torch.manual_seed(resolved.seed)
    dist_info = get_distributed_info()
    distributed_initialized, distributed_backend = maybe_init_process_group(dist_info)
    try:
        return _run_prepared_training_body(
            resolved,
            dist_info=dist_info,
            distributed_initialized=distributed_initialized,
            distributed_backend=distributed_backend,
            paths=run_paths,
        )
    finally:
        if distributed_initialized:
            cleanup_process_group()


def _run_prepared_training_body(
    resolved: PreparedTrainingConfig,
    *,
    dist_info: object,
    distributed_initialized: bool,
    distributed_backend: str | None,
    paths: PreparedTrainingRunPaths,
) -> PreparedTrainingResult:
    selection = _load_and_select_prepared_tasks(resolved)
    runtime = build_prepared_runtime_context(dist_info, paths)
    training_state = _train_selected_prepared_tasks(
        selection.selected_tasks,
        config=resolved,
        runtime=runtime,
    )
    result = _build_prepared_training_result(
        resolved,
        selection,
        training_state,
        runtime=runtime,
        distributed_initialized=distributed_initialized,
        distributed_backend=distributed_backend,
    )
    _persist_prepared_training_artifacts(runtime.paths, result, training_state)
    return result


def _load_and_select_prepared_tasks(resolved: PreparedTrainingConfig) -> PreparedTaskSelection:
    batches = load_event_batches(resolved.event_manifest)
    split = load_split_manifest(resolved.split_manifest)
    suite_config = resolved.suite_config()
    tasks, skipped = build_prepared_window_tasks(
        batches,
        split,
        window_length=suite_config.window_length,
        stride=suite_config.stride,
        seed=resolved.seed,
    )
    selected_tasks = _select_tasks(tasks, resolved.requested_task)
    if not selected_tasks:
        available = ", ".join(task.task_id for task in tasks) or "none"
        raise ValueError(f"No runnable prepared training task matched config task. Available: {available}")
    return PreparedTaskSelection(
        batches=batches,
        selected_tasks=selected_tasks,
        skipped_tasks=tuple(skipped),
    )


def _train_selected_prepared_tasks(
    selected_tasks: tuple[Any, ...],
    *,
    config: PreparedTrainingConfig,
    runtime: PreparedRuntimeContext,
) -> PreparedTaskTrainingState:
    task_results: list[dict[str, Any]] = []
    task_states: dict[str, dict[str, Any]] = {}
    best_task_id: str | None = None
    best_eval_mse: float | None = None
    best_model_state: dict[str, Any] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    first_model_state: dict[str, Any] | None = None
    first_optimizer_state: dict[str, Any] | None = None
    for task_index, task in enumerate(selected_tasks):
        artifact = train_single_task(
            task,
            config=config,
            runtime=runtime,
            objective_weight=float(config.objective_weights.get(task.task_id, 1.0)),
        )
        task_result = artifact.result.to_dict()
        _mark_task_quarantine(task_result)
        task_results.append(task_result)
        task_states[task.task_id] = {
            "model_config": task_result["model_config"],
            "model_state_dict": artifact.model_state,
            "optimizer_state_dict": artifact.optimizer_state,
        }
        if task_index == 0:
            first_model_state = artifact.model_state
            first_optimizer_state = artifact.optimizer_state
        task_best = None if _is_quarantined_task(task_result) else task_result.get("best_val_mse")
        if task_best is not None and (best_eval_mse is None or float(task_best) < best_eval_mse):
            best_task_id = task.task_id
            best_eval_mse = float(task_best)
            best_model_state = artifact.best_model_state or artifact.model_state
            best_optimizer_state = artifact.best_optimizer_state or artifact.optimizer_state
            if runtime.paths.best_checkpoint_path is not None:
                save_task_checkpoint(
                    runtime.paths.best_checkpoint_path,
                    status="best_prepared_training",
                    task_result=task_result,
                    model_state=best_model_state,
                    optimizer_state=best_optimizer_state,
                )

    return PreparedTaskTrainingState(
        task_results=tuple(task_results),
        task_states=task_states,
        best_task_id=best_task_id,
        best_eval_mse=best_eval_mse,
        first_model_state=first_model_state,
        first_optimizer_state=first_optimizer_state,
        best_model_state=best_model_state,
        best_optimizer_state=best_optimizer_state,
        start_step=runtime.start_step,
    )


def _build_prepared_training_result(
    resolved: PreparedTrainingConfig,
    selection: PreparedTaskSelection,
    training_state: PreparedTaskTrainingState,
    *,
    runtime: PreparedRuntimeContext,
    distributed_initialized: bool,
    distributed_backend: str | None,
) -> PreparedTrainingResult:
    task_results = training_state.task_results
    primary = task_results[0]
    aggregateable = tuple(row for row in task_results if not _is_quarantined_task(row))
    aggregate = aggregate_task_results(aggregateable or task_results)
    summary = event_manifest_summary(resolved.event_manifest)
    quarantined_tasks = tuple(
        {
            "task_id": str(row.get("task_id")),
            "reason": str(row.get("quarantine_reason", "nonfinite task metric")),
        }
        for row in task_results
        if _is_quarantined_task(row)
    )
    return PreparedTrainingResult(
        status="completed_prepared_training",
        task_id=resolved.requested_task if len(task_results) > 1 else str(primary["task_id"]),
        source_modality="multi" if len(task_results) > 1 else str(primary["source_modality"]),
        target_modality="multi" if len(task_results) > 1 else str(primary["target_modality"]),
        initial_loss=float(aggregate["initial_loss"]),
        final_loss=float(aggregate["final_loss"]),
        eval_mse=float(aggregate["eval_mse"]),
        eval_mae=float(aggregate["eval_mae"]),
        eval_pearsonr=float(aggregate["eval_pearsonr"]),
        eval_r2=float(aggregate["eval_r2"]),
        eval_spearmanr=float(aggregate["eval_spearmanr"]),
        best_val_mse=float(aggregate["best_val_mse"]),
        final_val_mse=float(aggregate["final_val_mse"]),
        best_step=_best_step(task_results, training_state.best_task_id),
        best_checkpoint_path=str(runtime.paths.best_checkpoint_path) if runtime.paths.best_checkpoint_path is not None and training_state.best_model_state is not None else None,
        final_checkpoint_path=str(runtime.paths.checkpoint_path) if runtime.paths.checkpoint_path is not None else None,
        checkpoint_selection_metric="val_mse",
        checkpoint_selection_mode="min",
        test_mse=float(aggregate["test_mse"]),
        test_mae=float(aggregate["test_mae"]),
        test_pearsonr=float(aggregate["test_pearsonr"]),
        test_r2=float(aggregate["test_r2"]),
        test_spearmanr=float(aggregate["test_spearmanr"]),
        selection_split="val" if all(str(row.get("selection_split")) == "val" for row in task_results) else "mixed_non_test",
        report_split="test",
        steps=resolved.steps,
        start_step=training_state.start_step,
        completed_steps=training_state.start_step + resolved.steps,
        gradient_accumulation_steps=resolved.gradient_accumulation_steps,
        max_grad_norm=resolved.max_grad_norm,
        precision=resolved.precision,
        device=str(runtime.device),
        resumed_from=str(runtime.paths.resume_path) if runtime.paths.resume_path is not None else None,
        distributed_initialized=distributed_initialized,
        distributed_backend=distributed_backend,
        rank=runtime.dist_info.rank,
        world_size=runtime.dist_info.world_size,
        train_samples=int(sum(int(task_result["train_samples"]) for task_result in task_results)),
        test_samples=int(sum(int(task_result["test_samples"]) for task_result in task_results)),
        synthetic_only=_is_synthetic_manifest(summary, selection.batches),
        skipped_tasks=selection.skipped_tasks,
        event_summary=summary,
        task_results=tuple(task_results),
        best_task_id=training_state.best_task_id,
        best_eval_mse=training_state.best_eval_mse,
        eval_every_steps=resolved.eval_every_steps,
        checkpoint_every_steps=resolved.checkpoint_every_steps,
        quarantined_tasks=quarantined_tasks,
        stimulus_evidence=_stimulus_evidence_from_tasks(selection.selected_tasks),
    )


def _persist_prepared_training_artifacts(
    paths: PreparedTrainingRunPaths,
    result: PreparedTrainingResult,
    training_state: PreparedTaskTrainingState,
) -> None:
    primary = training_state.task_results[0]
    if paths.checkpoint_path is not None:
        out = Path(paths.checkpoint_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "status": result.status,
                "model_config": primary["model_config"],
                "task": {key: primary[key] for key in ("task_id", "source_modality", "target_modality")},
                "metrics": result.to_dict(),
                "model_state_dict": training_state.first_model_state,
                "optimizer_state_dict": training_state.first_optimizer_state,
                "task_states": training_state.task_states,
                "completed_steps": result.completed_steps,
                "final_checkpoint_path": result.final_checkpoint_path,
                "best_checkpoint_path": result.best_checkpoint_path,
                "best_step": result.best_step,
                "checkpoint_selection_metric": result.checkpoint_selection_metric,
                "checkpoint_selection_mode": result.checkpoint_selection_mode,
            },
            out,
        )
    if paths.metrics_csv_path is not None:
        write_metrics_csv(paths.metrics_csv_path, result)


def _select_tasks(tasks: tuple[Any, ...], requested: str) -> tuple[Any, ...]:
    if requested in {"neural_translation_v1", "translation_smoke", "prepared"}:
        return tasks
    for task in tasks:
        if task.task_id == requested:
            return (task,)
    return ()


def _all_synthetic(batches: list[Any]) -> bool:
    return bool(batches) and all(bool(batch.metadata.get("synthetic")) for batch in batches)


def _is_synthetic_manifest(summary: dict[str, Any], batches: list[Any]) -> bool:
    return str(summary.get("schema")) in {
        "neurotwin.event_manifest.v1",
        "neurotwin.event_manifest.v2",
    } and _all_synthetic(batches)


def _best_step(task_results: tuple[dict[str, Any], ...], best_task_id: str | None) -> int | None:
    if best_task_id is None:
        return None
    for row in task_results:
        if row.get("task_id") == best_task_id:
            value = row.get("best_step")
            return int(value) if value is not None else None
    return None


def _mark_task_quarantine(task_result: dict[str, Any]) -> None:
    nonfinite = sorted(
        key
        for key, value in task_result.items()
        if isinstance(value, (int, float)) and not math.isfinite(float(value))
    )
    if nonfinite:
        task_result["status"] = "quarantined_nonfinite"
        task_result["quarantine_reason"] = "nonfinite metric(s): " + ",".join(nonfinite)
        for key in nonfinite:
            task_result[key] = None


def _is_quarantined_task(task_result: dict[str, Any]) -> bool:
    return str(task_result.get("status", "")).startswith("quarantined")


def _stimulus_evidence_from_tasks(tasks: tuple[Any, ...]) -> dict[str, Any] | None:
    for task in tasks:
        metadata = getattr(task, "metadata", {})
        if isinstance(metadata, dict) and isinstance(metadata.get("stimulus_evidence"), dict):
            return dict(metadata["stimulus_evidence"])
    return None
