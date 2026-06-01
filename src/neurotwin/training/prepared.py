from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from neurotwin.config_types import (
    ConfigPath,
    PreparedTrainingConfigInput,
    ResolvedPreparedModelConfig,
    ResolvedPreparedRuntimeConfig,
    resolve_prepared_config,
)
from neurotwin.data.event_io import event_manifest_summary, load_event_batches
from neurotwin.data.manifest_io import load_split_manifest
from neurotwin.data.prepared_tasks import PreparedSuiteConfig, build_prepared_window_tasks
from neurotwin.models.torch_models import NeuralStateSpaceTranslator, NeuralStateSpaceTranslatorConfig
from neurotwin.repro import append_jsonl
from neurotwin.runtime.distributed import (
    DistributedInfo,
    cleanup_process_group,
    get_distributed_info,
    maybe_init_process_group,
    unwrap_model,
    wrap_ddp_if_initialized,
)
from neurotwin.scoring.metrics import bandpower_error, mae, mse, pearsonr, r2_score, regionwise_pearsonr, spearmanr


@dataclass(frozen=True)
class PreparedTrainingConfig:
    event_manifest: ConfigPath
    split_manifest: ConfigPath
    seed: int
    window_length: int
    stride: int
    steps: int
    requested_task: str
    model: ResolvedPreparedModelConfig
    runtime: ResolvedPreparedRuntimeConfig

    @classmethod
    def from_mapping(cls, config: PreparedTrainingConfigInput) -> "PreparedTrainingConfig":
        resolved = resolve_prepared_config(
            config,
            require_manifests=True,
            latent_dim_default=64,
            n_layers_default=1,
            projection_dim_default=32,
        )
        return cls(
            event_manifest=resolved.event_manifest or "",
            split_manifest=resolved.split_manifest or "",
            seed=resolved.seed,
            window_length=resolved.window_length,
            stride=resolved.stride,
            steps=resolved.steps,
            requested_task=resolved.requested_task,
            model=resolved.model,
            runtime=resolved.runtime,
        )

    @property
    def objective_weights(self) -> dict[str, float]:
        return self.runtime.objective_weights

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.runtime.gradient_accumulation_steps

    @property
    def precision(self) -> str:
        return self.runtime.precision

    @property
    def eval_every_steps(self) -> int:
        return self.runtime.eval_every_steps

    @property
    def checkpoint_every_steps(self) -> int:
        return self.runtime.checkpoint_every_steps

    @property
    def compile(self) -> bool:
        return self.runtime.compile

    @property
    def learning_rate(self) -> float:
        return self.runtime.learning_rate

    @property
    def batch_size(self) -> int | None:
        return self.runtime.batch_size

    @property
    def eval_batch_size(self) -> int | None:
        return self.runtime.eval_batch_size

    @property
    def gradient_checkpointing(self) -> bool:
        return self.model.gradient_checkpointing

    def suite_config(self) -> PreparedSuiteConfig:
        return PreparedSuiteConfig(
            event_manifest=self.event_manifest,
            split_manifest=self.split_manifest,
            window_length=self.window_length,
            stride=self.stride,
            seed=self.seed,
            train_steps=self.steps,
        )


@dataclass(frozen=True)
class PreparedTrainingRunPaths:
    checkpoint_path: str | Path | None = None
    resume_path: str | Path | None = None
    metrics_csv_path: str | Path | None = None
    metrics_jsonl_path: str | Path | None = None
    best_checkpoint_path: str | Path | None = None


@dataclass(frozen=True)
class PreparedTrainingResult:
    status: str
    task_id: str
    source_modality: str
    target_modality: str
    initial_loss: float
    final_loss: float
    eval_mse: float
    eval_mae: float
    eval_pearsonr: float
    eval_r2: float
    eval_spearmanr: float
    best_val_mse: float | None
    test_mse: float
    test_mae: float
    test_pearsonr: float
    test_r2: float
    test_spearmanr: float
    selection_split: str
    report_split: str
    steps: int
    start_step: int
    completed_steps: int
    gradient_accumulation_steps: int
    precision: str
    device: str
    resumed_from: str | None
    distributed_initialized: bool
    distributed_backend: str | None
    rank: int
    world_size: int
    train_samples: int
    test_samples: int
    synthetic_only: bool
    skipped_tasks: tuple[dict[str, str], ...]
    event_summary: dict[str, Any]
    task_results: tuple[dict[str, Any], ...] = ()
    best_task_id: str | None = None
    best_eval_mse: float | None = None
    eval_every_steps: int = 0
    checkpoint_every_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedTaskSelection:
    batches: list[Any]
    selected_tasks: tuple[Any, ...]
    skipped_tasks: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class PreparedRuntimeContext:
    device: torch.device
    dist_info: DistributedInfo
    resume_checkpoint: dict[str, Any] | None
    paths: PreparedTrainingRunPaths
    start_step: int
    checkpoint_dir: Path | None


@dataclass(frozen=True)
class PreparedTaskTrainingState:
    task_results: tuple[dict[str, Any], ...]
    task_states: dict[str, dict[str, Any]]
    best_task_id: str | None
    best_eval_mse: float | None
    first_model_state: dict[str, Any] | None
    first_optimizer_state: dict[str, Any] | None
    start_step: int


@dataclass(frozen=True)
class TaskTrainingTensors:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    selection_split: str
    batch_size: int
    eval_batch_size: int
    train_sampler: "PreparedBatchSampler"


@dataclass(frozen=True)
class TaskTrainingArtifacts:
    result: dict[str, Any]
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]


@dataclass(frozen=True)
class PreparedBatchSampler:
    num_samples: int
    batch_size: int
    gradient_accumulation_steps: int
    rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.world_size < 1:
            raise ValueError("world_size must be positive")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}), got {self.rank}")

    def indices(self, step: int, micro_step: int) -> list[int]:
        global_micro_step = step * self.gradient_accumulation_steps + micro_step
        if self.world_size == 1:
            start = (global_micro_step * self.batch_size) % self.num_samples
            end = min(start + self.batch_size, self.num_samples)
            return list(range(start, end))

        start = ((global_micro_step * self.world_size + self.rank) * self.batch_size) % self.num_samples
        return [(start + offset) % self.num_samples for offset in range(self.batch_size)]


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
    run_paths = _resolve_run_paths(
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
    dist_info: DistributedInfo,
    distributed_initialized: bool,
    distributed_backend: str | None,
    paths: PreparedTrainingRunPaths,
) -> PreparedTrainingResult:
    selection = _load_and_select_prepared_tasks(resolved)
    runtime = _build_prepared_runtime_context(dist_info, paths)
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


def _build_prepared_runtime_context(dist_info: DistributedInfo, paths: PreparedTrainingRunPaths) -> PreparedRuntimeContext:
    device = torch.device(f"cuda:{dist_info.local_rank}" if torch.cuda.is_available() else "cpu")
    resume_checkpoint = torch.load(Path(paths.resume_path), map_location=device, weights_only=True) if paths.resume_path is not None else None
    return PreparedRuntimeContext(
        device=device,
        dist_info=dist_info,
        resume_checkpoint=resume_checkpoint,
        paths=paths,
        start_step=_resume_start_step(resume_checkpoint),
        checkpoint_dir=Path(paths.checkpoint_path).parent if paths.checkpoint_path is not None else None,
    )


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
    first_model_state: dict[str, Any] | None = None
    first_optimizer_state: dict[str, Any] | None = None
    for task_index, task in enumerate(selected_tasks):
        artifact = _train_single_task(
            task,
            config=config,
            runtime=runtime,
            objective_weight=float(config.objective_weights.get(task.task_id, 1.0)),
        )
        task_result = artifact.result
        task_results.append(task_result)
        task_states[task.task_id] = {
            "model_config": task_result["model_config"],
            "model_state_dict": artifact.model_state,
            "optimizer_state_dict": artifact.optimizer_state,
        }
        if task_index == 0:
            first_model_state = artifact.model_state
            first_optimizer_state = artifact.optimizer_state
        if best_eval_mse is None or float(task_result["best_val_mse"]) < best_eval_mse:
            best_task_id = task.task_id
            best_eval_mse = float(task_result["best_val_mse"])
            if runtime.paths.best_checkpoint_path is not None:
                _save_task_checkpoint(
                    runtime.paths.best_checkpoint_path,
                    status="best_prepared_training",
                    task_result=task_result,
                    model_state=artifact.model_state,
                    optimizer_state=artifact.optimizer_state,
                )

    return PreparedTaskTrainingState(
        task_results=tuple(task_results),
        task_states=task_states,
        best_task_id=best_task_id,
        best_eval_mse=best_eval_mse,
        first_model_state=first_model_state,
        first_optimizer_state=first_optimizer_state,
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
    aggregate = _aggregate_task_results(task_results)
    summary = event_manifest_summary(resolved.event_manifest)
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
            },
            out,
        )
    if paths.metrics_csv_path is not None:
        _write_metrics_csv(paths.metrics_csv_path, result)


def _resolve_run_paths(
    paths: PreparedTrainingRunPaths | None,
    *,
    checkpoint_path: str | Path | None,
    resume_path: str | Path | None,
    metrics_csv_path: str | Path | None,
    metrics_jsonl_path: str | Path | None,
    best_checkpoint_path: str | Path | None,
) -> PreparedTrainingRunPaths:
    if paths is None:
        return PreparedTrainingRunPaths(
            checkpoint_path=checkpoint_path,
            resume_path=resume_path,
            metrics_csv_path=metrics_csv_path,
            metrics_jsonl_path=metrics_jsonl_path,
            best_checkpoint_path=best_checkpoint_path,
        )
    if any(value is not None for value in (checkpoint_path, resume_path, metrics_csv_path, metrics_jsonl_path, best_checkpoint_path)):
        raise ValueError("pass either paths=PreparedTrainingRunPaths(...) or legacy path keywords, not both")
    return paths


def _predict(model: nn.Module, task: Any, x: torch.Tensor, precision: str = "fp32") -> torch.Tensor:
    enabled = precision == "bf16" and x.device.type in {"cuda", "cpu"}
    task_model = unwrap_model(model)
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=enabled):
        output = task_model.forward_task(
            {task.source_modality: x},
            target_modality=task.target_modality,
            task="forecast" if task.task_id == "future_state_forecasting" else "reconstruction",
        )
    return output["prediction"]


def _mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(prediction.float(), target.float())


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


def _write_metrics_csv(path: str | Path, result: PreparedTrainingResult) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "status",
                "task_id",
                "initial_loss",
                "final_loss",
                "eval_mse",
                "eval_mae",
                "eval_pearsonr",
                "eval_r2",
                "eval_spearmanr",
                "best_val_mse",
                "test_mse",
                "test_mae",
                "test_pearsonr",
                "test_r2",
                "test_spearmanr",
                "selection_split",
                "report_split",
                "steps",
                "start_step",
                "completed_steps",
                "gradient_accumulation_steps",
                "precision",
                "device",
                "resumed_from",
                "distributed_initialized",
                "distributed_backend",
                "rank",
                "world_size",
            ],
        )
        writer.writeheader()
        writer.writerow({key: result.to_dict().get(key) for key in writer.fieldnames})
    return out


def _train_single_task(
    task: Any,
    *,
    config: PreparedTrainingConfig,
    runtime: PreparedRuntimeContext,
    objective_weight: float,
) -> TaskTrainingArtifacts:
    model_config, model, optimizer, tensors = _prepare_task_training_state(task, config=config, runtime=runtime)
    initial_loss, final_loss = _run_task_training_loop(
        task,
        config=config,
        runtime=runtime,
        tensors=tensors,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        objective_weight=objective_weight,
    )
    return _finalize_task_result(
        task,
        config=config,
        runtime=runtime,
        tensors=tensors,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        initial_loss=initial_loss,
        final_loss=final_loss,
        objective_weight=objective_weight,
    )


def _prepare_task_training_state(
    task: Any,
    *,
    config: PreparedTrainingConfig,
    runtime: PreparedRuntimeContext,
) -> tuple[dict[str, Any], nn.Module, torch.optim.Optimizer, TaskTrainingTensors]:
    model_config = _model_config_for_task(task, config)
    model = _translator_from_model_config(model_config)
    model.to(runtime.device)
    if config.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]  # compiled module preserves nn.Module behavior at runtime.
    tensors = _prepare_task_training_tensors(task, config, runtime)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    _load_task_resume(model, optimizer, task.task_id, runtime.resume_checkpoint)
    model = wrap_ddp_if_initialized(model, local_rank=runtime.dist_info.local_rank)
    return model_config, model, optimizer, tensors


def _run_task_training_loop(
    task: Any,
    *,
    config: PreparedTrainingConfig,
    runtime: PreparedRuntimeContext,
    tensors: TaskTrainingTensors,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: dict[str, Any],
    objective_weight: float,
) -> tuple[float, float]:

    with torch.no_grad():
        initial_loss = _batched_loss(model, task, tensors.x_train, tensors.y_train, precision=config.precision, batch_size=tensors.eval_batch_size)
    _append_task_metric(
        runtime.paths.metrics_jsonl_path,
        {
            "task_id": task.task_id,
            "step": runtime.start_step,
            "loss": initial_loss,
            "phase": "initial",
            "source_modality": task.source_modality,
            "target_modality": task.target_modality,
        },
    )

    model.train()
    final_loss = initial_loss
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for micro_step in range(config.gradient_accumulation_steps):
            batch_indices = tensors.train_sampler.indices(step=step, micro_step=micro_step)
            index_tensor = torch.as_tensor(batch_indices, dtype=torch.long, device=runtime.device)
            xb = tensors.x_train.index_select(0, index_tensor)
            yb = tensors.y_train.index_select(0, index_tensor)
            loss = (_mse_loss(_predict(model, task, xb, precision=config.precision), yb) * objective_weight) / config.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += float(loss.detach())
        optimizer.step()
        final_loss = accumulated_loss
        completed_step = runtime.start_step + step + 1
        if config.eval_every_steps and completed_step % config.eval_every_steps == 0:
            val_snapshot = _evaluate_task(
                model,
                task,
                tensors.x_val,
                tensors.y_val,
                precision=config.precision,
                prefix="val",
                batch_size=tensors.eval_batch_size,
            )
            _append_task_metric(
                runtime.paths.metrics_jsonl_path,
                {"task_id": task.task_id, "step": completed_step, "phase": "eval", "selection_split": tensors.selection_split, **val_snapshot},
            )
        if config.checkpoint_every_steps and runtime.checkpoint_dir is not None and completed_step % config.checkpoint_every_steps == 0:
            _save_task_checkpoint(
                runtime.checkpoint_dir / f"checkpoint_{task.task_id}_step_{completed_step}.pt",
                status="periodic_prepared_training",
                task_result={"task_id": task.task_id, "completed_steps": completed_step, "model_config": model_config},
                model_state=unwrap_model(model).state_dict(),
                optimizer_state=optimizer.state_dict(),
            )

    return initial_loss, final_loss


def _finalize_task_result(
    task: Any,
    *,
    config: PreparedTrainingConfig,
    runtime: PreparedRuntimeContext,
    tensors: TaskTrainingTensors,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: dict[str, Any],
    initial_loss: float,
    final_loss: float,
    objective_weight: float,
) -> TaskTrainingArtifacts:
    val_metrics = _evaluate_task(model, task, tensors.x_val, tensors.y_val, precision=config.precision, prefix="val", batch_size=tensors.eval_batch_size)
    test_metrics = _evaluate_task(model, task, tensors.x_test, tensors.y_test, precision=config.precision, prefix="test", batch_size=tensors.eval_batch_size)
    model_state = unwrap_model(model).state_dict()
    optimizer_state = optimizer.state_dict()
    result = _build_task_training_result(
        task,
        config=config,
        runtime=runtime,
        tensors=tensors,
        model_config=model_config,
        initial_loss=initial_loss,
        final_loss=final_loss,
        objective_weight=objective_weight,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    _append_task_metric(
        runtime.paths.metrics_jsonl_path,
        {"task_id": task.task_id, "step": runtime.start_step + config.steps, "phase": "final", **val_metrics, **test_metrics},
    )
    return TaskTrainingArtifacts(result=result, model_state=model_state, optimizer_state=optimizer_state)


def _prepare_task_training_tensors(
    task: Any,
    config: PreparedTrainingConfig,
    runtime: PreparedRuntimeContext,
) -> TaskTrainingTensors:
    selection_split = "val" if task.x_val is not None and task.y_val is not None else "train"
    x_train = torch.as_tensor(task.x_train, dtype=torch.float32, device=runtime.device)
    y_train = torch.as_tensor(task.y_train, dtype=torch.float32, device=runtime.device)
    x_val = torch.as_tensor(task.x_val if task.x_val is not None else task.x_train, dtype=torch.float32, device=runtime.device)
    y_val = torch.as_tensor(task.y_val if task.y_val is not None else task.y_train, dtype=torch.float32, device=runtime.device)
    x_test = torch.as_tensor(task.x_test, dtype=torch.float32, device=runtime.device)
    y_test = torch.as_tensor(task.y_test, dtype=torch.float32, device=runtime.device)
    batch_size = max(1, config.batch_size or int(x_train.shape[0]))
    eval_batch_size = max(1, config.eval_batch_size or batch_size)
    return TaskTrainingTensors(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        selection_split=selection_split,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        train_sampler=PreparedBatchSampler(
            num_samples=int(x_train.shape[0]),
            batch_size=batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            rank=runtime.dist_info.rank,
            world_size=runtime.dist_info.world_size,
        ),
    )


def _build_task_training_result(
    task: Any,
    *,
    config: PreparedTrainingConfig,
    runtime: PreparedRuntimeContext,
    tensors: TaskTrainingTensors,
    model_config: dict[str, Any],
    initial_loss: float,
    final_loss: float,
    objective_weight: float,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "status": "completed",
        "task_id": task.task_id,
        "source_modality": task.source_modality,
        "target_modality": task.target_modality,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "steps": config.steps,
        "start_step": runtime.start_step,
        "completed_steps": runtime.start_step + config.steps,
        "train_samples": int(tensors.x_train.shape[0]),
        "val_samples": int(tensors.x_val.shape[0]),
        "test_samples": int(tensors.x_test.shape[0]),
        "resumed_from": str(runtime.paths.resume_path) if runtime.paths.resume_path is not None else None,
        "objective_weight": objective_weight,
        "compile": config.compile,
        "selection_split": tensors.selection_split,
        "report_split": "test",
        "model_config": model_config,
        "best_val_mse": val_metrics["val_mse"],
        "eval_mse": test_metrics["test_mse"],
        "eval_mae": test_metrics["test_mae"],
        "eval_pearsonr": test_metrics["test_pearsonr"],
        "eval_r2": test_metrics["test_r2"],
        "eval_spearmanr": test_metrics["test_spearmanr"],
        **val_metrics,
        **test_metrics,
    }


def _batched_loss(
    model: nn.Module,
    task: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    precision: str,
    batch_size: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            pred = _predict(model, task, x[start:end], precision=precision)
            batch_samples = end - start
            total_loss += float(_mse_loss(pred, y[start:end])) * batch_samples
            total_samples += batch_samples
    model.train()
    return total_loss / max(1, total_samples)


def _predict_numpy_batches(
    model: nn.Module,
    task: Any,
    x: torch.Tensor,
    precision: str,
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            pred = _predict(model, task, x[start:end], precision=precision)
            predictions.append(pred.detach().float().cpu().numpy())
    model.train()
    return np.concatenate(predictions, axis=0) if predictions else np.empty((0,), dtype=np.float32)


def _evaluate_task(
    model: nn.Module,
    task: Any,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    precision: str,
    prefix: str,
    batch_size: int,
) -> dict[str, float]:
    y_true_np = y_test.detach().cpu().numpy()
    y_pred_np = _predict_numpy_batches(model, task, x_test, precision=precision, batch_size=batch_size)
    eval_loss = mse(y_true_np, y_pred_np)
    metrics = {
        f"{prefix}_mse": eval_loss,
        f"{prefix}_mae": mae(y_true_np, y_pred_np),
        f"{prefix}_pearsonr": pearsonr(y_true_np, y_pred_np),
        f"{prefix}_spearmanr": spearmanr(y_true_np, y_pred_np),
        f"{prefix}_r2": r2_score(y_true_np, y_pred_np),
    }
    if task.target_modality in {"eeg", "meg"} or task.source_modality in {"eeg", "meg"}:
        metrics[f"{prefix}_bandpower_error"] = bandpower_error(y_true_np, y_pred_np)
    if task.target_modality == "fmri":
        metrics[f"{prefix}_regionwise_pearsonr"] = regionwise_pearsonr(y_true_np, y_pred_np)
    return metrics


def _model_config_for_task(task: Any, config: PreparedTrainingConfig) -> dict[str, Any]:
    model_cfg = config.model
    adapter_mode = model_cfg.adapter_mode
    use_subject_embeddings = model_cfg.use_subject_embeddings and adapter_mode in {"few_shot", "enabled", "subject"}
    architecture = NeuralStateSpaceTranslatorConfig(
        latent_dim=model_cfg.latent_dim,
        n_layers=model_cfg.n_layers,
        subject_adapter_dim=model_cfg.subject_adapter_dim,
        projection_dim=model_cfg.projection_dim,
        metadata_dim=model_cfg.metadata_dim,
        geometry_dim=model_cfg.geometry_dim,
        backbone=model_cfg.backbone,
        encoder=model_cfg.encoder,
        n_heads=model_cfg.n_heads,
        subject_vocab_size=model_cfg.subject_vocab_size,
        use_subject_embeddings=use_subject_embeddings,
        adapter_mode=adapter_mode,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    return {
        "input_dims": {task.source_modality: task.x_train.shape[-1]},
        "output_dims": {task.target_modality: task.y_train.shape[-1]},
        **asdict(architecture),
    }


def _translator_from_model_config(model_config: dict[str, Any]) -> NeuralStateSpaceTranslator:
    return NeuralStateSpaceTranslator(
        input_dims=dict(model_config["input_dims"]),
        output_dims=dict(model_config["output_dims"]),
        config=NeuralStateSpaceTranslatorConfig.from_mapping(model_config),
    )


def _load_task_resume(model: nn.Module, optimizer: torch.optim.Optimizer, task_id: str, checkpoint: dict[str, Any] | None) -> None:
    if checkpoint is None:
        return
    task_states = checkpoint.get("task_states")
    if isinstance(task_states, dict) and task_id in task_states:
        task_state = task_states[task_id]
        model.load_state_dict(task_state["model_state_dict"])
        if "optimizer_state_dict" in task_state:
            optimizer.load_state_dict(task_state["optimizer_state_dict"])
        return
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def _resume_start_step(checkpoint: dict[str, Any] | None) -> int:
    if checkpoint is None:
        return 0
    return int(checkpoint.get("completed_steps", checkpoint.get("metrics", {}).get("completed_steps", 0)))


def _aggregate_task_results(task_results: Sequence[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "initial_loss",
        "final_loss",
        "eval_mse",
        "eval_mae",
        "eval_pearsonr",
        "eval_r2",
        "eval_spearmanr",
        "best_val_mse",
        "test_mse",
        "test_mae",
        "test_pearsonr",
        "test_r2",
        "test_spearmanr",
    )
    return {key: float(np.mean([float(result[key]) for result in task_results])) for key in keys}


def _append_task_metric(path: str | Path | None, row: dict[str, Any]) -> None:
    if path is not None:
        append_jsonl(path, row)


def _save_task_checkpoint(
    path: str | Path,
    status: str,
    task_result: dict[str, Any],
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "status": status,
            "task": {key: task_result.get(key) for key in ("task_id", "source_modality", "target_modality")},
            "metrics": task_result,
            "model_config": task_result.get("model_config"),
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "completed_steps": task_result.get("completed_steps"),
        },
        out,
    )
