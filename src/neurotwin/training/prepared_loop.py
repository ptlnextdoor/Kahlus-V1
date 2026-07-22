from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
from typing import Any

import torch
from torch import nn

from neurotwin.models.architecture_registry import build_architecture_model, normalize_architecture_type
from neurotwin.models.pair_operator import NeuroTwinPairOperatorConfig
from neurotwin.models.nfc import NeuralFieldCompilerConfig
from neurotwin.models.torch_models import NeuralStateSpaceTranslatorConfig
from neurotwin.repro import append_jsonl
from neurotwin.runtime.distributed import distributed_any, unwrap_model, wrap_ddp_if_initialized
from neurotwin.training.prepared_checkpoints import load_task_resume, save_task_checkpoint
from neurotwin.training.prepared_metrics import (
    batched_objective_loss,
    evaluate_task,
    indexed_metric_mask,
    mse_loss,
    predict,
    probabilistic_loss,
)
from neurotwin.training.prepared_types import (
    PreparedBatchSampler,
    PreparedCheckpointInfo,
    PreparedEvalSnapshot,
    PreparedRuntimeContext,
    PreparedTaskRunResult,
    PreparedTrainingConfig,
    TaskTrainingTensors,
)


class PreparedTrainingMonitor:
    def __init__(self, *, best_checkpoint_path: str | None, final_checkpoint_path: str | None) -> None:
        self.snapshots: list[PreparedEvalSnapshot] = []
        self.best_snapshot: PreparedEvalSnapshot | None = None
        self.best_model_state: dict[str, Any] | None = None
        self.best_optimizer_state: dict[str, Any] | None = None
        self.best_checkpoint_path = best_checkpoint_path
        self.final_checkpoint_path = final_checkpoint_path
        self.checkpoint_selection_metric = "val_mse"
        self.checkpoint_selection_mode = "min"

    def record_eval(
        self,
        *,
        step: int,
        metrics: dict[str, float],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if self.checkpoint_selection_metric not in metrics:
            return
        selection_value = float(metrics[self.checkpoint_selection_metric])
        if not math.isfinite(selection_value):
            return
        snapshot = PreparedEvalSnapshot(step=step, metrics=dict(metrics), selection_metric=self.checkpoint_selection_metric)
        self.snapshots.append(snapshot)
        if self.best_snapshot is None or snapshot.selection_value < self.best_snapshot.selection_value:
            self.best_snapshot = snapshot
            self.best_model_state = copy.deepcopy(unwrap_model(model).state_dict())
            self.best_optimizer_state = copy.deepcopy(optimizer.state_dict())

    def checkpoint_info(self, final_val_mse: float | None) -> PreparedCheckpointInfo:
        return PreparedCheckpointInfo(
            best_step=self.best_snapshot.step if self.best_snapshot is not None else None,
            best_val_mse=self.best_snapshot.selection_value if self.best_snapshot is not None else None,
            final_val_mse=final_val_mse,
            best_checkpoint_path=self.best_checkpoint_path if self.best_snapshot is not None else None,
            final_checkpoint_path=self.final_checkpoint_path,
            checkpoint_selection_metric=self.checkpoint_selection_metric,
            checkpoint_selection_mode=self.checkpoint_selection_mode,
        )

    def load_best_model(self, model: nn.Module) -> None:
        if self.best_model_state is not None:
            unwrap_model(model).load_state_dict(self.best_model_state)


@dataclass(frozen=True)
class TaskTrainingArtifacts:
    result: PreparedTaskRunResult
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    best_model_state: dict[str, Any] | None
    best_optimizer_state: dict[str, Any] | None


def train_single_task(
    task: Any,
    *,
    config: PreparedTrainingConfig,
    runtime: PreparedRuntimeContext,
    objective_weight: float,
) -> TaskTrainingArtifacts:
    model_config, model, optimizer, tensors = _prepare_task_training_state(task, config=config, runtime=runtime)
    monitor = PreparedTrainingMonitor(
        best_checkpoint_path=str(runtime.paths.best_checkpoint_path) if runtime.paths.best_checkpoint_path is not None else None,
        final_checkpoint_path=str(runtime.paths.checkpoint_path) if runtime.paths.checkpoint_path is not None else None,
    )
    initial_loss, final_loss = _run_task_training_loop(
        task,
        config=config,
        runtime=runtime,
        tensors=tensors,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        objective_weight=objective_weight,
        monitor=monitor,
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
        monitor=monitor,
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
    load_task_resume(model, optimizer, task.task_id, runtime.resume_checkpoint)
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
    monitor: PreparedTrainingMonitor,
) -> tuple[float, float]:

    with torch.no_grad():
        initial_loss = batched_objective_loss(
            model,
            task,
            tensors.x_train,
            tensors.y_train,
            precision=config.precision,
            batch_size=tensors.eval_batch_size,
            probabilistic=_uses_probabilistic_objective(model_config),
        )
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
        runtime=runtime,
    )
    if distributed_any(not math.isfinite(float(initial_loss)), device=runtime.device):
        _append_task_metric(
            runtime.paths.metrics_jsonl_path,
            {
                "task_id": task.task_id,
                "step": runtime.start_step,
                "phase": "nonfinite_loss",
                "loss": initial_loss,
                "optimizer_step_skipped": True,
            },
            runtime=runtime,
        )
        return initial_loss, float("nan")

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
            batch_mask = indexed_metric_mask(task, batch_indices)
            objective = (
                probabilistic_loss(model, task, xb, yb, precision=config.precision, metric_mask=batch_mask)
                if _uses_probabilistic_objective(model_config)
                else mse_loss(predict(model, task, xb, precision=config.precision), yb, metric_mask=batch_mask)
            )
            loss = (objective * objective_weight) / config.gradient_accumulation_steps
            loss_value = float(loss.detach())
            if distributed_any(not math.isfinite(loss_value), device=runtime.device):
                optimizer.zero_grad(set_to_none=True)
                _append_task_metric(
                    runtime.paths.metrics_jsonl_path,
                    {
                        "task_id": task.task_id,
                        "step": runtime.start_step + step + 1,
                        "phase": "nonfinite_loss",
                        "micro_step": micro_step,
                        "loss": loss_value,
                        "optimizer_step_skipped": True,
                    },
                    runtime=runtime,
                )
                return initial_loss, float("nan")
            loss.backward()
            accumulated_loss += loss_value
        grad_norm = _clip_or_measure_grad_norm(model, config.max_grad_norm)
        if distributed_any(grad_norm is not None and not math.isfinite(grad_norm), device=runtime.device):
            optimizer.zero_grad(set_to_none=True)
            _append_task_metric(
                runtime.paths.metrics_jsonl_path,
                {
                    "task_id": task.task_id,
                    "step": runtime.start_step + step + 1,
                    "phase": "nonfinite_gradient",
                    "grad_norm": grad_norm,
                    "optimizer_step_skipped": True,
                },
                runtime=runtime,
            )
            return initial_loss, float("nan")
        optimizer.step()
        final_loss = accumulated_loss
        completed_step = runtime.start_step + step + 1
        if config.eval_every_steps and completed_step % config.eval_every_steps == 0:
            val_snapshot = evaluate_task(
                model,
                task,
                tensors.x_val,
                tensors.y_val,
                precision=config.precision,
                prefix="val",
                batch_size=tensors.eval_batch_size,
            )
            monitor.record_eval(step=completed_step, metrics=val_snapshot, model=model, optimizer=optimizer)
            _append_task_metric(
                runtime.paths.metrics_jsonl_path,
                {"task_id": task.task_id, "step": completed_step, "phase": "eval", "selection_split": tensors.selection_split, **val_snapshot},
                runtime=runtime,
            )
        if (
            config.checkpoint_every_steps
            and runtime.dist_info.is_rank_zero
            and runtime.checkpoint_dir is not None
            and completed_step % config.checkpoint_every_steps == 0
        ):
            save_task_checkpoint(
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
    monitor: PreparedTrainingMonitor,
) -> TaskTrainingArtifacts:
    final_val_metrics = evaluate_task(model, task, tensors.x_val, tensors.y_val, precision=config.precision, prefix="val", batch_size=tensors.eval_batch_size)
    completed_step = runtime.start_step + config.steps
    monitor.record_eval(step=completed_step, metrics=final_val_metrics, model=model, optimizer=optimizer)
    final_model_state = copy.deepcopy(unwrap_model(model).state_dict())
    final_optimizer_state = copy.deepcopy(optimizer.state_dict())
    checkpoint_info = monitor.checkpoint_info(final_val_mse=final_val_metrics["val_mse"])
    monitor.load_best_model(model)
    selected_val_metrics = dict(monitor.best_snapshot.metrics) if monitor.best_snapshot is not None else final_val_metrics
    test_metrics = evaluate_task(model, task, tensors.x_test, tensors.y_test, precision=config.precision, prefix="test", batch_size=tensors.eval_batch_size)
    result = _build_task_training_result(
        task,
        config=config,
        runtime=runtime,
        tensors=tensors,
        model_config=model_config,
        initial_loss=initial_loss,
        final_loss=final_loss,
        objective_weight=objective_weight,
        val_metrics=selected_val_metrics,
        test_metrics=test_metrics,
        checkpoint_info=checkpoint_info,
    )
    _append_task_metric(
        runtime.paths.metrics_jsonl_path,
        {
            "task_id": task.task_id,
            "step": completed_step,
            "phase": "final",
            "checkpoint_role": "selected_best",
            "best_step": checkpoint_info.best_step,
            "best_val_mse": checkpoint_info.best_val_mse,
            "final_val_mse": checkpoint_info.final_val_mse,
            **selected_val_metrics,
            **test_metrics,
        },
        runtime=runtime,
    )
    return TaskTrainingArtifacts(
        result=result,
        model_state=final_model_state,
        optimizer_state=final_optimizer_state,
        best_model_state=monitor.best_model_state,
        best_optimizer_state=monitor.best_optimizer_state,
    )


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
    checkpoint_info: PreparedCheckpointInfo,
) -> PreparedTaskRunResult:
    return PreparedTaskRunResult(
        status="completed",
        task_id=task.task_id,
        source_modality=task.source_modality,
        target_modality=task.target_modality,
        initial_loss=initial_loss,
        final_loss=final_loss,
        steps=config.steps,
        start_step=runtime.start_step,
        completed_steps=runtime.start_step + config.steps,
        train_samples=int(tensors.x_train.shape[0]),
        val_samples=int(tensors.x_val.shape[0]),
        test_samples=int(tensors.x_test.shape[0]),
        resumed_from=str(runtime.paths.resume_path) if runtime.paths.resume_path is not None else None,
        objective_weight=objective_weight,
        compile=config.compile,
        selection_split=tensors.selection_split,
        report_split="test",
        model_config=model_config,
        best_val_mse=checkpoint_info.best_val_mse,
        final_val_mse=checkpoint_info.final_val_mse,
        best_step=checkpoint_info.best_step,
        best_checkpoint_path=checkpoint_info.best_checkpoint_path,
        final_checkpoint_path=checkpoint_info.final_checkpoint_path,
        checkpoint_selection_metric=checkpoint_info.checkpoint_selection_metric,
        checkpoint_selection_mode=checkpoint_info.checkpoint_selection_mode,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )


def _model_config_for_task(task: Any, config: PreparedTrainingConfig) -> dict[str, Any]:
    model_cfg = config.model
    adapter_mode = model_cfg.adapter_mode
    use_subject_embeddings = model_cfg.use_subject_embeddings and adapter_mode in {"few_shot", "enabled", "subject"}
    model_type = _normalize_model_type(model_cfg.type)
    if model_type == "NeuroTwinPairOperator":
        pair_config = NeuroTwinPairOperatorConfig(
            latent_dim=model_cfg.latent_dim,
            n_layers=model_cfg.n_layers,
            backbone=model_cfg.backbone,
            n_heads=model_cfg.n_heads,
            projection_dim=model_cfg.projection_dim,
            pair_rank=model_cfg.pair_rank,
            pair_top_k=model_cfg.pair_top_k,
            network_blocks=model_cfg.network_blocks,
            pair_confidence_max_parcels=model_cfg.pair_confidence_max_parcels,
            use_pair_state=model_cfg.use_pair_state,
            use_uncertainty_head=model_cfg.use_uncertainty_head,
            use_pair_uncertainty=model_cfg.use_pair_uncertainty,
            refinement_steps=model_cfg.refinement_steps,
            hrf_delay_steps=model_cfg.hrf_delay_steps,
        )
        return {
            "type": model_type,
            "input_dims": {task.source_modality: task.x_train.shape[-1]},
            "output_dims": {task.target_modality: task.y_train.shape[-1]},
            **asdict(pair_config),
        }
    if model_type == "NeuralFieldCompiler":
        nfc_config = NeuralFieldCompilerConfig(
            latent_dim=model_cfg.latent_dim,
            n_layers=model_cfg.n_layers,
            n_heads=model_cfg.n_heads,
            backbone=model_cfg.backbone,
            projection_dim=model_cfg.projection_dim,
            pair_rank=model_cfg.pair_rank,
            use_pair_kernel=model_cfg.use_pair_kernel,
            use_observation_operator=model_cfg.use_observation_operator,
            use_uncertainty=model_cfg.use_uncertainty,
            stimulus_lag_steps=model_cfg.stimulus_lag_steps,
            hrf_delay_steps=model_cfg.hrf_delay_steps,
            subject_state_dim=model_cfg.subject_state_dim,
            geometry_dim=model_cfg.geometry_dim,
        )
        return {
            "type": model_type,
            "input_dims": {task.source_modality: task.x_train.shape[-1]},
            "output_dims": {task.target_modality: task.y_train.shape[-1]},
            **asdict(nfc_config),
        }
    translator_config = NeuralStateSpaceTranslatorConfig(
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
        "type": model_type,
        "input_dims": {task.source_modality: task.x_train.shape[-1]},
        "output_dims": {task.target_modality: task.y_train.shape[-1]},
        **asdict(translator_config),
    }


def _uses_probabilistic_objective(model_config: dict[str, Any]) -> bool:
    model_type = str(model_config.get("type", ""))
    if model_type == "NeuralFieldCompiler":
        return bool(model_config.get("use_uncertainty"))
    if model_type == "NeuroTwinPairOperator":
        return bool(model_config.get("use_uncertainty_head"))
    return False


def _translator_from_model_config(model_config: dict[str, Any]) -> nn.Module:
    return build_architecture_model(model_config)


def _normalize_model_type(value: str) -> str:
    return normalize_architecture_type(value)


def _append_task_metric(path: str | None, row: dict[str, Any], *, runtime: PreparedRuntimeContext) -> None:
    if path is not None:
        enriched = dict(row)
        enriched.setdefault("rank", runtime.dist_info.rank)
        enriched.setdefault("world_size", runtime.dist_info.world_size)
        append_jsonl(path, _json_safe_metric_row(enriched))


def _json_safe_metric_row(row: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            safe[key] = None
        else:
            safe[key] = value
    return safe


def _clip_or_measure_grad_norm(model: nn.Module, max_grad_norm: float | None) -> float | None:
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if not parameters:
        return None
    if max_grad_norm is not None:
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_grad_norm)
        return float(norm.detach().cpu())
    total = torch.zeros((), device=parameters[0].device)
    for parameter in parameters:
        grad = parameter.grad.detach()
        total = total + grad.pow(2).sum()
    return float(torch.sqrt(total).detach().cpu())
