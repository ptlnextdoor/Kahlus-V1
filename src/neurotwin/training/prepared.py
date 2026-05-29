from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from neurotwin.benchmarks.prepared_suite import PreparedSuiteConfig, build_prepared_window_tasks
from neurotwin.data.event_io import event_manifest_summary, load_event_batches
from neurotwin.data.manifest_io import load_split_manifest
from neurotwin.eval.metrics import bandpower_error, mae, mse, pearsonr, r2_score, regionwise_pearsonr, spearmanr
from neurotwin.models.torch_models import NeuralStateSpaceTranslator
from neurotwin.repro import append_jsonl
from neurotwin.runtime.distributed import (
    cleanup_process_group,
    get_distributed_info,
    maybe_init_process_group,
    unwrap_model,
    wrap_ddp_if_initialized,
)


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


def run_prepared_training(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    resume_path: str | Path | None = None,
    metrics_csv_path: str | Path | None = None,
    metrics_jsonl_path: str | Path | None = None,
    best_checkpoint_path: str | Path | None = None,
) -> PreparedTrainingResult:
    data_config = dict(config.get("data", {}))
    event_manifest = config.get("event_manifest") or data_config.get("event_manifest")
    split_manifest = config.get("split_manifest") or data_config.get("split_manifest")
    if not event_manifest or not split_manifest:
        raise ValueError("prepared training requires event_manifest and split_manifest")

    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    dist_info = get_distributed_info()
    distributed_initialized, distributed_backend = maybe_init_process_group(dist_info)
    batches = load_event_batches(event_manifest)
    split = load_split_manifest(split_manifest)
    suite_config = PreparedSuiteConfig(
        event_manifest=event_manifest,
        split_manifest=split_manifest,
        window_length=int(config.get("window_size", config.get("window_length", 8))),
        stride=int(config.get("stride", config.get("window_size", config.get("window_length", 8)))),
        seed=seed,
        train_steps=int(config.get("steps", 24)),
    )
    tasks, skipped = build_prepared_window_tasks(
        batches,
        split,
        window_length=suite_config.window_length,
        stride=suite_config.stride,
        seed=seed,
    )
    requested_task = str(config.get("task", "future_state_forecasting"))
    selected_tasks = _select_tasks(tasks, requested_task)
    if not selected_tasks:
        available = ", ".join(task.task_id for task in tasks) or "none"
        raise ValueError(f"No runnable prepared training task matched config task. Available: {available}")
    model_cfg = dict(config.get("model", {}))
    device = torch.device(f"cuda:{dist_info.local_rank}" if torch.cuda.is_available() else "cpu")
    steps = int(config.get("steps", 24))
    grad_accum = max(1, int(config.get("gradient_accumulation_steps", dict(config.get("training", {})).get("gradient_accumulation_steps", 1))))
    precision = str(config.get("precision", dict(config.get("training", {})).get("precision", "fp32"))).lower()
    training_config = dict(config.get("training", {}))
    eval_every_steps = max(0, int(config.get("eval_every_steps", training_config.get("eval_every_steps", 0))))
    checkpoint_every_steps = max(0, int(config.get("checkpoint_every_steps", training_config.get("checkpoint_every_steps", 0))))
    objective_weights = dict(training_config.get("objective_weights", config.get("objective_weights", {})))
    resume_checkpoint = torch.load(Path(resume_path), map_location=device) if resume_path is not None else None
    task_results: list[dict[str, Any]] = []
    task_states: dict[str, dict[str, Any]] = {}
    best_task_id: str | None = None
    best_eval_mse: float | None = None
    first_model_state: dict[str, Any] | None = None
    first_optimizer_state: dict[str, Any] | None = None
    start_step = _resume_start_step(resume_checkpoint)
    for task_index, task in enumerate(selected_tasks):
        task_result, model_state, optimizer_state = _train_single_task(
            task,
            config=config,
            model_cfg=model_cfg,
            device=device,
            dist_local_rank=dist_info.local_rank,
            resume_checkpoint=resume_checkpoint,
            resume_path=resume_path,
            metrics_jsonl_path=metrics_jsonl_path,
            precision=precision,
            steps=steps,
            start_step=start_step,
            gradient_accumulation_steps=grad_accum,
            eval_every_steps=eval_every_steps,
            checkpoint_every_steps=checkpoint_every_steps,
            checkpoint_dir=Path(checkpoint_path).parent if checkpoint_path is not None else None,
            objective_weight=float(objective_weights.get(task.task_id, 1.0)),
        )
        task_results.append(task_result)
        task_states[task.task_id] = {
            "model_config": task_result["model_config"],
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
        }
        if task_index == 0:
            first_model_state = model_state
            first_optimizer_state = optimizer_state
        if best_eval_mse is None or float(task_result["best_val_mse"]) < best_eval_mse:
            best_task_id = task.task_id
            best_eval_mse = float(task_result["best_val_mse"])
            if best_checkpoint_path is not None:
                _save_task_checkpoint(
                    best_checkpoint_path,
                    status="best_prepared_training",
                    task_result=task_result,
                    model_state=model_state,
                    optimizer_state=optimizer_state,
                )

    primary = task_results[0]
    aggregate = _aggregate_task_results(task_results)
    summary = event_manifest_summary(event_manifest)
    result = PreparedTrainingResult(
        status="completed_prepared_training",
        task_id=requested_task if len(task_results) > 1 else str(primary["task_id"]),
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
        steps=steps,
        start_step=start_step,
        completed_steps=start_step + steps,
        gradient_accumulation_steps=grad_accum,
        precision=precision,
        device=str(device),
        resumed_from=str(resume_path) if resume_path is not None else None,
        distributed_initialized=distributed_initialized,
        distributed_backend=distributed_backend,
        rank=dist_info.rank,
        world_size=dist_info.world_size,
        train_samples=int(sum(int(task_result["train_samples"]) for task_result in task_results)),
        test_samples=int(sum(int(task_result["test_samples"]) for task_result in task_results)),
        synthetic_only=summary.get("schema") == "neurotwin.event_manifest.v1" and _all_synthetic(batches),
        skipped_tasks=tuple(skipped),
        event_summary=summary,
        task_results=tuple(task_results),
        best_task_id=best_task_id,
        best_eval_mse=best_eval_mse,
        eval_every_steps=eval_every_steps,
        checkpoint_every_steps=checkpoint_every_steps,
    )
    if checkpoint_path is not None:
        out = Path(checkpoint_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "status": result.status,
                "model_config": primary["model_config"],
                "task": {key: primary[key] for key in ("task_id", "source_modality", "target_modality")},
                "metrics": result.to_dict(),
                "model_state_dict": first_model_state,
                "optimizer_state_dict": first_optimizer_state,
                "task_states": task_states,
                "completed_steps": result.completed_steps,
            },
            out,
        )
    if metrics_csv_path is not None:
        _write_metrics_csv(metrics_csv_path, result)
    if distributed_initialized:
        cleanup_process_group()
    return result


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
    return (tasks[0],) if tasks else ()


def _all_synthetic(batches: list[Any]) -> bool:
    return bool(batches) and all(bool(batch.metadata.get("synthetic")) for batch in batches)


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
    config: dict[str, Any],
    model_cfg: dict[str, Any],
    device: torch.device,
    dist_local_rank: int,
    resume_checkpoint: dict[str, Any] | None,
    resume_path: str | Path | None,
    metrics_jsonl_path: str | Path | None,
    precision: str,
    steps: int,
    start_step: int,
    gradient_accumulation_steps: int,
    eval_every_steps: int,
    checkpoint_every_steps: int,
    checkpoint_dir: Path | None,
    objective_weight: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model_config = _model_config_for_task(task, model_cfg, config)
    model = NeuralStateSpaceTranslator(**model_config)
    model.to(device)
    training_cfg = dict(config.get("training", {}))
    compile_enabled = bool(config.get("compile", training_cfg.get("compile", False)))
    if compile_enabled and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    selection_split = "val" if task.x_val is not None and task.y_val is not None else "train"
    x_train = torch.as_tensor(task.x_train, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(task.y_train, dtype=torch.float32, device=device)
    x_val = torch.as_tensor(task.x_val if task.x_val is not None else task.x_train, dtype=torch.float32, device=device)
    y_val = torch.as_tensor(task.y_val if task.y_val is not None else task.y_train, dtype=torch.float32, device=device)
    x_test = torch.as_tensor(task.x_test, dtype=torch.float32, device=device)
    y_test = torch.as_tensor(task.y_test, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", config.get("learning_rate", 2e-3))))
    _load_task_resume(model, optimizer, task.task_id, resume_checkpoint)
    model = wrap_ddp_if_initialized(model, local_rank=dist_local_rank)
    batch_size = max(1, int(config.get("batch_size", x_train.shape[0])))
    eval_batch_size = max(1, int(config.get("eval_batch_size", training_cfg.get("eval_batch_size", batch_size))))

    with torch.no_grad():
        initial_loss = _batched_loss(model, task, x_train, y_train, precision=precision, batch_size=eval_batch_size)
    _append_task_metric(
        metrics_jsonl_path,
        {
            "task_id": task.task_id,
            "step": start_step,
            "loss": initial_loss,
            "phase": "initial",
            "source_modality": task.source_modality,
            "target_modality": task.target_modality,
        },
    )

    model.train()
    final_loss = initial_loss
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for micro_step in range(gradient_accumulation_steps):
            offset = (step * gradient_accumulation_steps + micro_step) * batch_size
            start = offset % x_train.shape[0]
            end = min(start + batch_size, x_train.shape[0])
            if end <= start:
                start, end = 0, min(batch_size, x_train.shape[0])
            xb = x_train[start:end]
            yb = y_train[start:end]
            loss = (_mse_loss(_predict(model, task, xb, precision=precision), yb) * objective_weight) / gradient_accumulation_steps
            loss.backward()
            accumulated_loss += float(loss.detach())
        optimizer.step()
        final_loss = accumulated_loss
        completed_step = start_step + step + 1
        if eval_every_steps and completed_step % eval_every_steps == 0:
            val_snapshot = _evaluate_task(
                model,
                task,
                x_val,
                y_val,
                precision=precision,
                prefix="val",
                batch_size=eval_batch_size,
            )
            _append_task_metric(metrics_jsonl_path, {"task_id": task.task_id, "step": completed_step, "phase": "eval", "selection_split": selection_split, **val_snapshot})
        if checkpoint_every_steps and checkpoint_dir is not None and completed_step % checkpoint_every_steps == 0:
            _save_task_checkpoint(
                checkpoint_dir / f"checkpoint_{task.task_id}_step_{completed_step}.pt",
                status="periodic_prepared_training",
                task_result={"task_id": task.task_id, "completed_steps": completed_step, "model_config": model_config},
                model_state=unwrap_model(model).state_dict(),
                optimizer_state=optimizer.state_dict(),
            )

    val_metrics = _evaluate_task(model, task, x_val, y_val, precision=precision, prefix="val", batch_size=eval_batch_size)
    test_metrics = _evaluate_task(model, task, x_test, y_test, precision=precision, prefix="test", batch_size=eval_batch_size)
    model_state = unwrap_model(model).state_dict()
    optimizer_state = optimizer.state_dict()
    result = {
        "status": "completed",
        "task_id": task.task_id,
        "source_modality": task.source_modality,
        "target_modality": task.target_modality,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "steps": steps,
        "start_step": start_step,
        "completed_steps": start_step + steps,
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "objective_weight": objective_weight,
        "compile": compile_enabled,
        "selection_split": selection_split,
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
    _append_task_metric(metrics_jsonl_path, {"task_id": task.task_id, "step": start_step + steps, "phase": "final", **val_metrics, **test_metrics})
    return result, model_state, optimizer_state


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


def _model_config_for_task(task: Any, model_cfg: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    training_cfg = dict(config.get("training", {}))
    adapter_mode = str(model_cfg.get("adapter_mode", training_cfg.get("adapter_mode", "disabled")))
    use_subject_embeddings = bool(model_cfg.get("use_subject_embeddings", False)) and adapter_mode in {"few_shot", "enabled", "subject"}
    return {
        "input_dims": {task.source_modality: task.x_train.shape[-1]},
        "output_dims": {task.target_modality: task.y_train.shape[-1]},
        "latent_dim": int(model_cfg.get("latent_dim", 64)),
        "n_layers": int(model_cfg.get("n_layers", 1)),
        "subject_adapter_dim": int(model_cfg.get("subject_adapter_dim", 0)),
        "projection_dim": int(model_cfg.get("projection_dim", 32)),
        "metadata_dim": int(model_cfg.get("metadata_dim", 0)),
        "geometry_dim": int(model_cfg.get("geometry_dim", 0)),
        "backbone": str(model_cfg.get("backbone", "ssm_fallback")),
        "encoder": str(model_cfg.get("encoder", "auto")),
        "n_heads": int(model_cfg.get("n_heads", 4)),
        "subject_vocab_size": int(model_cfg.get("subject_vocab_size", 0)),
        "use_subject_embeddings": use_subject_embeddings,
        "adapter_mode": adapter_mode,
        "gradient_checkpointing": bool(
            config.get("gradient_checkpointing", model_cfg.get("gradient_checkpointing", training_cfg.get("gradient_checkpointing", False)))
        ),
    }


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


def _aggregate_task_results(task_results: list[dict[str, Any]]) -> dict[str, float]:
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
