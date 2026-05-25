from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from neurotwin.benchmarks.prepared_suite import PreparedSuiteConfig, build_prepared_window_tasks
from neurotwin.data.event_io import event_manifest_summary, load_event_batches
from neurotwin.data.manifest_io import load_split_manifest
from neurotwin.eval.metrics import mae, mse, pearsonr, r2_score
from neurotwin.models.torch_models import NeuralStateSpaceTranslator
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_prepared_training(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    resume_path: str | Path | None = None,
    metrics_csv_path: str | Path | None = None,
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
    selected = _select_task(tasks, str(config.get("task", "future_state_forecasting")))
    if selected is None:
        available = ", ".join(task.task_id for task in tasks) or "none"
        raise ValueError(f"No runnable prepared training task matched config task. Available: {available}")

    model_cfg = dict(config.get("model", {}))
    model = NeuralStateSpaceTranslator(
        input_dims={selected.source_modality: selected.x_train.shape[-1]},
        output_dims={selected.target_modality: selected.y_train.shape[-1]},
        latent_dim=int(model_cfg.get("latent_dim", 64)),
        n_layers=int(model_cfg.get("n_layers", 1)),
        subject_adapter_dim=int(model_cfg.get("subject_adapter_dim", 0)),
        projection_dim=int(model_cfg.get("projection_dim", 32)),
    )
    device = torch.device(f"cuda:{dist_info.local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x_train = torch.as_tensor(selected.x_train, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(selected.y_train, dtype=torch.float32, device=device)
    x_test = torch.as_tensor(selected.x_test, dtype=torch.float32, device=device)
    y_test = torch.as_tensor(selected.y_test, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", config.get("learning_rate", 2e-3))))
    loss_fn = nn.MSELoss()
    batch_size = max(1, int(config.get("batch_size", x_train.shape[0])))
    steps = int(config.get("steps", 24))
    grad_accum = max(1, int(config.get("gradient_accumulation_steps", dict(config.get("training", {})).get("gradient_accumulation_steps", 1))))
    precision = str(config.get("precision", dict(config.get("training", {})).get("precision", "fp32"))).lower()
    start_step = 0
    if resume_path is not None:
        checkpoint = torch.load(Path(resume_path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint.get("completed_steps", checkpoint.get("metrics", {}).get("completed_steps", 0)))
    model = wrap_ddp_if_initialized(model, local_rank=dist_info.local_rank)

    with torch.no_grad():
        initial_loss = float(loss_fn(_predict(model, selected, x_train, precision=precision), y_train))

    model.train()
    final_loss = initial_loss
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for micro_step in range(grad_accum):
            offset = (step * grad_accum + micro_step) * batch_size
            start = offset % x_train.shape[0]
            end = min(start + batch_size, x_train.shape[0])
            if end <= start:
                start, end = 0, min(batch_size, x_train.shape[0])
            xb = x_train[start:end]
            yb = y_train[start:end]
            loss = loss_fn(_predict(model, selected, xb, precision=precision), yb) / grad_accum
            loss.backward()
            accumulated_loss += float(loss.detach())
        optimizer.step()
        final_loss = accumulated_loss

    model.eval()
    with torch.no_grad():
        pred = _predict(model, selected, x_test, precision=precision)
        eval_loss = float(loss_fn(pred, y_test))
    y_true_np = y_test.detach().cpu().numpy()
    y_pred_np = pred.detach().cpu().numpy()
    summary = event_manifest_summary(event_manifest)
    result = PreparedTrainingResult(
        status="completed_prepared_training",
        task_id=selected.task_id,
        source_modality=selected.source_modality,
        target_modality=selected.target_modality,
        initial_loss=initial_loss,
        final_loss=final_loss,
        eval_mse=eval_loss,
        eval_mae=mae(y_true_np, y_pred_np),
        eval_pearsonr=pearsonr(y_true_np, y_pred_np),
        eval_r2=r2_score(y_true_np, y_pred_np),
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
        train_samples=int(x_train.shape[0]),
        test_samples=int(x_test.shape[0]),
        synthetic_only=summary.get("schema") == "neurotwin.event_manifest.v1" and _all_synthetic(batches),
        skipped_tasks=tuple(skipped),
        event_summary=summary,
    )
    if checkpoint_path is not None:
        out = Path(checkpoint_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "status": result.status,
                "model_config": {
                    "input_dims": {selected.source_modality: selected.x_train.shape[-1]},
                    "output_dims": {selected.target_modality: selected.y_train.shape[-1]},
                    "latent_dim": int(model_cfg.get("latent_dim", 64)),
                    "n_layers": int(model_cfg.get("n_layers", 1)),
                    "subject_adapter_dim": int(model_cfg.get("subject_adapter_dim", 0)),
                    "projection_dim": int(model_cfg.get("projection_dim", 32)),
                },
                "task": {
                    "task_id": selected.task_id,
                    "source_modality": selected.source_modality,
                    "target_modality": selected.target_modality,
                },
                "metrics": result.to_dict(),
                "model_state_dict": unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
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


def _select_task(tasks: tuple[Any, ...], requested: str) -> Any | None:
    if requested in {"neural_translation_v1", "translation_smoke", "prepared"}:
        return tasks[0] if tasks else None
    for task in tasks:
        if task.task_id == requested:
            return task
    return tasks[0] if tasks else None


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
