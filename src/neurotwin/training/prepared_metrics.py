from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from neurotwin.runtime.distributed import unwrap_model
from neurotwin.scoring.metrics import bandpower_error, mae, mse, pearsonr, r2_score, regionwise_pearsonr, spearmanr
from neurotwin.training.prepared_types import PreparedTrainingResult


def predict_output(model: nn.Module, task: Any, x: torch.Tensor, precision: str = "fp32") -> dict[str, torch.Tensor]:
    enabled = precision == "bf16" and x.device.type in {"cuda", "cpu"}
    task_model = unwrap_model(model)
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=enabled):
        return task_model.forward_task(
            {task.source_modality: x},
            target_modality=task.target_modality,
            task="forecast" if task.task_id == "future_state_forecasting" else "reconstruction",
        )


def predict(model: nn.Module, task: Any, x: torch.Tensor, precision: str = "fp32") -> torch.Tensor:
    output = predict_output(model, task, x, precision=precision)
    return output["prediction"]


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(prediction.float(), target.float())


def batched_loss(
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
            pred = predict(model, task, x[start:end], precision=precision)
            batch_samples = end - start
            total_loss += float(mse_loss(pred, y[start:end])) * batch_samples
            total_samples += batch_samples
    model.train()
    return total_loss / max(1, total_samples)


def predict_numpy_batches(
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
            pred = predict(model, task, x[start:end], precision=precision)
            predictions.append(pred.detach().float().cpu().numpy())
    model.train()
    return np.concatenate(predictions, axis=0) if predictions else np.empty((0,), dtype=np.float32)


def predict_numpy_outputs(
    model: nn.Module,
    task: Any,
    x: torch.Tensor,
    precision: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    predictions: list[np.ndarray] = []
    uncertainties: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            output = predict_output(model, task, x[start:end], precision=precision)
            predictions.append(output["prediction"].detach().float().cpu().numpy())
            uncertainty = output.get("uncertainty")
            if uncertainty is not None:
                uncertainties.append(uncertainty.detach().float().cpu().numpy())
    model.train()
    prediction = np.concatenate(predictions, axis=0) if predictions else np.empty((0,), dtype=np.float32)
    uncertainty = np.concatenate(uncertainties, axis=0) if uncertainties else None
    return prediction, uncertainty


def evaluate_task(
    model: nn.Module,
    task: Any,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    precision: str,
    prefix: str,
    batch_size: int,
) -> dict[str, float]:
    y_true_np = y_test.detach().cpu().numpy()
    y_pred_np, uncertainty_np = predict_numpy_outputs(model, task, x_test, precision=precision, batch_size=batch_size)
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
    if uncertainty_np is not None and uncertainty_np.shape == y_pred_np.shape:
        abs_error = np.abs(np.asarray(y_true_np, dtype=np.float64) - np.asarray(y_pred_np, dtype=np.float64)).ravel()
        uncertainty = np.asarray(uncertainty_np, dtype=np.float64).ravel()
        finite = np.isfinite(abs_error) & np.isfinite(uncertainty)
        if finite.any():
            metrics[f"{prefix}_mean_uncertainty"] = float(np.mean(uncertainty[finite]))
            metrics[f"{prefix}_error_uncertainty_correlation"] = pearsonr(abs_error[finite], uncertainty[finite])
    return metrics


def aggregate_task_results(task_results: Sequence[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "initial_loss",
        "final_loss",
        "eval_mse",
        "eval_mae",
        "eval_pearsonr",
        "eval_r2",
        "eval_spearmanr",
        "best_val_mse",
        "final_val_mse",
        "test_mse",
        "test_mae",
        "test_pearsonr",
        "test_r2",
        "test_spearmanr",
    )
    aggregate: dict[str, float] = {}
    for key in keys:
        values = [
            float(result[key])
            for result in task_results
            if result.get(key) is not None and np.isfinite(float(result[key]))
        ]
        aggregate[key] = float(np.mean(values)) if values else 0.0
    return aggregate


def write_metrics_csv(path: str | Path, result: PreparedTrainingResult) -> Path:
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
                "final_val_mse",
                "best_step",
                "best_checkpoint_path",
                "final_checkpoint_path",
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
