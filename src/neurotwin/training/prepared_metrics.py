from __future__ import annotations

import csv
import math
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
            task=_task_mode(task),
        )


def predict(model: nn.Module, task: Any, x: torch.Tensor, precision: str = "fp32") -> torch.Tensor:
    enabled = precision == "bf16" and x.device.type in {"cuda", "cpu"}
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=enabled):
        return model(
            {task.source_modality: x},
            target_modality=task.target_modality,
            task=_task_mode(task),
        )


def _task_mode(task: Any) -> str:
    return "forecast" if task.task_id == "future_state_forecasting" else "reconstruction"


def mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    metric_mask: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    pred = prediction.float()
    tgt = target.float()
    if metric_mask is None:
        return nn.functional.mse_loss(pred, tgt)
    mask = _as_bool_mask(metric_mask, pred)
    squared = (pred - tgt).square()
    if not bool(mask.any()):
        raise ValueError("metric_mask selected zero elements for mse_loss")
    return squared[mask].mean()


def gaussian_nll_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    *,
    minimum_scale: float = 1e-6,
    metric_mask: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    if prediction.shape != target.shape or scale.shape != target.shape:
        raise ValueError("prediction, target, and scale must have identical shapes")
    safe_scale = scale.float().clamp_min(float(minimum_scale))
    standardized = (target.float() - prediction.float()) / safe_scale
    nll = 0.5 * standardized.square() + safe_scale.log() + 0.5 * math.log(2.0 * math.pi)
    if metric_mask is None:
        return nll.mean()
    mask = _as_bool_mask(metric_mask, nll)
    if not bool(mask.any()):
        raise ValueError("metric_mask selected zero elements for gaussian_nll_loss")
    return nll[mask].mean()


def _as_bool_mask(metric_mask: torch.Tensor | np.ndarray, reference: torch.Tensor) -> torch.Tensor:
    mask = torch.as_tensor(metric_mask, device=reference.device, dtype=torch.bool)
    if mask.shape != reference.shape:
        raise ValueError(f"metric_mask shape {tuple(mask.shape)} must match tensor shape {tuple(reference.shape)}")
    return mask


def _split_metric_mask(task: Any, prefix: str) -> np.ndarray | None:
    if prefix == "test":
        return getattr(task, "metric_mask", None)
    if prefix == "val":
        val_mask = getattr(task, "val_metric_mask", None)
        if val_mask is not None:
            return val_mask
        if getattr(task, "x_val", None) is None:
            return getattr(task, "train_metric_mask", None)
        return None
    if prefix == "train":
        return getattr(task, "train_metric_mask", None)
    return None


def batch_metric_mask(task: Any, start: int, end: int) -> np.ndarray | None:
    mask = getattr(task, "train_metric_mask", None)
    if mask is None:
        return None
    return np.asarray(mask)[start:end]


def indexed_metric_mask(task: Any, indices: Sequence[int] | torch.Tensor) -> np.ndarray | None:
    mask = getattr(task, "train_metric_mask", None)
    if mask is None:
        return None
    index_list = [int(v) for v in indices]
    return np.asarray(mask)[index_list]


def probabilistic_output(
    model: nn.Module,
    task: Any,
    x: torch.Tensor,
    precision: str = "fp32",
) -> dict[str, torch.Tensor]:
    enabled = precision == "bf16" and x.device.type in {"cuda", "cpu"}
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=enabled):
        output = model(
            {task.source_modality: x},
            target_modality=task.target_modality,
            task=_task_mode(task),
            return_output=True,
        )
    if not isinstance(output, dict):
        raise TypeError("probabilistic model must return an output mapping")
    if "prediction" not in output or "uncertainty" not in output:
        raise ValueError("probabilistic model output requires prediction and uncertainty tensors")
    return output


def probabilistic_loss(
    model: nn.Module,
    task: Any,
    x: torch.Tensor,
    target: torch.Tensor,
    precision: str = "fp32",
    metric_mask: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    output = probabilistic_output(model, task, x, precision=precision)
    return gaussian_nll_loss(
        output["prediction"],
        target,
        output["uncertainty"],
        metric_mask=metric_mask,
    )


def batched_objective_loss(
    model: nn.Module,
    task: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    precision: str,
    batch_size: int,
    *,
    probabilistic: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            batch_mask = batch_metric_mask(task, start, end)
            loss = (
                probabilistic_loss(
                    model,
                    task,
                    x[start:end],
                    y[start:end],
                    precision=precision,
                    metric_mask=batch_mask,
                )
                if probabilistic
                else mse_loss(
                    predict(model, task, x[start:end], precision=precision),
                    y[start:end],
                    metric_mask=batch_mask,
                )
            )
            batch_samples = end - start
            total_loss += float(loss) * batch_samples
            total_samples += batch_samples
    model.train()
    return total_loss / max(1, total_samples)


def batched_loss(
    model: nn.Module,
    task: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    precision: str,
    batch_size: int,
) -> float:
    return batched_objective_loss(
        model,
        task,
        x,
        y,
        precision,
        batch_size,
        probabilistic=False,
    )


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
    metric_mask = _split_metric_mask(task, prefix)
    if metric_mask is not None:
        mask = np.asarray(metric_mask, dtype=bool)
        if mask.shape != y_true_np.shape:
            raise ValueError(
                f"{prefix} metric_mask shape {mask.shape} must match target shape {y_true_np.shape}"
            )
        y_true_metric = y_true_np[mask]
        y_pred_metric = y_pred_np[mask]
    else:
        y_true_metric = y_true_np
        y_pred_metric = y_pred_np
    eval_loss = mse(y_true_metric, y_pred_metric)
    metrics = {
        f"{prefix}_mse": eval_loss,
        f"{prefix}_mae": mae(y_true_metric, y_pred_metric),
        f"{prefix}_pearsonr": pearsonr(y_true_metric, y_pred_metric),
        f"{prefix}_spearmanr": spearmanr(y_true_metric, y_pred_metric),
        f"{prefix}_r2": r2_score(y_true_metric, y_pred_metric),
    }
    if task.target_modality in {"eeg", "meg"} or task.source_modality in {"eeg", "meg"}:
        metrics[f"{prefix}_bandpower_error"] = bandpower_error(y_true_np, y_pred_np)
    if task.target_modality == "fmri":
        metrics[f"{prefix}_regionwise_pearsonr"] = regionwise_pearsonr(y_true_np, y_pred_np)
    if uncertainty_np is not None and uncertainty_np.shape == y_pred_np.shape:
        abs_error = np.abs(np.asarray(y_true_metric, dtype=np.float64) - np.asarray(y_pred_metric, dtype=np.float64)).ravel()
        if metric_mask is not None:
            uncertainty = np.asarray(uncertainty_np, dtype=np.float64)[np.asarray(metric_mask, dtype=bool)].ravel()
        else:
            uncertainty = np.asarray(uncertainty_np, dtype=np.float64).ravel()
        finite = np.isfinite(abs_error) & np.isfinite(uncertainty)
        if finite.any():
            metrics[f"{prefix}_mean_uncertainty"] = float(np.mean(uncertainty[finite]))
            if int(np.sum(finite)) >= 2:
                metrics[f"{prefix}_error_uncertainty_correlation"] = pearsonr(abs_error[finite], uncertainty[finite])
    if metric_mask is not None:
        metrics[f"{prefix}_masked_mse"] = eval_loss
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
