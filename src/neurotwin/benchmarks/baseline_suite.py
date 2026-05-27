from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from neurotwin.eval.metrics import (
    bandpower_error,
    bootstrap_ci,
    mae,
    mse,
    pearsonr,
    r2_score,
    rank_models,
    regionwise_pearsonr,
    spectral_error,
    spearmanr,
)
from neurotwin.models.baselines import NumpyRidgeBaseline, TorchMLPBaseline, TorchTCNBaseline
from neurotwin.models.torch_models import NeuralStateSpaceTranslator, TinySSMBaseline, TinyTransformerBaseline


@dataclass(frozen=True)
class SupervisedWindowTask:
    task_id: str
    source_modality: str
    target_modality: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    metric_mask: np.ndarray | None = None
    x_val: np.ndarray | None = None
    y_val: np.ndarray | None = None
    val_metric_mask: np.ndarray | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BaselineFailure:
    model_id: str
    task_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"model_id": self.model_id, "task_id": self.task_id, "reason": self.reason}


def run_synthetic_baseline_suite(seed: int = 0, train_steps: int = 10) -> dict[str, object]:
    """Run tiny local baselines on paired synthetic windows.

    This is intentionally a plumbing benchmark. It validates that all local
    baselines see identical tensor shapes and metrics under one task surface.
    """

    data = _make_paired_windows(seed=seed)
    tasks = (
        _future_task(data),
        _masked_reconstruction_task(data, seed=seed),
        _cross_modal_task(data),
    )
    return run_supervised_window_tasks(
        tasks,
        seed=seed,
        train_steps=train_steps,
        scope_status="synthetic-only",
        scope_notes=(
            "Validates baseline/task plumbing only.",
            "Do not interpret these rankings as scientific evidence.",
            "Real benchmark claims require prepared public data, strict held-out splits, and bootstrap CIs.",
        ),
    )


def run_supervised_window_tasks(
    tasks: tuple[SupervisedWindowTask, ...],
    seed: int = 0,
    train_steps: int = 10,
    scope_status: str = "prepared-data",
    scope_notes: tuple[str, ...] = (),
) -> dict[str, object]:
    task_payloads = {}
    rank_accumulator: dict[str, list[int]] = {}
    all_failures: list[dict[str, str]] = []
    for task in tasks:
        task_result = _run_task_models(task, seed=seed, train_steps=train_steps)
        task_payloads[task.task_id] = task_result
        all_failures.extend(task_result.get("failures", []))
        for row in task_result["ranking"]:
            rank_accumulator.setdefault(str(row["model_id"]), []).append(int(row["rank"]))

    aggregate_rank = sorted(
        (
            {
                "model_id": model_id,
                "mean_rank": float(np.mean(ranks)),
                "tasks_ranked": len(ranks),
            }
            for model_id, ranks in rank_accumulator.items()
        ),
        key=lambda row: (float(row["mean_rank"]), str(row["model_id"])),
    )
    return {
        "scope": {
            "status": scope_status,
            "notes": list(scope_notes),
        },
        "tasks": task_payloads,
        "aggregate": {
            "selection_metric": "mse",
            "higher_is_better": False,
            "aggregate_rank": aggregate_rank,
        },
        "baseline_catalog": _baseline_catalog(tasks),
        "baseline_failures": all_failures,
    }


def _make_paired_windows(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_samples = 30
    n_time = 8
    latent_dim = 4
    latent = rng.normal(size=(n_samples, n_time, latent_dim)).astype(np.float32)
    latent[:, 1:] += 0.55 * latent[:, :-1]
    eeg_weights = rng.normal(size=(latent_dim, 6)).astype(np.float32)
    fmri_weights = rng.normal(size=(latent_dim, 5)).astype(np.float32)
    eeg = latent @ eeg_weights + 0.04 * rng.normal(size=(n_samples, n_time, 6)).astype(np.float32)
    fmri = latent @ fmri_weights + 0.04 * rng.normal(size=(n_samples, n_time, 5)).astype(np.float32)
    return {"eeg": eeg.astype(np.float32), "fmri": fmri.astype(np.float32)}


def _split(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_idx = max(1, int(round(x.shape[0] * 0.7)))
    return x[:split_idx], y[:split_idx], x[split_idx:], y[split_idx:]


def _future_task(data: dict[str, np.ndarray]) -> SupervisedWindowTask:
    x_train, y_train, x_test, y_test = _split(data["eeg"][:, :-1], data["eeg"][:, 1:])
    return SupervisedWindowTask(
        task_id="future_state_forecasting",
        source_modality="eeg",
        target_modality="eeg",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        notes=("predict next EEG window from previous neural state",),
    )


def _masked_reconstruction_task(data: dict[str, np.ndarray], seed: int) -> SupervisedWindowTask:
    rng = np.random.default_rng(seed + 17)
    clean = data["eeg"]
    mask = rng.random(clean.shape) < 0.2
    masked = clean.copy()
    masked[mask] = 0.0
    x_train, y_train, x_test, y_test = _split(masked, clean)
    _, _, mask_test, _ = _split(mask.astype(bool), mask.astype(bool))
    return SupervisedWindowTask(
        task_id="masked_neural_reconstruction",
        source_modality="eeg",
        target_modality="eeg",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        metric_mask=mask_test,
        notes=("mask time/channel entries and reconstruct clean EEG",),
    )


def _cross_modal_task(data: dict[str, np.ndarray]) -> SupervisedWindowTask:
    x_train, y_train, x_test, y_test = _split(data["eeg"], data["fmri"])
    return SupervisedWindowTask(
        task_id="cross_modal_translation",
        source_modality="eeg",
        target_modality="fmri",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        notes=("paired synthetic EEG to fMRI-like latent proxy",),
    )


def _run_task_models(task: SupervisedWindowTask, seed: int, train_steps: int) -> dict[str, object]:
    runners: dict[str, Callable[[], np.ndarray]] = {
        "linear_ridge": lambda: _fit_ridge(task.x_train, task.y_train, task.x_test),
        "mlp": lambda: _fit_torch_sequence_model(
            lambda: TorchMLPBaseline(task.x_train.shape[-1], task.y_train.shape[-1], hidden_dim=24),
            task,
            seed=seed + 1,
            steps=train_steps,
        ),
        "tcn": lambda: _fit_torch_sequence_model(
            lambda: TorchTCNBaseline(task.x_train.shape[-1], task.y_train.shape[-1], hidden_dim=24),
            task,
            seed=seed + 2,
            steps=train_steps,
        ),
        "transformer": lambda: _fit_torch_sequence_model(
            lambda: TinyTransformerBaseline(
                task.x_train.shape[-1],
                task.y_train.shape[-1],
                latent_dim=24,
                n_heads=4,
                n_layers=1,
            ),
            task,
            seed=seed + 3,
            steps=train_steps,
        ),
        "ssm_fallback": lambda: _fit_torch_sequence_model(
            lambda: TinySSMBaseline(task.x_train.shape[-1], task.y_train.shape[-1], latent_dim=24, n_layers=1),
            task,
            seed=seed + 4,
            steps=train_steps,
        ),
        "neurotwin": lambda: _fit_neurotwin(task, seed=seed + 5, steps=train_steps),
    }
    predictions: dict[str, np.ndarray] = {}
    failures: list[BaselineFailure] = []
    for model_id, runner in runners.items():
        try:
            prediction = runner()
            _validate_prediction(task, model_id, prediction)
            predictions[model_id] = prediction
        except Exception as exc:  # noqa: BLE001 - benchmark failures are payload data.
            failures.append(BaselineFailure(model_id=model_id, task_id=task.task_id, reason=str(exc)))

    metrics_by_model: dict[str, dict[str, float]] = {}
    for model_id, prediction in predictions.items():
        try:
            model_metrics = _metrics(
                task.y_test,
                prediction,
                task.metric_mask,
                source_modality=task.source_modality,
                target_modality=task.target_modality,
                seed=seed,
            )
            _validate_metrics(model_id, model_metrics)
            metrics_by_model[model_id] = model_metrics
        except Exception as exc:  # noqa: BLE001 - benchmark failures are payload data.
            failures.append(BaselineFailure(model_id=model_id, task_id=task.task_id, reason=f"metric failure: {exc}"))
    ranking = [
        {
            "model_id": row.model_id,
            "metric": row.metric,
            "value": row.value,
            "rank": row.rank,
        }
        for row in rank_models(metrics_by_model, metric="mse", higher_is_better=False)
    ]
    return {
        "status": "completed",
        "source_modality": task.source_modality,
        "target_modality": task.target_modality,
        "metrics_by_model": metrics_by_model,
        "ranking": ranking,
        "failures": [failure.to_dict() for failure in failures],
        "notes": list(task.notes),
    }


def _fit_ridge(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    model = NumpyRidgeBaseline(alpha=1e-2)
    model.fit(_flatten_time(x_train), _flatten_time(y_train))
    pred = model.predict(_flatten_time(x_test))
    return pred.reshape(x_test.shape[0], x_test.shape[1], y_train.shape[-1])


def _fit_torch_sequence_model(
    factory: Callable[[], nn.Module],
    task: SupervisedWindowTask,
    seed: int,
    steps: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    model = factory()
    x_train = torch.as_tensor(task.x_train, dtype=torch.float32)
    y_train = torch.as_tensor(task.y_train, dtype=torch.float32)
    x_test = torch.as_tensor(task.x_test, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(x_test).detach().cpu().numpy()


def _fit_neurotwin(task: SupervisedWindowTask, seed: int, steps: int) -> np.ndarray:
    torch.manual_seed(seed)
    model = NeuralStateSpaceTranslator(
        input_dims={task.source_modality: task.x_train.shape[-1]},
        output_dims={task.target_modality: task.y_train.shape[-1]},
        latent_dim=24,
        n_layers=1,
        subject_adapter_dim=8,
    )
    x_train = torch.as_tensor(task.x_train, dtype=torch.float32)
    y_train = torch.as_tensor(task.y_train, dtype=torch.float32)
    x_test = torch.as_tensor(task.x_test, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_task(
            {task.source_modality: x_train},
            target_modality=task.target_modality,
            task="forecast" if task.task_id == "future_state_forecasting" else "reconstruction",
        )
        loss = loss_fn(output["prediction"], y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        output = model.forward_task(
            {task.source_modality: x_test},
            target_modality=task.target_modality,
            task="forecast" if task.task_id == "future_state_forecasting" else "reconstruction",
        )
    return output["prediction"].detach().cpu().numpy()


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_mask: np.ndarray | None,
    source_modality: str,
    target_modality: str,
    seed: int = 0,
) -> dict[str, float]:
    if metric_mask is not None:
        y_true_metric = y_true[metric_mask]
        y_pred_metric = y_pred[metric_mask]
    else:
        y_true_metric = y_true
        y_pred_metric = y_pred
    values = {
        "mse": mse(y_true_metric, y_pred_metric),
        "mae": mae(y_true_metric, y_pred_metric),
        "pearsonr": pearsonr(y_true_metric, y_pred_metric),
        "spearmanr": spearmanr(y_true_metric, y_pred_metric),
        "r2": r2_score(y_true_metric, y_pred_metric),
    }
    squared_error = (np.asarray(y_true_metric, dtype=float).ravel() - np.asarray(y_pred_metric, dtype=float).ravel()) ** 2
    absolute_error = np.abs(np.asarray(y_true_metric, dtype=float).ravel() - np.asarray(y_pred_metric, dtype=float).ravel())
    values["mse_ci_low"], values["mse_ci_high"] = bootstrap_ci(squared_error, seed=seed, n_boot=200)
    values["mae_ci_low"], values["mae_ci_high"] = bootstrap_ci(absolute_error, seed=seed + 1, n_boot=200)
    if source_modality in {"eeg", "meg"} or target_modality in {"eeg", "meg"}:
        values["spectral_error"] = spectral_error(y_true, y_pred)
        values["bandpower_error"] = bandpower_error(y_true, y_pred)
    if target_modality == "fmri":
        values["regionwise_pearsonr"] = regionwise_pearsonr(y_true, y_pred)
    if metric_mask is not None:
        values["masked_mse"] = values["mse"]
    return values


def _flatten_time(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])


def _validate_prediction(task: SupervisedWindowTask, model_id: str, prediction: np.ndarray) -> None:
    prediction = np.asarray(prediction)
    if prediction.shape != task.y_test.shape:
        raise ValueError(f"{model_id} prediction shape {prediction.shape} does not match target {task.y_test.shape}")
    if not np.isfinite(prediction).all():
        raise ValueError(f"{model_id} prediction contains NaN or Inf")


def _validate_metrics(model_id: str, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        if not isinstance(value, (int, float, np.floating)) or not np.isfinite(float(value)):
            raise ValueError(f"{model_id} metric {key} is not finite: {value}")


def _baseline_catalog(tasks: tuple[SupervisedWindowTask, ...]) -> list[dict[str, object]]:
    task_ids = {task.task_id for task in tasks}
    modalities = {task.source_modality for task in tasks} | {task.target_modality for task in tasks}
    catalog = [
        {"model_id": "linear_ridge", "status": "local_baseline", "notes": "Closed-form sanity baseline on identical prepared windows."},
        {"model_id": "mlp", "status": "local_baseline", "notes": "Per-timepoint neural-window baseline."},
        {"model_id": "tcn", "status": "local_baseline", "notes": "Local temporal convolution baseline."},
        {"model_id": "transformer", "status": "local_baseline", "notes": "Small local Transformer with shared splits."},
        {"model_id": "ssm_fallback", "status": "local_baseline", "notes": "GRU-based SSM fallback until Mamba is pinned."},
        {"model_id": "neurotwin", "status": "local_baseline", "notes": "Current NeuroTwin implementation under the same task API."},
        {
            "model_id": "tribe_style",
            "status": "approximation" if "cross_modal_translation" in task_ids and "fmri" in modalities else "unavailable",
            "notes": "Approximate stimulus/history-to-fMRI lane only when fMRI-aligned inputs exist; not an exact TRIBE v2 reproduction.",
        },
        {
            "model_id": "brainvista_style",
            "status": "approximation" if "future_state_forecasting" in task_ids and "fmri" in modalities else "unavailable",
            "notes": "Approximate autoregressive fMRI rollout lane; not an exact BrainVista reproduction.",
        },
        {
            "model_id": "brain_of_style",
            "status": "approximation" if "masked_neural_reconstruction" in task_ids and len(modalities) >= 2 else "unavailable",
            "notes": "Approximate multimodal masked reconstruction lane; not an exact Brain-OF reproduction.",
        },
        {
            "model_id": "brainomni_style",
            "status": "approximation" if modalities & {"eeg", "meg"} else "unavailable",
            "notes": "Approximate EEG/MEG tokenizer lane; not an exact BrainOmni reproduction.",
        },
        {
            "model_id": "braindecode_wrapper",
            "status": "unavailable",
            "notes": "Optional EEG wrapper slot; exact use requires installed Braindecode and compatible task protocols.",
        },
        {
            "model_id": "cebra_wrapper",
            "status": "unavailable",
            "notes": "Optional neural-behavior embedding wrapper slot; exact use requires installed CEBRA and aligned behavior data.",
        },
    ]
    return catalog
