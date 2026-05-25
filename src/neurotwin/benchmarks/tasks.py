from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurotwin.eval.metrics import mse, pearsonr


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    status: str
    metrics: dict[str, float]
    notes: tuple[str, ...] = ()


def run_future_forecasting_task(signal: np.ndarray, history: int = 8, horizon: int = 1) -> TaskResult:
    signal = np.asarray(signal, dtype=float)
    if signal.shape[0] <= history + horizon:
        return TaskResult("future_state_forecasting", "skipped", {}, ("not enough timepoints",))
    y_true = signal[history + horizon - 1 :]
    naive = signal[history - 1 : -horizon]
    return TaskResult(
        "future_state_forecasting",
        "completed",
        {"mse": mse(y_true, naive), "pearsonr": pearsonr(y_true, naive)},
        ("naive persistence baseline",),
    )


def run_masked_reconstruction_task(signal: np.ndarray, mask_fraction: float = 0.15, seed: int = 0) -> TaskResult:
    signal = np.asarray(signal, dtype=float)
    rng = np.random.default_rng(seed)
    mask = rng.random(signal.shape) < mask_fraction
    if not np.any(mask):
        mask.flat[0] = True
    baseline = np.broadcast_to(np.mean(signal, axis=0, keepdims=True), signal.shape)
    return TaskResult(
        "masked_neural_reconstruction",
        "completed",
        {"masked_mse": mse(signal[mask], baseline[mask])},
        ("channel/time mean baseline",),
    )


def run_cross_modal_translation_task(modalities: dict[str, np.ndarray], source: str, target: str) -> TaskResult:
    if source not in modalities or target not in modalities:
        return TaskResult("cross_modal_translation", "skipped", {}, ("missing paired modalities",))
    source_signal = np.asarray(modalities[source], dtype=float)
    target_signal = np.asarray(modalities[target], dtype=float)
    n = min(source_signal.shape[0], target_signal.shape[0])
    source_flat = source_signal[:n].reshape(n, -1)
    target_flat = target_signal[:n].reshape(n, -1)
    width = min(source_flat.shape[1], target_flat.shape[1])
    pred = source_flat[:, :width]
    truth = target_flat[:, :width]
    return TaskResult("cross_modal_translation", "completed", {"mse": mse(truth, pred), "pearsonr": pearsonr(truth, pred)})


def run_subject_adaptation_task(support: np.ndarray, query: np.ndarray) -> TaskResult:
    support = np.asarray(support, dtype=float)
    query = np.asarray(query, dtype=float)
    global_pred = np.zeros_like(query) + np.mean(query)
    adapted_pred = np.zeros_like(query) + np.mean(support)
    global_loss = mse(query, global_pred)
    adapted_loss = mse(query, adapted_pred)
    return TaskResult(
        "few_shot_subject_adaptation",
        "completed",
        {
            "global_mse": global_loss,
            "adapted_mse": adapted_loss,
            "adaptation_gain": global_loss - adapted_loss,
        },
        ("mean-adapter sanity baseline",),
    )


def run_dataset_site_generalization_task(
    source_signal: np.ndarray,
    target_signal: np.ndarray,
    source_name: str = "source",
    target_name: str = "target",
) -> TaskResult:
    source_signal = np.asarray(source_signal, dtype=float)
    target_signal = np.asarray(target_signal, dtype=float)
    pred = np.zeros_like(target_signal) + np.mean(source_signal, axis=0, keepdims=True)
    return TaskResult(
        "dataset_site_generalization",
        "completed",
        {
            "generalization_mse": mse(target_signal, pred),
            "generalization_pearsonr": pearsonr(target_signal, pred),
        },
        (f"source={source_name} target={target_name}",),
    )
