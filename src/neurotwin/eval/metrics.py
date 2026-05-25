from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankingRow:
    model_id: str
    metric: str
    value: float
    rank: int


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if np.isclose(denom, 0.0):
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same flattened shape")
    if y_true.size < 2:
        raise ValueError("pearsonr requires at least two samples")
    if np.isclose(np.std(y_true), 0.0) or np.isclose(np.std(y_pred), 0.0):
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spectral_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_fft = np.abs(np.fft.rfft(np.asarray(y_true, dtype=float), axis=0))
    y_pred_fft = np.abs(np.fft.rfft(np.asarray(y_pred, dtype=float), axis=0))
    return mse(y_true_fft, y_pred_fft)


def retrieval_accuracy(query: np.ndarray, candidates: np.ndarray) -> float:
    query = np.asarray(query, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    sims = query @ candidates.T
    pred = np.argmax(sims, axis=1)
    truth = np.arange(query.shape[0])
    return float(np.mean(pred == truth))


def bootstrap_ci(
    values: np.ndarray,
    seed: int = 0,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("bootstrap_ci requires at least one value")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True)
    means = np.mean(samples, axis=1)
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return low, high


def rank_models(
    model_metrics: dict[str, dict[str, float]],
    metric: str,
    higher_is_better: bool,
) -> list[RankingRow]:
    rows = []
    for model_id, metrics in model_metrics.items():
        if metric not in metrics:
            raise ValueError(f"Model {model_id!r} is missing metric {metric!r}")
        rows.append((model_id, float(metrics[metric])))
    rows.sort(key=lambda item: item[1], reverse=higher_is_better)
    return [
        RankingRow(model_id=model_id, metric=metric, value=value, rank=idx + 1)
        for idx, (model_id, value) in enumerate(rows)
    ]
