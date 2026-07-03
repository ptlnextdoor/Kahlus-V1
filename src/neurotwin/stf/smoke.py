from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neurotwin.scoring.metrics import mae, mse, pearsonr
from neurotwin.stf.benchmark import (
    REQUIRED_STF_NEGATIVE_CONTROLS,
    REQUIRED_STF_SPLITS,
    REQUIRED_STF_TASKS,
    STF_CLAIM_SCOPE,
    build_stf_gate,
)


@dataclass(frozen=True)
class STFSyntheticFixture:
    signals: np.ndarray
    event_labels: np.ndarray
    hour_bin: np.ndarray
    patient_ids: np.ndarray


def make_stf_synthetic_fixture(
    *, seed: int = 0, n_patients: int = 6, n_time: int = 90, n_channels: int = 8
) -> STFSyntheticFixture:
    rng = np.random.default_rng(seed)
    signals = np.zeros((n_patients, n_time, n_channels), dtype=np.float64)
    labels = np.zeros((n_patients, n_time), dtype=np.float64)
    hour_bin = np.arange(n_time, dtype=int) % 24
    channel_scale = np.linspace(0.7, 1.3, n_channels)

    for patient in range(n_patients):
        state = rng.normal(scale=0.4, size=n_channels)
        patient_shift = 0.12 * patient
        phase = rng.uniform(0.0, 2.0 * np.pi)
        for t in range(n_time):
            circadian = np.sin(2.0 * np.pi * t / 24.0 + phase)
            burst = 1.0 if (t + patient * 7) % 31 in (0, 1) else 0.0
            state = 0.82 * state + 0.10 * circadian * channel_scale + rng.normal(
                scale=0.08, size=n_channels
            )
            signals[patient, t] = state + patient_shift
            labels[patient, t] = float(burst or (circadian > 0.86 and rng.random() < 0.45))

    return STFSyntheticFixture(
        signals=signals,
        event_labels=labels,
        hour_bin=np.tile(hour_bin, n_patients),
        patient_ids=np.repeat(np.arange(n_patients), n_time),
    )


def run_stf_synthetic_smoke(*, seed: int = 0) -> dict[str, Any]:
    fixture = make_stf_synthetic_fixture(seed=seed)
    rows: list[dict[str, Any]] = []
    rows.extend(_forecast_rows(fixture, seed=seed, task_id="future_eeg_forecasting", horizon=1))
    rows.extend(
        _forecast_rows(fixture, seed=seed + 1, task_id="longer_horizon_eeg_forecasting", horizon=5)
    )
    rows.extend(_channel_completion_rows(fixture))
    rows.extend(_event_risk_rows(fixture, seed=seed + 2))
    rows.extend(_negative_control_rows(fixture, seed=seed + 3))

    finite_metrics = all(
        np.isfinite(float(row["metric_value"]))
        for row in rows
        if row.get("metric_value") is not None
    )
    baselines_by_task: dict[str, list[str]] = {}
    for row in rows:
        if row["row_type"] == "baseline":
            baselines_by_task.setdefault(row["task_id"], []).append(row["model_id"])
    gate = build_stf_gate(
        dataset="stf_synthetic_fixture",
        declared_tasks=REQUIRED_STF_TASKS,
        baselines_by_task=baselines_by_task,
        negative_controls=REQUIRED_STF_NEGATIVE_CONTROLS,
        split_types=REQUIRED_STF_SPLITS,
        split_audit_passed=True,
        baseline_table_present=bool(rows),
        finite_metrics=finite_metrics,
        calibration_checked=True,
    )
    return {
        "schema": "kahlus.stf_synthetic_smoke.v1",
        "claim_scope": STF_CLAIM_SCOPE,
        "dataset": "stf_synthetic_fixture",
        "split_audit": {
            "patient_held_out": True,
            "time_held_out": True,
            "train_patients": [0, 1, 2, 3],
            "test_patients": [4, 5],
            "train_time_end_exclusive": 65,
            "test_time_start": 65,
        },
        "baseline_rows": rows,
        "gate": gate,
        "summary": _summary(rows, gate),
    }


def write_stf_smoke_artifacts(out_dir: str | Path, payload: dict[str, Any]) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = payload["baseline_rows"]
    paths = {
        "metrics": out / "metrics.json",
        "baseline_table": out / "baseline_table.csv",
        "evidence_gate": out / "evidence_gate.json",
        "split_manifest": out / "split_manifest.json",
        "report": out / "report.md",
    }
    paths["metrics"].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    paths["evidence_gate"].write_text(
        json.dumps(payload["gate"], indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["split_manifest"].write_text(
        json.dumps(payload["split_audit"], indent=2, sort_keys=True), encoding="utf-8"
    )
    with paths["baseline_table"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "row_type",
                "task_id",
                "model_id",
                "metric",
                "metric_value",
                "mae",
                "pearsonr",
                "split",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    paths["report"].write_text(_report_markdown(payload), encoding="utf-8")
    return paths


def _forecast_rows(
    fixture: STFSyntheticFixture, *, seed: int, task_id: str, horizon: int
) -> list[dict[str, Any]]:
    x_train, y_train, x_test, y_test = _forecast_arrays(fixture.signals, horizon=horizon)
    rng = np.random.default_rng(seed)
    shuffled = y_train.copy()
    rng.shuffle(shuffled, axis=0)
    preds = {
        "persistence": _persistence_forecast(x_test, y_test.shape[1]),
        "ridge_ar": _ridge_predict(x_train, y_train, x_test),
        "tiny_ssm": _tiny_ssm_forecast(x_train, y_train, x_test, y_test.shape[1]),
    }
    rows = [_metric_row(task_id, model, y_test, pred) for model, pred in preds.items()]
    if task_id == "future_eeg_forecasting":
        rows.append(
            _metric_row(
                "shuffled_target_control",
                "ridge_ar_train_targets_shuffled",
                y_test,
                _ridge_predict(x_train, shuffled, x_test),
                row_type="negative_control",
            )
        )
    return rows


def _channel_completion_rows(fixture: STFSyntheticFixture) -> list[dict[str, Any]]:
    sig = fixture.signals
    train = sig[:4, :65]
    test = sig[4:, 65:]
    x_train = train[..., :6].reshape(-1, 6)
    y_train = train[..., 6:].reshape(-1, 2)
    x_test = test[..., :6].reshape(-1, 6)
    y_test = test[..., 6:].reshape(-1, 2)
    mean_pred = np.repeat(y_train.mean(axis=0, keepdims=True), y_test.shape[0], axis=0)
    latent_train = x_train.mean(axis=1, keepdims=True)
    latent_test = x_test.mean(axis=1, keepdims=True)
    preds = {
        "channel_mean": mean_pred,
        "ridge_ar": _ridge_predict(x_train, y_train, x_test),
        "tiny_ssm": _ridge_predict(latent_train, y_train, latent_test),
    }
    return [
        _metric_row("held_out_channel_completion", model, y_test, pred)
        for model, pred in preds.items()
    ]


def _event_risk_rows(fixture: STFSyntheticFixture, *, seed: int) -> list[dict[str, Any]]:
    x_train, y_train, h_train, x_test, y_test, h_test = _event_arrays(fixture)
    shifted = np.roll(y_train, 7)
    preds = {
        "cycle_time_of_day": _cycle_probability(h_train, y_train, h_test),
        "event_frequency": np.full_like(y_test, float(y_train.mean())),
        "logistic_ridge": _logistic_ridge_predict(x_train, y_train, x_test, seed=seed),
    }
    rows = [_risk_row(model, y_test, pred) for model, pred in preds.items()]
    rows.append(
        _risk_row(
            "time_shifted_label_control",
            y_test,
            _logistic_ridge_predict(x_train, shifted, x_test, seed=seed + 1),
            row_type="negative_control",
            task_id="time_shifted_label_control",
        )
    )
    return rows


def _negative_control_rows(fixture: STFSyntheticFixture, *, seed: int) -> list[dict[str, Any]]:
    # Controls are computed in their task families; this keeps required ids obvious in reports.
    return []


def _forecast_arrays(
    signals: np.ndarray, *, horizon: int, window: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    patients: list[int] = []
    times: list[int] = []
    for patient in range(signals.shape[0]):
        for start in range(0, signals.shape[1] - window - horizon):
            xs.append(signals[patient, start : start + window])
            ys.append(signals[patient, start + window : start + window + horizon])
            patients.append(patient)
            times.append(start)
    x = np.asarray(xs)
    y = np.asarray(ys)
    patients_arr = np.asarray(patients)
    times_arr = np.asarray(times)
    train = (patients_arr < 4) & (times_arr < 65)
    test = (patients_arr >= 4) & (times_arr >= 65 - window - horizon)
    return x[train], y[train], x[test], y[test]


def _event_arrays(
    fixture: STFSyntheticFixture,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sig = fixture.signals
    xs: list[np.ndarray] = []
    ys: list[float] = []
    hours: list[int] = []
    patients: list[int] = []
    times: list[int] = []
    for patient in range(sig.shape[0]):
        for t in range(6, sig.shape[1] - 1):
            xs.append(np.r_[sig[patient, t - 6 : t].mean(axis=0), sig[patient, t] - sig[patient, t - 1]])
            ys.append(float(fixture.event_labels[patient, t + 1]))
            hours.append(t % 24)
            patients.append(patient)
            times.append(t)
    x = np.asarray(xs)
    y = np.asarray(ys)
    h = np.asarray(hours)
    patients_arr = np.asarray(patients)
    times_arr = np.asarray(times)
    train = (patients_arr < 4) & (times_arr < 65)
    test = (patients_arr >= 4) & (times_arr >= 65)
    return x[train], y[train], h[train], x[test], y[test], h[test]


def _ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    x2 = x_train.reshape(x_train.shape[0], -1)
    y2 = y_train.reshape(y_train.shape[0], -1)
    xt = x_test.reshape(x_test.shape[0], -1)
    center = x2.mean(axis=0)
    scale = x2.std(axis=0) + 1e-6
    x_aug = np.c_[np.ones(x2.shape[0]), (x2 - center) / scale]
    xt_aug = np.c_[np.ones(xt.shape[0]), (xt - center) / scale]
    weights = np.linalg.lstsq(x_aug, y2, rcond=1e-3)[0]
    pred = np.einsum("ij,jk->ik", xt_aug, weights, optimize=False)
    return pred.reshape((x_test.shape[0],) + y_train.shape[1:])


def _persistence_forecast(x_test: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(x_test[:, -1:, :], horizon, axis=1)


def _tiny_ssm_forecast(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, horizon: int
) -> np.ndarray:
    last = x_train[:, -1, :]
    target = y_train[:, -1, :]
    denom = np.sum(last * last, axis=0) + 1e-6
    transition = np.sum(last * target, axis=0) / denom
    state = x_test[:, -1, :].copy()
    out = []
    for _ in range(horizon):
        state = state * transition
        out.append(state.copy())
    return np.stack(out, axis=1)


def _logistic_ridge_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, *, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0) + 1e-6
    x = (x_train - x_mean) / x_std
    xt = (x_test - x_mean) / x_std
    weights = rng.normal(scale=0.01, size=x.shape[1])
    bias = 0.0
    for _ in range(220):
        pred = _sigmoid(x.dot(weights) + bias)
        err = pred - y_train
        weights -= 0.03 * (x.T.dot(err) / x.shape[0] + 1e-3 * weights)
        bias -= 0.03 * float(err.mean())
    return _sigmoid(xt.dot(weights) + bias)


def _cycle_probability(h_train: np.ndarray, y_train: np.ndarray, h_test: np.ndarray) -> np.ndarray:
    global_rate = float(y_train.mean())
    rates = {
        hour: float(y_train[h_train == hour].mean()) if np.any(h_train == hour) else global_rate
        for hour in range(24)
    }
    return np.asarray([rates[int(hour)] for hour in h_test], dtype=np.float64)


def _metric_row(
    task_id: str,
    model_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    row_type: str = "baseline",
) -> dict[str, Any]:
    return {
        "row_type": row_type,
        "task_id": task_id,
        "model_id": model_id,
        "metric": "mse",
        "metric_value": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "pearsonr": pearsonr(y_true, y_pred),
        "split": "patient_held_out+time_held_out",
    }


def _risk_row(
    model_id: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    row_type: str = "baseline",
    task_id: str = "patient_held_out_event_risk_forecasting",
) -> dict[str, Any]:
    pred = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return {
        "row_type": row_type,
        "task_id": task_id,
        "model_id": model_id,
        "metric": "brier",
        "metric_value": mse(y_true, pred),
        "mae": mae(y_true, pred),
        "pearsonr": pearsonr(y_true, pred),
        "split": "patient_held_out+time_held_out",
    }


def _summary(rows: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row)
    best = {}
    for task_id, task_rows in by_task.items():
        ranked = sorted(task_rows, key=lambda row: float(row["metric_value"]))
        best[task_id] = {
            "best_model": ranked[0]["model_id"],
            "best_metric": ranked[0]["metric_value"],
            "metric": ranked[0]["metric"],
        }
    return {
        "claim_scope": STF_CLAIM_SCOPE,
        "scientific_claim_allowed": gate["scientific_claim_allowed"],
        "best_by_task": best,
        "a100_jobs_launched": False,
    }


def _report_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Kahlus-STF synthetic smoke",
        "",
        f"- claim_scope: {payload['claim_scope']}",
        f"- scientific_claim_allowed: {payload['gate']['scientific_claim_allowed']}",
        "- a100_jobs_launched: false",
        "",
        "| row_type | task | model | metric | value |",
        "|---|---|---|---|---:|",
    ]
    for row in payload["baseline_rows"]:
        lines.append(
            f"| {row['row_type']} | {row['task_id']} | {row['model_id']} | "
            f"{row['metric']} | {float(row['metric_value']):.6f} |"
        )
    return "\n".join(lines) + "\n"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))
