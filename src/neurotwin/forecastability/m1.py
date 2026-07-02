from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TransitionFixture:
    windows: np.ndarray
    nuisance: np.ndarray
    y: np.ndarray
    patient: np.ndarray
    site: np.ndarray
    time_bucket: np.ndarray
    session: np.ndarray


def run_m1_gate(out_dir: str | Path, *, seed: int = 0) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    known = _run_fixture(make_transition_fixture(seed=seed, residual_signal=True), seed=seed)
    null = _run_fixture(make_transition_fixture(seed=seed + 100, residual_signal=False), seed=seed + 100)
    gate = {
        "milestone": "M1",
        "known_signal": known,
        "synthetic_null": null,
        "gate_passed": _known_passes(known) and _null_passes(null),
        "stop_reason": "M1 gate reached; do not proceed to M2 until this report is reviewed.",
    }
    _write_json(out / "m1_gate_report.json", gate)
    _write_report(out / "M1_EVIDENCE_REPORT.md", gate)
    return gate


def make_transition_fixture(
    *,
    seed: int,
    residual_signal: bool,
    n_patients: int = 12,
    steps_per_patient: int = 120,
    window: int = 32,
    channels: int = 4,
    horizon: int = 1,
) -> TransitionFixture:
    rng = np.random.default_rng(seed)
    windows: list[np.ndarray] = []
    nuisance: list[list[float]] = []
    labels: list[int] = []
    patients: list[int] = []
    sites: list[int] = []
    time_buckets: list[int] = []
    sessions: list[int] = []
    t_axis = np.linspace(0.0, 1.0, window, dtype=np.float32)
    for patient in range(n_patients):
        site = patient % 2
        precursor = 0.0
        events = [0]
        raw_windows: list[np.ndarray] = []
        logits: list[float] = []
        nuisance_rows: list[list[float]] = []
        for t in range(steps_per_patient + horizon):
            cycle = np.sin(2.0 * np.pi * t / 48.0)
            recent = float(np.mean(events[-12:]))
            precursor = 0.82 * precursor + rng.normal(0.0, 0.8)
            visible = precursor if residual_signal else rng.normal(0.0, 1.0)
            signal = rng.normal(0.0, 0.25, size=(window, channels)).astype(np.float32)
            signal[:, 0] += (0.6 + 0.28 * visible) * np.sin(2.0 * np.pi * (6.0 + 0.25 * cycle) * t_axis)
            signal[:, 1] += 0.18 * visible * np.linspace(-1.0, 1.0, window)
            signal[:, 2] += 0.15 * np.sin(2.0 * np.pi * 3.0 * t_axis + float(site))
            logit = -2.25 + 0.85 * cycle + 1.15 * recent + (1.75 * precursor if residual_signal else 0.0)
            event = int(rng.random() < _sigmoid(logit))
            raw_windows.append(signal)
            logits.append(logit)
            nuisance_rows.append([1.0, cycle, np.cos(2.0 * np.pi * t / 48.0), recent, float(t % 48) / 47.0])
            events.append(event)
        for t in range(steps_per_patient):
            windows.append(raw_windows[t])
            nuisance.append(nuisance_rows[t])
            labels.append(int(events[t + horizon]))
            patients.append(patient)
            sites.append(site)
            time_buckets.append((t % 48) // 12)
            sessions.append(t // max(1, steps_per_patient // 2))
    return TransitionFixture(
        windows=np.asarray(windows, dtype=np.float32),
        nuisance=np.asarray(nuisance, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int64),
        patient=np.asarray(patients, dtype=np.int64),
        site=np.asarray(sites, dtype=np.int64),
        time_bucket=np.asarray(time_buckets, dtype=np.int64),
        session=np.asarray(sessions, dtype=np.int64),
    )


def handcrafted_eeg_features(windows: np.ndarray) -> np.ndarray:
    x = np.asarray(windows, dtype=np.float32)
    diffs = np.diff(x, axis=1)
    line_length = np.mean(np.abs(diffs), axis=(1, 2), keepdims=False)[:, None]
    amplitude_variance = np.var(x, axis=(1, 2), keepdims=False)[:, None]
    centered_t = np.linspace(-1.0, 1.0, x.shape[1], dtype=np.float32)
    trend = np.mean(x * centered_t[None, :, None], axis=(1, 2), keepdims=False)[:, None]
    spectrum = np.abs(np.fft.rfft(x, axis=1)) ** 2
    total_power = np.sum(spectrum, axis=(1, 2), keepdims=False)[:, None] + 1e-8
    bandpower = np.concatenate(
        [
            np.sum(spectrum[:, 1:4, :], axis=(1, 2), keepdims=False)[:, None] / total_power,
            np.sum(spectrum[:, 4:9, :], axis=(1, 2), keepdims=False)[:, None] / total_power,
            np.sum(spectrum[:, 9:, :], axis=(1, 2), keepdims=False)[:, None] / total_power,
        ],
        axis=1,
    )
    p = spectrum / (np.sum(spectrum, axis=1, keepdims=True) + 1e-8)
    spectral_entropy = -np.mean(np.sum(p * np.log(p + 1e-8), axis=1), axis=1)[:, None]
    coherence = np.asarray([_mean_abs_corr(row) for row in x], dtype=np.float32)[:, None]
    return np.concatenate([bandpower, line_length, spectral_entropy, coherence, amplitude_variance, trend], axis=1).astype(np.float32)


def discrete_survival_labels(events: np.ndarray, *, bins: int = 3) -> np.ndarray:
    y = np.asarray(events, dtype=np.int64)
    labels = np.zeros((len(y), bins), dtype=np.int64)
    for idx in range(len(y)):
        for horizon in range(1, bins + 1):
            if idx + horizon < len(y):
                labels[idx, horizon - 1] = y[idx + horizon]
    return labels


def _run_fixture(fixture: TransitionFixture, *, seed: int) -> dict[str, Any]:
    y = fixture.y
    z = handcrafted_eeg_features(fixture.windows)
    b = fixture.nuisance
    x_full = np.concatenate([b, z], axis=1)
    baseline = _crossfit_proba(b, y, fixture.patient, _logistic_factory)
    full = _crossfit_residual_proba(b, z, y, fixture.patient)
    gbm = _crossfit_proba(x_full, y, fixture.patient, _gbm_factory(seed))
    shuffled = _crossfit_residual_proba(b, z, y, fixture.patient, control="shuffle", seed=seed + 11)
    shifted = _crossfit_residual_proba(b, z, y, fixture.patient, control="time_shift", seed=seed + 12)
    moving = _moving_average_proba(y, fixture.patient)
    random_warning = np.full_like(full, float(np.mean(y)), dtype=np.float64)
    alarm_time = _within_patient_roll(full, fixture.patient, shift=17)
    survival = discrete_survival_labels(y, bins=3)
    return {
        "n": int(len(y)),
        "positive_events": int(np.sum(y)),
        "event_patients": int(len(set(fixture.patient[y == 1]))),
        "survival_label_shape": list(survival.shape),
        "baseline_nll": _nll(y, baseline),
        "moving_average_nll": _nll(y, moving),
        "random_warning_nll": _nll(y, random_warning),
        "alarm_time_nll": _nll(y, alarm_time),
        "logistic_full": _rfs_payload(y, baseline, full, fixture.patient, seed=seed),
        "gbm_full": _rfs_payload(y, baseline, gbm, fixture.patient, seed=seed + 1),
        "shuffled_target_control": _rfs_payload(y, baseline, shuffled, fixture.patient, seed=seed + 2),
        "time_shift_control": _rfs_payload(y, baseline, shifted, fixture.patient, seed=seed + 3),
        "nuisance_probes": {
            "patient": _probe_accuracy(z, fixture.patient),
            "site": _probe_accuracy(z, fixture.site),
            "time_bucket": _probe_accuracy(z, fixture.time_bucket),
            "session": _probe_accuracy(z, fixture.session),
        },
    }


def _crossfit_proba(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    factory: Callable[[], Any],
    *,
    control: str | None = None,
    seed: int = 0,
) -> np.ndarray:
    pred = np.zeros(len(y), dtype=np.float64)
    folds = GroupKFold(n_splits=min(4, len(set(groups))))
    rng = np.random.default_rng(seed)
    for train_idx, test_idx in folds.split(x, y, groups):
        y_train = y[train_idx].copy()
        if control == "shuffle":
            rng.shuffle(y_train)
        elif control == "time_shift":
            y_train = _shift_train_labels(y_train, groups[train_idx], shift=9)
        pred[test_idx] = _fit_predict(factory(), x[train_idx], y_train, x[test_idx])
    return np.clip(pred, 1e-5, 1.0 - 1e-5)


def _crossfit_residual_proba(
    b: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    control: str | None = None,
    seed: int = 0,
) -> np.ndarray:
    pred = np.zeros(len(y), dtype=np.float64)
    folds = GroupKFold(n_splits=min(4, len(set(groups))))
    rng = np.random.default_rng(seed)
    for train_idx, test_idx in folds.split(z, y, groups):
        y_train = y[train_idx].copy()
        if control == "shuffle":
            rng.shuffle(y_train)
        elif control == "time_shift":
            y_train = _shift_train_labels(y_train, groups[train_idx], shift=9)
        nuisance = _logistic_factory()
        q0_train = _fit_predict(nuisance, b[train_idx], y[train_idx], b[train_idx])
        q0_test = _fit_predict(nuisance, b[train_idx], y[train_idx], b[test_idx])
        pred[test_idx] = _fit_residual_offset_predict(z[train_idx], y_train, q0_train, z[test_idx], q0_test)
    return np.clip(pred, 1e-5, 1.0 - 1e-5)


def _fit_predict(model: Any, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    if len(set(y_train)) < 2:
        return np.full(x_test.shape[0], float(np.mean(y_train)), dtype=np.float64)
    model.fit(x_train, y_train)
    return np.asarray(model.predict_proba(x_test)[:, 1], dtype=np.float64)


def _fit_residual_offset_predict(
    z_train: np.ndarray,
    y_train: np.ndarray,
    q0_train: np.ndarray,
    z_test: np.ndarray,
    q0_test: np.ndarray,
    *,
    steps: int = 500,
    lr: float = 0.15,
    l2: float = 0.2,
) -> np.ndarray:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(z_train)
    x_test = scaler.transform(z_test)
    x_train = np.concatenate([np.ones((x_train.shape[0], 1)), x_train], axis=1)
    x_test = np.concatenate([np.ones((x_test.shape[0], 1)), x_test], axis=1)
    w = np.zeros(x_train.shape[1], dtype=np.float64)
    offset = _logit(q0_train)
    for _ in range(steps):
        p = _sigmoid_array(offset + x_train @ w)
        grad = x_train.T @ (p - y_train) / len(y_train)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return _sigmoid_array(_logit(q0_test) + x_test @ w)


def _rfs_payload(y: np.ndarray, baseline: np.ndarray, pred: np.ndarray, patient: np.ndarray, *, seed: int) -> dict[str, float]:
    values = _cluster_bootstrap_rfs(y, baseline, pred, patient, seed=seed)
    values["rfs_bits"] = _rfs_bits(y, baseline, pred)
    values["nll"] = _nll(y, pred)
    return values


def _cluster_bootstrap_rfs(
    y: np.ndarray,
    baseline: np.ndarray,
    pred: np.ndarray,
    patient: np.ndarray,
    *,
    seed: int,
    n_boot: int = 300,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(patient)
    samples = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(patient == group) for group in chosen])
        samples.append(_rfs_bits(y[idx], baseline[idx], pred[idx]))
    return {"rfs_ci_low": float(np.percentile(samples, 2.5)), "rfs_ci_high": float(np.percentile(samples, 97.5))}


def _rfs_bits(y: np.ndarray, baseline: np.ndarray, pred: np.ndarray) -> float:
    return float((_nll(y, baseline) - _nll(y, pred)) / np.log(2.0))


def _nll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _moving_average_proba(y: np.ndarray, patient: np.ndarray, *, window: int = 20) -> np.ndarray:
    out = np.zeros(len(y), dtype=np.float64)
    base = float(np.mean(y))
    for group in np.unique(patient):
        idx = np.flatnonzero(patient == group)
        history: list[int] = []
        for row in idx:
            out[row] = float(np.mean(history[-window:])) if history else base
            history.append(int(y[row]))
    return np.clip(out, 1e-5, 1.0 - 1e-5)


def _within_patient_roll(values: np.ndarray, patient: np.ndarray, *, shift: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    for group in np.unique(patient):
        idx = np.flatnonzero(patient == group)
        out[idx] = np.roll(values[idx], shift)
    return np.clip(out, 1e-5, 1.0 - 1e-5)


def _shift_train_labels(y: np.ndarray, groups: np.ndarray, *, shift: int) -> np.ndarray:
    out = y.copy()
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        out[idx] = np.roll(out[idx], shift)
    return out


def _probe_accuracy(z: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) < 2:
        return {"accuracy": 1.0, "chance": 1.0}
    train = np.arange(len(labels)) % 3 != 0
    test = ~train
    model = _logistic_factory()
    model.fit(z[train], labels[train])
    accuracy = float(np.mean(model.predict(z[test]) == labels[test]))
    return {"accuracy": accuracy, "chance": float(1.0 / len(classes))}


def _logistic_factory() -> Any:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.75, random_state=0))


def _gbm_factory(seed: int) -> Callable[[], Any]:
    def factory() -> Any:
        return HistGradientBoostingClassifier(max_iter=64, learning_rate=0.05, max_leaf_nodes=15, random_state=seed)

    return factory


def _known_passes(payload: dict[str, Any]) -> bool:
    full = payload["logistic_full"]
    shuffled = payload["shuffled_target_control"]
    shifted = payload["time_shift_control"]
    return bool(
        payload["positive_events"] >= 40
        and payload["event_patients"] >= 8
        and full["rfs_bits"] > 0.03
        and full["rfs_ci_low"] > 0.0
        and shuffled["rfs_bits"] < full["rfs_bits"] * 0.4
        and shifted["rfs_bits"] < full["rfs_bits"] * 0.4
    )


def _null_passes(payload: dict[str, Any]) -> bool:
    full = payload["logistic_full"]
    return bool(payload["positive_events"] >= 20 and full["rfs_ci_low"] <= 0.0 <= full["rfs_ci_high"] and abs(full["rfs_bits"]) < 0.03)


def _mean_abs_corr(window: np.ndarray) -> float:
    if window.shape[1] < 2:
        return 0.0
    corr = np.corrcoef(window, rowvar=False)
    tri = corr[np.triu_indices_from(corr, k=1)]
    finite = np.abs(tri[np.isfinite(tri)])
    return float(np.mean(finite)) if finite.size else 0.0


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _sigmoid_array(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_report(path: Path, gate: dict[str, Any]) -> None:
    known = gate["known_signal"]
    null = gate["synthetic_null"]
    lines = [
        "# Kahlus Forecastability Trial 0 - M1 Evidence Report",
        "",
        f"Gate passed: `{gate['gate_passed']}`",
        "",
        "## Synthetic Known Signal",
        "",
        _fixture_lines(known),
        "",
        "## Synthetic Null",
        "",
        _fixture_lines(null),
        "",
        "M1 stops here. M2 should not start until this instrumentation gate is reviewed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fixture_lines(payload: dict[str, Any]) -> str:
    full = payload["logistic_full"]
    shuffled = payload["shuffled_target_control"]
    shifted = payload["time_shift_control"]
    return "\n".join(
        [
            f"- rows/events/event-patients: `{payload['n']}` / `{payload['positive_events']}` / `{payload['event_patients']}`",
            f"- logistic RFS bits: `{full['rfs_bits']:.6f}` CI `[ {full['rfs_ci_low']:.6f}, {full['rfs_ci_high']:.6f} ]`",
            f"- GBM RFS bits: `{payload['gbm_full']['rfs_bits']:.6f}`",
            f"- shuffled-target RFS bits: `{shuffled['rfs_bits']:.6f}`",
            f"- time-shift RFS bits: `{shifted['rfs_bits']:.6f}`",
            f"- baseline/moving/random/alarm NLL: `{payload['baseline_nll']:.6f}` / `{payload['moving_average_nll']:.6f}` / `{payload['random_warning_nll']:.6f}` / `{payload['alarm_time_nll']:.6f}`",
        ]
    )
