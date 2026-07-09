from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from neurotwin.stf.benchmark import (
    REQUIRED_STF_NEGATIVE_CONTROLS,
    REQUIRED_STF_SPLITS,
    REQUIRED_STF_TASKS,
    build_stf_gate,
)
from neurotwin.stf.public_data import CHBMITRootAudit, audit_chb_mit_root
from neurotwin.stf.smoke import (
    _cycle_probability,
    _logistic_ridge_predict,
    _metric_row,
    _persistence_forecast,
    _ridge_predict,
    _risk_row,
    _tiny_ssm_forecast,
)


def run_chb_mit_public_smoke(
    data_root: str | Path,
    *,
    seed: int = 0,
    max_records: int = 6,
    max_samples_per_record: int = 512,
    max_channels: int = 8,
) -> dict[str, Any]:
    audit = audit_chb_mit_root(data_root)
    if not audit.passed:
        return _blocked_payload(audit, ["CHB-MIT root audit failed"])

    records = _load_record_list(Path(data_root), max_records=max_records)
    if len({patient for patient, _path in records}) < 2:
        return _blocked_payload(audit, ["at least two patients are required for patient-held-out smoke"])

    signals, patient_ids, record_ids, sampling_frequencies = _load_edf_records(
        Path(data_root),
        records,
        max_samples_per_record=max_samples_per_record,
        max_channels=max_channels,
    )
    if len({str(patient) for patient in patient_ids}) < 2:
        return _blocked_payload(audit, ["EDF records did not yield at least two patients"])

    rows = []
    rows.extend(
        _forecast_rows_from_records(
            signals, patient_ids, seed=seed, task_id="future_eeg_forecasting", horizon=1
        )
    )
    rows.extend(
        _forecast_rows_from_records(
            signals, patient_ids, seed=seed + 1, task_id="longer_horizon_eeg_forecasting", horizon=5
        )
    )
    rows.extend(_channel_completion_rows_from_records(signals, patient_ids))
    seizure_intervals = parse_chb_mit_summary_dir(Path(data_root))
    rows.extend(
        _event_risk_rows_from_records(
            signals,
            patient_ids,
            record_ids,
            sampling_frequencies,
            seizure_intervals,
            seed=seed + 2,
        )
    )
    finite_metrics = all(
        np.isfinite(float(row["metric_value"]))
        for row in rows
        if row.get("metric_value") is not None
    )
    baselines_by_task: dict[str, list[str]] = {}
    negative_controls: list[str] = []
    for row in rows:
        if row["row_type"] == "baseline":
            baselines_by_task.setdefault(row["task_id"], []).append(row["model_id"])
        elif row["row_type"] == "negative_control":
            negative_controls.append(row["task_id"])
    gate = build_stf_gate(
        dataset=audit.dataset_id,
        declared_tasks=REQUIRED_STF_TASKS,
        baselines_by_task=baselines_by_task,
        negative_controls=negative_controls,
        split_types=REQUIRED_STF_SPLITS,
        split_audit_passed=True,
        baseline_table_present=bool(rows),
        finite_metrics=finite_metrics,
        calibration_checked=True,
    )
    return {
        "schema": "kahlus.stf.chb_mit_public_smoke.v1",
        "dataset": audit.dataset_id,
        "source_data_root": str(Path(data_root).expanduser().resolve()),
        "record_ids": record_ids,
        "patient_ids": patient_ids,
        "sampling_frequencies_hz": sampling_frequencies,
        "audit": audit.as_dict(),
        "baseline_rows": rows,
        "gate": gate,
        "public_smoke_passed": bool(gate["scientific_claim_allowed"]),
        "failure_reasons": list(gate["failure_reasons"]),
        "claim_boundary": (
            "public-data EEG/event-risk smoke only; full gates and A100 handoff remain blocked"
        ),
        "a100_jobs_launched": False,
    }


def write_chb_mit_public_smoke(out_dir: str | Path, payload: dict[str, Any]) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": out / "chb_mit_public_smoke.json",
        "baseline_table": out / "chb_mit_public_baseline_table.csv",
        "report": out / "chb_mit_public_smoke_report.md",
    }
    paths["metrics"].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
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
        writer.writerows(payload.get("baseline_rows", []))
    paths["report"].write_text(_report(payload), encoding="utf-8")
    return paths


def _load_record_list(root: Path, *, max_records: int) -> list[tuple[str, str]]:
    by_patient: dict[str, list[str]] = {}
    for row in (root / "RECORDS").read_text(encoding="utf-8").splitlines():
        record = row.strip()
        if record.endswith(".edf") and "/" in record:
            by_patient.setdefault(record.split("/", 1)[0], []).append(record)
    records = []
    while len(records) < max_records:
        grew = False
        for patient in sorted(by_patient):
            if by_patient[patient]:
                records.append((patient, by_patient[patient].pop(0)))
                grew = True
                if len(records) >= max_records:
                    break
        if not grew:
            break
    return records


def _load_edf_records(
    root: Path,
    records: list[tuple[str, str]],
    *,
    max_samples_per_record: int,
    max_channels: int,
) -> tuple[list[np.ndarray], list[str], list[str], list[float]]:
    import edfio

    matrices: list[np.ndarray] = []
    patient_ids: list[str] = []
    record_ids: list[str] = []
    sampling_frequencies: list[float] = []
    min_channels = max_channels
    for patient, record in records:
        edf = edfio.read_edf(root / record, lazy_load_data=True)
        channels = []
        record_sampling_frequency: float | None = None
        for signal in edf.signals[:max_channels]:
            sampling_frequency = float(signal.sampling_frequency)
            if record_sampling_frequency is None:
                record_sampling_frequency = sampling_frequency
            elif not np.isclose(record_sampling_frequency, sampling_frequency):
                raise ValueError(f"mixed sampling frequencies in EDF record {record}")
            data = np.asarray(signal.data[:max_samples_per_record], dtype=np.float64)
            channels.append(data)
        if not channels:
            continue
        if record_sampling_frequency is None or record_sampling_frequency <= 0:
            raise ValueError(f"missing positive sampling frequency in EDF record {record}")
        min_channels = min(min_channels, len(channels))
        matrix = np.stack(channels, axis=1)
        matrices.append(matrix)
        patient_ids.append(patient)
        record_ids.append(record)
        sampling_frequencies.append(record_sampling_frequency)
    if len(matrices) < 2:
        raise ValueError("at least two readable EDF records are required")
    trimmed = [matrix[:, :min_channels] for matrix in matrices]
    return trimmed, patient_ids, record_ids, sampling_frequencies


def parse_chb_mit_summary_dir(data_root: str | Path) -> dict[str, list[tuple[float, float]]]:
    root = Path(data_root)
    intervals: dict[str, list[tuple[float, float]]] = {}
    for summary in sorted(root.glob("chb*/chb*-summary.txt")):
        intervals.update(parse_chb_mit_summary_text(summary.read_text(encoding="utf-8")))
    return intervals


def parse_chb_mit_summary_text(text: str) -> dict[str, list[tuple[float, float]]]:
    intervals: dict[str, list[tuple[float, float]]] = {}
    current: str | None = None
    pending_start: float | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("File Name:"):
            current = line.split(":", 1)[1].strip()
            intervals.setdefault(current, [])
            pending_start = None
        elif re.match(r"^Seizure(?:\s+\d+)?\s+Start Time:", line) and current:
            pending_start = _seconds_field(line)
        elif (
            re.match(r"^Seizure(?:\s+\d+)?\s+End Time:", line)
            and current
            and pending_start is not None
        ):
            intervals[current].append((pending_start, _seconds_field(line)))
            pending_start = None
    return {record: spans for record, spans in intervals.items() if spans}


def _seconds_field(line: str) -> float:
    return float(line.split(":", 1)[1].strip().split()[0])


def _forecast_rows_from_records(
    signals: list[np.ndarray],
    patient_ids: list[str],
    *,
    seed: int,
    task_id: str,
    horizon: int,
    window: int = 32,
) -> list[dict[str, Any]]:
    x_train, y_train, x_test, y_test = _record_forecast_arrays(
        signals, patient_ids, horizon=horizon, window=window
    )
    if x_train.size == 0 or x_test.size == 0:
        raise ValueError("patient/time-held-out EDF smoke produced no train or test windows")
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


def _record_forecast_arrays(
    signals: list[np.ndarray], patient_ids: list[str], *, horizon: int, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    patients = sorted(set(patient_ids))
    test_patient = patients[-1]
    xs_train: list[np.ndarray] = []
    ys_train: list[np.ndarray] = []
    xs_test: list[np.ndarray] = []
    ys_test: list[np.ndarray] = []
    for record_idx, patient in enumerate(patient_ids):
        signal = signals[record_idx]
        split_time = int(signal.shape[0] * 0.65)
        limit = signal.shape[0] - window - horizon
        for start in _background_starts(limit):
            x = signal[start : start + window]
            y = signal[start + window : start + window + horizon]
            if patient != test_patient and start + window + horizon <= split_time:
                xs_train.append(x)
                ys_train.append(y)
            elif patient == test_patient and start >= split_time:
                xs_test.append(x)
                ys_test.append(y)
    return (
        np.asarray(xs_train),
        np.asarray(ys_train),
        np.asarray(xs_test),
        np.asarray(ys_test),
    )


def _channel_completion_rows_from_records(
    signals: list[np.ndarray],
    patient_ids: list[str],
) -> list[dict[str, Any]]:
    if not signals or signals[0].shape[1] < 2:
        return []
    patients = sorted(set(patient_ids))
    test_patient = patients[-1]
    observed_channels = max(1, signals[0].shape[1] - 2)
    x_train: list[np.ndarray] = []
    y_train: list[np.ndarray] = []
    x_test: list[np.ndarray] = []
    y_test: list[np.ndarray] = []
    for record_idx, patient in enumerate(patient_ids):
        signal = signals[record_idx]
        split_time = int(signal.shape[0] * 0.65)
        if patient != test_patient:
            x_train.append(signal[:split_time, :observed_channels])
            y_train.append(signal[:split_time, observed_channels:])
        else:
            x_test.append(signal[split_time:, :observed_channels])
            y_test.append(signal[split_time:, observed_channels:])
    if not x_train or not x_test:
        return []
    xtr = np.concatenate(x_train, axis=0)
    ytr = np.concatenate(y_train, axis=0)
    xte = np.concatenate(x_test, axis=0)
    yte = np.concatenate(y_test, axis=0)
    mean_pred = np.repeat(ytr.mean(axis=0, keepdims=True), yte.shape[0], axis=0)
    latent_train = xtr.mean(axis=1, keepdims=True)
    latent_test = xte.mean(axis=1, keepdims=True)
    preds = {
        "channel_mean": mean_pred,
        "ridge_ar": _ridge_predict(xtr, ytr, xte),
        "tiny_ssm": _ridge_predict(latent_train, ytr, latent_test),
    }
    return [
        _metric_row("held_out_channel_completion", model, yte, pred)
        for model, pred in preds.items()
    ]


def _event_risk_rows_from_records(
    signals: list[np.ndarray],
    patient_ids: list[str],
    record_ids: list[str],
    sampling_frequencies: list[float],
    seizure_intervals: dict[str, list[tuple[float, float]]],
    *,
    seed: int,
    window: int = 32,
    horizon: int = 32,
) -> list[dict[str, Any]]:
    if not seizure_intervals:
        return []
    x_train, y_train, h_train, x_test, y_test, h_test = _record_event_arrays(
        signals,
        patient_ids,
        record_ids,
        sampling_frequencies,
        seizure_intervals,
        window=window,
        horizon=horizon,
    )
    if x_train.size == 0 or x_test.size == 0 or len(set(y_train.tolist())) < 2:
        return []
    shifted = np.roll(y_train, max(1, y_train.size // 5))
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


def _record_event_arrays(
    signals: list[np.ndarray],
    patient_ids: list[str],
    record_ids: list[str],
    sampling_frequencies: list[float],
    seizure_intervals: dict[str, list[tuple[float, float]]],
    *,
    window: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    patients = sorted(set(patient_ids))
    test_patient = patients[-1]
    x_train: list[np.ndarray] = []
    y_train: list[float] = []
    h_train: list[int] = []
    x_test: list[np.ndarray] = []
    y_test: list[float] = []
    h_test: list[int] = []
    for record_idx, patient in enumerate(patient_ids):
        signal = signals[record_idx]
        split_time = int(signal.shape[0] * 0.65)
        record_name = Path(record_ids[record_idx]).name
        labels = _sample_event_labels(
            signal.shape[0],
            seizure_intervals.get(record_name, ()),
            sampling_frequency=sampling_frequencies[record_idx],
        )
        limit = signal.shape[0] - window - horizon
        for start in _event_starts(labels, limit, window=window, horizon=horizon):
            x = np.r_[
                signal[start : start + window].mean(axis=0),
                signal[start + window - 1] - signal[start],
            ]
            y = float(labels[start + window : start + window + horizon].max())
            if patient != test_patient and start + window + horizon <= split_time:
                x_train.append(x)
                y_train.append(y)
                h_train.append(start % 24)
            elif patient == test_patient and start >= split_time:
                x_test.append(x)
                y_test.append(y)
                h_test.append(start % 24)
    return (
        np.asarray(x_train),
        np.asarray(y_train),
        np.asarray(h_train),
        np.asarray(x_test),
        np.asarray(y_test),
        np.asarray(h_test),
    )


def _sample_event_labels(
    n_samples: int,
    intervals: list[tuple[float, float]],
    *,
    sampling_frequency: float,
) -> np.ndarray:
    labels = np.zeros(n_samples, dtype=np.float64)
    for start_s, end_s in intervals:
        start = max(0, int(round(start_s * sampling_frequency)))
        end = min(n_samples, int(round(end_s * sampling_frequency)))
        if end > start:
            labels[start:end] = 1.0
    return labels


def _background_starts(limit: int, *, max_windows: int = 2048) -> range:
    if limit <= 0:
        return range(0)
    return range(0, limit, max(1, limit // max_windows))


def _event_starts(labels: np.ndarray, limit: int, *, window: int, horizon: int) -> list[int]:
    starts = set(_background_starts(limit))
    positive = np.flatnonzero(labels > 0)
    if positive.size:
        breaks = np.flatnonzero(np.diff(positive) > 1) + 1
        spans = np.split(positive, breaks)
        step = max(1, horizon // 4)
        for span in spans:
            start_min = max(0, int(span[0]) - window - horizon + 1)
            start_max = min(limit - 1, int(span[-1]) - window)
            starts.update(range(start_min, start_max + 1, step))
    return sorted(starts)


def _blocked_payload(audit: CHBMITRootAudit, reasons: list[str]) -> dict[str, Any]:
    return {
        "schema": "kahlus.stf.chb_mit_public_smoke.v1",
        "dataset": audit.dataset_id,
        "source_data_root": audit.data_root,
        "audit": audit.as_dict(),
        "baseline_rows": [],
        "public_smoke_passed": False,
        "failure_reasons": reasons + list(audit.failure_reasons),
        "a100_jobs_launched": False,
    }


def _report(payload: dict[str, Any]) -> str:
    lines = [
        "# Kahlus-STF CHB-MIT public smoke",
        "",
        f"- dataset: {payload['dataset']}",
        f"- public_smoke_passed: {payload['public_smoke_passed']}",
        "- a100_jobs_launched: false",
        "",
        "## Baselines",
        "| task | model | metric | value |",
        "|---|---|---|---:|",
    ]
    if "gate" in payload:
        lines.insert(4, f"- scientific_claim_allowed: {payload['gate']['scientific_claim_allowed']}")
    for row in payload.get("baseline_rows", []):
        lines.append(
            f"| {row['task_id']} | {row['model_id']} | {row['metric']} | "
            f"{float(row['metric_value']):.6f} |"
        )
    if payload.get("failure_reasons"):
        lines.extend(["", "## Failure Reasons"])
        lines.extend(f"- {reason}" for reason in payload["failure_reasons"])
    return "\n".join(lines) + "\n"
