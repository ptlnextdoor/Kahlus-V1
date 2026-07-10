"""Local-only export of real EEG windows for mentor-facing diagnostics.

This module deliberately supports the historical one-sample-shift task because
its purpose is to show why that task is easy, not to revive it as valid
future-window forecasting evidence. Public waveform arrays are written only to
the caller-selected output directory and are never repository artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from neurotwin.data.schemas import NeuralEventBatch
from neurotwin.data.split_manifest import RecordingRecord, SplitManifest
from neurotwin.models.baselines import NumpyRidgeBaseline


HISTORICAL_OVERLAP_PROTOCOL = "kahlus.forecast.v1_overlap"


@dataclass(frozen=True)
class WindowEvidenceConfig:
    """Bounded local export configuration for a fixed shifted-window task."""

    context_samples: int = 127
    forecast_horizon_samples: int = 1
    stride_samples: int = 127
    max_train_windows: int = 512
    max_test_windows: int = 3
    ridge_alpha: float = 1e-2

    def __post_init__(self) -> None:
        for name in ("context_samples", "forecast_horizon_samples", "stride_samples", "max_train_windows", "max_test_windows"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if not np.isfinite(float(self.ridge_alpha)) or self.ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be finite and non-negative")


@dataclass(frozen=True)
class WindowEvidenceExport:
    """Arrays plus a minimal provenance summary for plotting outside the repo."""

    arrays: dict[str, np.ndarray]
    manifest: dict[str, Any]


def build_historical_window_evidence(
    batches: list[NeuralEventBatch],
    split: SplitManifest,
    *,
    config: WindowEvidenceConfig,
) -> WindowEvidenceExport:
    """Create a bounded raw-window export and ridge/persistence predictions.

    The task remains explicitly ineligible for scientific claims because target
    samples overlap the context whenever ``forecast_horizon_samples`` is less
    than ``context_samples``.
    """

    by_record = {batch.recording_id: batch for batch in batches if batch.modality == "eeg"}
    rows_by_split: dict[str, list[_WindowRow]] = {"train": [], "val": [], "test": []}
    for split_name, records in (("train", split.train), ("val", split.val), ("test", split.test)):
        limit = config.max_train_windows if split_name == "train" else config.max_test_windows
        for record in records:
            batch = by_record.get(record.record_id)
            if batch is None:
                continue
            rows_by_split[split_name].extend(_record_windows(batch, record, config=config, limit=limit - len(rows_by_split[split_name])))
            if len(rows_by_split[split_name]) >= limit:
                break

    if not rows_by_split["train"] or not rows_by_split["test"]:
        raise ValueError("window evidence requires at least one EEG training and test window")
    _validate_uniform_grid([*rows_by_split["train"], *rows_by_split["test"]])

    x_train = np.asarray([row.x for row in rows_by_split["train"]], dtype=np.float32)
    y_train = np.asarray([row.y for row in rows_by_split["train"]], dtype=np.float32)
    x_test = np.asarray([row.x for row in rows_by_split["test"]], dtype=np.float32)
    y_test = np.asarray([row.y for row in rows_by_split["test"]], dtype=np.float32)
    ridge = NumpyRidgeBaseline(alpha=config.ridge_alpha)
    ridge.fit(_flatten_time(x_train), _flatten_time(y_train))
    ridge_prediction = ridge.predict(_flatten_time(x_test)).reshape(y_test.shape).astype(np.float32)
    persistence_prediction = x_test.copy()
    first = rows_by_split["test"][0]
    overlap_samples = max(0, config.context_samples - config.forecast_horizon_samples)
    manifest = {
        "schema": "neurotwin.moabb_window_evidence.v1",
        "purpose": "mentor-facing diagnostic; not paper-ready benchmark evidence",
        "claim_eligible": False,
        "protocol_id": HISTORICAL_OVERLAP_PROTOCOL,
        "protocol_reason": "target is shifted by forecast_horizon_samples relative to context",
        "context_samples": config.context_samples,
        "target_samples": config.context_samples,
        "forecast_horizon_samples": config.forecast_horizon_samples,
        "shared_samples_per_example": overlap_samples,
        "shared_sample_fraction": overlap_samples / config.context_samples,
        "sampling_rate_hz": first.sampling_rate_hz,
        "signal_unit": first.signal_unit,
        "channel_names": list(first.channel_names),
        "split_policy": split.policy,
        "split_counts": {name: len(rows) for name, rows in rows_by_split.items()},
        "ridge": {"implementation": "NumpyRidgeBaseline", "alpha": config.ridge_alpha},
        "model_prediction_status": "not_available_without_a_provenance_matched_checkpoint_or_prediction_export",
        "raw_public_data_not_committed": True,
    }
    arrays = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "ridge_prediction_test": ridge_prediction,
        "y_pred_test": ridge_prediction,
        "persistence_prediction_test": persistence_prediction,
        "channel_names": np.asarray(first.channel_names, dtype="U"),
        "test_record_ids": np.asarray([row.record_id for row in rows_by_split["test"]], dtype="U"),
        "test_subject_ids": np.asarray([row.subject_id for row in rows_by_split["test"]], dtype="U"),
        "test_window_starts": np.asarray([row.start for row in rows_by_split["test"]], dtype=np.int64),
        "sampling_rate_hz": np.asarray(first.sampling_rate_hz, dtype=np.float64),
        "sfreq": np.asarray(first.sampling_rate_hz, dtype=np.float64),
        "signal_unit": np.asarray(first.signal_unit),
        "protocol_id": np.asarray(HISTORICAL_OVERLAP_PROTOCOL),
        "context_samples": np.asarray(config.context_samples, dtype=np.int64),
        "forecast_horizon_samples": np.asarray(config.forecast_horizon_samples, dtype=np.int64),
        "per_channel_lag_correlation": _per_channel_lag_correlation(x_test, y_test),
    }
    return WindowEvidenceExport(arrays=arrays, manifest=manifest)


def write_historical_window_evidence(export: WindowEvidenceExport, out_dir: str | Path) -> dict[str, Path]:
    """Write bounded local diagnostic arrays and an adjacent provenance manifest."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / "moabb_historical_window_evidence.npz"
    manifest_path = out / "moabb_historical_window_evidence.json"
    np.savez_compressed(npz_path, **export.arrays)
    manifest_path.write_text(json.dumps(export.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"npz": npz_path, "manifest": manifest_path}


@dataclass(frozen=True)
class _WindowRow:
    x: np.ndarray
    y: np.ndarray
    record_id: str
    subject_id: str
    start: int
    sampling_rate_hz: float
    signal_unit: str
    channel_names: tuple[str, ...]


def _record_windows(
    batch: NeuralEventBatch,
    record: RecordingRecord,
    *,
    config: WindowEvidenceConfig,
    limit: int,
) -> list[_WindowRow]:
    if limit <= 0:
        return []
    sampling_rate_hz = batch.sampling_rate
    if sampling_rate_hz is None:
        raise ValueError(f"record {record.record_id!r} has no sampling rate")
    signal_unit = str(batch.metadata.get("signal_unit", "unknown"))
    if signal_unit != "uV":
        raise ValueError(f"record {record.record_id!r} has signal unit {signal_unit!r}, expected 'uV'")
    channel_names = tuple(str(name) for name in batch.metadata.get("channel_names", ()))
    if len(channel_names) != batch.n_space:
        raise ValueError(f"record {record.record_id!r} does not have one channel name per EEG channel")
    signal = np.asarray(batch.signal, dtype=np.float32)
    last_start = signal.shape[0] - config.context_samples - config.forecast_horizon_samples
    rows: list[_WindowRow] = []
    for start in range(0, last_start + 1, config.stride_samples):
        target_start = start + config.forecast_horizon_samples
        rows.append(
            _WindowRow(
                x=signal[start : start + config.context_samples],
                y=signal[target_start : target_start + config.context_samples],
                record_id=record.record_id,
                subject_id=record.subject_id,
                start=start,
                sampling_rate_hz=float(sampling_rate_hz),
                signal_unit=signal_unit,
                channel_names=channel_names,
            )
        )
        if len(rows) >= limit:
            break
    return rows


def _validate_uniform_grid(rows: list[_WindowRow]) -> None:
    first = rows[0]
    for row in rows[1:]:
        if row.sampling_rate_hz != first.sampling_rate_hz:
            raise ValueError("diagnostic export requires one sampling rate")
        if row.signal_unit != first.signal_unit:
            raise ValueError("diagnostic export requires one signal unit")
        if row.channel_names != first.channel_names:
            raise ValueError("diagnostic export requires one channel order")


def _flatten_time(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1, values.shape[-1])


def _per_channel_lag_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for channel in range(x.shape[-1]):
        a = x[..., channel].reshape(-1)
        b = y[..., channel].reshape(-1)
        if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
            values.append(float("nan"))
        else:
            values.append(float(np.corrcoef(a, b)[0, 1]))
    return np.asarray(values, dtype=np.float64)
