"""Local-only export of MOABB-preprocessed EEG epochs for diagnostics.

This module deliberately supports the historical one-sample-shift task because
its purpose is to show why that task is easy, not to revive it as valid
future-window forecasting evidence. Public waveform arrays are written only to
the caller-selected output directory and are never repository artifacts.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any
import uuid

import numpy as np

from neurotwin.data.schemas import NeuralEventBatch
from neurotwin.data.forecast_contract import (
    FORECAST_PROTOCOL_V1_OVERLAP,
    legacy_overlapping_forecast_spec,
)
from neurotwin.data.split_manifest import RecordingRecord, SplitManifest
from neurotwin.models.baselines import NumpyRidgeBaseline


HISTORICAL_OVERLAP_PROTOCOL = FORECAST_PROTOCOL_V1_OVERLAP


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
        legacy_overlapping_forecast_spec(
            window_samples=self.context_samples,
            forecast_horizon_samples=self.forecast_horizon_samples,
            stride_samples=self.stride_samples,
        )
        for name in ("max_train_windows", "max_test_windows"):
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
    """Create a bounded MOABB-preprocessed epoch export and predictions.

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
    protocol = legacy_overlapping_forecast_spec(
        window_samples=config.context_samples,
        forecast_horizon_samples=config.forecast_horizon_samples,
        stride_samples=config.stride_samples,
        sampling_rate_hz=first.sampling_rate_hz,
    )
    input_start, input_stop, target_start, target_stop = protocol.ranges(0)
    overlap_samples = max(0, min(input_stop, target_stop) - max(input_start, target_start))
    source_provenance = dict(first.source_provenance)
    manifest = {
        "schema": "neurotwin.moabb_window_evidence.v1",
        "purpose": "mentor-facing diagnostic; not paper-ready benchmark evidence",
        "claim_eligible": protocol.claim_eligible,
        "protocol_id": protocol.spec.protocol_id,
        "protocol_reason": "target is shifted by forecast_horizon_samples relative to context",
        "context_samples": config.context_samples,
        "target_samples": protocol.target_samples,
        "forecast_horizon_samples": config.forecast_horizon_samples,
        "shared_samples_per_example": overlap_samples,
        "shared_sample_fraction": overlap_samples / config.context_samples,
        "sampling_rate_hz": first.sampling_rate_hz,
        "signal_unit": first.signal_unit,
        "channel_names": list(first.channel_names),
        "signal_source": source_provenance["signal_source"],
        "dataset": source_provenance["moabb_dataset"],
        "paradigm": source_provenance["moabb_paradigm"],
        "moabb_version": source_provenance["moabb_version"],
        "filters": source_provenance["moabb_filters"],
        "preprocessing": source_provenance["moabb_preprocessing"],
        "unit_factor_provenance": source_provenance["unit_factor_provenance"],
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
        "train_record_ids": np.asarray([row.record_id for row in rows_by_split["train"]], dtype="U"),
        "train_subject_ids": np.asarray([row.subject_id for row in rows_by_split["train"]], dtype="U"),
        "train_window_starts": np.asarray([row.start for row in rows_by_split["train"]], dtype=np.int64),
        "test_record_ids": np.asarray([row.record_id for row in rows_by_split["test"]], dtype="U"),
        "test_subject_ids": np.asarray([row.subject_id for row in rows_by_split["test"]], dtype="U"),
        "test_window_starts": np.asarray([row.start for row in rows_by_split["test"]], dtype=np.int64),
        "sampling_rate_hz": np.asarray(first.sampling_rate_hz, dtype=np.float64),
        "sfreq": np.asarray(first.sampling_rate_hz, dtype=np.float64),
        "signal_unit": np.asarray(first.signal_unit),
        "protocol_id": np.asarray(protocol.spec.protocol_id),
        "dataset": np.asarray(source_provenance["moabb_dataset"]),
        "paradigm": np.asarray(source_provenance["moabb_paradigm"]),
        "moabb_version": np.asarray(source_provenance["moabb_version"]),
        "signal_source": np.asarray(source_provenance["signal_source"]),
        "context_samples": np.asarray(config.context_samples, dtype=np.int64),
        "forecast_horizon_samples": np.asarray(config.forecast_horizon_samples, dtype=np.int64),
        "per_channel_lag_correlation": recordwise_autocorrelation(
            x_train,
            np.asarray([row.record_id for row in rows_by_split["train"]], dtype="U"),
            max_lag=config.forecast_horizon_samples,
        )[config.forecast_horizon_samples],
    }
    return WindowEvidenceExport(arrays=arrays, manifest=manifest)


def write_historical_window_evidence(export: WindowEvidenceExport, out_dir: str | Path) -> dict[str, Path]:
    """Write bounded local diagnostic arrays and an adjacent provenance manifest."""

    out = Path(out_dir)
    with fresh_atomic_output_directory(out) as staging:
        np.savez_compressed(staging / "moabb_historical_window_evidence.npz", **export.arrays)
        (staging / "moabb_historical_window_evidence.json").write_text(
            json.dumps(export.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    npz_path = out / "moabb_historical_window_evidence.npz"
    manifest_path = out / "moabb_historical_window_evidence.json"
    return {"npz": npz_path, "manifest": manifest_path}


def reject_git_managed_output_dir(out_dir: str | Path) -> Path:
    """Reject output paths located anywhere inside a Git worktree or repository."""

    out = Path(out_dir).expanduser().resolve(strict=False)
    existing = out
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    try:
        result = subprocess.run(
            ["git", "-C", str(existing), "rev-parse", "--is-inside-work-tree", "--is-inside-git-dir"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        result = None
    if result is not None and result.returncode == 0:
        states = {line.strip() for line in result.stdout.splitlines()}
        if "true" in states:
            raise ValueError(f"output directory must be outside every Git worktree/repository: {out}")
    if any((parent / ".git").exists() for parent in (out, *out.parents)):
        raise ValueError(f"output directory must be outside every Git worktree/repository: {out}")
    return out


@contextmanager
def fresh_atomic_output_directory(out_dir: str | Path):
    """Stage a complete output directory, then replace any prior directory."""

    out = reject_git_managed_output_dir(out_dir)
    if not out.name:
        raise ValueError("output directory must not be a filesystem root")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))
    backup: Path | None = None
    try:
        yield staging
        if out.exists():
            if not out.is_dir():
                raise ValueError(f"output path exists and is not a directory: {out}")
            backup = out.parent / f".{out.name}.previous-{uuid.uuid4().hex}"
            os.replace(out, backup)
        try:
            os.replace(staging, out)
        except BaseException:
            if backup is not None and backup.exists() and not out.exists():
                os.replace(backup, out)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


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
    source_provenance: dict[str, Any]


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
    source_provenance = _source_provenance(batch, record)
    protocol = legacy_overlapping_forecast_spec(
        window_samples=config.context_samples,
        forecast_horizon_samples=config.forecast_horizon_samples,
        stride_samples=config.stride_samples,
        sampling_rate_hz=float(sampling_rate_hz),
    )
    last_start = signal.shape[0] - protocol.ranges(0)[3]
    rows: list[_WindowRow] = []
    for start in range(0, last_start + 1, protocol.stride_samples):
        input_start, input_stop, target_start, target_stop = protocol.ranges(start)
        rows.append(
            _WindowRow(
                x=signal[input_start:input_stop],
                y=signal[target_start:target_stop],
                record_id=record.record_id,
                subject_id=record.subject_id,
                start=start,
                sampling_rate_hz=float(sampling_rate_hz),
                signal_unit=signal_unit,
                channel_names=channel_names,
                source_provenance=source_provenance,
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
        if row.source_provenance != first.source_provenance:
            raise ValueError("diagnostic export requires uniform MOABB preprocessing provenance")


def _flatten_time(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1, values.shape[-1])


def recordwise_autocorrelation(windows: np.ndarray, record_ids: np.ndarray, *, max_lag: int) -> np.ndarray:
    """Compute ACF inside windows/records and then average across records."""

    values = np.asarray(windows, dtype=np.float64)
    ids = np.asarray(record_ids).astype("U")
    if values.ndim != 3:
        raise ValueError("windows must have shape [windows, time, channels]")
    if ids.ndim != 1 or ids.shape[0] != values.shape[0]:
        raise ValueError("record_ids must contain one ID per window")
    if max_lag < 0 or max_lag >= values.shape[1]:
        raise ValueError("max_lag must be non-negative and shorter than each window")

    per_record: list[np.ndarray] = []
    for record_id in dict.fromkeys(ids.tolist()):
        record_windows = values[ids == record_id]
        numerators = np.zeros((max_lag + 1, values.shape[-1]), dtype=np.float64)
        denominators = np.zeros(values.shape[-1], dtype=np.float64)
        for window in record_windows:
            centered = window - window.mean(axis=0, keepdims=True)
            denominators += np.sum(centered * centered, axis=0)
            for lag in range(max_lag + 1):
                numerators[lag] += np.sum(centered[: centered.shape[0] - lag or None] * centered[lag:], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_record.append(numerators / denominators[None, :])
    return np.nanmean(np.asarray(per_record), axis=0)


def _source_provenance(batch: NeuralEventBatch, record: RecordingRecord) -> dict[str, Any]:
    keys = (
        "signal_source",
        "moabb_dataset",
        "moabb_paradigm",
        "moabb_version",
        "moabb_filters",
        "moabb_preprocessing",
        "unit_factor_provenance",
    )
    provenance = {key: batch.metadata.get(key) for key in keys}
    missing = [key for key, value in provenance.items() if value is None]
    if missing:
        raise ValueError(f"record {record.record_id!r} is missing MOABB provenance: {', '.join(missing)}")
    if provenance["signal_source"] != "MOABB-preprocessed epochs":
        raise ValueError(f"record {record.record_id!r} is not labeled as MOABB-preprocessed epochs")
    return provenance
