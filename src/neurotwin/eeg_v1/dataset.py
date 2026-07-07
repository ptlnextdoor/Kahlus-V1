from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neurotwin.data.prepared_tasks import SupervisedWindowTask
from neurotwin.data.schemas import NeuralEventBatch
from neurotwin.data.split_manifest import RecordingRecord, SplitManifest, build_split_manifest


@dataclass(frozen=True)
class EEGV1Dataset:
    """EEG batches plus the pre-window split manifest used for v1 baseline evaluation."""

    dataset_id: str
    batches: tuple[NeuralEventBatch, ...]
    records: tuple[RecordingRecord, ...]
    split_manifest: SplitManifest
    split_subjects: dict[str, tuple[str, ...]]
    source: str = "synthetic_fixture"


def make_synthetic_eeg_v1_dataset(
    *,
    seed: int = 0,
    n_subjects: int = 12,
    sessions_per_subject: int = 2,
    n_time: int = 72,
    n_channels: int = 6,
) -> EEGV1Dataset:
    """Build a deterministic EEG-like fixture with subject-held-out metadata."""

    if n_subjects < 3:
        raise ValueError("n_subjects must be >= 3 for train/val/test subject-held-out splits")
    if sessions_per_subject < 1:
        raise ValueError("sessions_per_subject must be >= 1")
    if n_time < 16:
        raise ValueError("n_time must be >= 16")
    if n_channels < 1:
        raise ValueError("n_channels must be >= 1")

    records: list[RecordingRecord] = []
    batches: list[NeuralEventBatch] = []
    projection = _stable_projection(n_channels)
    for subject_idx in range(n_subjects):
        subject_id = f"sub-{subject_idx:03d}"
        site_id = "site-a" if subject_idx % 2 == 0 else "site-b"
        subject_gain = 0.9 + 0.03 * subject_idx
        for session_idx in range(sessions_per_subject):
            session_id = f"ses-{session_idx:02d}"
            record_id = f"synthetic_eeg_v1_{subject_id}_{session_id}_eeg"
            start_time = float(session_idx * 100)
            signal = _synthetic_signal(
                seed=seed,
                subject_idx=subject_idx,
                session_idx=session_idx,
                n_time=n_time,
                projection=projection,
                subject_gain=subject_gain,
            )
            time = np.arange(n_time, dtype=np.float32) + start_time
            metadata: dict[str, Any] = {
                "record_id": record_id,
                "source_record_id": record_id,
                "synthetic": True,
                "sampling_rate": 128.0,
            }
            records.append(
                RecordingRecord(
                    record_id=record_id,
                    modality="eeg",
                    dataset="synthetic_eeg_v1",
                    subject_id=subject_id,
                    session_id=session_id,
                    site_id=site_id,
                    start_time=start_time,
                    end_time=start_time + float(n_time - 1),
                    path=f"synthetic://{record_id}",
                    metadata=metadata,
                )
            )
            batches.append(
                NeuralEventBatch(
                    modality="eeg",
                    dataset="synthetic_eeg_v1",
                    subject_id=subject_id,
                    session_id=session_id,
                    site_id=site_id,
                    time=time,
                    signal=signal,
                    mask=np.ones_like(signal, dtype=bool),
                    stimulus_embedding=None,
                    behavior={},
                    space_index=np.arange(n_channels),
                    uncertainty=np.full_like(signal, 0.05, dtype=np.float32),
                    provenance={"source": "synthetic_eeg_v1"},
                    metadata=metadata,
                )
            )
    split = build_split_manifest(records, policy="subject", seed=seed)
    return EEGV1Dataset(
        dataset_id="synthetic_eeg_v1",
        batches=tuple(batches),
        records=tuple(records),
        split_manifest=split,
        split_subjects=_split_subjects(split),
    )


def build_future_forecasting_task(
    dataset: EEGV1Dataset,
    *,
    window_length: int = 8,
    forecast_horizon: int = 1,
    stride: int = 1,
) -> SupervisedWindowTask:
    """Create a subject-held-out EEG future-window task with equal input/target lengths."""

    if window_length <= 1:
        raise ValueError("window_length must be > 1")
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    by_record = {batch.recording_id: batch for batch in dataset.batches}
    arrays: dict[str, list[np.ndarray]] = {"train_x": [], "train_y": [], "val_x": [], "val_y": [], "test_x": [], "test_y": []}
    test_subjects: list[str] = []
    test_records: list[str] = []
    test_starts: list[int] = []
    for split_name, records in (
        ("train", dataset.split_manifest.train),
        ("val", dataset.split_manifest.val),
        ("test", dataset.split_manifest.test),
    ):
        for record in records:
            batch = by_record.get(record.record_id)
            if batch is None:
                continue
            x, y, starts = _future_windows(batch.signal, window_length, forecast_horizon, stride)
            if x.size == 0:
                continue
            arrays[f"{split_name}_x"].append(x)
            arrays[f"{split_name}_y"].append(y)
            if split_name == "test":
                test_subjects.extend([record.subject_id] * x.shape[0])
                test_records.extend([record.record_id] * x.shape[0])
                test_starts.extend(starts)

    x_train, y_train = _stack(arrays["train_x"]), _stack(arrays["train_y"])
    x_val, y_val = _stack(arrays["val_x"]), _stack(arrays["val_y"])
    x_test, y_test = _stack(arrays["test_x"]), _stack(arrays["test_y"])
    if x_train.size == 0 or x_test.size == 0:
        raise ValueError("future forecasting task requires nonempty train and test windows")
    return SupervisedWindowTask(
        task_id="future_state_forecasting",
        source_modality="eeg",
        target_modality="eeg",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_val=x_val if x_val.size else None,
        y_val=y_val if y_val.size else None,
        notes=("Kahlus v1 EEG future-window forecasting under held-out subject split",),
        metadata={
            "dataset_id": dataset.dataset_id,
            "source": dataset.source,
            "benchmark_status": _benchmark_status(dataset),
            "split_type": "subject_held_out",
            "window_length": int(window_length),
            "forecast_horizon": int(forecast_horizon),
            "window_stride": int(stride),
            "sampling_rate_hz": _dataset_sampling_rate_hz(dataset),
            "test_subject_ids": tuple(test_subjects),
            "test_record_ids": tuple(test_records),
            "test_window_starts": tuple(int(v) for v in test_starts),
        },
    )


def _future_windows(
    signal: np.ndarray,
    window_length: int,
    forecast_horizon: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    signal = np.asarray(signal, dtype=np.float32)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    starts: list[int] = []
    last_start = signal.shape[0] - window_length - forecast_horizon + 1
    for start in range(0, max(0, last_start), stride):
        target_start = start + forecast_horizon
        xs.append(signal[start : start + window_length])
        ys.append(signal[target_start : target_start + window_length])
        starts.append(start)
    if not xs:
        return np.empty((0, window_length, signal.shape[-1]), dtype=np.float32), np.empty((0, window_length, signal.shape[-1]), dtype=np.float32), []
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), starts


def _stack(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def _split_subjects(split: SplitManifest) -> dict[str, tuple[str, ...]]:
    return {
        name: tuple(sorted({record.subject_id for record in records}))
        for name, records in (("train", split.train), ("val", split.val), ("test", split.test))
    }


def _benchmark_status(dataset: EEGV1Dataset) -> str:
    if dataset.dataset_id == "hbn_eeg" and dataset.source == "hbn_eeg_local":
        return "local_manifest_not_public_hbn_benchmark"
    if dataset.source == "synthetic_fixture":
        return "synthetic_fixture_not_public_benchmark"
    return "user_provided_local_data"


def _dataset_sampling_rate_hz(dataset: EEGV1Dataset) -> float | None:
    values: set[float] = set()
    for record in dataset.records:
        value = record.metadata.get("sampling_rate")
        if value is None:
            continue
        try:
            rate = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(rate) and rate > 0.0:
            values.add(rate)
    return values.pop() if len(values) == 1 else None


def _stable_projection(n_channels: int) -> np.ndarray:
    rng = np.random.default_rng(20260618)
    return rng.normal(size=(4, n_channels)).astype(np.float32)


def _synthetic_signal(
    *,
    seed: int,
    subject_idx: int,
    session_idx: int,
    n_time: int,
    projection: np.ndarray,
    subject_gain: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed * 1009 + subject_idx * 37 + session_idx)
    latent = rng.normal(scale=0.3, size=(n_time, 4)).astype(np.float32)
    for t in range(1, n_time):
        latent[t] += 0.72 * latent[t - 1]
    time = np.arange(n_time, dtype=np.float32)
    latent[:, 0] += np.sin(time / 5.0 + subject_idx * 0.17)
    latent[:, 1] += np.cos(time / 9.0 + session_idx * 0.11)
    signal = subject_gain * (latent @ projection)
    signal += rng.normal(scale=0.03, size=signal.shape).astype(np.float32)
    return np.ascontiguousarray(signal, dtype=np.float32)
