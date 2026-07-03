from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from neurotwin.data.schemas import NeuralEventBatch
from neurotwin.data.split_manifest import RecordingRecord, build_split_manifest
from neurotwin.eeg_v1.dataset import EEGV1Dataset

HBN_MISSING_ROOT = "HBN-EEG data root not found. Provide --data-root."


def load_hbn_eeg_local_dataset(data_root: str | Path | None, *, seed: int = 0) -> EEGV1Dataset:
    """Load a local HBN-style EEG manifest.

    This adapter deliberately supports only user-provided local files. Expected manifest:
    ``manifest.jsonl`` with one JSON object per recording and a relative or absolute ``path``
    to a ``.npy`` array or ``.npz`` containing ``signal`` shaped ``[time, channels]``.
    """

    if data_root is None:
        raise FileNotFoundError(HBN_MISSING_ROOT)
    root = Path(data_root).resolve()
    if not root.exists():
        raise FileNotFoundError(HBN_MISSING_ROOT)
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"HBN-EEG local manifest not found: {manifest_path}")

    batches: list[NeuralEventBatch] = []
    records: list[RecordingRecord] = []
    expected_channels: int | None = None
    expected_channel_names: tuple[str, ...] | None = None
    expected_sampling_rate: float | None = None
    seen_record_ids: set[str] = set()
    for idx, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = _parse_manifest_row(line, line_number=idx + 1)
        signal_path = _resolve_path(root, row.get("path"))
        signal = _load_signal(signal_path)
        channel_names = _channel_names(row, n_channels=signal.shape[1], line_number=idx + 1)
        if expected_channels is None:
            expected_channels = int(signal.shape[1])
        elif signal.shape[1] != expected_channels:
            raise ValueError(
                "HBN-EEG manifest must use a consistent channel count "
                f"(expected {expected_channels}, got {signal.shape[1]} on line {idx + 1})"
            )
        if channel_names is not None:
            if expected_channel_names is None:
                expected_channel_names = channel_names
            elif channel_names != expected_channel_names:
                raise ValueError(f"HBN-EEG manifest must use consistent channel_names on line {idx + 1}")
        subject_id = _required_text(row, "subject_id")
        session_id = _optional_text(row, "session_id", default="ses-00")
        site_id = _optional_text(row, "site_id", default="hbn_local")
        record_id = _optional_text(row, "record_id", default=f"hbn_eeg_{subject_id}_{session_id}_{idx:05d}")
        if record_id in seen_record_ids:
            raise ValueError(f"HBN-EEG manifest duplicate record_id on line {idx + 1}: {record_id}")
        seen_record_ids.add(record_id)
        sampling_rate = _sampling_rate(row.get("sampling_rate", 1.0))
        if expected_sampling_rate is None:
            expected_sampling_rate = sampling_rate
        elif sampling_rate != expected_sampling_rate:
            raise ValueError(
                "HBN-EEG manifest must use a consistent sampling_rate "
                f"(expected {expected_sampling_rate}, got {sampling_rate} on line {idx + 1})"
            )
        time = np.arange(signal.shape[0], dtype=np.float32) / sampling_rate
        metadata: dict[str, Any] = {
            "record_id": record_id,
            "source_record_id": record_id,
            "adapter": "hbn_eeg_local",
            "sampling_rate": sampling_rate,
            "path": str(signal_path),
        }
        if channel_names is not None:
            metadata["channel_names"] = list(channel_names)
        records.append(
            RecordingRecord(
                record_id=record_id,
                modality="eeg",
                dataset="hbn_eeg",
                subject_id=subject_id,
                session_id=session_id,
                site_id=site_id,
                start_time=float(time[0]) if time.size else 0.0,
                end_time=float(time[-1]) if time.size else 0.0,
                path=str(signal_path),
                metadata=metadata,
            )
        )
        batches.append(
            NeuralEventBatch(
                modality="eeg",
                dataset="hbn_eeg",
                subject_id=subject_id,
                session_id=session_id,
                site_id=site_id,
                time=time,
                signal=signal,
                mask=np.ones_like(signal, dtype=bool),
                stimulus_embedding=None,
                behavior={},
                space_index=np.arange(signal.shape[1]),
                provenance={"adapter": "hbn_eeg_local"},
                metadata=metadata,
            )
        )
    if not records:
        raise ValueError(f"HBN-EEG local manifest is empty: {manifest_path}")
    split = build_split_manifest(records, policy="subject", seed=seed)
    return EEGV1Dataset(
        dataset_id="hbn_eeg",
        batches=tuple(batches),
        records=tuple(records),
        split_manifest=split,
        split_subjects={
            name: tuple(sorted({record.subject_id for record in recs}))
            for name, recs in (("train", split.train), ("val", split.val), ("test", split.test))
        },
        source="hbn_eeg_local",
    )


def _parse_manifest_row(line: str, *, line_number: int) -> dict[str, Any]:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"HBN-EEG manifest line {line_number} is not valid JSON") from exc
    if not isinstance(row, dict):
        raise ValueError(f"HBN-EEG manifest line {line_number} must be a JSON object")
    return row


def _resolve_path(root: Path, value: Any) -> Path:
    if not value:
        raise ValueError("HBN-EEG manifest row missing path")
    path = Path(str(value))
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not path.is_absolute() and not _is_relative_to(resolved, root):
        raise ValueError(f"HBN-EEG manifest path escapes data root: {value}")
    return resolved


def _required_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"HBN-EEG manifest row missing {field}")
    return text


def _optional_text(row: dict[str, Any], field: str, *, default: str) -> str:
    if field not in row or row[field] is None:
        return default
    text = str(row[field]).strip()
    if not text:
        raise ValueError(f"HBN-EEG manifest row missing {field}")
    return text


def _channel_names(row: dict[str, Any], *, n_channels: int, line_number: int) -> tuple[str, ...] | None:
    if "channel_names" not in row or row["channel_names"] is None:
        return None
    value = row["channel_names"]
    if not isinstance(value, list):
        raise ValueError(f"HBN-EEG manifest channel_names must be a JSON array on line {line_number}")
    if len(value) != n_channels:
        raise ValueError(
            f"HBN-EEG manifest channel_names length must match signal channels on line {line_number}"
        )
    names = tuple(str(item).strip() for item in value)
    if any(not name for name in names):
        raise ValueError(f"HBN-EEG manifest channel_names must be nonempty strings on line {line_number}")
    return names


def _load_signal(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"HBN-EEG signal file not found: {path}")
    if path.suffix not in {".npy", ".npz"}:
        raise ValueError(f"HBN-EEG unsupported signal file extension: {path.suffix or '<none>'}")
    if path.suffix == ".npz":
        payload = np.load(path)
        if "signal" not in payload:
            raise ValueError(f"HBN-EEG npz file missing 'signal': {path}")
        signal = payload["signal"]
    else:
        signal = np.load(path)
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError(f"HBN-EEG signal must be [time, channels], got {signal.shape}")
    if signal.shape[0] < 1 or signal.shape[1] < 1:
        raise ValueError("HBN-EEG signal must have at least one time sample and one channel")
    if not np.isfinite(signal).all():
        raise ValueError(f"HBN-EEG signal contains NaN or Inf: {path}")
    return np.ascontiguousarray(signal, dtype=np.float32)


def _sampling_rate(value: Any) -> float:
    try:
        sampling_rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("HBN-EEG sampling_rate must be finite and > 0") from exc
    if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise ValueError("HBN-EEG sampling_rate must be finite and > 0")
    return sampling_rate


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
