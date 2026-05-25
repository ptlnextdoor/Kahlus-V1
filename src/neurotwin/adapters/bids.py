from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from neurotwin.data.schemas import NeuralEventBatch
from neurotwin.data.split_manifest import RecordingRecord


_ENTITY_RE = re.compile(r"(?P<key>sub|ses|task|run)-(?P<value>[^_]+)")


def scan_bids_manifest(root: str | Path, dataset_id: str, site_id: str = "bids") -> list[RecordingRecord]:
    root_path = Path(root)
    participants = _read_tsv(root_path / "participants.tsv", key_field="participant_id")
    records = []
    for file_path in sorted(root_path.rglob("*")):
        if not file_path.is_file() or not _is_bids_signal(file_path):
            continue
        entities = _parse_entities(file_path.name)
        subject_id = f"sub-{entities.get('sub', 'unknown')}"
        session_id = f"ses-{entities.get('ses', 'none')}"
        run_id = f"run-{entities.get('run', 'none')}"
        task = entities.get("task")
        modality = _infer_modality(file_path)
        events = _events_for(file_path)
        scans = _scans_for(root_path, file_path)
        derivative = _timeseries_derivative_for(root_path, file_path)
        rel_path = file_path.relative_to(root_path)
        record_id = f"{dataset_id}_{subject_id}_{session_id}_{task or 'task-none'}_{run_id}_{modality}"
        records.append(
            RecordingRecord(
                record_id=record_id,
                modality=modality,
                dataset=dataset_id,
                subject_id=subject_id,
                session_id=session_id,
                site_id=site_id,
                start_time=0.0,
                end_time=float(len(events) if events else 1.0),
                stimulus_id=task,
                path=str(file_path),
                metadata={
                    "adapter": "bids",
                    "relative_path": str(rel_path),
                    "run_id": run_id,
                    "task": task,
                    "suffix": _suffix(file_path.name),
                    "participants": participants.get(subject_id, {}),
                    "events": events,
                    "scans": scans,
                    "timeseries_derivative": str(derivative.relative_to(root_path)) if derivative else None,
                },
            )
        )
    return records


def records_to_event_batches(records: list[RecordingRecord]) -> list[NeuralEventBatch]:
    batches: list[NeuralEventBatch] = []
    for record in records:
        derivative = record.metadata.get("timeseries_derivative")
        if not derivative:
            continue
        root = Path(record.path).parents[_relative_depth(str(record.metadata.get("relative_path", "")))] if record.path else Path(".")
        derivative_path = root / str(derivative)
        if not derivative_path.exists():
            derivative_path = Path(record.path).with_name(Path(str(derivative)).name) if record.path else derivative_path
        signal, labels = _load_timeseries_derivative(derivative_path)
        if signal.ndim != 2:
            raise ValueError(f"BIDS time-series derivative must be 2D [time, space]: {derivative_path}")
        n_time, n_space = signal.shape
        batches.append(
            NeuralEventBatch(
                modality=record.modality,
                dataset=record.dataset,
                subject_id=record.subject_id,
                session_id=record.session_id,
                site_id=record.site_id,
                time=np.arange(n_time, dtype=np.float32),
                signal=signal.astype(np.float32),
                mask=np.ones_like(signal, dtype=bool),
                stimulus_embedding=None,
                behavior={
                    "task": record.metadata.get("task"),
                    "events": record.metadata.get("events", []),
                },
                space_index=np.arange(n_space),
                provenance={
                    "adapter": "bids",
                    "source_path": record.path,
                    "timeseries_derivative": str(derivative_path),
                    "split_stage": "recording_manifest",
                },
                metadata={
                    "record_id": record.record_id,
                    "source_record_id": record.record_id,
                    "run_id": record.metadata.get("run_id"),
                    "task": record.metadata.get("task"),
                    "space_labels": labels,
                    "timeseries_derivative": str(derivative_path),
                },
            )
        )
    return batches


def _is_bids_signal(path: Path) -> bool:
    name = path.name
    return name.endswith(("_bold.nii", "_bold.nii.gz", "_eeg.set", "_eeg.edf", "_meg.fif"))


def _timeseries_derivative_for(root: Path, signal_path: Path) -> Path | None:
    base = _strip_signal_suffix(signal_path.name)
    for suffix in ("_timeseries.npy", "_timeseries.npz", "_timeseries.tsv", "_timeseries.csv"):
        candidate = signal_path.with_name(base + suffix)
        if candidate.exists():
            return candidate
    relative_parent = signal_path.parent.relative_to(root)
    derivative_parent = root / "derivatives" / "neurotwin" / relative_parent
    for suffix in ("_timeseries.npy", "_timeseries.npz", "_timeseries.tsv", "_timeseries.csv"):
        candidate = derivative_parent / (base + suffix)
        if candidate.exists():
            return candidate
    return None


def _infer_modality(path: Path) -> str:
    name = path.name
    if "_bold" in name:
        return "fmri"
    if "_meg" in name:
        return "meg"
    return "eeg"


def _suffix(name: str) -> str:
    stem = name
    for ext in (".nii.gz", ".nii", ".set", ".edf", ".fif"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    return stem.split("_")[-1]


def _parse_entities(name: str) -> dict[str, str]:
    return {match.group("key"): match.group("value") for match in _ENTITY_RE.finditer(name)}


def _events_for(signal_path: Path) -> list[dict[str, str]]:
    event_path = signal_path.with_name(_strip_signal_suffix(signal_path.name) + "_events.tsv")
    return list(_read_tsv_rows(event_path))


def _scans_for(root: Path, signal_path: Path) -> dict[str, str]:
    scans_path = next((parent / f"{parent.name}_scans.tsv" for parent in [signal_path.parent, *signal_path.parents] if (parent / f"{parent.name}_scans.tsv").exists()), None)
    if scans_path is None:
        return {}
    rel = signal_path.relative_to(scans_path.parent)
    for row in _read_tsv_rows(scans_path):
        if row.get("filename") in {str(rel), str(signal_path.relative_to(root))}:
            return row
    return {}


def _strip_signal_suffix(name: str) -> str:
    for suffix in ("_bold.nii.gz", "_bold.nii", "_eeg.set", "_eeg.edf", "_meg.fif"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _load_timeseries_derivative(path: Path) -> tuple[np.ndarray, list[str]]:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32), []
    if path.suffix == ".npz":
        with np.load(path) as payload:
            key = "signal" if "signal" in payload else payload.files[0]
            labels = _listlike(payload["labels"].tolist()) if "labels" in payload else []
            return np.asarray(payload[key], dtype=np.float32), [str(label) for label in labels]
    if path.suffix in {".tsv", ".csv"}:
        delimiter = "\t" if path.suffix == ".tsv" else ","
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"Empty BIDS time-series derivative: {path}")
        header = lines[0].split(delimiter)
        rows = []
        for line in lines[1:]:
            values = line.split(delimiter)
            rows.append([float(value) for value in values])
        return np.asarray(rows, dtype=np.float32), header
    raise ValueError(f"Unsupported BIDS time-series derivative extension: {path}")


def _relative_depth(relative_path: str) -> int:
    if not relative_path:
        return 0
    return max(0, len(Path(relative_path).parts) - 1)


def _listlike(value: object) -> list[object]:
    if value is None or isinstance(value, str):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return []


def _read_tsv(path: Path, key_field: str) -> dict[str, dict[str, str]]:
    rows = {}
    for row in _read_tsv_rows(path):
        key = row.get(key_field)
        if key:
            rows[key] = row
    return rows


def _read_tsv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        values = line.split("\t")
        rows.append({key: values[idx] if idx < len(values) else "" for idx, key in enumerate(header)})
    return rows
