from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from neurotwin.data.schemas import NeuralEventBatch
from neurotwin.repro import hash_file, stable_hash, write_json


EVENT_MANIFEST_SCHEMA = "neurotwin.event_manifest.v2"


def save_event_batches(
    batches: list[NeuralEventBatch],
    out_dir: str | Path,
    manifest_name: str = "event_manifest.json",
    manifest_metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist prepared event batches for offline training/eval jobs."""

    root = Path(out_dir)
    events_dir = root / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, batch in enumerate(batches):
        record_id = str(batch.metadata.get("record_id") or batch.metadata.get("source_record_id") or f"event-{idx:05d}")
        file_path = events_dir / f"{_safe_name(record_id)}.npz"
        np.savez_compressed(
            file_path,
            time=batch.time,
            signal=batch.signal,
            mask=batch.mask,
            space_index=batch.space_index,
            stimulus_embedding=batch.stimulus_embedding if batch.stimulus_embedding is not None else np.asarray([], dtype=np.float32),
            uncertainty=batch.uncertainty if batch.uncertainty is not None else np.asarray([], dtype=np.float32),
            behavior_json=np.asarray(json.dumps(_jsonable(batch.behavior))),
            provenance_json=np.asarray(json.dumps(_jsonable(batch.provenance))),
            metadata_json=np.asarray(json.dumps(_jsonable(batch.metadata))),
        )
        rows.append(
            {
                "record_id": record_id,
                "recording_id": batch.recording_id,
                "modality": batch.modality,
                "dataset": batch.dataset,
                "dataset_id": batch.dataset_id,
                "subject_id": batch.subject_id,
                "session_id": batch.session_id,
                "site_id": batch.site_id,
                "task_id": batch.task_id,
                "sampling_rate": batch.sampling_rate,
                "time_start": batch.time_start,
                "time_end": batch.time_end,
                "source_hash": batch.source_hash,
                "preprocessing_hash": batch.preprocessing_hash,
                "split_assignment": batch.split_assignment,
                "n_time": batch.n_time,
                "n_space": batch.n_space,
                "path": str(file_path.relative_to(root)),
                "sha256": hash_file(file_path),
            }
        )
    return write_json(
        root / manifest_name,
        {
            "schema": EVENT_MANIFEST_SCHEMA,
            "event_count": len(rows),
            "manifest_hash": stable_hash(rows),
            "metadata": _jsonable(manifest_metadata or {}),
            "events": rows,
        },
    )


def load_event_batches(manifest_path: str | Path) -> list[NeuralEventBatch]:
    manifest_file = Path(manifest_path)
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    root = manifest_file.parent
    batches = []
    for row in payload.get("events", []):
        file_path = root / row["path"]
        if hash_file(file_path) != row.get("sha256"):
            raise ValueError(f"Prepared event hash mismatch: {file_path}")
        with np.load(file_path, allow_pickle=False) as data:
            stimulus = data["stimulus_embedding"]
            uncertainty = data["uncertainty"]
            metadata = _loads_json(data["metadata_json"])
            metadata.setdefault("record_id", row["record_id"])
            for key in (
                "recording_id",
                "dataset_id",
                "task_id",
                "sampling_rate",
                "time_start",
                "time_end",
                "source_hash",
                "preprocessing_hash",
                "split_assignment",
            ):
                if key in row and row.get(key) is not None:
                    metadata.setdefault(key, row.get(key))
            batches.append(
                NeuralEventBatch(
                    modality=str(row["modality"]),
                    dataset=str(row["dataset"]),
                    subject_id=str(row["subject_id"]),
                    session_id=str(row["session_id"]),
                    site_id=str(row["site_id"]),
                    time=data["time"],
                    signal=data["signal"],
                    mask=data["mask"],
                    stimulus_embedding=stimulus if stimulus.size else None,
                    behavior=_loads_json(data["behavior_json"]),
                    space_index=data["space_index"],
                    uncertainty=uncertainty if uncertainty.size else None,
                    provenance=_loads_json(data["provenance_json"]),
                    metadata=metadata,
                )
            )
    return batches


def event_manifest_summary(manifest_path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    modalities = sorted({str(row["modality"]) for row in payload.get("events", [])})
    datasets = sorted({str(row["dataset"]) for row in payload.get("events", [])})
    subjects = sorted({str(row["subject_id"]) for row in payload.get("events", [])})
    task_ids = sorted({str(row.get("task_id")) for row in payload.get("events", []) if row.get("task_id") is not None})
    return {
        "schema": payload.get("schema"),
        "event_count": int(payload.get("event_count", 0)),
        "manifest_hash": payload.get("manifest_hash"),
        "metadata": payload.get("metadata", {}),
        "modalities": modalities,
        "datasets": datasets,
        "subjects": subjects,
        "task_ids": task_ids,
    }


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in value)


def _loads_json(value: np.ndarray) -> dict[str, Any]:
    text = str(value.item())
    payload = json.loads(text)
    return dict(payload)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
