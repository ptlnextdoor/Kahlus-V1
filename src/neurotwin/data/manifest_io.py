from __future__ import annotations

from pathlib import Path
from typing import Any

from neurotwin.data.leakage import LeakageReport
from neurotwin.data.split_manifest import RecordingRecord, SplitManifest
from neurotwin.repro import write_json


def record_to_dict(record: RecordingRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "modality": record.modality,
        "dataset": record.dataset,
        "subject_id": record.subject_id,
        "session_id": record.session_id,
        "site_id": record.site_id,
        "start_time": record.start_time,
        "end_time": record.end_time,
        "stimulus_id": record.stimulus_id,
        "path": record.path,
        "metadata": record.metadata,
    }


def record_from_dict(payload: dict[str, Any]) -> RecordingRecord:
    return RecordingRecord(
        record_id=str(payload["record_id"]),
        modality=str(payload["modality"]),
        dataset=str(payload["dataset"]),
        subject_id=str(payload["subject_id"]),
        session_id=str(payload["session_id"]),
        site_id=str(payload["site_id"]),
        start_time=float(payload["start_time"]),
        end_time=float(payload["end_time"]),
        stimulus_id=payload.get("stimulus_id"),
        path=payload.get("path"),
        metadata=dict(payload.get("metadata", {})),
    )


def split_manifest_to_dict(manifest: SplitManifest) -> dict[str, Any]:
    return {
        "policy": manifest.policy,
        "seed": manifest.seed,
        "split_stage": manifest.split_stage,
        "notes": manifest.notes,
        "record_hashes": manifest.record_hashes,
        "train": [record_to_dict(record) for record in manifest.train],
        "val": [record_to_dict(record) for record in manifest.val],
        "test": [record_to_dict(record) for record in manifest.test],
    }


def split_manifest_from_dict(payload: dict[str, Any]) -> SplitManifest:
    return SplitManifest(
        policy=str(payload["policy"]),
        seed=int(payload["seed"]),
        train=[record_from_dict(record) for record in payload.get("train", [])],
        val=[record_from_dict(record) for record in payload.get("val", [])],
        test=[record_from_dict(record) for record in payload.get("test", [])],
        record_hashes=dict(payload.get("record_hashes", {})),
        split_stage=str(payload.get("split_stage", "recording_manifest")),
        notes=list(payload.get("notes", [])),
    )


def save_split_manifest(manifest: SplitManifest, path: str | Path) -> Path:
    return write_json(path, split_manifest_to_dict(manifest))


def load_split_manifest(path: str | Path) -> SplitManifest:
    import json

    return split_manifest_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_data_manifest(records: list[RecordingRecord], path: str | Path) -> Path:
    return write_json(path, {"records": [record_to_dict(record) for record in records]})


def save_leakage_report(report: LeakageReport, path: str | Path) -> Path:
    return write_json(
        path,
        {
            "passed": report.passed,
            "violations": report.violations,
            "checked_keys": report.checked_keys,
        },
    )
