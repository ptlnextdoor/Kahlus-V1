"""Immutable registry for physical HNPH recording contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from neurotwin.forecastability.contracts import LeadGeometry, PhysicalSignalRecord, QualityInterval


@dataclass(frozen=True)
class PhysicalRecordRegistry:
    """A duplicate-free, serialization-safe collection of physical records."""

    records: tuple[PhysicalSignalRecord, ...]
    schema_version: str = "hnph_physical_registry_v1"

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("physical record registry cannot be empty")
        record_ids = [record.record_id for record in self.records]
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("physical record registry contains duplicate record_id values")

    @classmethod
    def from_records(cls, records: Iterable[PhysicalSignalRecord]) -> "PhysicalRecordRegistry":
        return cls(records=tuple(records))

    def by_id(self, record_id: str) -> PhysicalSignalRecord:
        for record in self.records:
            if record.record_id == record_id:
                return record
        raise KeyError(f"physical record not found: {record_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "records": [asdict(record) for record in self.records],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PhysicalRecordRegistry":
        schema_version = str(payload.get("schema_version", ""))
        if schema_version != "hnph_physical_registry_v1":
            raise ValueError(f"unsupported physical registry schema_version {schema_version!r}")
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise ValueError("physical registry records must be a list")
        records = tuple(_record_from_dict(raw_record) for raw_record in raw_records)
        return cls(records=records, schema_version=schema_version)


def _record_from_dict(payload: Any) -> PhysicalSignalRecord:
    if not isinstance(payload, dict):
        raise ValueError("physical registry record must be an object")
    leads = tuple(
        LeadGeometry(
            lead_id=str(lead["lead_id"]),
            positive_xyz_m=_coordinate_or_none(lead.get("positive_xyz_m")),
            negative_xyz_m=_coordinate_or_none(lead.get("negative_xyz_m")),
            reference_kind=str(lead["reference_kind"]),
            position_source=str(lead["position_source"]),
        )
        for lead in _require_object_sequence(payload.get("leads"), "leads")
    )
    quality_intervals = tuple(
        QualityInterval(
            start_s=float(interval["start_s"]),
            end_s=float(interval["end_s"]),
            state=str(interval["state"]),
            reason=interval.get("reason"),
        )
        for interval in _require_object_sequence(payload.get("quality_intervals"), "quality_intervals")
    )
    return PhysicalSignalRecord(
        record_id=str(payload["record_id"]),
        subject_id=str(payload["subject_id"]),
        session_id=str(payload["session_id"]),
        dataset_id=str(payload["dataset_id"]),
        site_id=payload.get("site_id"),
        modality=str(payload["modality"]),
        sampling_rate_hz=float(payload["sampling_rate_hz"]),
        physical_unit=str(payload["physical_unit"]),
        duration_s=float(payload["duration_s"]),
        leads=leads,
        quality_intervals=quality_intervals,
        raw_source_uri=str(payload["raw_source_uri"]),
        source_sha256=payload.get("source_sha256"),
        annotation_uri=payload.get("annotation_uri"),
    )


def _coordinate_or_none(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("lead coordinate must be a three-value list or tuple")
    return (float(value[0]), float(value[1]), float(value[2]))


def _require_object_sequence(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{name} must be a sequence of objects")
    return tuple(value)
