"""Minimal, physical-metadata-first EDF ingestion for HNPH adapters.

The module intentionally reads headers and EDF+ annotations without choosing a
dataset-specific montage, sleep ontology, resampling rule, or model target.
Dataset adapters own those later policies.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurotwin.forecastability.contracts import LeadGeometry, PhysicalSignalRecord, QualityInterval
from neurotwin.repro import hash_file, write_json


UNIT_ALIASES = {
    "v": "V",
    "mv": "mV",
    "uv": "uV",
    "nv": "nV",
}


class EdfReadError(ValueError):
    """Raised when EDF headers cannot support an auditable physical record."""


@dataclass(frozen=True)
class EdfSignalHeader:
    label: str
    sampling_rate_hz: float
    raw_physical_unit: str
    canonical_physical_unit: str | None


@dataclass(frozen=True)
class EdfAnnotationHeader:
    onset_s: float
    duration_s: float | None
    text: str


@dataclass(frozen=True)
class EdfHeaderRecord:
    source_sha256: str
    duration_s: float
    signals: tuple[EdfSignalHeader, ...]
    annotations: tuple[EdfAnnotationHeader, ...]

    def signal_by_label(self) -> dict[str, EdfSignalHeader]:
        return {signal.label: signal for signal in self.signals}


def normalize_edf_physical_unit(value: str) -> str | None:
    """Normalize known voltage units without guessing unknown physical dimensions."""

    normalized = value.strip().replace("µ", "u").replace("μ", "u").lower()
    return UNIT_ALIASES.get(normalized)


def read_edf_header(
    path: str | Path,
    *,
    reader: Callable[[Path], Any] | None = None,
) -> EdfHeaderRecord:
    """Read EDF header metadata and EDF+ annotations without materializing samples."""

    source_path = Path(path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"EDF source does not exist or is not a file: {source_path}")
    edf = (reader or _default_edf_reader)(source_path)
    signals = tuple(_signal_header(signal) for signal in tuple(getattr(edf, "signals", ())))
    if not signals:
        raise EdfReadError("EDF contains no ordinary signals")
    labels = [signal.label for signal in signals]
    if len(set(labels)) != len(labels):
        raise EdfReadError("EDF contains duplicate ordinary signal labels")
    duration_s = float(getattr(edf, "duration"))
    if duration_s <= 0:
        raise EdfReadError("EDF duration must be positive")
    annotations = tuple(_annotation_header(annotation) for annotation in tuple(getattr(edf, "annotations", ())))
    return EdfHeaderRecord(
        source_sha256=hash_file(source_path),
        duration_s=duration_s,
        signals=signals,
        annotations=annotations,
    )


def build_physical_record_from_edf(
    header: EdfHeaderRecord,
    *,
    record_id: str,
    subject_id: str,
    session_id: str,
    dataset_id: str,
    site_id: str | None,
    raw_source_uri: str,
    annotation_uri: str | None,
    quality_intervals: Iterable[QualityInterval],
    include_labels: Iterable[str] | None = None,
    lead_geometry: Mapping[str, LeadGeometry] | None = None,
    modality: str = "eeg",
) -> PhysicalSignalRecord:
    """Build a physical record only when unit/rate and quality choices are explicit."""

    headers_by_label = header.signal_by_label()
    selected_labels = tuple(include_labels) if include_labels is not None else tuple(headers_by_label)
    if not selected_labels:
        raise EdfReadError("at least one EDF signal label must be selected")
    if len(set(selected_labels)) != len(selected_labels):
        raise EdfReadError("selected EDF signal labels must be unique")
    missing = [label for label in selected_labels if label not in headers_by_label]
    if missing:
        raise EdfReadError(f"selected EDF signal labels are missing: {missing}")
    selected = tuple(headers_by_label[label] for label in selected_labels)
    units = {signal.canonical_physical_unit for signal in selected}
    if None in units or len(units) != 1:
        raise EdfReadError("selected EDF signals must have one recognized, shared voltage unit")
    sampling_rates = {signal.sampling_rate_hz for signal in selected}
    if len(sampling_rates) != 1:
        raise EdfReadError("selected EDF signals must have one shared sampling rate before resampling")
    geometry = dict(lead_geometry or {})
    leads = tuple(
        geometry.get(
            signal.label,
            LeadGeometry(
                lead_id=signal.label,
                positive_xyz_m=None,
                negative_xyz_m=None,
                reference_kind="unspecified",
                position_source="unavailable",
            ),
        )
        for signal in selected
    )
    return PhysicalSignalRecord(
        record_id=record_id,
        subject_id=subject_id,
        session_id=session_id,
        dataset_id=dataset_id,
        site_id=site_id,
        modality=modality,
        sampling_rate_hz=next(iter(sampling_rates)),
        physical_unit=next(iter(units)),
        duration_s=header.duration_s,
        leads=leads,
        quality_intervals=tuple(quality_intervals),
        raw_source_uri=raw_source_uri,
        source_sha256=header.source_sha256,
        annotation_uri=annotation_uri,
    )


def build_edf_data_card(header: EdfHeaderRecord, record: PhysicalSignalRecord) -> dict[str, Any]:
    """Build a redacted data card from a header and its physical-record contract."""

    if record.source_sha256 != header.source_sha256:
        raise EdfReadError("physical record source checksum does not match the EDF header checksum")
    return {
        "schema_version": "hnph_edf_data_card_v1",
        "record_id": record.record_id,
        "dataset_id": record.dataset_id,
        "modality": record.modality,
        "duration_s": record.duration_s,
        "sampling_rate_hz": record.sampling_rate_hz,
        "physical_unit": record.physical_unit,
        "lead_count": len(record.leads),
        "lead_ids": [lead.lead_id for lead in record.leads],
        "annotation_count": len(header.annotations),
        "raw_source_uri": record.raw_source_uri,
        "source_sha256": record.source_sha256,
        "annotation_uri": record.annotation_uri,
        "quality_interval_count": len(record.quality_intervals),
        "valid_intervals_s": [[start_s, end_s] for start_s, end_s in record.valid_intervals_s],
        "local_source_path_recorded": False,
    }


def write_edf_data_card(path: str | Path, header: EdfHeaderRecord, record: PhysicalSignalRecord) -> Path:
    """Write a derived JSON data card; raw EDF samples and local paths are excluded."""

    return write_json(path, build_edf_data_card(header, record))


def _default_edf_reader(path: Path) -> Any:
    try:
        import edfio
    except ImportError as exc:  # pragma: no cover - depends on optional installation.
        raise RuntimeError("EDF ingestion requires the optional dependency: pip install 'neurotwin[edf]'") from exc
    return edfio.read_edf(path, lazy_load_data=True)


def _signal_header(signal: Any) -> EdfSignalHeader:
    label = str(getattr(signal, "label", "")).strip()
    if not label:
        raise EdfReadError("EDF signal label must be non-empty")
    sampling_rate_hz = float(getattr(signal, "sampling_frequency"))
    if sampling_rate_hz <= 0:
        raise EdfReadError(f"EDF signal {label!r} has non-positive sampling frequency")
    raw_unit = str(getattr(signal, "physical_dimension", "")).strip()
    return EdfSignalHeader(
        label=label,
        sampling_rate_hz=sampling_rate_hz,
        raw_physical_unit=raw_unit,
        canonical_physical_unit=normalize_edf_physical_unit(raw_unit),
    )


def _annotation_header(annotation: Any) -> EdfAnnotationHeader:
    onset_s = float(getattr(annotation, "onset"))
    duration = getattr(annotation, "duration")
    duration_s = None if duration is None else float(duration)
    if onset_s < 0 or duration_s is not None and duration_s < 0:
        raise EdfReadError("EDF annotations must have non-negative onset and duration")
    return EdfAnnotationHeader(onset_s=onset_s, duration_s=duration_s, text=str(getattr(annotation, "text", "")))
