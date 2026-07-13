"""Manifest-driven Sleep-EDF Cassette ingestion for the HNPH Phase-0 path."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

from neurotwin.adapters.edf_common import (
    EdfAnnotationHeader,
    EdfHeaderRecord,
    build_edf_data_card,
    build_physical_record_from_edf,
    read_edf_header,
)
from neurotwin.data.split_manifest import RecordingRecord, SplitManifest, build_split_manifest
from neurotwin.forecastability import PhysicalRecordRegistry, QualityInterval, TransitionEpoch


SLEEP_EDF_DATASET_ID = "sleep-edf-expanded-sleep-cassette"
SLEEP_EDF_CADENCE_S = 30.0
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RK_TO_MACROSTATE = {
    "sleep stage w": "Wake",
    "sleep stage 1": "NREM",
    "sleep stage 2": "NREM",
    "sleep stage 3": "NREM",
    "sleep stage 4": "NREM",
    "sleep stage r": "REM",
    "sleep stage ?": "Unknown",
    "movement time": "Unknown",
}


class SleepEdfAdapterError(ValueError):
    """Raised when an explicit Sleep-EDF pair cannot support the frozen contract."""


@dataclass(frozen=True)
class SleepEdfPair:
    """One predeclared PSG/hypnogram pair; paths are used locally and never emitted."""

    record_id: str
    subject_id: str
    session_id: str
    psg_path: str | Path
    hypnogram_path: str | Path
    raw_source_uri: str
    annotation_uri: str
    eeg_labels: tuple[str, ...] = ("EEG Fpz-Cz", "EEG Pz-Oz")
    expected_psg_sha256: str | None = None
    expected_hypnogram_sha256: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("record_id", self.record_id),
            ("subject_id", self.subject_id),
            ("session_id", self.session_id),
            ("raw_source_uri", self.raw_source_uri),
            ("annotation_uri", self.annotation_uri),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not self.eeg_labels or len(set(self.eeg_labels)) != len(self.eeg_labels):
            raise ValueError("eeg_labels must be a non-empty, duplicate-free selection")
        for name, value in (
            ("expected_psg_sha256", self.expected_psg_sha256),
            ("expected_hypnogram_sha256", self.expected_hypnogram_sha256),
        ):
            if value is not None and not _SHA256.fullmatch(value.lower()):
                raise ValueError(f"{name} must be a 64-character SHA-256 value when provided")


@dataclass(frozen=True)
class SleepEdfRegistryBundle:
    registry: PhysicalRecordRegistry
    stage_epochs: tuple[TransitionEpoch, ...]
    data_cards: tuple[dict[str, Any], ...]
    recording_records: tuple[RecordingRecord, ...]


def map_sleep_edf_rk_annotation(text: str) -> str:
    """Map declared Sleep-EDF R&K labels without silently inventing ontology."""

    normalized = " ".join(str(text).strip().lower().split())
    try:
        return _RK_TO_MACROSTATE[normalized]
    except KeyError as exc:
        raise SleepEdfAdapterError(f"unsupported Sleep-EDF R&K annotation {text!r}") from exc


def build_sleep_edf_stage_epochs(
    record_id: str,
    annotations: Iterable[EdfAnnotationHeader],
    *,
    duration_s: float,
    cadence_s: float = SLEEP_EDF_CADENCE_S,
) -> tuple[TransitionEpoch, ...]:
    """Expand R&K annotations onto the frozen natural 30-second evaluation grid."""

    if not math.isfinite(float(duration_s)) or duration_s <= 0:
        raise SleepEdfAdapterError("PSG duration must be finite and positive")
    if not math.isfinite(float(cadence_s)) or cadence_s <= 0:
        raise SleepEdfAdapterError("cadence must be finite and positive")
    epoch_count = round(duration_s / cadence_s)
    if not math.isclose(duration_s, epoch_count * cadence_s, rel_tol=0.0, abs_tol=1e-6):
        raise SleepEdfAdapterError("PSG duration must align to the frozen 30-second epoch grid")
    states = ["Unknown"] * epoch_count
    for annotation in annotations:
        macrostate = map_sleep_edf_rk_annotation(annotation.text)
        duration = cadence_s if annotation.duration_s is None else annotation.duration_s
        if not math.isfinite(float(duration)) or duration <= 0:
            raise SleepEdfAdapterError("Sleep-EDF stage annotation duration must be positive")
        start = round(annotation.onset_s / cadence_s)
        count = round(duration / cadence_s)
        if not math.isclose(annotation.onset_s, start * cadence_s, rel_tol=0.0, abs_tol=1e-6):
            raise SleepEdfAdapterError("Sleep-EDF stage annotation onset is off the natural epoch grid")
        if not math.isclose(duration, count * cadence_s, rel_tol=0.0, abs_tol=1e-6):
            raise SleepEdfAdapterError("Sleep-EDF stage annotation duration is off the natural epoch grid")
        if start < 0 or count <= 0 or start + count > epoch_count:
            raise SleepEdfAdapterError("Sleep-EDF stage annotation extends outside PSG duration")
        for index in range(start, start + count):
            if states[index] != "Unknown":
                raise SleepEdfAdapterError("Sleep-EDF stage annotations overlap on the natural grid")
            states[index] = macrostate
    return tuple(
        TransitionEpoch(record_id=record_id, start_s=index * cadence_s, macrostate=state)
        for index, state in enumerate(states)
    )


def build_sleep_edf_registry(
    pairs: Iterable[SleepEdfPair],
    *,
    reader: Callable[[Path], Any] | None = None,
) -> SleepEdfRegistryBundle:
    """Build physical records, redacted cards, and person-split records from an explicit manifest."""

    pair_rows = tuple(pairs)
    if not pair_rows:
        raise SleepEdfAdapterError("Sleep-EDF registry requires at least one declared PSG/hypnogram pair")
    _validate_pairs(pair_rows)
    records = []
    stage_epochs: list[TransitionEpoch] = []
    data_cards: list[dict[str, Any]] = []
    recording_records: list[RecordingRecord] = []
    for pair in pair_rows:
        _validate_pair_files(pair)
        psg = read_edf_header(pair.psg_path, reader=reader)
        hypnogram = read_edf_header(pair.hypnogram_path, reader=reader, require_signals=False)
        _verify_hashes(pair, psg, hypnogram)
        _validate_clock(psg, hypnogram)
        epochs = build_sleep_edf_stage_epochs(pair.record_id, hypnogram.annotations, duration_s=psg.duration_s)
        record = build_physical_record_from_edf(
            psg,
            record_id=pair.record_id,
            subject_id=pair.subject_id,
            session_id=pair.session_id,
            dataset_id=SLEEP_EDF_DATASET_ID,
            site_id=None,
            raw_source_uri=pair.raw_source_uri,
            annotation_uri=pair.annotation_uri,
            quality_intervals=(QualityInterval(0.0, psg.duration_s, "valid"),),
            include_labels=pair.eeg_labels,
        )
        card = build_edf_data_card(psg, record)
        card.update(
            {
                "sleep_ontology": "R&K_to_Wake_NREM_REM_Unknown_v1",
                "annotation_sha256": hypnogram.source_sha256,
                "stage_epoch_count": len(epochs),
                "unknown_stage_epoch_count": sum(epoch.macrostate == "Unknown" for epoch in epochs),
            }
        )
        records.append(record)
        stage_epochs.extend(epochs)
        data_cards.append(card)
        recording_records.append(
            RecordingRecord(
                record_id=record.record_id,
                modality=record.modality,
                dataset=record.dataset_id,
                subject_id=record.subject_id,
                session_id=record.session_id,
                site_id=record.site_id or "unknown_site",
                start_time=0.0,
                end_time=record.duration_s,
                metadata=record.manifest_metadata(),
            )
        )
    return SleepEdfRegistryBundle(
        registry=PhysicalRecordRegistry.from_records(records),
        stage_epochs=tuple(stage_epochs),
        data_cards=tuple(data_cards),
        recording_records=tuple(recording_records),
    )


def build_sleep_edf_person_split(bundle: SleepEdfRegistryBundle, *, seed: int) -> SplitManifest:
    """Split declared recordings by person before target construction or windowing."""

    return build_split_manifest(bundle.recording_records, policy="subject", seed=seed)


def _validate_pairs(pairs: tuple[SleepEdfPair, ...]) -> None:
    record_ids = [pair.record_id for pair in pairs]
    if len(set(record_ids)) != len(record_ids):
        raise SleepEdfAdapterError("Sleep-EDF manifest contains duplicate record IDs")
    subject_sessions = [(pair.subject_id, pair.session_id) for pair in pairs]
    if len(set(subject_sessions)) != len(subject_sessions):
        raise SleepEdfAdapterError("Sleep-EDF manifest contains duplicate subject/session pairs")


def _validate_pair_files(pair: SleepEdfPair) -> None:
    psg = Path(pair.psg_path)
    hypnogram = Path(pair.hypnogram_path)
    if not psg.is_file() or not hypnogram.is_file():
        raise FileNotFoundError("declared Sleep-EDF PSG and hypnogram files must both exist")
    if not psg.name.endswith("-PSG.edf") or not hypnogram.name.endswith("-Hypnogram.edf"):
        raise SleepEdfAdapterError("Sleep-EDF pair must use PSG and Hypnogram EDF filenames")


def _verify_hashes(pair: SleepEdfPair, psg: EdfHeaderRecord, hypnogram: EdfHeaderRecord) -> None:
    if pair.expected_psg_sha256 is not None and pair.expected_psg_sha256.lower() != psg.source_sha256:
        raise SleepEdfAdapterError("PSG SHA-256 does not match the declared manifest")
    if pair.expected_hypnogram_sha256 is not None and pair.expected_hypnogram_sha256.lower() != hypnogram.source_sha256:
        raise SleepEdfAdapterError("hypnogram SHA-256 does not match the declared manifest")


def _validate_clock(psg: EdfHeaderRecord, hypnogram: EdfHeaderRecord) -> None:
    if not math.isclose(psg.duration_s, hypnogram.duration_s, rel_tol=0.0, abs_tol=SLEEP_EDF_CADENCE_S):
        raise SleepEdfAdapterError("PSG and hypnogram clocks disagree by more than one epoch")
    for annotation in hypnogram.annotations:
        duration = SLEEP_EDF_CADENCE_S if annotation.duration_s is None else annotation.duration_s
        if annotation.onset_s + duration > psg.duration_s + 1e-6:
            raise SleepEdfAdapterError("hypnogram annotation extends beyond PSG clock")
