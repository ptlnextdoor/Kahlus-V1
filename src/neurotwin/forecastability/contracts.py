"""Immutable physical-record and evidence contracts for HNPH.

These contracts deliberately sit beside the historical event-window schema. They
describe a continuous acquisition before splitting, windowing, target extraction,
or model fitting, so later stages can audit what was physically observed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal
from urllib.parse import urlparse


CANONICAL_PHYSICAL_UNITS = frozenset({"V", "mV", "uV", "nV"})
QUALITY_STATES = frozenset({"valid", "artifact", "missing"})
OUTCOME_CLASSES = frozenset(
    {
        "full_pass",
        "dynamics_only_pass",
        "transition_prior_result",
        "within_dataset_only_result",
        "calibrated_null",
        "invalid_experiment",
    }
)


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_finite_positive(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and > 0")


def _require_coordinate(value: tuple[float, float, float] | None, name: str) -> None:
    if value is None:
        return
    if len(value) != 3 or any(not math.isfinite(float(component)) for component in value):
        raise ValueError(f"{name} must contain exactly three finite coordinates")


@dataclass(frozen=True)
class LeadGeometry:
    """Physical definition of one measured lead, without invented coordinates."""

    lead_id: str
    positive_xyz_m: tuple[float, float, float] | None
    negative_xyz_m: tuple[float, float, float] | None
    reference_kind: str
    position_source: str

    def __post_init__(self) -> None:
        _require_text(self.lead_id, "lead_id")
        _require_text(self.reference_kind, "reference_kind")
        _require_text(self.position_source, "position_source")
        _require_coordinate(self.positive_xyz_m, "positive_xyz_m")
        _require_coordinate(self.negative_xyz_m, "negative_xyz_m")


@dataclass(frozen=True)
class QualityInterval:
    """A non-overlapping physical-time quality segment for one recording."""

    start_s: float
    end_s: float
    state: Literal["valid", "artifact", "missing"]
    reason: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.start_s)) or not math.isfinite(float(self.end_s)):
            raise ValueError("quality interval boundaries must be finite")
        if float(self.start_s) < 0 or float(self.end_s) <= float(self.start_s):
            raise ValueError("quality interval must satisfy 0 <= start_s < end_s")
        if self.state not in QUALITY_STATES:
            supported = ", ".join(sorted(QUALITY_STATES))
            raise ValueError(f"unsupported quality state {self.state!r}; expected one of: {supported}")
        if self.reason is not None:
            _require_text(self.reason, "reason")


@dataclass(frozen=True)
class PhysicalSignalRecord:
    """A continuous physiological acquisition and the metadata needed to audit it."""

    record_id: str
    subject_id: str
    session_id: str
    dataset_id: str
    site_id: str | None
    modality: str
    sampling_rate_hz: float
    physical_unit: str
    duration_s: float
    leads: tuple[LeadGeometry, ...]
    quality_intervals: tuple[QualityInterval, ...]
    raw_source_uri: str
    source_sha256: str | None = None
    annotation_uri: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("record_id", self.record_id),
            ("subject_id", self.subject_id),
            ("session_id", self.session_id),
            ("dataset_id", self.dataset_id),
            ("modality", self.modality),
            ("raw_source_uri", self.raw_source_uri),
        ):
            _require_text(value, name)
        if self.site_id is not None:
            _require_text(self.site_id, "site_id")
        if self.annotation_uri is not None:
            _require_text(self.annotation_uri, "annotation_uri")
        _require_finite_positive(self.sampling_rate_hz, "sampling_rate_hz")
        _require_finite_positive(self.duration_s, "duration_s")
        if self.physical_unit not in CANONICAL_PHYSICAL_UNITS:
            supported = ", ".join(sorted(CANONICAL_PHYSICAL_UNITS))
            raise ValueError(f"physical_unit must be canonical; expected one of: {supported}")
        if not self.leads:
            raise ValueError("leads must contain at least one measured lead")
        lead_ids = [lead.lead_id for lead in self.leads]
        if len(set(lead_ids)) != len(lead_ids):
            raise ValueError("lead IDs must be unique within a physical record")
        self._validate_quality_intervals()
        if self.source_sha256 is not None:
            normalized_hash = self.source_sha256.lower()
            if len(normalized_hash) != 64 or any(character not in "0123456789abcdef" for character in normalized_hash):
                raise ValueError("source_sha256 must be a lowercase or uppercase 64-character hexadecimal digest")

    def _validate_quality_intervals(self) -> None:
        previous_end = 0.0
        for interval in self.quality_intervals:
            if interval.end_s > self.duration_s:
                raise ValueError("quality interval cannot extend beyond duration_s")
            if interval.start_s < previous_end:
                raise ValueError("quality intervals must be ordered and non-overlapping")
            previous_end = interval.end_s

    @property
    def valid_intervals_s(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (interval.start_s, interval.end_s)
            for interval in self.quality_intervals
            if interval.state == "valid"
        )

    def manifest_metadata(self) -> dict[str, Any]:
        """Return a publication-safe sidecar; local source locations stay local."""

        return {
            "physical_record_schema_version": "hnph_physical_record_v1",
            "dataset_id": self.dataset_id,
            "sampling_rate_hz": self.sampling_rate_hz,
            "physical_unit": self.physical_unit,
            "duration_s": self.duration_s,
            "raw_source_uri": _public_source_uri(self.raw_source_uri),
            "source_sha256": self.source_sha256,
            "annotation_uri": _public_source_uri(self.annotation_uri),
            "lead_geometry": [
                {
                    "lead_id": lead.lead_id,
                    "positive_xyz_m": lead.positive_xyz_m,
                    "negative_xyz_m": lead.negative_xyz_m,
                    "reference_kind": lead.reference_kind,
                    "position_source": lead.position_source,
                }
                for lead in self.leads
            ],
            "quality_intervals": [
                {
                    "start_s": interval.start_s,
                    "end_s": interval.end_s,
                    "state": interval.state,
                    "reason": interval.reason,
                }
                for interval in self.quality_intervals
            ],
        }


def _public_source_uri(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlparse(value)
    if parsed.scheme in {"", "file"} or value.startswith(("/", "~", "\\")):
        return None
    if len(value) > 2 and value[1] == ":" and value[2] in {"/", "\\"}:
        return None
    return value


@dataclass(frozen=True)
class StateTargetSpec:
    """Frozen definition of a future state-scale target transform."""

    version: str
    bands_hz: tuple[tuple[float, float], ...]
    target_window_s: float
    include_aperiodic: bool
    include_spatial_covariance: bool
    include_complex_spectrum: bool

    def __post_init__(self) -> None:
        _require_text(self.version, "version")
        _require_finite_positive(self.target_window_s, "target_window_s")
        if not self.bands_hz:
            raise ValueError("bands_hz must contain at least one band")
        previous_high = 0.0
        for low_hz, high_hz in self.bands_hz:
            if not math.isfinite(float(low_hz)) or not math.isfinite(float(high_hz)):
                raise ValueError("state target bands must be finite")
            if float(low_hz) < 0 or float(high_hz) <= float(low_hz):
                raise ValueError("each state target band must satisfy 0 <= low_hz < high_hz")
            if float(low_hz) < previous_high:
                raise ValueError("state target bands must be ordered and non-overlapping")
            previous_high = float(high_hz)


@dataclass(frozen=True)
class EvidenceDecision:
    """Machine-readable evidence gate output with explicit claim boundaries."""

    protocol_version: str
    gate_passed: bool
    outcome_class: str
    failed_requirements: tuple[str, ...]
    allowed_claims: tuple[str, ...]
    blocked_claims: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.protocol_version, "protocol_version")
        if self.outcome_class not in OUTCOME_CLASSES:
            supported = ", ".join(sorted(OUTCOME_CLASSES))
            raise ValueError(f"unsupported outcome_class {self.outcome_class!r}; expected one of: {supported}")
        for field_name, values in (
            ("failed_requirements", self.failed_requirements),
            ("allowed_claims", self.allowed_claims),
            ("blocked_claims", self.blocked_claims),
        ):
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{field_name} must contain only non-empty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} must not contain duplicate values")
        overlap = set(self.allowed_claims) & set(self.blocked_claims)
        if overlap:
            raise ValueError(f"claims cannot be both allowed and blocked: {sorted(overlap)}")
        if self.gate_passed and self.failed_requirements:
            raise ValueError("a passing evidence decision cannot contain failed requirements")
        if not self.gate_passed and self.outcome_class != "invalid_experiment" and not self.failed_requirements:
            raise ValueError("a failed non-invalid decision must identify failed requirements")
