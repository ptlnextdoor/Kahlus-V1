"""Causal forecast-anchor contracts and fail-closed leakage audits for HNPH."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal, Mapping

from neurotwin.forecastability.contracts import PhysicalSignalRecord
from neurotwin.forecastability.registry import PhysicalRecordRegistry


FORBIDDEN_MODEL_INPUT_FIELDS = frozenset(
    {
        "subject_id",
        "session_id",
        "dataset_id",
        "site_id",
        "record_id",
        "source_record_id",
        "path",
        "raw_source_uri",
        "source_sha256",
        "annotation_uri",
        "future_label",
        "target_label",
        "diagnosis",
    }
)


@dataclass(frozen=True)
class TimeInterval:
    """A half-open interval in physical seconds from a recording start."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.start_s)) or not math.isfinite(float(self.end_s)):
            raise ValueError("interval boundaries must be finite")
        if float(self.start_s) < 0 or float(self.end_s) <= float(self.start_s):
            raise ValueError("interval must satisfy 0 <= start_s < end_s")

    def overlaps(self, other: "TimeInterval") -> bool:
        return max(self.start_s, other.start_s) < min(self.end_s, other.end_s)

    def is_contained_by(self, support: "TimeInterval") -> bool:
        return support.start_s <= self.start_s and self.end_s <= support.end_s


@dataclass(frozen=True)
class ForecastAnchor:
    """One causal context-target pair with an explicit physical-time firebreak."""

    anchor_id: str
    record_id: str
    context: TimeInterval
    target: TimeInterval
    filter_guard_s: float

    def __post_init__(self) -> None:
        if not self.anchor_id.strip() or not self.record_id.strip():
            raise ValueError("anchor_id and record_id must be non-empty strings")
        if not math.isfinite(float(self.filter_guard_s)) or float(self.filter_guard_s) < 0:
            raise ValueError("filter_guard_s must be finite and >= 0")
        if self.target.start_s < self.context.end_s + self.filter_guard_s:
            raise ValueError("target must begin after context end plus filter_guard_s")

    @property
    def guard(self) -> TimeInterval | None:
        if self.filter_guard_s == 0:
            return None
        return TimeInterval(self.context.end_s, self.context.end_s + self.filter_guard_s)


@dataclass(frozen=True)
class CausalPreprocessingSpec:
    """Declares preprocessing constraints that prevent future-record leakage."""

    normalization_scope: Literal["context_only", "train_only"]
    filter_mode: Literal["causal", "none"]
    filter_guard_s: float

    def __post_init__(self) -> None:
        if self.normalization_scope not in {"context_only", "train_only"}:
            raise ValueError("normalization_scope must be context_only or train_only")
        if self.filter_mode not in {"causal", "none"}:
            raise ValueError("filter_mode must be causal or none")
        if not math.isfinite(float(self.filter_guard_s)) or float(self.filter_guard_s) < 0:
            raise ValueError("filter_guard_s must be finite and >= 0")
        if self.filter_mode == "none" and self.filter_guard_s != 0:
            raise ValueError("filter_guard_s must be zero when filter_mode is none")


@dataclass(frozen=True)
class FirebreakAuditReport:
    passed: bool
    violations: tuple[str, ...]
    checked_anchor_count: int
    checked_model_input_fields: tuple[str, ...]


def audit_forecast_firebreak(
    anchors: Iterable[ForecastAnchor],
    registry: PhysicalRecordRegistry,
    preprocessing: CausalPreprocessingSpec,
) -> FirebreakAuditReport:
    """Audit anchor support, uniqueness, and causal preprocessing declarations."""

    violations: list[str] = []
    seen_anchor_ids: set[str] = set()
    anchor_list = tuple(anchors)
    for anchor in anchor_list:
        if anchor.anchor_id in seen_anchor_ids:
            violations.append(f"duplicate anchor_id {anchor.anchor_id!r}")
            continue
        seen_anchor_ids.add(anchor.anchor_id)
        try:
            record = registry.by_id(anchor.record_id)
        except KeyError:
            violations.append(f"anchor {anchor.anchor_id!r} references unknown record {anchor.record_id!r}")
            continue
        violations.extend(_anchor_support_violations(anchor, record))
        if preprocessing.filter_mode == "causal" and anchor.filter_guard_s < preprocessing.filter_guard_s:
            violations.append(
                f"anchor {anchor.anchor_id!r} filter guard {anchor.filter_guard_s:g}s "
                f"is shorter than preprocessing guard {preprocessing.filter_guard_s:g}s"
            )
    return FirebreakAuditReport(
        passed=not violations,
        violations=tuple(violations),
        checked_anchor_count=len(anchor_list),
        checked_model_input_fields=(),
    )


def audit_model_input_fields(field_names: Iterable[str]) -> FirebreakAuditReport:
    """Reject metadata fields that encode identity, paths, provenance, or targets."""

    fields = tuple(str(field_name) for field_name in field_names)
    violations = [
        f"forbidden model input field {field!r}"
        for field in fields
        if field.lower() in FORBIDDEN_MODEL_INPUT_FIELDS
    ]
    return FirebreakAuditReport(
        passed=not violations,
        violations=tuple(violations),
        checked_anchor_count=0,
        checked_model_input_fields=fields,
    )


def audit_model_input_mapping(mapping: Mapping[str, object]) -> FirebreakAuditReport:
    """Convenience wrapper for a concrete model-input metadata mapping."""

    return audit_model_input_fields(mapping.keys())


def _anchor_support_violations(anchor: ForecastAnchor, record: PhysicalSignalRecord) -> list[str]:
    supports = tuple(TimeInterval(start_s=start_s, end_s=end_s) for start_s, end_s in record.valid_intervals_s)
    if not supports:
        return [f"record {record.record_id!r} has no valid intervals"]
    violations: list[str] = []
    if not any(anchor.context.is_contained_by(support) for support in supports):
        violations.append(f"anchor {anchor.anchor_id!r} context is outside a declared valid interval")
    if not any(anchor.target.is_contained_by(support) for support in supports):
        violations.append(f"anchor {anchor.anchor_id!r} target is outside a declared valid interval")
    guard = anchor.guard
    if guard is not None and not any(guard.is_contained_by(support) for support in supports):
        violations.append(f"anchor {anchor.anchor_id!r} filter guard is outside a declared valid interval")
    return violations
