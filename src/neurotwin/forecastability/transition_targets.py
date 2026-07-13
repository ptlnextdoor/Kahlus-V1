"""Natural-grid HNPH sleep-transition targets with causal forecast anchors."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Literal

from neurotwin.forecastability.firebreak import ForecastAnchor, TimeInterval


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class TransitionLeadBand:
    """One frozen next-transition lead-time band in seconds."""

    band_id: str
    lower_exclusive_s: float
    upper_inclusive_s: float

    def __post_init__(self) -> None:
        _text(self.band_id, "band_id")
        if not math.isfinite(float(self.lower_exclusive_s)) or not math.isfinite(float(self.upper_inclusive_s)):
            raise ValueError("lead-band boundaries must be finite")
        if self.lower_exclusive_s < 0 or self.upper_inclusive_s <= self.lower_exclusive_s:
            raise ValueError("lead band must satisfy 0 <= lower_exclusive_s < upper_inclusive_s")


@dataclass(frozen=True)
class TransitionTargetSpec:
    """Frozen target semantics for the HNPH natural 30-second epoch grid."""

    version: str
    cadence_s: float
    context_s: float
    stable_destination_epochs: int
    lead_bands: tuple[TransitionLeadBand, ...]
    primary_band_id: str
    ontology: tuple[str, ...] = ("Wake", "NREM", "REM", "Unknown")
    complete_follow_up_required: bool = True
    outcome_mode: Literal["next_stable_transition_with_ambiguity"] = "next_stable_transition_with_ambiguity"
    ambiguous_outcome: str = "Ambiguous"

    def __post_init__(self) -> None:
        _text(self.version, "version")
        _text(self.primary_band_id, "primary_band_id")
        if not math.isfinite(float(self.cadence_s)) or self.cadence_s <= 0:
            raise ValueError("cadence_s must be finite and > 0")
        if not math.isfinite(float(self.context_s)) or self.context_s <= 0:
            raise ValueError("context_s must be finite and > 0")
        context_epochs = self.context_s / self.cadence_s
        if not math.isclose(context_epochs, round(context_epochs), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("context_s must contain an integer number of cadence epochs")
        if self.stable_destination_epochs < 2:
            raise ValueError("stable_destination_epochs must be at least two")
        if not self.lead_bands:
            raise ValueError("lead_bands must not be empty")
        band_ids = [band.band_id for band in self.lead_bands]
        if len(set(band_ids)) != len(band_ids):
            raise ValueError("lead-band IDs must be unique")
        if self.primary_band_id not in band_ids:
            raise ValueError("primary_band_id must identify a frozen lead band")
        previous_upper = 0.0
        for band in self.lead_bands:
            if band.lower_exclusive_s < previous_upper:
                raise ValueError("lead bands must be ordered and non-overlapping")
            previous_upper = band.upper_inclusive_s
        if not self.ontology or len(set(self.ontology)) != len(self.ontology):
            raise ValueError("ontology must contain unique states")
        if self.outcome_mode != "next_stable_transition_with_ambiguity":
            raise ValueError("unsupported transition outcome mode")
        _text(self.ambiguous_outcome, "ambiguous_outcome")
        if self.ambiguous_outcome in {*self.ontology, "no_event"}:
            raise ValueError("ambiguous_outcome must be distinct from ontology states and no_event")


@dataclass(frozen=True)
class TransitionEpoch:
    """One scored macrostate epoch; samples remain outside this metadata contract."""

    record_id: str
    start_s: float
    macrostate: str

    def __post_init__(self) -> None:
        _text(self.record_id, "record_id")
        _text(self.macrostate, "macrostate")
        if not math.isfinite(float(self.start_s)) or self.start_s < 0:
            raise ValueError("start_s must be finite and >= 0")


@dataclass(frozen=True)
class TransitionTarget:
    """One natural-grid categorical next-transition outcome for one lead band."""

    target_id: str
    record_id: str
    anchor: ForecastAnchor
    issue_time_s: float
    current_macrostate: str
    destination_macrostate: str | None
    event_time_s: float | None
    band_id: str
    complete_follow_up: bool
    ambiguous: bool = False
    ambiguous_outcome: str = "Ambiguous"

    def __post_init__(self) -> None:
        for name, value in (
            ("target_id", self.target_id),
            ("record_id", self.record_id),
            ("current_macrostate", self.current_macrostate),
            ("band_id", self.band_id),
        ):
            _text(value, name)
        if self.anchor.record_id != self.record_id:
            raise ValueError("transition target and forecast anchor must share a record_id")
        if not math.isfinite(float(self.issue_time_s)) or self.issue_time_s < 0:
            raise ValueError("issue_time_s must be finite and >= 0")
        if not math.isclose(self.anchor.context.end_s, self.issue_time_s, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("forecast-anchor context must end at the transition issue time")
        if self.ambiguous:
            if self.destination_macrostate is not None or self.event_time_s is None:
                raise ValueError("ambiguous transition targets require a boundary time but no destination")
            _text(self.ambiguous_outcome, "ambiguous_outcome")
        elif (self.destination_macrostate is None) != (self.event_time_s is None):
            raise ValueError("event time and destination must be present or absent together")
        if self.destination_macrostate is not None:
            _text(self.destination_macrostate, "destination_macrostate")
            if self.destination_macrostate == self.current_macrostate:
                raise ValueError("transition destination must differ from the current macrostate")
        if self.event_time_s is not None and not self.issue_time_s <= self.event_time_s <= self.anchor.target.end_s:
            raise ValueError("transition boundary must occur on or after issue time and before lead-band end")

    @property
    def outcome(self) -> str:
        if self.ambiguous:
            return self.ambiguous_outcome
        return self.destination_macrostate or "no_event"


@dataclass(frozen=True)
class TransitionTargetBuildResult:
    """Targets plus explicit right-censoring counts for the frozen lead bands."""

    targets: tuple[TransitionTarget, ...]
    complete_follow_up_excluded_count_by_band: dict[str, int]


def build_natural_transition_targets(
    epochs: Iterable[TransitionEpoch],
    spec: TransitionTargetSpec,
    *,
    filter_guard_s: float = 0.0,
) -> tuple[TransitionTarget, ...]:
    """Build complete-follow-up HNPH targets without event-enriched resampling."""

    return build_natural_transition_target_result(epochs, spec, filter_guard_s=filter_guard_s).targets


def build_natural_transition_target_result(
    epochs: Iterable[TransitionEpoch],
    spec: TransitionTargetSpec,
    *,
    filter_guard_s: float = 0.0,
) -> TransitionTargetBuildResult:
    """Build targets and report each right-censored lead-band exclusion."""

    if not math.isfinite(float(filter_guard_s)) or filter_guard_s < 0:
        raise ValueError("filter_guard_s must be finite and >= 0")
    by_record: dict[str, list[TransitionEpoch]] = defaultdict(list)
    for epoch in epochs:
        if epoch.macrostate not in spec.ontology:
            raise ValueError(f"epoch macrostate {epoch.macrostate!r} is outside the frozen ontology")
        by_record[epoch.record_id].append(epoch)
    targets: list[TransitionTarget] = []
    excluded = {band.band_id: 0 for band in spec.lead_bands}
    for record_id, record_epochs in sorted(by_record.items()):
        ordered = sorted(record_epochs, key=lambda row: row.start_s)
        _validate_natural_grid(ordered, spec.cadence_s)
        record_targets, record_excluded = _targets_for_record(record_id, ordered, spec, filter_guard_s)
        targets.extend(record_targets)
        for band_id, count in record_excluded.items():
            excluded[band_id] += count
    return TransitionTargetBuildResult(
        targets=tuple(targets),
        complete_follow_up_excluded_count_by_band=excluded,
    )


def _validate_natural_grid(epochs: list[TransitionEpoch], cadence_s: float) -> None:
    if not epochs:
        return
    for index, epoch in enumerate(epochs):
        expected_start = index * cadence_s
        if not math.isclose(epoch.start_s, expected_start, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("transition epochs must form a natural zero-based cadence grid")


def _targets_for_record(
    record_id: str,
    epochs: list[TransitionEpoch],
    spec: TransitionTargetSpec,
    filter_guard_s: float,
) -> tuple[list[TransitionTarget], dict[str, int]]:
    context_epochs = int(round(spec.context_s / spec.cadence_s))
    record_end_s = epochs[-1].start_s + spec.cadence_s
    targets: list[TransitionTarget] = []
    excluded = {band.band_id: 0 for band in spec.lead_bands}
    for issue_index in range(context_epochs - 1, len(epochs)):
        current = epochs[issue_index].macrostate
        issue_time_s = epochs[issue_index].start_s + spec.cadence_s
        context = TimeInterval(issue_time_s - spec.context_s, issue_time_s)
        for band in spec.lead_bands:
            target = TimeInterval(
                issue_time_s + band.lower_exclusive_s,
                issue_time_s + band.upper_inclusive_s,
            )
            complete = target.end_s <= record_end_s
            if spec.complete_follow_up_required and not complete:
                excluded[band.band_id] += 1
                continue
            if current == "Unknown":
                event_time_s, destination, ambiguous = issue_time_s, None, True
            else:
                event_time_s, destination, ambiguous = _transition_status(
                    epochs,
                    issue_index,
                    target.start_s,
                    target.end_s,
                    spec,
                )
            anchor = ForecastAnchor(
                anchor_id=f"{record_id}:{issue_index:06d}:{band.band_id}",
                record_id=record_id,
                context=context,
                target=target,
                filter_guard_s=filter_guard_s,
            )
            targets.append(
                TransitionTarget(
                    target_id=anchor.anchor_id,
                    record_id=record_id,
                    anchor=anchor,
                    issue_time_s=issue_time_s,
                    current_macrostate=current,
                    destination_macrostate=destination,
                    event_time_s=event_time_s,
                    band_id=band.band_id,
                    complete_follow_up=complete,
                    ambiguous=ambiguous,
                    ambiguous_outcome=spec.ambiguous_outcome,
                )
            )
    return targets, excluded


def _transition_status(
    epochs: list[TransitionEpoch],
    issue_index: int,
    band_start_s: float,
    band_end_s: float,
    spec: TransitionTargetSpec,
) -> tuple[float | None, str | None, bool]:
    """Return the first stable transition in one band, or an adjudication ambiguity.

    A failed one-epoch candidate is not itself ambiguous: it is simply not a
    stable transition.  Unknown annotation segments and an unresolved candidate
    that begins inside the band remain scored as ``Ambiguous``.
    """

    current = epochs[issue_index].macrostate
    band_end_index = int(round(band_end_s / spec.cadence_s)) - 1
    for index in range(issue_index + 1, min(band_end_index + 1, len(epochs))):
        destination = epochs[index].macrostate
        if destination == current:
            continue
        if destination == "Unknown":
            return epochs[index].start_s, None, True
        stable_end = index + spec.stable_destination_epochs - 1
        if stable_end > band_end_index or stable_end >= len(epochs):
            if band_start_s < epochs[index].start_s <= band_end_s:
                return epochs[index].start_s, None, True
            return None, None, False
        if all(epochs[offset].macrostate == destination for offset in range(index, stable_end + 1)):
            if band_start_s < epochs[index].start_s <= band_end_s:
                return epochs[index].start_s, destination, False
            return None, None, False
    return None, None, False
