"""HNPH B2 subject-level power analysis.

This module answers the question that gates the whole B2 experiment: given a real
sleep cohort and literature-grounded transition base rates, does the available
number of *event subjects* provide the preregistered target power (default 80%)
to detect the preregistered per-anchor effect (default 0.02 bits) against the
chief nuisance comparator?

Scientific contract
--------------------
- Inference is at the **subject-cluster** level, never the anchor level. Anchors
  within a subject are dependent (they share physiology, montage, night); a night
  with ~3 NREM->REM transitions contributes ~3 correlated anchors, not 3
  independent samples. The binding constraint is therefore the count of subjects
  who contribute at least one usable primary-band event, not the total anchor
  count. This module makes that distinction explicit and refuses to let a large
  anchor count paper over a small subject count.
- The power model is the one frozen in
  ``docs/research/hnph_b2_preregistration_addendum.md`` and the canonical
  notebook:

      m >= [ (z_{1 - alpha/|G|} + z_{1 - beta0}) * sigma / d ]^2

  where ``d`` is the per-subject margin over the comparator (in the same
  subject-balanced log-skill bits as the primary score), ``sigma`` is the
  between-subject standard deviation of the per-subject gain, ``alpha`` is the
  familywise one-sided level, ``|G|`` is the number of (lead band x transition
  type x named evaluation) hypotheses (Bonferroni split), and ``beta0`` is the
  target Type-II rate.
- ``sigma`` is not knowable before data. This module therefore reports required
  subjects across a grid of ``sigma`` and, given the cohort's available event
  subjects, the largest ``sigma`` the cohort can tolerate at target power. That
  ceiling, compared to plausible ``sigma``, is the honest feasibility signal.

Base rates are literature-grounded (see ``LITERATURE_BASE_RATES``): healthy-adult
overnight PSG shows ~3 NREM->REM cycles/night and ~10-14 stage transitions/hour.
Those set expected event-subject and anchor availability, not the effect size.

No heavy dependencies: standard library only, so a lean cluster or CI environment
can run it. Nothing here downloads or touches raw neural data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import NormalDist
from typing import Mapping

HNPH_POWER_SCHEMA = "kahlus.hnph.power_analysis.v1"

# Literature-grounded healthy-adult overnight-PSG base rates, used only to derive
# expected event-subject and anchor availability. Sources (Consensus, 2026):
#   - ~3.0-3.2 NREM->REM cycles/night: Geisler 2025 (40-yr PSG follow-up),
#     Kishi 2023 (phenotypic dynamics).
#   - ~10-14 stage transitions/hour overall: Laffan 2010 (SHHS), Geisler 2025.
# These are population expectations, not a performance prior for HNPH.
LITERATURE_BASE_RATES: dict[str, float] = {
    "nrem_to_rem_transitions_per_night": 3.0,
    "overall_transitions_per_hour": 10.11,  # SHHS median, Laffan 2010
}

_MIN_TARGET_POWER = 0.5
_MAX_TARGET_POWER = 0.999


class HnphPowerError(ValueError):
    """Raised when power inputs are physically or statistically incoherent."""


@dataclass(frozen=True)
class HnphBandSpec:
    """One preregistered lead band and the transition it targets."""

    band_id: str
    transition: str
    # Expected count of qualifying stable transitions of this kind per subject-night.
    expected_events_per_subject_night: float

    def __post_init__(self) -> None:
        if not self.band_id or not self.transition:
            raise HnphPowerError("band_id and transition must be non-empty")
        if not math.isfinite(self.expected_events_per_subject_night) or self.expected_events_per_subject_night < 0:
            raise HnphPowerError(f"band {self.band_id}: expected events must be finite and >= 0")


@dataclass(frozen=True)
class HnphCohortSpec:
    """The physical cohort available for B2, before outcome analysis."""

    name: str
    n_subjects: int
    mean_nights_per_subject: float
    usable_subject_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.n_subjects < 1:
            raise HnphPowerError("n_subjects must be >= 1")
        if self.mean_nights_per_subject <= 0:
            raise HnphPowerError("mean_nights_per_subject must be > 0")
        if not 0.0 < self.usable_subject_fraction <= 1.0:
            raise HnphPowerError("usable_subject_fraction must be in (0, 1]")

    @property
    def usable_subjects(self) -> int:
        return int(math.floor(self.n_subjects * self.usable_subject_fraction))


@dataclass(frozen=True)
class HnphPowerInputs:
    """Preregistered design-sensitivity inputs. Frozen before claim-mode analysis."""

    epsilon_bits_per_anchor: float = 0.02
    familywise_alpha: float = 0.05
    target_power: float = 0.80
    family_count: int = 12  # 4 bands x 3 transition types x 1 internal evaluation
    # Grid of plausible between-subject SD of per-subject log-skill gain (bits).
    sigma_grid_bits: tuple[float, ...] = (0.02, 0.03, 0.05, 0.08, 0.12, 0.20)

    def __post_init__(self) -> None:
        if not 0.0 < self.epsilon_bits_per_anchor:
            raise HnphPowerError("epsilon must be > 0")
        if not 0.0 < self.familywise_alpha < 0.5:
            raise HnphPowerError("familywise_alpha must be in (0, 0.5)")
        if not _MIN_TARGET_POWER <= self.target_power <= _MAX_TARGET_POWER:
            raise HnphPowerError(f"target_power must be in [{_MIN_TARGET_POWER}, {_MAX_TARGET_POWER}]")
        if self.family_count < 1:
            raise HnphPowerError("family_count must be >= 1")
        if not self.sigma_grid_bits or any(s <= 0 for s in self.sigma_grid_bits):
            raise HnphPowerError("sigma_grid_bits must be non-empty and strictly positive")


@dataclass(frozen=True)
class HnphBandPower:
    band_id: str
    transition: str
    available_event_subjects: int
    expected_primary_anchors: int
    required_subjects_by_sigma: dict[str, int]
    max_supportable_sigma_bits: float
    powered_at_target: bool
    limiting_factor: str


@dataclass(frozen=True)
class HnphPowerReport:
    schema: str
    cohort: dict
    inputs: dict
    per_family_alpha: float
    z_alpha: float
    z_power: float
    bands: list[HnphBandPower]
    overall_powered: bool
    verdict: str

    def to_dict(self) -> dict:
        out = asdict(self)
        return out

    def frozen_power_inputs(self, sigma_bits_by_band: Mapping[str, float]) -> dict:
        """Emit the mapping shape the HNPH gate consumes as ``frozen_power_inputs``.

        ``sigma_bits_by_band`` is the analyst's frozen per-band sigma estimate (from
        training folds or synthetic pilots), not a value learned from held-out
        outcomes. Required subjects are recomputed at exactly that sigma.
        """
        required: dict[str, int] = {}
        available: dict[str, int] = {}
        simulated: dict[str, float] = {}
        z_sum = self.z_alpha + self.z_power
        for band in self.bands:
            sigma = float(sigma_bits_by_band.get(band.band_id, float("nan")))
            if not math.isfinite(sigma) or sigma <= 0:
                raise HnphPowerError(f"band {band.band_id}: frozen sigma must be finite and > 0")
            required[band.band_id] = _required_subjects(z_sum, sigma, _effect(self.inputs))
            available[band.band_id] = band.available_event_subjects
            simulated[band.band_id] = _power_at_subjects(
                band.available_event_subjects, sigma, _effect(self.inputs), self.z_alpha
            )
        return {
            "source": "hnph_power.compute_power_analysis",
            "sigma_bits_by_band": dict(sigma_bits_by_band),
            "required_subjects_by_band": required,
            "available_event_subjects_by_band": available,
            "simulated_power_by_band": simulated,
        }


def _effect(inputs: Mapping) -> float:
    return float(inputs["epsilon_bits_per_anchor"])


def _required_subjects(z_sum: float, sigma: float, effect: float) -> int:
    """m >= [ z_sum * sigma / d ]^2, rounded up. At least 2 subjects for a variance."""
    raw = (z_sum * sigma / effect) ** 2
    return max(2, int(math.ceil(raw)))


def _power_at_subjects(m: int, sigma: float, effect: float, z_alpha: float) -> float:
    """Achieved one-sided power for m subjects at the given effect and sigma.

    power = Phi( sqrt(m) * d / sigma - z_alpha ).
    """
    if m < 1:
        return 0.0
    ncp = math.sqrt(m) * effect / sigma
    return float(NormalDist().cdf(ncp - z_alpha))


def _max_supportable_sigma(m: int, z_sum: float, effect: float) -> float:
    """Largest sigma for which m subjects still reach target power."""
    if m < 1:
        return 0.0
    return math.sqrt(m) * effect / z_sum


def compute_power_analysis(
    cohort: HnphCohortSpec,
    bands: list[HnphBandSpec],
    inputs: HnphPowerInputs | None = None,
) -> HnphPowerReport:
    """Compute subject-level feasibility for each preregistered band.

    An event subject is a usable subject expected to contribute at least one
    qualifying transition of the band's type. With ~3 NREM->REM transitions per
    subject-night, essentially every usable multi-night subject is an event
    subject for that band; sparse transition types shrink this pool.
    """
    if inputs is None:
        inputs = HnphPowerInputs()
    if not bands:
        raise HnphPowerError("at least one band is required")

    per_family_alpha = inputs.familywise_alpha / inputs.family_count
    z_alpha = NormalDist().inv_cdf(1.0 - per_family_alpha)
    z_power = NormalDist().inv_cdf(inputs.target_power)
    z_sum = z_alpha + z_power
    effect = inputs.epsilon_bits_per_anchor
    usable = cohort.usable_subjects

    band_reports: list[HnphBandPower] = []
    overall_powered = True
    for band in bands:
        expected_events_per_subject = band.expected_events_per_subject_night * cohort.mean_nights_per_subject
        # P(subject contributes >= 1 event) under a Poisson count with this mean.
        p_event_subject = 1.0 - math.exp(-expected_events_per_subject) if expected_events_per_subject > 0 else 0.0
        available_event_subjects = int(math.floor(usable * p_event_subject))
        expected_primary_anchors = int(math.floor(usable * expected_events_per_subject))

        required_by_sigma = {
            f"{sigma:.3f}": _required_subjects(z_sum, sigma, effect) for sigma in inputs.sigma_grid_bits
        }
        max_sigma = _max_supportable_sigma(available_event_subjects, z_sum, effect)

        # Powered if the smallest sigma in the grid is already supportable.
        smallest_required = _required_subjects(z_sum, min(inputs.sigma_grid_bits), effect)
        powered = available_event_subjects >= smallest_required
        if not powered:
            overall_powered = False

        if available_event_subjects < 2:
            limiting = "too_few_event_subjects_for_variance"
        elif available_event_subjects < smallest_required:
            limiting = "event_subject_count_below_requirement_even_at_smallest_sigma"
        else:
            limiting = "sigma_dependent"

        band_reports.append(
            HnphBandPower(
                band_id=band.band_id,
                transition=band.transition,
                available_event_subjects=available_event_subjects,
                expected_primary_anchors=expected_primary_anchors,
                required_subjects_by_sigma=required_by_sigma,
                max_supportable_sigma_bits=round(max_sigma, 6),
                powered_at_target=powered,
                limiting_factor=limiting,
            )
        )

    if overall_powered:
        verdict = "cohort_may_be_powered_pending_frozen_sigma"
    else:
        verdict = "cohort_underpowered_at_smallest_plausible_sigma_expect_bounded_null"

    return HnphPowerReport(
        schema=HNPH_POWER_SCHEMA,
        cohort=asdict(cohort),
        inputs=asdict(inputs),
        per_family_alpha=per_family_alpha,
        z_alpha=round(z_alpha, 6),
        z_power=round(z_power, 6),
        bands=band_reports,
        overall_powered=overall_powered,
        verdict=verdict,
    )


# Canonical Sleep-EDF Sleep Cassette cohort (SC-78): ~78 subjects, ~2 nights each.
SLEEP_EDF_SC78 = HnphCohortSpec(
    name="sleep_edf_sc78",
    n_subjects=78,
    mean_nights_per_subject=1.96,
    usable_subject_fraction=1.0,
)

# Primary and secondary bands. Expected events/night are literature-grounded:
# NREM->REM ~3/night; W/NREM/REM omnibus far denser. Sparser transition types
# would shrink the event-subject pool and should be added explicitly when frozen.
DEFAULT_BANDS = [
    HnphBandSpec("band_2_5_min", "nrem_to_rem", LITERATURE_BASE_RATES["nrem_to_rem_transitions_per_night"]),
]
