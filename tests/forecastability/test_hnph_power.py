"""Hand-worked verification of the HNPH B2 subject-level power analysis."""

from __future__ import annotations

import math
from statistics import NormalDist

import pytest

from neurotwin.forecastability.hnph_power import (
    DEFAULT_BANDS,
    SLEEP_EDF_SC78,
    HnphBandSpec,
    HnphCohortSpec,
    HnphPowerError,
    HnphPowerInputs,
    compute_power_analysis,
)


def test_z_values_match_hand_computation():
    # familywise alpha 0.05 over 12 families -> per-family 0.0041667 -> z ~ 2.6383.
    # target power 0.80 -> z ~ 0.8416.
    report = compute_power_analysis(SLEEP_EDF_SC78, DEFAULT_BANDS, HnphPowerInputs())
    assert report.per_family_alpha == pytest.approx(0.05 / 12)
    assert report.z_alpha == pytest.approx(NormalDist().inv_cdf(1 - 0.05 / 12), abs=1e-4)
    assert report.z_alpha == pytest.approx(2.6383, abs=1e-3)
    assert report.z_power == pytest.approx(0.8416, abs=1e-3)


def test_required_subjects_matches_closed_form():
    # m = ceil( (z_sum * sigma / d)^2 ). Hand-check at sigma=0.05, d=0.02.
    inputs = HnphPowerInputs()
    report = compute_power_analysis(SLEEP_EDF_SC78, DEFAULT_BANDS, inputs)
    z_sum = report.z_alpha + report.z_power  # ~3.4799
    expected = math.ceil((z_sum * 0.05 / 0.02) ** 2)  # ~ 76
    band = report.bands[0]
    assert band.required_subjects_by_sigma["0.050"] == expected
    # Sanity: this lands right around the Sleep-EDF-78 cohort size, the knife's edge.
    assert 70 <= expected <= 80


def test_power_and_required_subjects_are_consistent():
    # At exactly the required subject count, achieved power should meet the target.
    inputs = HnphPowerInputs(target_power=0.80)
    report = compute_power_analysis(SLEEP_EDF_SC78, DEFAULT_BANDS, inputs)
    fip = report.frozen_power_inputs({"band_2_5_min": 0.05})
    required = fip["required_subjects_by_band"]["band_2_5_min"]
    z_alpha = report.z_alpha
    ncp = math.sqrt(required) * 0.02 / 0.05
    achieved = NormalDist().cdf(ncp - z_alpha)
    assert achieved >= 0.80 - 1e-6


def test_nrem_rem_event_subjects_track_base_rate():
    # ~3 events/night over ~1.96 nights -> mean ~5.88 -> P(>=1 event) ~ 0.997.
    report = compute_power_analysis(SLEEP_EDF_SC78, DEFAULT_BANDS)
    band = report.bands[0]
    # Essentially every usable subject is an event subject for NREM->REM.
    assert band.available_event_subjects >= 77
    # Anchor count is abundant even though subject count is the binding wall.
    assert band.expected_primary_anchors >= 400
    assert band.available_event_subjects < band.expected_primary_anchors


def test_sparse_transition_shrinks_event_pool():
    # A rare transition (0.21/night, ~ Wake->REM from Laffan 2010) yields far fewer
    # event subjects than NREM->REM despite the same cohort, which lowers the
    # supportable-sigma ceiling even where the smallest grid sigma is still met.
    sparse = compute_power_analysis(SLEEP_EDF_SC78, [HnphBandSpec("band_wake_rem", "wake_to_rem", 0.21)]).bands[0]
    dense = compute_power_analysis(SLEEP_EDF_SC78, DEFAULT_BANDS).bands[0]
    # mean ~0.41 events/subject -> P(>=1) ~ 0.336 -> ~26 event subjects.
    assert 20 <= sparse.available_event_subjects <= 32
    assert sparse.available_event_subjects < dense.available_event_subjects
    # Fewer event subjects -> strictly lower tolerable between-subject SD.
    assert sparse.max_supportable_sigma_bits < dense.max_supportable_sigma_bits


def test_underpowered_cohort_flagged():
    tiny = HnphCohortSpec(name="pilot", n_subjects=8, mean_nights_per_subject=1.0)
    report = compute_power_analysis(tiny, DEFAULT_BANDS)
    assert report.overall_powered is False
    assert report.verdict.startswith("cohort_underpowered")


def test_max_supportable_sigma_is_the_honest_ceiling():
    report = compute_power_analysis(SLEEP_EDF_SC78, DEFAULT_BANDS)
    band = report.bands[0]
    # With ~77 event subjects the cohort can only support a modest between-subject
    # SD before losing 80% power. Verify it round-trips to ~target power.
    ncp = math.sqrt(band.available_event_subjects) * 0.02 / band.max_supportable_sigma_bits
    achieved = NormalDist().cdf(ncp - report.z_alpha)
    assert achieved == pytest.approx(0.80, abs=1e-3)


def test_frozen_power_inputs_shape_matches_gate_contract():
    report = compute_power_analysis(SLEEP_EDF_SC78, DEFAULT_BANDS)
    fip = report.frozen_power_inputs({"band_2_5_min": 0.05})
    assert set(fip) == {
        "source",
        "sigma_bits_by_band",
        "required_subjects_by_band",
        "available_event_subjects_by_band",
        "simulated_power_by_band",
    }
    assert fip["required_subjects_by_band"]["band_2_5_min"] > 0
    assert 0.0 <= fip["simulated_power_by_band"]["band_2_5_min"] <= 1.0


def test_rejects_incoherent_inputs():
    with pytest.raises(HnphPowerError):
        HnphPowerInputs(epsilon_bits_per_anchor=0.0)
    with pytest.raises(HnphPowerError):
        HnphPowerInputs(target_power=0.3)
    with pytest.raises(HnphPowerError):
        HnphCohortSpec(name="bad", n_subjects=0, mean_nights_per_subject=1.0)
    with pytest.raises(HnphPowerError):
        compute_power_analysis(SLEEP_EDF_SC78, [])
