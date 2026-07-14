from __future__ import annotations

import unittest

from neurotwin.forecastability import (
    CausalPreprocessingSpec,
    LeadGeometry,
    PhysicalRecordRegistry,
    PhysicalSignalRecord,
    QualityInterval,
    TransitionEpoch,
    TransitionLeadBand,
    TransitionTargetSpec,
    audit_forecast_firebreak,
    build_natural_transition_target_result,
    build_natural_transition_targets,
)


def _spec() -> TransitionTargetSpec:
    return TransitionTargetSpec(
        version="hnph-v0.2",
        cadence_s=30.0,
        context_s=600.0,
        stable_destination_epochs=2,
        lead_bands=(
            TransitionLeadBand("B1", 30.0, 120.0),
            TransitionLeadBand("B2", 120.0, 300.0),
        ),
        primary_band_id="B2",
    )


def _epochs() -> tuple[TransitionEpoch, ...]:
    return tuple(
        TransitionEpoch("sleep-001", index * 30.0, "NREM" if index < 25 else "REM")
        for index in range(48)
    )


class TransitionTargetTests(unittest.TestCase):
    def test_builds_natural_grid_b2_target_for_first_stable_transition(self) -> None:
        targets = build_natural_transition_targets(_epochs(), _spec())
        target = next(row for row in targets if row.target_id == "sleep-001:000019:B2")

        self.assertEqual(target.current_macrostate, "NREM")
        self.assertEqual(target.destination_macrostate, "REM")
        self.assertEqual(target.outcome, "REM")
        self.assertEqual(target.issue_time_s, 600.0)
        self.assertEqual(target.event_time_s, 750.0)
        self.assertTrue(target.complete_follow_up)

    def test_complete_targets_pass_the_existing_firebreak_audit(self) -> None:
        targets = build_natural_transition_targets(_epochs(), _spec())
        registry = PhysicalRecordRegistry.from_records(
            (
                PhysicalSignalRecord(
                    record_id="sleep-001",
                    subject_id="subject-001",
                    session_id="night-1",
                    dataset_id="sleep-edf-expanded-sleep-cassette",
                    site_id=None,
                    modality="eeg",
                    sampling_rate_hz=100.0,
                    physical_unit="uV",
                    duration_s=48 * 30.0,
                    leads=(LeadGeometry("Fpz-Cz", None, None, "bipolar", "fixture"),),
                    quality_intervals=(QualityInterval(0.0, 48 * 30.0, "valid"),),
                    raw_source_uri="physionet://sleep-edf/sleep-001.edf",
                ),
            )
        )
        report = audit_forecast_firebreak(
            (target.anchor for target in targets),
            registry,
            CausalPreprocessingSpec("context_only", "none", 0.0),
        )
        self.assertTrue(report.passed, report.violations)

    def test_rejects_non_natural_epoch_grid(self) -> None:
        epochs = list(_epochs())
        epochs[4] = TransitionEpoch("sleep-001", 121.0, "NREM")
        with self.assertRaisesRegex(ValueError, "natural zero-based cadence grid"):
            build_natural_transition_targets(epochs, _spec())

    def test_earlier_transition_is_a_no_event_for_a_later_band(self) -> None:
        targets = build_natural_transition_targets(_epochs(), _spec())
        target = next(row for row in targets if row.target_id == "sleep-001:000019:B1")

        self.assertEqual(target.outcome, "no_event")
        self.assertIsNone(target.event_time_s)

    def test_one_epoch_reversal_is_not_silently_promoted_to_ambiguous(self) -> None:
        epochs = list(_epochs())
        epochs[25] = TransitionEpoch("sleep-001", 25 * 30.0, "REM")
        for index in range(26, len(epochs)):
            epochs[index] = TransitionEpoch("sleep-001", index * 30.0, "NREM")
        targets = build_natural_transition_targets(epochs, _spec())
        target = next(row for row in targets if row.target_id == "sleep-001:000019:B2")

        self.assertEqual(target.outcome, "no_event")
        self.assertFalse(target.ambiguous)

    def test_unknown_annotation_boundary_is_a_scored_ambiguity(self) -> None:
        epochs = list(_epochs())
        epochs[25] = TransitionEpoch("sleep-001", 25 * 30.0, "Unknown")
        targets = build_natural_transition_targets(epochs, _spec())
        target = next(row for row in targets if row.target_id == "sleep-001:000019:B2")

        self.assertEqual(target.outcome, "Ambiguous")
        self.assertTrue(target.ambiguous)
        self.assertEqual(target.event_time_s, 750.0)

    def test_unknown_source_epoch_is_scored_as_ambiguous_not_silently_dropped(self) -> None:
        epochs = list(_epochs())
        epochs[19] = TransitionEpoch("sleep-001", 19 * 30.0, "Unknown")
        targets = build_natural_transition_targets(epochs, _spec())
        target = next(row for row in targets if row.target_id == "sleep-001:000019:B2")

        self.assertEqual(target.outcome, "Ambiguous")
        self.assertTrue(target.ambiguous)
        self.assertEqual(target.event_time_s, 600.0)

    def test_right_censored_anchors_are_excluded_and_counted_by_band(self) -> None:
        result = build_natural_transition_target_result(_epochs(), _spec())

        self.assertGreater(result.complete_follow_up_excluded_count_by_band["B1"], 0)
        self.assertGreater(result.complete_follow_up_excluded_count_by_band["B2"], 0)


if __name__ == "__main__":
    unittest.main()
