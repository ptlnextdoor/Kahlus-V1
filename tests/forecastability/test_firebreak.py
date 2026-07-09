from __future__ import annotations

import unittest

from neurotwin.forecastability import (
    CausalPreprocessingSpec,
    ForecastAnchor,
    PhysicalRecordRegistry,
    TimeInterval,
    audit_forecast_firebreak,
    audit_model_input_mapping,
)
from tests.forecastability.test_contracts import _record


class ForecastFirebreakTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = PhysicalRecordRegistry.from_records((_record(),))
        self.preprocessing = CausalPreprocessingSpec(
            normalization_scope="context_only",
            filter_mode="causal",
            filter_guard_s=2.0,
        )

    def test_anchor_requires_a_strict_future_gap(self) -> None:
        with self.assertRaisesRegex(ValueError, "filter_guard_s"):
            ForecastAnchor(
                anchor_id="overlap",
                record_id="sleep-001",
                context=TimeInterval(0.0, 20.0),
                target=TimeInterval(21.0, 30.0),
                filter_guard_s=2.0,
            )

    def test_audit_requires_context_target_and_guard_inside_valid_support(self) -> None:
        anchor = ForecastAnchor(
            anchor_id="artifact-crossing",
            record_id="sleep-001",
            context=TimeInterval(10.0, 20.0),
            target=TimeInterval(42.0, 52.0),
            filter_guard_s=12.0,
        )
        report = audit_forecast_firebreak((anchor,), self.registry, self.preprocessing)
        self.assertFalse(report.passed)
        self.assertIn("filter guard is outside", " ".join(report.violations))

    def test_audit_passes_a_fully_supported_causal_anchor(self) -> None:
        anchor = ForecastAnchor(
            anchor_id="valid",
            record_id="sleep-001",
            context=TimeInterval(0.0, 20.0),
            target=TimeInterval(22.0, 28.0),
            filter_guard_s=2.0,
        )
        report = audit_forecast_firebreak((anchor,), self.registry, self.preprocessing)
        self.assertTrue(report.passed, report.violations)

    def test_audit_rejects_undersized_anchor_guard_and_unknown_record(self) -> None:
        undersized_anchor = ForecastAnchor(
            anchor_id="undersized-guard",
            record_id="sleep-001",
            context=TimeInterval(0.0, 10.0),
            target=TimeInterval(11.0, 20.0),
            filter_guard_s=1.0,
        )
        unknown_record_anchor = ForecastAnchor(
            anchor_id="wrong-record",
            record_id="unknown",
            context=TimeInterval(0.0, 10.0),
            target=TimeInterval(11.0, 20.0),
            filter_guard_s=1.0,
        )
        report = audit_forecast_firebreak((undersized_anchor, unknown_record_anchor), self.registry, self.preprocessing)
        self.assertFalse(report.passed)
        violations = " ".join(report.violations)
        self.assertIn("unknown record", violations)
        self.assertIn("shorter than preprocessing guard", violations)

    def test_preprocessing_rejects_full_record_normalization_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "normalization_scope"):
            CausalPreprocessingSpec(
                normalization_scope="full_record",  # type: ignore[arg-type]
                filter_mode="causal",
                filter_guard_s=2.0,
            )

    def test_model_input_audit_rejects_identity_and_target_fields(self) -> None:
        report = audit_model_input_mapping(
            {
                "signal": object(),
                "subject_id": object(),
                "target_label": object(),
            }
        )
        self.assertFalse(report.passed)
        self.assertEqual(len(report.violations), 2)


if __name__ == "__main__":
    unittest.main()
