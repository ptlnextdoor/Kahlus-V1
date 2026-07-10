from __future__ import annotations

import unittest

import numpy as np

from neurotwin.config_types import resolve_forecast_task_for_sampling, resolve_prepared_config
from neurotwin.data.forecast_contract import (
    FORECAST_PROTOCOL_V1_OVERLAP,
    FORECAST_PROTOCOL_V2_NONOVERLAP,
    ForecastProtocolError,
    ForecastTaskSpec,
    WindowExampleProvenance,
    legacy_overlapping_forecast_spec,
)
from neurotwin.data.prepared_tasks import SupervisedWindowTask


class ForecastContractTests(unittest.TestCase):
    def test_v2_protocol_resolves_disjoint_ranges(self) -> None:
        spec = ForecastTaskSpec(
            protocol_id=FORECAST_PROTOCOL_V2_NONOVERLAP,
            schema_version=2,
            context_seconds=1.0,
            target_seconds=0.5,
            gap_seconds=0.25,
            stride_seconds=0.5,
            claim_eligible=True,
        ).resolve(128.0)

        self.assertEqual((spec.context_samples, spec.target_samples, spec.gap_samples, spec.stride_samples), (128, 64, 32, 64))
        self.assertEqual(spec.ranges(10), (10, 138, 170, 234))
        self.assertTrue(spec.claim_eligible)

    def test_claim_eligible_v2_requires_positive_gap(self) -> None:
        with self.assertRaisesRegex(ForecastProtocolError, "positive gap_seconds"):
            ForecastTaskSpec(
                protocol_id=FORECAST_PROTOCOL_V2_NONOVERLAP,
                schema_version=2,
                context_seconds=1.0,
                target_seconds=0.5,
                gap_seconds=0.0,
                stride_seconds=0.5,
                claim_eligible=True,
            )

    def test_legacy_protocol_is_explicitly_ineligible(self) -> None:
        spec = legacy_overlapping_forecast_spec(window_samples=8, forecast_horizon_samples=1, stride_samples=1)

        self.assertEqual(spec.spec.protocol_id, FORECAST_PROTOCOL_V1_OVERLAP)
        self.assertEqual(spec.ranges(0), (0, 8, 1, 9))
        with self.assertRaises(ForecastProtocolError):
            spec.assert_claim_eligible()

    def test_provenance_must_match_the_resolved_protocol(self) -> None:
        spec = ForecastTaskSpec(
            protocol_id=FORECAST_PROTOCOL_V2_NONOVERLAP,
            schema_version=2,
            context_seconds=1.0,
            target_seconds=0.5,
            gap_seconds=0.25,
            stride_seconds=0.5,
            claim_eligible=True,
        ).resolve(4.0)
        provenance = WindowExampleProvenance(
            dataset_id="fixture",
            subject_id="sub-01",
            session_id="ses-01",
            site_id="site-01",
            record_id="record-01",
            source_hash="abc",
            split="train",
            input_start=0,
            input_stop=4,
            target_start=5,
            target_stop=7,
        )

        provenance.validate_against(spec)
        self.assertFalse(provenance.target_overlaps_input)

    def test_claim_eligible_task_requires_matching_provenance(self) -> None:
        spec = ForecastTaskSpec(
            protocol_id=FORECAST_PROTOCOL_V2_NONOVERLAP,
            schema_version=2,
            context_seconds=1.0,
            target_seconds=1.0,
            gap_seconds=0.25,
            stride_seconds=1.0,
            claim_eligible=True,
        ).resolve(4.0)
        values = np.zeros((1, 4, 2), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "require train and test provenance"):
            SupervisedWindowTask(
                task_id="future_state_forecasting",
                source_modality="eeg",
                target_modality="eeg",
                x_train=values,
                y_train=values,
                x_test=values,
                y_test=values,
                forecast_spec=spec,
            )

    def test_config_parses_v2_forecast_task_without_legacy_fallback(self) -> None:
        resolved = resolve_prepared_config(
            {
                "task": "future_state_forecasting",
                "forecast_task": {
                    "protocol_id": FORECAST_PROTOCOL_V2_NONOVERLAP,
                    "schema_version": 2,
                    "context_seconds": 1.0,
                    "target_seconds": 0.5,
                    "gap_seconds": 0.25,
                    "stride_seconds": 0.5,
                    "claim_eligible": True,
                },
            }
        )

        self.assertIsNotNone(resolved.forecast_task)
        assert resolved.forecast_task is not None
        self.assertEqual(resolved.forecast_task.protocol_id, FORECAST_PROTOCOL_V2_NONOVERLAP)

    def test_flat_legacy_config_is_classified_as_overlap_only(self) -> None:
        spec = resolve_forecast_task_for_sampling(
            {"task": "future_state_forecasting", "window_length": 8, "forecast_horizon": 1, "stride": 8},
            sampling_rate_hz=128.0,
        )

        assert spec is not None
        self.assertEqual(spec.spec.protocol_id, FORECAST_PROTOCOL_V1_OVERLAP)
        self.assertFalse(spec.claim_eligible)


if __name__ == "__main__":
    unittest.main()
