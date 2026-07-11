import copy
import tempfile
import unittest
from pathlib import Path

from neurotwin.eval.forecast_eligibility import (
    build_forecast_eligibility_artifact,
    validate_forecast_eligibility_artifact,
)
from tests.forecast_eligibility_fixtures import build_bound_forecast_eligibility


class ForecastEligibilityTests(unittest.TestCase):
    def test_bound_v2_artifacts_are_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = build_bound_forecast_eligibility(Path(tmp))
        decision = validate_forecast_eligibility_artifact(artifact)
        self.assertTrue(decision.claim_eligible)
        self.assertFalse(decision.violations)

    def test_unbound_caller_mapping_fails_closed(self):
        artifact = build_forecast_eligibility_artifact(
            {
                "protocol": {
                    "protocol_id": "kahlus.forecast.v2_nonoverlap",
                    "schema_version": 2,
                },
                "source_hash_verification_passed": True,
                "transform_lineage_complete": True,
            }
        )
        decision = validate_forecast_eligibility_artifact(artifact)
        self.assertFalse(decision.claim_eligible)
        self.assertIn(
            "forecast eligibility was not derived from bound filesystem artifacts",
            decision.violations,
        )

    def test_legacy_overlap_protocol_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = build_bound_forecast_eligibility(Path(tmp))
        edited = copy.deepcopy(artifact)
        edited["evidence"]["artifact_payloads"]["protocol"]["protocol_id"] = (
            "kahlus.forecast.v1_overlap"
        )
        decision = validate_forecast_eligibility_artifact(edited)
        self.assertFalse(decision.claim_eligible)
        self.assertTrue(any("ineligible" in item for item in decision.violations))
        self.assertTrue(any("bindings" in item for item in decision.violations))

    def test_cross_split_subject_overlap_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = build_bound_forecast_eligibility(
                Path(tmp), subject_overlap_count=26
            )
        decision = validate_forecast_eligibility_artifact(artifact)
        self.assertFalse(decision.claim_eligible)
        self.assertIn(
            "split audit subject_overlap_count must equal zero", decision.violations
        )

    def test_invalidated_dependency_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = build_bound_forecast_eligibility(
                Path(tmp),
                result_dependency_ids=("legacy-overlap",),
            )
        decision = validate_forecast_eligibility_artifact(artifact)
        self.assertFalse(decision.claim_eligible)
        self.assertTrue(
            any(
                "invalidated historical results" in item for item in decision.violations
            )
        )

    def test_edited_decision_or_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = build_bound_forecast_eligibility(Path(tmp))
        edited = copy.deepcopy(artifact)
        edited["evidence"]["source_hashes"] = []
        edited["decision"]["claim_eligible"] = True
        decision = validate_forecast_eligibility_artifact(edited)
        self.assertFalse(decision.claim_eligible)
        self.assertTrue(any("hash mismatch" in item for item in decision.violations))
        self.assertTrue(any("does not match" in item for item in decision.violations))


if __name__ == "__main__":
    unittest.main()
