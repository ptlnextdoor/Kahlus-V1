import copy
import unittest

from neurotwin.eval.forecast_eligibility import (
    build_forecast_eligibility_artifact,
    derive_forecast_eligibility,
    validate_forecast_eligibility_artifact,
)


def valid_evidence() -> dict[str, object]:
    return {
        "protocol": {"protocol_id": "kahlus.forecast.v2_nonoverlap", "schema_version": 2},
        "source_hashes": ["a" * 64],
        "source_hash_verification_passed": True,
        "transform_lineage_hash": "b" * 64,
        "transform_lineage_complete": True,
        "split_audit": {
            "passed": True,
            "violations": [],
            "subject_overlap_count": 0,
            "recording_overlap_count": 0,
            "session_overlap_count": 0,
        },
        "firebreak_audit": {"passed": True, "violations": [], "target_overlaps_context": False},
        "invalidated_result_ids": [],
    }


class ForecastEligibilityTests(unittest.TestCase):
    def test_valid_v2_evidence_is_derived_as_eligible(self):
        artifact = build_forecast_eligibility_artifact(valid_evidence())
        decision = validate_forecast_eligibility_artifact(artifact)
        self.assertTrue(decision.claim_eligible)
        self.assertFalse(decision.violations)

    def test_legacy_overlap_protocol_fails_closed(self):
        evidence = valid_evidence()
        evidence["protocol"] = {"protocol_id": "kahlus.forecast.v1_overlap", "schema_version": 1}
        evidence["firebreak_audit"] = {"passed": True, "violations": [], "target_overlaps_context": True}
        decision = derive_forecast_eligibility(evidence)
        self.assertFalse(decision.claim_eligible)
        self.assertTrue(any("ineligible" in item for item in decision.violations))
        self.assertTrue(any("overlaps" in item for item in decision.violations))

    def test_missing_hashes_and_failed_firebreak_fail_closed(self):
        evidence = valid_evidence()
        evidence["source_hashes"] = []
        evidence["firebreak_audit"] = {"passed": False, "violations": ["future sample"]}
        decision = derive_forecast_eligibility(evidence)
        self.assertFalse(decision.claim_eligible)
        self.assertTrue(any("SHA-256" in item for item in decision.violations))
        self.assertTrue(any("firebreak" in item for item in decision.violations))

    def test_unknown_protocol_and_invalidated_dependency_fail_closed(self):
        evidence = valid_evidence()
        evidence["protocol"] = {"protocol_id": "unknown", "schema_version": 9}
        evidence["invalidated_result_ids"] = ["INV-001"]
        decision = derive_forecast_eligibility(evidence)
        self.assertFalse(decision.claim_eligible)
        self.assertTrue(any("unknown" in item for item in decision.violations))
        self.assertTrue(any("invalidated" in item for item in decision.violations))

    def test_cross_split_subject_overlap_fails_closed(self):
        evidence = valid_evidence()
        evidence["split_audit"]["subject_overlap_count"] = 26
        decision = derive_forecast_eligibility(evidence)
        self.assertFalse(decision.claim_eligible)
        self.assertIn("split audit subject_overlap_count must equal zero", decision.violations)

    def test_edited_or_self_attested_artifact_is_rejected(self):
        artifact = build_forecast_eligibility_artifact(valid_evidence())
        edited = copy.deepcopy(artifact)
        edited["evidence"]["source_hash_verification_passed"] = False
        edited["decision"]["claim_eligible"] = True
        decision = validate_forecast_eligibility_artifact(edited)
        self.assertFalse(decision.claim_eligible)
        self.assertTrue(any("hash mismatch" in item for item in decision.violations))
        self.assertTrue(any("does not match" in item for item in decision.violations))


if __name__ == "__main__":
    unittest.main()
