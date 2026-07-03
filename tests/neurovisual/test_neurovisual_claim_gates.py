import unittest

from neurotwin.neurovisual import (
    NEUROVISUAL_ALLOWED_CLAIM_SCOPES,
    NEUROVISUAL_BLOCKED_CLAIMS,
    evaluate_neurovisual_claim_gate,
)


class NeurovisualClaimGateTests(unittest.TestCase):
    def test_allowed_claim_scope_passes_without_blocked_terms(self):
        gate = evaluate_neurovisual_claim_gate(
            claim_scope="structured_intake_schema_ready",
            payloads=["structured symptom mapping for clinician or researcher review"],
        )

        self.assertTrue(gate["passed"], gate)
        self.assertEqual(sorted(gate["allowed_claim_scopes"]), sorted(NEUROVISUAL_ALLOWED_CLAIM_SCOPES))

    def test_blocked_claims_fail_hard_in_any_payload_text(self):
        for blocked in NEUROVISUAL_BLOCKED_CLAIMS:
            gate = evaluate_neurovisual_claim_gate(
                claim_scope="structured_intake_schema_ready",
                payloads=[f"this output {blocked}"],
            )
            self.assertFalse(gate["passed"], blocked)
            self.assertIn(blocked, gate["blocked_claims_found"])

    def test_unknown_claim_scope_fails(self):
        gate = evaluate_neurovisual_claim_gate(
            claim_scope="clinical_diagnostic_report",
            payloads=["structured symptom mapping only"],
        )

        self.assertFalse(gate["passed"], gate)
        self.assertIn("unsupported_claim_scope:clinical_diagnostic_report", gate["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
