import json
import tempfile
import unittest
from pathlib import Path


from neurotwin.falsification import (
    Outcome,
    assemble_gate,
    build_report,
    outcomes_finite,
    write_report,
)


def _outcome(name, passed, detail, reason=""):
    return Outcome(name=name, passed=passed, detail=detail, reason=reason)


class FalsificationCoreTests(unittest.TestCase):
    def test_outcomes_finite_handles_nested_dicts(self):
        ok = [_outcome("a", True, {"x": 1.0, "nested": {"y": 2.0, "z": 3.0}})]
        self.assertTrue(outcomes_finite(ok))

    def test_outcomes_finite_detects_nested_non_finite(self):
        bad = [_outcome("a", True, {"nested": {"y": float("nan")}})]
        self.assertFalse(outcomes_finite(bad))
        bad_inf = [_outcome("a", True, {"v": float("inf")})]
        self.assertFalse(outcomes_finite(bad_inf))

    def test_outcomes_finite_ignores_bools_and_empty(self):
        self.assertTrue(outcomes_finite([_outcome("a", True, {"flag": True, "label": "ok"})]))
        self.assertTrue(outcomes_finite([_outcome("a", True, {})]))

    def test_gate_allows_narrow_claim_when_required_pass(self):
        outcomes = [_outcome("recovery", True, {"score": 1.0}), _outcome("aux", False, {"d": 0.0}, "aux failed")]
        gate = assemble_gate(
            branch="v3", dataset="transition_gym_synthetic",
            claim_scope="synthetic_transition_operator_recovery",
            outcomes=outcomes, required=["recovery"],  # aux not required
        )
        self.assertTrue(gate["scientific_claim_allowed"])
        self.assertEqual(gate["failure_reasons"], [])

    def test_gate_blocks_when_required_fails(self):
        outcomes = [_outcome("recovery", False, {"score": 0.1}, "recovery too low")]
        gate = assemble_gate(
            branch="v3", dataset="transition_gym_synthetic",
            claim_scope="synthetic_transition_operator_recovery",
            outcomes=outcomes, required=["recovery"],
        )
        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertTrue(any("recovery too low" in r for r in gate["failure_reasons"]))

    def test_gate_blocks_broad_scope(self):
        gate = assemble_gate(
            branch="v3", dataset="d", claim_scope="real_brain_control",
            outcomes=[_outcome("r", True, {"s": 1.0})], required=["r"],
        )
        self.assertFalse(gate["scientific_claim_allowed"])

    def test_extra_finite_blocks_when_false(self):
        gate = assemble_gate(
            branch="v2", dataset="dual_field_synthetic", claim_scope="synthetic_dual_field_recovery",
            outcomes=[_outcome("r", True, {"s": 1.0})], required=["r"], extra_finite=False,
        )
        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertTrue(any("non-finite" in r for r in gate["failure_reasons"]))

    def test_build_and_write_report(self):
        outcomes = [_outcome("r", True, {"s": 1.0})]
        gate = assemble_gate(branch="v2", dataset="dual_field_synthetic",
                             claim_scope="synthetic_dual_field_recovery", outcomes=outcomes, required=["r"])
        report = build_report(schema="kahlus.x.v1", branch="v2", claim_scope="synthetic_dual_field_recovery",
                              seed=0, config={"a": 1}, outcomes=outcomes, gate=gate, extra={"leaderboard": {}})
        for key in ("schema", "branch", "claim_scope", "seed", "config", "diagnostics",
                    "falsification_passed", "scientific_claim_allowed", "failure_reasons", "evidence_gate",
                    "leaderboard"):
            self.assertIn(key, report)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_report(tmp, report=report, gate=gate, prefix="v2")
            self.assertTrue(Path(paths["report"]).name == "v2_benchmark_report.json")
            self.assertTrue(Path(paths["evidence_gate"]).exists())
            loaded = json.loads(Path(paths["report"]).read_text())
            self.assertEqual(loaded["branch"], "v2")


if __name__ == "__main__":
    unittest.main()
