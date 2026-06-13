import unittest

import numpy as np

from neurotwin.gates import evaluate_gate
from neurotwin.transition_gym import SyntheticWorldConfig, build_transition_gym
from neurotwin.transition_gym.benchmark import benchmark_report, run_v3_benchmark
from neurotwin.transition_gym.operator_recovery import (
    heldout_composition_recovery,
    non_commutativity_score,
    operator_recovery,
    response_profile_distances,
)

_CFG = SyntheticWorldConfig(n_episodes=96, seed=0)


class OperatorRecoveryDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.bundle = build_transition_gym(_CFG)

    def _finite(self, outcome):
        for value in outcome.detail.values():
            if isinstance(value, dict):
                self.assertTrue(np.isfinite(list(value.values())).all(), outcome.name)
            elif isinstance(value, (int, float)):
                self.assertTrue(np.isfinite(value), outcome.name)

    def test_operator_recovery_passes_and_finite(self):
        out = operator_recovery(self.bundle)
        self._finite(out)
        self.assertTrue(out.passed)
        self.assertGreaterEqual(out.detail["mean_recovery_score"], 0.9)

    def test_heldout_compositions_actually_held_out(self):
        heldout = set(self.bundle.splits.heldout_compositions)
        train = set(self.bundle.splits.train_compositions)
        self.assertTrue(len(heldout) > 0)
        self.assertEqual(heldout & train, set())  # genuinely held out

    def test_heldout_composition_recovery_passes(self):
        out = heldout_composition_recovery(self.bundle)
        self._finite(out)
        self.assertTrue(out.passed)

    def test_non_commutativity_present(self):
        out = non_commutativity_score(self.bundle)
        self._finite(out)
        self.assertTrue(out.passed)
        self.assertGreater(out.detail["normalized_gap"], 0.0)

    def test_response_profile_distances_finite(self):
        out = response_profile_distances(self.bundle)
        self._finite(out)
        self.assertGreater(out.detail["operator_induced_distance"], 0.0)

    def test_split_integrity_no_leakage(self):
        self.bundle.splits.assert_no_episode_leakage()
        self.bundle.splits.assert_no_composition_leakage()


class V3BenchmarkTests(unittest.TestCase):
    def test_benchmark_passes_and_allows_narrow_claim(self):
        result = run_v3_benchmark(_CFG, seed=0)
        self.assertTrue(result.passed)
        self.assertEqual(result.gate["branch"], "v3")
        self.assertEqual(result.gate["claim_scope"], "synthetic_transition_operator_recovery")
        self.assertTrue(result.gate["scientific_claim_allowed"])
        self.assertEqual(result.failure_reasons, [])

    def test_leaderboard_non_empty(self):
        result = run_v3_benchmark(_CFG, seed=0)
        self.assertGreater(len(result.leaderboard), 0)
        self.assertIn("ridge", result.leaderboard)
        self.assertIn("retrieval_knn", result.leaderboard)
        self.assertIn("ktm", result.leaderboard)
        for metrics in result.leaderboard.values():
            self.assertTrue(np.isfinite(list(metrics.values())).all())

    def test_ktm_vs_baseline_reported(self):
        result = run_v3_benchmark(_CFG, seed=0)
        self.assertIsInstance(result.ktm_beats_baselines, bool)
        report = benchmark_report(result)
        self.assertIn("ktm_beats_baselines", report)
        self.assertIn("baseline_leaderboard", report)

    def test_broad_claim_blocked(self):
        gate = evaluate_gate(
            branch="v3", dataset="transition_gym_synthetic", split_audit_passed=True,
            baseline_table_present=True, finite_metrics=True, calibration_checked=True,
            claim_scope="real_brain_state_control",  # broad/forbidden scope
        )
        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertTrue(any("claim scope too broad" in r for r in gate["failure_reasons"]))

    def test_narrow_claim_requires_all_checks(self):
        # Missing calibration (a required diagnostic failed) must block the narrow claim.
        gate = evaluate_gate(
            branch="v3", dataset="transition_gym_synthetic", split_audit_passed=True,
            baseline_table_present=True, finite_metrics=True, calibration_checked=False,
            claim_scope="synthetic_transition_operator_recovery",
        )
        self.assertFalse(gate["scientific_claim_allowed"])

    def test_honest_fail_on_underdetermined_world(self):
        # Too few episodes -> operator recovery underdetermined -> claim blocked honestly.
        degenerate = SyntheticWorldConfig(n_episodes=10, state_dim=6, seed=0)
        result = run_v3_benchmark(degenerate, seed=0)
        self.assertFalse(result.passed)
        self.assertFalse(result.gate["scientific_claim_allowed"])
        self.assertTrue(len(result.failure_reasons) > 0)

    def test_report_schema(self):
        report = benchmark_report(run_v3_benchmark(_CFG, seed=0))
        for key in ("branch", "claim_scope", "seed", "diagnostics", "baseline_leaderboard",
                    "operator_recovery_scores", "heldout_composition_scores", "non_commutativity_gap",
                    "ktm_beats_baselines", "falsification_passed", "scientific_claim_allowed",
                    "failure_reasons", "evidence_gate"):
            self.assertIn(key, report)


if __name__ == "__main__":
    unittest.main()
