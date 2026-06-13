import unittest

import numpy as np

from neurotwin.models.dual_field import DualFieldConfig, simulate_dual_field
from neurotwin.models.dual_field.benchmark import benchmark_report, run_v2_benchmark
from neurotwin.models.dual_field.diagnostics import (
    bold_dependence,
    eeg_dependence,
    fast_latent_recovery,
    lag_recovery,
    long_rollout_stability,
    one_vs_two_field_forecast,
    slow_latent_recovery,
)

_CFG = DualFieldConfig(n_samples=64, time_steps=48, seed=0)


class DualFieldDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.rollout = simulate_dual_field(_CFG)

    def _assert_finite_detail(self, outcome):
        for value in outcome.detail.values():
            self.assertTrue(np.isfinite(value), f"{outcome.name}:{value}")

    def test_fast_latent_recovery_passes(self):
        out = fast_latent_recovery(self.rollout)
        self._assert_finite_detail(out)
        self.assertTrue(out.passed)
        self.assertGreater(out.detail["r2"], 0.5)

    def test_slow_latent_recovery_passes(self):
        out = slow_latent_recovery(self.rollout)
        self._assert_finite_detail(out)
        self.assertTrue(out.passed)

    def test_eeg_is_fast_dominated(self):
        out = eeg_dependence(self.rollout)
        self._assert_finite_detail(out)
        self.assertTrue(out.passed)
        self.assertGreater(out.detail["r2_from_fast_N"], out.detail["r2_from_slow_H"])

    def test_bold_is_slow_dominated(self):
        out = bold_dependence(self.rollout)
        self._assert_finite_detail(out)
        self.assertTrue(out.passed)
        self.assertGreater(out.detail["r2_from_slow_H"], out.detail["r2_from_fast_N_current"])

    def test_bold_is_lagged(self):
        out = lag_recovery(self.rollout)
        self._assert_finite_detail(out)
        self.assertTrue(out.passed)
        self.assertGreaterEqual(out.detail["recovered_lag"], 1.0)
        self.assertGreater(out.detail["r2_at_recovered"], out.detail["r2_at_lag0"])

    def test_long_rollout_stable(self):
        out = long_rollout_stability(_CFG, time_steps=200)
        self._assert_finite_detail(out)
        self.assertTrue(out.passed)
        self.assertLessEqual(out.detail["max_abs"], _CFG.state_clip)

    def test_two_field_beats_one_field(self):
        out = one_vs_two_field_forecast(self.rollout)
        self._assert_finite_detail(out)
        self.assertTrue(out.passed)
        self.assertLessEqual(out.detail["two_field_bold_mse"], out.detail["one_field_bold_mse"])


class DualFieldBenchmarkTests(unittest.TestCase):
    def test_benchmark_passes_and_allows_narrow_claim(self):
        result = run_v2_benchmark(_CFG, seed=0)
        self.assertTrue(result.passed)
        self.assertEqual(result.gate["branch"], "v2")
        self.assertEqual(result.gate["claim_scope"], "synthetic_dual_field_recovery")
        self.assertTrue(result.gate["scientific_claim_allowed"])
        self.assertEqual(result.failure_reasons, [])
        self.assertEqual(len(result.outcomes), 7)

    def test_benchmark_deterministic(self):
        a = run_v2_benchmark(_CFG, seed=3)
        b = run_v2_benchmark(_CFG, seed=3)
        self.assertEqual(
            [o.detail for o in a.outcomes],
            [o.detail for o in b.outcomes],
        )

    def test_report_schema(self):
        report = benchmark_report(run_v2_benchmark(_CFG, seed=0))
        for key in ("branch", "claim_scope", "seed", "diagnostics", "falsification_passed",
                    "scientific_claim_allowed", "failure_reasons", "evidence_gate"):
            self.assertIn(key, report)
        self.assertEqual(len(report["diagnostics"]), 7)

    def test_honest_fail_on_degenerate_data(self):
        # Tiny noisy config: recovery/structure cannot be established -> claim must be blocked.
        degenerate = DualFieldConfig(n_samples=6, time_steps=12, hemo_lag=2, noise_scale=0.8, seed=0)
        result = run_v2_benchmark(degenerate, seed=0, long_rollout_steps=60)
        self.assertFalse(result.passed)
        self.assertFalse(result.gate["scientific_claim_allowed"])
        self.assertTrue(len(result.failure_reasons) > 0)


if __name__ == "__main__":
    unittest.main()
