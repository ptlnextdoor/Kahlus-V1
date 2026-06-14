import unittest

from neurotwin.training_v3.metrics_eval import BASELINE_BUDGET_POLICY, ktm_vs_baselines

_BASELINES = {
    "ridge": {"mse": 0.0030},
    "mlp": {"mse": 0.0010},
}


class KtmVsBaselinesTests(unittest.TestCase):
    def test_clear_win_matched_budget_earns_claim(self):
        out = ktm_vs_baselines(
            0.0006, _BASELINES,
            ktm_train_steps=400, baseline_train_steps=400,
            ktm_world_size=7, ktm_global_batch_size=511, margin=0.05,
        )
        self.assertEqual(out["best_baseline"], "mlp")
        self.assertTrue(out["budget_matched"])
        self.assertTrue(out["comparison_locked"])
        self.assertGreater(out["relative_improvement"], 0.05)
        self.assertTrue(out["ktm_beats_baselines"])
        self.assertEqual(out["budget"]["baseline_budget_policy"], BASELINE_BUDGET_POLICY)
        self.assertEqual(out["budget"]["ktm_world_size"], 7)
        self.assertEqual(out["budget"]["ktm_global_batch_size"], 511)

    def test_unmatched_budget_blocks_claim_even_if_mse_lower(self):
        # The Sprint 2D regression: KTM MSE < baseline MSE, but baselines ran a shorter budget.
        out = ktm_vs_baselines(
            0.0006, _BASELINES,
            ktm_train_steps=400, baseline_train_steps=60, margin=0.05,
        )
        self.assertLess(out["ktm_mse"], out["best_baseline_mse"])
        self.assertFalse(out["budget_matched"])
        self.assertFalse(out["comparison_locked"])
        self.assertFalse(out["ktm_beats_baselines"])

    def test_win_below_margin_blocks_claim(self):
        # Matched budget, but the KTM edge is under the required relative margin.
        out = ktm_vs_baselines(
            0.00099, _BASELINES,
            ktm_train_steps=400, baseline_train_steps=400, margin=0.05,
        )
        self.assertTrue(out["budget_matched"])
        self.assertLess(out["relative_improvement"], 0.05)
        self.assertFalse(out["ktm_beats_baselines"])

    def test_missing_step_metadata_is_unlocked(self):
        out = ktm_vs_baselines(0.0006, _BASELINES, margin=0.05)
        self.assertFalse(out["budget_matched"])
        self.assertFalse(out["ktm_beats_baselines"])

    def test_no_baselines_blocks_claim(self):
        out = ktm_vs_baselines(
            0.0006, {}, ktm_train_steps=400, baseline_train_steps=400, margin=0.05,
        )
        self.assertIsNone(out["best_baseline"])
        self.assertFalse(out["ktm_beats_baselines"])


if __name__ == "__main__":
    unittest.main()
