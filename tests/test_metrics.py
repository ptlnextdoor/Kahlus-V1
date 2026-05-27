import unittest

import numpy as np

from neurotwin.eval.metrics import bandpower_error, bootstrap_ci, mse, pearsonr, rank_models, regionwise_pearsonr, spearmanr


class MetricTests(unittest.TestCase):
    def test_mse_and_pearsonr(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.5, 2.5])

        self.assertAlmostEqual(mse(y_true, y_pred), 1.0 / 6.0)
        self.assertGreater(pearsonr(y_true, y_pred), 0.75)
        self.assertGreater(spearmanr(y_true, y_pred), 0.75)

    def test_neural_signal_metrics_are_finite(self):
        y_true = np.ones((2, 8, 3), dtype=np.float32)
        y_pred = y_true * 0.9

        self.assertTrue(np.isfinite(bandpower_error(y_true, y_pred)))
        self.assertTrue(np.isfinite(regionwise_pearsonr(y_true, y_pred)))

    def test_rank_models_lower_is_better(self):
        ranked = rank_models(
            {
                "transformer": {"mse": 0.31},
                "mamba_ssm": {"mse": 0.28},
                "neurotwin": {"mse": 0.24},
            },
            metric="mse",
            higher_is_better=False,
        )

        self.assertEqual([row.model_id for row in ranked], ["neurotwin", "mamba_ssm", "transformer"])

    def test_bootstrap_ci_is_deterministic(self):
        low, high = bootstrap_ci(np.array([1.0, 2.0, 3.0, 4.0]), seed=11, n_boot=128)

        self.assertLess(low, high)
        self.assertGreaterEqual(low, 1.0)
        self.assertLessEqual(high, 4.0)

    def test_bootstrap_ci_caps_large_inputs_deterministically(self):
        values = np.arange(50_000, dtype=float)

        first = bootstrap_ci(values, seed=7, n_boot=25, max_values=1_000)
        second = bootstrap_ci(values, seed=7, n_boot=25, max_values=1_000)

        self.assertEqual(first, second)
        self.assertLess(first[0], first[1])
