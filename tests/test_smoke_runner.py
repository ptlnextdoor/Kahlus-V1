import unittest

from neurotwin.benchmarks.smoke import format_smoke_results, run_translation_smoke


class SmokeRunnerTests(unittest.TestCase):
    def test_smoke_runner_returns_ranked_synthetic_results(self):
        result = run_translation_smoke(seed=5)

        self.assertTrue(result.leakage_passed)
        self.assertIn("neurotwin", result.metrics)
        self.assertIn("transformer", result.metrics)
        self.assertIn("mamba_ssm", result.metrics)
        self.assertEqual(result.ranking[0].metric, "mse")

    def test_smoke_report_mentions_synthetic_only(self):
        report = format_smoke_results(run_translation_smoke(seed=5))

        self.assertIn("synthetic-only", report)
        self.assertIn("aggregate_rank", report)
        self.assertIn("neurotwin", report)
