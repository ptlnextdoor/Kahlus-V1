import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurotwin.benchmarks.baseline_suite import run_synthetic_baseline_suite


class BaselineSuiteTests(unittest.TestCase):
    def test_synthetic_baseline_suite_runs_all_required_models(self):
        payload = run_synthetic_baseline_suite(seed=4, train_steps=1)

        self.assertEqual(payload["scope"]["status"], "synthetic-only")
        future = payload["tasks"]["future_state_forecasting"]
        self.assertEqual(future["status"], "completed")
        self.assertIn("linear_ridge", future["metrics_by_model"])
        self.assertIn("neurotwin", future["metrics_by_model"])
        self.assertIn("mse_ci_low", future["metrics_by_model"]["linear_ridge"])
        self.assertIn("mse_ci_high", future["metrics_by_model"]["linear_ridge"])
        self.assertTrue(payload["aggregate"]["aggregate_rank"])

    def test_eval_suite_writes_baseline_artifact(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "eval",
                    "--suite",
                    "neural_translation_v1",
                    "--out-dir",
                    str(out_dir),
                ],
                check=True,
                text=True,
                capture_output=True,
                env=env,
            )

            baseline_path = out_dir / "baseline_suite.json"
            self.assertTrue(baseline_path.exists())
            payload = json.loads(baseline_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["scope"]["status"], "synthetic-only")
            self.assertIn("local_baseline_suite", result.stdout)


if __name__ == "__main__":
    unittest.main()
