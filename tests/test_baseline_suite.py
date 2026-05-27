import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

from neurotwin.benchmarks.baseline_suite import SupervisedWindowTask, run_supervised_window_tasks, run_synthetic_baseline_suite
from neurotwin.models.baselines import NumpyRidgeBaseline


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
        self.assertIn("baseline_catalog", payload)

    def test_ridge_baseline_is_finite_for_ill_conditioned_windows(self):
        x = np.ones((12, 3), dtype=np.float64)
        x[:, 1] = np.linspace(0.0, 1e12, 12)
        x[:, 2] = x[:, 1] + 1e-6
        y = np.stack([x[:, 1] * 0.5, x[:, 2] * -0.25], axis=1)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            model = NumpyRidgeBaseline(alpha=1e-2).fit(x, y)
            pred = model.predict(x)

        self.assertTrue(np.isfinite(pred).all())
        self.assertEqual(pred.shape, y.shape)

    def test_failed_baseline_is_recorded_and_excluded_from_ranking(self):
        x = np.random.default_rng(0).normal(size=(6, 4, 2)).astype(np.float32)
        task = SupervisedWindowTask(
            task_id="future_state_forecasting",
            source_modality="eeg",
            target_modality="eeg",
            x_train=x[:4],
            y_train=x[:4],
            x_test=x[4:],
            y_test=x[4:],
        )

        with mock.patch(
            "neurotwin.benchmarks.baseline_suite._fit_ridge",
            return_value=np.full_like(task.y_test, np.nan),
        ):
            payload = run_supervised_window_tasks((task,), seed=0, train_steps=1)

        failures = payload["baseline_failures"]
        ranked = {row["model_id"] for row in payload["tasks"]["future_state_forecasting"]["ranking"]}
        self.assertTrue(any(row["model_id"] == "linear_ridge" for row in failures))
        self.assertNotIn("linear_ridge", ranked)

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
