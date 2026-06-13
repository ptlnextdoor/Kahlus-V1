import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurotwin.baseline_runner import (
    DEFAULT_MODELS,
    dual_field_regression_task,
    run_baselines,
    transition_gym_regression_task,
    write_run_artifacts,
)
from neurotwin.models.dual_field import DualFieldConfig
from neurotwin.transition_gym import SyntheticWorldConfig


class BaselineRunnerTests(unittest.TestCase):
    def test_dual_field_task_is_leakage_safe_shape(self):
        task = dual_field_regression_task(DualFieldConfig(seed=0, n_samples=16, time_steps=16), window=4)
        self.assertEqual(task.branch, "v2")
        self.assertEqual(task.x_train.shape[1], 4)
        self.assertEqual(task.x_train.shape[-1], task.y_train.shape[-1])
        self.assertGreater(task.x_test.shape[0], 0)

    def test_runner_produces_finite_metrics_and_ranking(self):
        task = dual_field_regression_task(DualFieldConfig(seed=1, n_samples=16, time_steps=16), window=4)
        result = run_baselines(task, train_steps=5, seed=0)
        self.assertGreater(len(result.metrics_by_model), 0)
        for metrics in result.metrics_by_model.values():
            for value in metrics.values():
                self.assertTrue(np.isfinite(value))
        self.assertEqual(len(result.ranking), len(result.metrics_by_model))

    def test_required_baselines_attempted(self):
        task = dual_field_regression_task(DualFieldConfig(seed=2, n_samples=12, time_steps=14), window=3)
        result = run_baselines(task, train_steps=3, seed=0)
        # ridge / autoregressive_ridge / mlp / transformer / ssm_fallback should complete.
        for model_id in ("ridge", "autoregressive_ridge", "mlp", "transformer", "ssm_fallback"):
            self.assertIn(model_id, result.metrics_by_model)
        # nfc is importable but intentionally skipped; recorded honestly.
        self.assertTrue(any(reason.startswith("nfc:") for reason in result.failure_reasons))

    def test_gate_blocks_claim_without_calibration(self):
        task = dual_field_regression_task(DualFieldConfig(seed=3, n_samples=12, time_steps=14), window=3)
        result = run_baselines(task, train_steps=3, seed=0)
        self.assertFalse(result.evidence_gate["scientific_claim_allowed"])
        self.assertTrue(result.evidence_gate["baseline_table_present"])
        self.assertTrue(any("calibration" in r for r in result.evidence_gate["failure_reasons"]))

    def test_transition_gym_task_runs(self):
        task = transition_gym_regression_task(SyntheticWorldConfig(seed=0, n_episodes=32, history_len=5, horizon=4))
        result = run_baselines(task, train_steps=3, seed=0)
        self.assertEqual(task.branch, "v3")
        self.assertGreater(len(result.metrics_by_model), 0)

    def test_write_artifacts(self):
        task = dual_field_regression_task(DualFieldConfig(seed=4, n_samples=12, time_steps=14), window=3)
        result = run_baselines(task, train_steps=3, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_run_artifacts(tmp, task, result)
            for key in ("metrics", "baseline_table_json", "baseline_table_csv", "evidence_gate", "run_config"):
                self.assertTrue(Path(paths[key]).exists())
            gate = json.loads(Path(paths["evidence_gate"]).read_text())
            for field_name in (
                "branch",
                "dataset",
                "split_audit_passed",
                "baseline_table_present",
                "finite_metrics",
                "calibration_checked",
                "claim_scope",
                "scientific_claim_allowed",
                "failure_reasons",
            ):
                self.assertIn(field_name, gate)
            csv_text = Path(paths["baseline_table_csv"]).read_text()
            self.assertIn("model_id", csv_text)

    def test_default_models_constant(self):
        self.assertIn("ridge", DEFAULT_MODELS)
        self.assertIn("nfc", DEFAULT_MODELS)


if __name__ == "__main__":
    unittest.main()
