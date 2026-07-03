import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurotwin.researchdock.observation_model import (
    ResearchDockObservationModelConfig,
    build_researchdock_observation_task,
    run_researchdock_observation_benchmark,
    write_researchdock_observation_artifacts,
)
from neurotwin.researchdock.synthetic import make_synthetic_researchdock_sessions


class ResearchDockRD2ObservationModelTests(unittest.TestCase):
    def test_observation_task_is_subject_held_out_and_finite(self):
        sessions = make_synthetic_researchdock_sessions(seed=0)
        task = build_researchdock_observation_task(sessions, seed=0)

        self.assertEqual(task.task_id, "researchdock_multimodal_observation")
        self.assertGreater(task.x_train.shape[0], 0)
        self.assertGreater(task.x_test.shape[0], 0)
        self.assertEqual(task.x_train.shape[1], len(task.feature_names))
        self.assertEqual(task.y_train.shape[1], len(task.target_names))
        self.assertTrue(np.isfinite(task.x_train).all())
        self.assertTrue(np.isfinite(task.y_test).all())
        self.assertFalse(set(task.train_subjects) & set(task.test_subjects))
        self.assertIn("pupil_diameter", task.target_names)
        self.assertIn("hrv_proxy", task.target_names)

    def test_baselines_are_reported_before_observation_operator(self):
        task = build_researchdock_observation_task(make_synthetic_researchdock_sessions(seed=2), seed=2)
        result = run_researchdock_observation_benchmark(
            task,
            config=ResearchDockObservationModelConfig(latent_dim=2, ridge_alpha=1e-2),
        )

        self.assertEqual(result["model_order"][:2], ["train_mean", "linear_ridge"])
        self.assertEqual(result["model_order"][-1], "researchdock_observation_operator")
        self.assertIn("baseline_ranking", result)
        self.assertIn("researchdock_observation_operator", result["metrics_by_model"])
        for metrics in result["metrics_by_model"].values():
            self.assertTrue(np.isfinite(metrics["mse"]))
            self.assertTrue(np.isfinite(metrics["mae"]))
        self.assertIsInstance(result["observation_operator_beats_best_baseline"], bool)
        self.assertEqual(result["claim_boundary"], "synthetic_pretraining_only_no_clinical_claim")

    def test_artifact_writer_outputs_metrics_baselines_and_report(self):
        task = build_researchdock_observation_task(make_synthetic_researchdock_sessions(seed=3), seed=3)
        result = run_researchdock_observation_benchmark(task)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_researchdock_observation_artifacts(tmp, task=task, result=result)

            self.assertTrue(paths["metrics"].exists())
            self.assertTrue(paths["baseline_table_csv"].exists())
            rows = list(csv.DictReader(paths["baseline_table_csv"].read_text(encoding="utf-8").splitlines()))
            self.assertEqual(rows[0]["model_id"], "train_mean")
            self.assertIn("researchdock_observation_operator", {row["model_id"] for row in rows})
            report = paths["report"].read_text(encoding="utf-8")
            self.assertIn("Baselines First", report)
            self.assertIn("not diagnosis", report)

    def test_script_writes_rd2_observation_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_researchdock_synthetic.py",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--run-observation-model",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )

            root = Path(tmp)
            self.assertIn("observation_model_metrics=", result.stdout)
            self.assertTrue((root / "researchdock_observation_metrics.json").exists())
            self.assertTrue((root / "researchdock_observation_baselines.csv").exists())
            payload = json.loads((root / "researchdock_observation_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["task_id"], "researchdock_multimodal_observation")
            self.assertEqual(payload["model_order"][0], "train_mean")


if __name__ == "__main__":
    unittest.main()
