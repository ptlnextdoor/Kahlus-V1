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
from neurotwin.eval.paper_gate import validate_paper_mode_payload
from neurotwin.models.baselines import NumpyRidgeBaseline
from neurotwin.models.tribe_style import TribeStyleModel


class BaselineSuiteTests(unittest.TestCase):
    def test_synthetic_baseline_suite_runs_all_required_models(self):
        payload = run_synthetic_baseline_suite(seed=4, train_steps=1)

        self.assertEqual(payload["scope"]["status"], "synthetic-only")
        future = payload["tasks"]["future_state_forecasting"]
        self.assertEqual(future["status"], "completed")
        self.assertIn("persistence", future["metrics_by_model"])
        self.assertIn("train_mean", future["metrics_by_model"])
        self.assertIn("random_permutation", future["metrics_by_model"])
        self.assertIn("linear_ridge", future["metrics_by_model"])
        self.assertIn("autoregressive_ridge", future["metrics_by_model"])
        self.assertIn("neurotwin", future["metrics_by_model"])
        self.assertIn("mse_ci_low", future["metrics_by_model"]["linear_ridge"])
        self.assertIn("mse_ci_high", future["metrics_by_model"]["linear_ridge"])
        self.assertTrue(payload["aggregate"]["aggregate_rank"])
        self.assertIn("baseline_catalog", payload)
        catalog_ids = {row["model_id"] for row in payload["baseline_catalog"]}
        self.assertIn("persistence", catalog_ids)
        self.assertIn("train_mean", catalog_ids)
        self.assertIn("random_permutation", catalog_ids)
        self.assertIn("autoregressive_ridge", catalog_ids)
        self.assertEqual(payload["seeds"], [4])
        self.assertEqual(payload["benchmark_contract"]["required_seeds"], [0, 1, 2])

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

    def test_tribe_style_runs_on_stimulus_fmri_task(self):
        rng = np.random.default_rng(7)
        x_train = rng.normal(size=(10, 4, 3)).astype(np.float32)
        y_train = rng.normal(size=(10, 4, 5)).astype(np.float32)
        x_test = rng.normal(size=(4, 4, 3)).astype(np.float32)
        y_test = rng.normal(size=(4, 4, 5)).astype(np.float32)
        task = SupervisedWindowTask(
            task_id="stimulus_to_fmri_response",
            source_modality="stimulus",
            target_modality="fmri",
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            notes=("tribe_style is a clean_room_approximation",),
        )

        payload = run_supervised_window_tasks((task,), seed=0, train_steps=1)
        result = payload["tasks"]["stimulus_to_fmri_response"]
        metrics = result["metrics_by_model"]["tribe_style"]
        catalog = {row["model_id"]: row for row in payload["baseline_catalog"]}

        self.assertIn("mse_ci_low", metrics)
        self.assertIn("mse_ci_high", metrics)
        self.assertTrue(np.isfinite(metrics["mse"]))
        self.assertIn("tribe_style", {row["model_id"] for row in result["ranking"]})
        self.assertEqual(catalog["tribe_style"]["status"], "clean_room_approximation")
        self.assertIn("not an exact TRIBE v2 reproduction", catalog["tribe_style"]["notes"])

    def test_tribe_style_facade_predicts_without_external_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            text_path = Path(tmp) / "stimulus.txt"
            text_path.write_text("visual auditory language stimulus", encoding="utf-8")
            model = TribeStyleModel.from_pretrained(
                "local",
                config_update={"stimulus_dim": 6, "output_dim": 4, "hidden_dim": 8},
            )

            events = model.get_events_dataframe(text_path=str(text_path))
            preds, segments = model.predict(events, verbose=False)

        self.assertEqual(model.model_id, "tribe_style")
        self.assertEqual(model.implementation_status, "clean_room_approximation")
        self.assertEqual(preds.shape, (4, 4))
        self.assertEqual(len(segments), 4)
        self.assertTrue(np.isfinite(preds).all())

    def test_paper_mode_gate_enforces_audit_seeds_ranking_and_ci_contract(self):
        payload = run_synthetic_baseline_suite(seed=0, train_steps=1)

        missing_seed_report = validate_paper_mode_payload(payload, audit_report={"passed": True})
        self.assertFalse(missing_seed_report.passed)
        self.assertTrue(any("missing 1,2" in violation for violation in missing_seed_report.violations))
        self.assertEqual(missing_seed_report.required_seeds, (0, 1, 2))
        with self.assertRaises(TypeError):
            validate_paper_mode_payload(payload, audit_report={"passed": True}, required_seeds=(0,))  # type: ignore[call-arg]

        payload["paper_mode_contract"] = {"observed_seeds": [0, 1, 2]}
        metadata_only_report = validate_paper_mode_payload(payload, audit_report={"passed": True})
        self.assertFalse(metadata_only_report.passed)
        self.assertEqual(metadata_only_report.observed_seeds, (0,))
        self.assertTrue(any("missing 1,2" in violation for violation in metadata_only_report.violations))

        invalid_seed_payload = json.loads(json.dumps(payload))
        invalid_seed_payload["seed_results"] = [
            {"seed": 1, "aggregate": {"aggregate_rank": [{"model_id": "linear_ridge", "mean_rank": 1.0}]}},
            {
                "seed": 2,
                "tasks": {
                    "future_state_forecasting": {
                        "status": "completed",
                        "metrics": {"mse": 0.42},
                    }
                },
            },
        ]
        invalid_seed_report = validate_paper_mode_payload(invalid_seed_payload, audit_report={"passed": True})
        self.assertFalse(invalid_seed_report.passed)
        self.assertNotIn(1, invalid_seed_report.observed_seeds)
        self.assertNotIn(2, invalid_seed_report.observed_seeds)
        self.assertTrue(any("seed 2:future_state_forecasting:metrics:mse lacks finite CI" in violation for violation in invalid_seed_report.violations))

        payload["seed_results"] = {
            "1": {
                "task_results": [
                    {
                        "task_id": "future_state_forecasting",
                        "test_mse": 0.31,
                        "test_mse_ci_low": 0.25,
                        "test_mse_ci_high": 0.37,
                    }
                ]
            },
            "2": {
                "tasks": {
                    "future_state_forecasting": {
                        "status": "completed",
                        "metrics": {"mse": 0.42, "mse_ci_low": 0.35, "mse_ci_high": 0.50},
                    }
                }
            },
        }
        passed_report = validate_paper_mode_payload(payload, audit_report={"passed": True})
        self.assertTrue(passed_report.passed)
        self.assertEqual(passed_report.observed_seeds, (0, 1, 2))

        failed_audit_report = validate_paper_mode_payload(payload, audit_report={"passed": False, "violations": ["leakage"]})
        self.assertFalse(failed_audit_report.passed)
        self.assertTrue(any("audit did not pass" in violation for violation in failed_audit_report.violations))

        no_ranking = dict(payload)
        no_ranking["aggregate"] = {"aggregate_rank": []}
        no_ranking_report = validate_paper_mode_payload(no_ranking, audit_report={"passed": True})
        self.assertFalse(no_ranking_report.passed)
        self.assertTrue(any("aggregate_rank is empty" in violation for violation in no_ranking_report.violations))

        no_ci = json.loads(json.dumps(payload))
        future_metrics = no_ci["tasks"]["future_state_forecasting"]["metrics_by_model"]
        first_model = next(iter(future_metrics))
        future_metrics[first_model].pop("mse_ci_low", None)
        no_ci_report = validate_paper_mode_payload(no_ci, audit_report={"passed": True})
        self.assertFalse(no_ci_report.passed)
        self.assertTrue(any("lacks finite mse CI" in violation for violation in no_ci_report.violations))

        auxiliary_no_ci = json.loads(json.dumps(payload))
        auxiliary_no_ci["tasks"]["few_shot_subject_adaptation"] = {
            "status": "completed",
            "metrics": {"adapted_mse": 0.25, "support_minutes": 1.0},
        }
        auxiliary_no_ci_report = validate_paper_mode_payload(auxiliary_no_ci, audit_report={"passed": True})
        self.assertFalse(auxiliary_no_ci_report.passed)
        self.assertTrue(
            any(
                "few_shot_subject_adaptation:metrics:adapted_mse lacks finite CI" in violation
                for violation in auxiliary_no_ci_report.violations
            )
        )

        auxiliary_no_ci["tasks"]["few_shot_subject_adaptation"]["metrics"]["adapted_mse_ci_low"] = 0.20
        auxiliary_no_ci["tasks"]["few_shot_subject_adaptation"]["metrics"]["adapted_mse_ci_high"] = 0.30
        auxiliary_ci_report = validate_paper_mode_payload(auxiliary_no_ci, audit_report={"passed": True})
        self.assertTrue(auxiliary_ci_report.passed)

        top_level_no_ci = json.loads(json.dumps(payload))
        top_level_no_ci["test_mse"] = 0.12
        top_level_no_ci_report = validate_paper_mode_payload(top_level_no_ci, audit_report={"passed": True})
        self.assertFalse(top_level_no_ci_report.passed)
        self.assertTrue(
            any("top-level test reporting:test_mse lacks finite CI" in violation for violation in top_level_no_ci_report.violations)
        )

        top_level_no_ci["test_mse_ci_low"] = 0.10
        top_level_no_ci["test_mse_ci_high"] = 0.14
        top_level_ci_report = validate_paper_mode_payload(top_level_no_ci, audit_report={"passed": True})
        self.assertTrue(top_level_ci_report.passed)

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
