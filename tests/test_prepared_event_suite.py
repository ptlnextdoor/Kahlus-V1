import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurotwin.adapters.synthetic import make_synthetic_event_batches, make_synthetic_recordings
from neurotwin.benchmarks.prepared_suite import PreparedSuiteConfig, build_prepared_window_tasks, run_prepared_baseline_suite
from neurotwin.data.event_io import event_manifest_summary, load_event_batches, save_event_batches
from neurotwin.data.split_manifest import build_split_manifest
from neurotwin.data.manifest_io import save_split_manifest


class PreparedEventSuiteTests(unittest.TestCase):
    def test_event_batches_roundtrip_through_manifest(self):
        batches = make_synthetic_event_batches(n_subjects=2, sessions_per_subject=1, modalities=("eeg", "fmri"))
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = save_event_batches(batches, tmp)
            loaded = load_event_batches(manifest_path)
            summary = event_manifest_summary(manifest_path)

        self.assertEqual(len(loaded), len(batches))
        self.assertEqual(loaded[0].signal.shape, batches[0].signal.shape)
        self.assertIn("eeg", summary["modalities"])
        self.assertIn("fmri", summary["modalities"])

    def test_prepared_window_tasks_use_split_manifest(self):
        records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=1, modalities=("eeg", "fmri"))
        batches = make_synthetic_event_batches(n_subjects=6, sessions_per_subject=1, modalities=("eeg", "fmri"))
        split = build_split_manifest(records, policy="subject", seed=0)

        tasks, skipped = build_prepared_window_tasks(batches, split, window_length=8, stride=8)

        self.assertFalse([row for row in skipped if row["task_id"] == "all"])
        self.assertIn("future_state_forecasting", {task.task_id for task in tasks})
        self.assertIn("cross_modal_translation", {task.task_id for task in tasks})

    def test_prepared_baseline_suite_and_cli_artifacts(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        with tempfile.TemporaryDirectory() as tmp:
            prep_dir = Path(tmp) / "prepared"
            eval_dir = Path(tmp) / "eval"
            prepare = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "data",
                    "prepare",
                    "--dataset",
                    "synthetic",
                    "--split",
                    "subject",
                    "--out-dir",
                    str(prep_dir),
                ],
                check=True,
                text=True,
                capture_output=True,
                env=env,
            )
            self.assertIn("event_manifest=", prepare.stdout)

            payload = run_prepared_baseline_suite(
                PreparedSuiteConfig(
                    event_manifest=prep_dir / "event_manifest.json",
                    split_manifest=prep_dir / "split_manifest.json",
                    train_steps=1,
                ),
                out_dir=eval_dir,
            )
            self.assertEqual(payload["scope"]["status"], "prepared-synthetic")
            self.assertTrue((eval_dir / "prepared_baseline_suite.json").exists())
            self.assertTrue((eval_dir / "baseline_failures.json").exists())
            self.assertIn("few_shot_subject_adaptation", payload["tasks"])
            self.assertIn("dataset_site_generalization", payload["tasks"])

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "eval",
                    "--suite",
                    "neural_translation_v1",
                    "--event-manifest",
                    str(prep_dir / "event_manifest.json"),
                    "--split-manifest",
                    str(prep_dir / "split_manifest.json"),
                    "--out-dir",
                    str(eval_dir),
                    "--train-steps",
                    "1",
                ],
                check=True,
                text=True,
                capture_output=True,
                env=env,
            )

            artifact = json.loads((eval_dir / "prepared_baseline_suite.json").read_text(encoding="utf-8"))
            self.assertIn("Prepared Baseline Suite", result.stdout)
            self.assertIn("ci95=", result.stdout)
            self.assertIn("baseline_catalog", result.stdout)
            self.assertIn("few_shot_subject_adaptation", result.stdout)
            self.assertIn("dataset_site_generalization", result.stdout)
            future = artifact["tasks"]["future_state_forecasting"]["metrics_by_model"]["linear_ridge"]
            self.assertIn("mse_ci_low", future)
            self.assertIn("mse_ci_high", future)
            self.assertTrue(artifact["aggregate"]["aggregate_rank"])

            paper_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "eval",
                    "--suite",
                    "neural_translation_v1",
                    "--event-manifest",
                    str(prep_dir / "event_manifest.json"),
                    "--split-manifest",
                    str(prep_dir / "split_manifest.json"),
                    "--out-dir",
                    str(eval_dir),
                    "--train-steps",
                    "1",
                    "--paper-mode",
                ],
                text=True,
                capture_output=True,
                env=env,
            )
            self.assertNotEqual(paper_result.returncode, 0)
            self.assertIn("paper_mode_gate=True", paper_result.stdout)
            self.assertIn("paper_mode_passed=False", paper_result.stdout)
            self.assertIn("missing 1,2", paper_result.stdout)
            self.assertTrue((eval_dir / "paper_mode_gate.json").exists())


if __name__ == "__main__":
    unittest.main()
