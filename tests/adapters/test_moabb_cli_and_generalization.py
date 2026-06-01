import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurotwin.benchmarks.suite import run_neural_translation_v1_synthetic
from neurotwin.benchmarks.tasks import run_dataset_site_generalization_task


class MoabbCliAndGeneralizationTests(unittest.TestCase):
    def run_cli_no_check(self, *args: str, env_overrides: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        if env_overrides:
            env.update(env_overrides)
        return subprocess.run(
            [sys.executable, "-m", "neurotwin.cli", *args],
            text=True,
            capture_output=True,
            env=env,
        )

    def test_moabb_prepare_missing_deps_fails_cleanly(self):
        result = self.run_cli_no_check(
            "data",
            "prepare",
            "--dataset",
            "moabb",
            "--split",
            "subject",
            "--moabb-dataset",
            "BNCI2014_001",
            "--max-trials",
            "2",
            env_overrides={"NEUROTWIN_FORCE_MISSING_MOABB": "1"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pip install -e", result.stderr + result.stdout)

    def test_moabb_smoke_missing_deps_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_cli_no_check(
                "data",
                "smoke",
                "--dataset",
                "moabb",
                "--out-dir",
                tmp,
                "--split",
                "subject",
                "--max-trials",
                "2",
                env_overrides={"NEUROTWIN_FORCE_MISSING_MOABB": "1"},
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pip install -e", result.stderr + result.stdout)

    def test_dataset_site_generalization_task(self):
        source = np.ones((5, 3), dtype=np.float32)
        target = np.ones((5, 3), dtype=np.float32) * 1.5

        result = run_dataset_site_generalization_task(source, target, source_name="site-a", target_name="site-b")

        self.assertEqual(result.status, "completed")
        self.assertIn("generalization_mse", result.metrics)
        self.assertIn("site-a", result.notes[0])

    def test_neural_translation_suite_includes_generalization(self):
        payload = run_neural_translation_v1_synthetic(seed=7)

        self.assertIn("dataset_site_generalization", payload)

    @unittest.skipUnless(os.environ.get("NEUROTWIN_RUN_REAL_MOABB_TESTS") == "1", "real MOABB integration disabled")
    def test_real_moabb_smoke_script_creates_windows(self):
        env = dict(os.environ)
        env.update({"MAX_TRIALS": "12", "SUBJECTS": "1 2 3", "TRAIN_STEPS": "1"})
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "smoke"

            subprocess.run(
                ["bash", "scripts/prepare_moabb_smoke.sh", str(out_dir)],
                check=True,
                text=True,
                capture_output=True,
                env=env,
                timeout=600,
            )

            audit = json.loads((out_dir / "eval_audit.json").read_text(encoding="utf-8"))
            self.assertGreater(audit["window_count"], 0)
            for split_name in ("train", "val", "test"):
                self.assertGreater(audit["window_counts_by_split"][split_name], 0)

    @unittest.skipUnless(os.environ.get("NEUROTWIN_RUN_REAL_MOABB_TESTS") == "1", "real MOABB integration disabled")
    def test_real_moabb_benchmark_script_creates_windows_and_tasks(self):
        env = dict(os.environ)
        env.update({"MAX_TRIALS": "12", "SUBJECTS": "1 2 3", "TRAIN_STEPS": "1"})
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "benchmark"

            subprocess.run(
                ["bash", "scripts/prepare_moabb_benchmark.sh", str(out_dir)],
                check=True,
                text=True,
                capture_output=True,
                env=env,
                timeout=600,
            )

            audit = json.loads((out_dir / "eval_audit.json").read_text(encoding="utf-8"))
            suite = json.loads((out_dir / "prepared_baseline_suite.json").read_text(encoding="utf-8"))
            self.assertGreater(audit["window_count"], 0)
            self.assertEqual(suite["tasks"]["future_state_forecasting"]["status"], "completed")
            self.assertEqual(suite["tasks"]["masked_neural_reconstruction"]["status"], "completed")
