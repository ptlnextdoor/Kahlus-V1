import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from neurotwin.benchmarks.suite import run_neural_translation_v1_synthetic
from neurotwin.benchmarks.tasks import run_dataset_site_generalization_task


class MoabbCliAndGeneralizationTests(unittest.TestCase):
    def run_cli_no_check(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
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
