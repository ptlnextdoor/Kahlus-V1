import os
import subprocess
import sys
import unittest

from neurotwin.training.smoke import run_synthetic_training


class TrainingSmokeTests(unittest.TestCase):
    def test_synthetic_training_reduces_loss(self):
        result = run_synthetic_training(seed=19, steps=18)

        self.assertLess(result.final_loss, result.initial_loss)
        self.assertEqual(result.steps, 18)

    def test_train_cli_runs_debug_config(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "neurotwin.cli",
                "train",
                "--config",
                "configs/train/synthetic_debug.yaml",
            ],
            check=True,
            text=True,
            capture_output=True,
            env=env,
        )

        self.assertIn("training_status=completed_synthetic_smoke", result.stdout)
        self.assertIn("final_loss=", result.stdout)
