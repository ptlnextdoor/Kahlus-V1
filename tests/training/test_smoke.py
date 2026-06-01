import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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

    def test_train_cli_honors_synthetic_seed_and_steps(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "synthetic_custom.yaml"
            run_root = root / "runs"
            config_path.write_text(
                "\n".join(
                    [
                        "experiment: synthetic_custom",
                        "dataset: synthetic",
                        "seed: 7",
                        "steps: 3",
                    ]
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "train",
                    "--config",
                    str(config_path),
                    "--run-root",
                    str(run_root),
                ],
                check=True,
                text=True,
                capture_output=True,
                env=env,
            )

            metrics = json.loads((run_root / "synthetic_custom" / "metrics.json").read_text(encoding="utf-8"))
            expected = run_synthetic_training(seed=7, steps=3)

        self.assertIn("steps=3", result.stdout)
        self.assertEqual(metrics["steps"], 3)
        self.assertAlmostEqual(metrics["initial_loss"], expected.initial_loss)
        self.assertAlmostEqual(metrics["final_loss"], expected.final_loss)
