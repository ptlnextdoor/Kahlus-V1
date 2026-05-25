import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class ExpandedCliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        return subprocess.run(
            [sys.executable, "-m", "neurotwin.cli", *args],
            check=True,
            text=True,
            capture_output=True,
            env=env,
        )

    def test_estimate_and_train_dry_run(self):
        estimate = self.run_cli("estimate", "--config", "configs/train/synthetic_debug.yaml")
        train = self.run_cli("train", "--dry-run", "--config", "configs/train/synthetic_debug.yaml")

        self.assertIn("estimated_parameters", estimate.stdout)
        self.assertIn("dry_run=True", train.stdout)

    def test_data_and_split_audits(self):
        data = self.run_cli("data", "audit", "--dataset", "synthetic")
        split = self.run_cli("split", "audit", "--dataset", "synthetic", "--split", "subject")

        self.assertIn("audit_passed=True", data.stdout)
        self.assertIn("leakage_passed=True", split.stdout)

    def test_report_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "metrics.json").write_text('{"mse": 0.1}', encoding="utf-8")
            result = self.run_cli("report", "--run-dir", str(run_dir))

        self.assertIn("NeuroTwin Run Report", result.stdout)
        self.assertIn("mse", result.stdout)

    def test_bids_prepare_writes_event_manifest_when_derivative_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bids"
            out_dir = Path(tmp) / "prepared"
            func = root / "sub-01" / "ses-01" / "func"
            func.mkdir(parents=True)
            bold = func / "sub-01_ses-01_task-rest_run-01_bold.nii.gz"
            bold.write_text("placeholder", encoding="utf-8")
            np.save(func / "sub-01_ses-01_task-rest_run-01_timeseries.npy", np.ones((6, 2), dtype=np.float32))

            result = self.run_cli(
                "data",
                "prepare",
                "--dataset",
                "bids",
                "--split",
                "subject",
                "--root",
                str(root),
                "--out-dir",
                str(out_dir),
            )

            self.assertIn("event_manifest=", result.stdout)
            self.assertTrue((out_dir / "event_manifest.json").exists())
