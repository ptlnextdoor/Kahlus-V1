import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurotwin.data.manifest_io import load_split_manifest, split_manifest_to_dict


class ManifestPersistenceAndEvalSuiteTests(unittest.TestCase):
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

    def test_data_prepare_writes_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            result = self.run_cli(
                "data",
                "prepare",
                "--dataset",
                "synthetic",
                "--split",
                "subject",
                "--out-dir",
                str(out_dir),
            )

            split_path = out_dir / "split_manifest.json"
            data_path = out_dir / "data_manifest.json"
            audit_path = out_dir / "leakage_report.json"

            self.assertTrue(split_path.exists())
            self.assertTrue(data_path.exists())
            self.assertTrue(audit_path.exists())
            self.assertIn("split_manifest=", result.stdout)
            self.assertTrue(load_split_manifest(split_path).train)

    def test_train_writes_split_manifest_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            self.run_cli(
                "train",
                "--config",
                "configs/train/synthetic_debug.yaml",
                "--run-root",
                str(run_root),
            )
            run_dir = run_root / "synthetic_debug"

            self.assertTrue((run_dir / "split_manifest.json").exists())
            self.assertTrue((run_dir / "checkpoint.pt").exists())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("split_manifest_hash", summary)

    def test_neural_translation_eval_suite_outputs_task_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            result = self.run_cli(
                "eval",
                "--suite",
                "neural_translation_v1",
                "--out-dir",
                str(out_dir),
            )

            metrics_path = out_dir / "metrics.json"
            self.assertTrue(metrics_path.exists())
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertIn("future_state_forecasting", metrics)
            self.assertIn("masked_neural_reconstruction", metrics)
            self.assertIn("cross_modal_translation", metrics)
            self.assertIn("synthetic-only", result.stdout)

    def test_split_manifest_roundtrip_dict(self):
        split = load_split_manifest("runs/synthetic_debug/split_manifest.json") if Path("runs/synthetic_debug/split_manifest.json").exists() else None
        if split is None:
            self.skipTest("synthetic_debug run not present")
        payload = split_manifest_to_dict(split)
        self.assertIn("train", payload)
        self.assertEqual(payload["split_stage"], "recording_manifest")
