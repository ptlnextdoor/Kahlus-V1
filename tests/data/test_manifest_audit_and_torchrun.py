import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from neurotwin.runtime.distributed import get_rank_metrics_path


class ManifestAuditAndTorchrunTests(unittest.TestCase):
    def run_cli(self, *args: str, env_extra: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        if env_extra:
            env.update(env_extra)
        return subprocess.run(
            [sys.executable, "-m", "neurotwin.cli", *args],
            check=check,
            text=True,
            capture_output=True,
            env=env,
        )

    def test_split_audit_from_saved_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            self.run_cli(
                "data",
                "prepare",
                "--dataset",
                "synthetic",
                "--split",
                "subject",
                "--out-dir",
                str(out_dir),
            )
            result = self.run_cli(
                "split",
                "audit",
                "--manifest",
                str(out_dir / "split_manifest.json"),
                "--split",
                "subject",
            )

        self.assertIn("manifest=", result.stdout)
        self.assertIn("leakage_passed=True", result.stdout)

    def test_rank_metrics_path_is_rank_specific_when_distributed(self):
        run_dir = Path("/tmp/example-run")
        with mock.patch.dict(os.environ, {"RANK": "2", "WORLD_SIZE": "4"}, clear=False):
            path = get_rank_metrics_path(run_dir)

        self.assertEqual(path.name, "metrics.rank2.jsonl")

    def test_train_rank_one_does_not_write_shared_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            result = self.run_cli(
                "train",
                "--config",
                "configs/train/synthetic_debug.yaml",
                "--run-root",
                str(run_root),
                env_extra={"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2"},
            )
            run_dir = run_root / "synthetic_debug"

            self.assertIn("rank=1", result.stdout)
            self.assertTrue((run_dir / "metrics.rank1.jsonl").exists())
            self.assertFalse((run_dir / "checkpoint.pt").exists())
            self.assertFalse((run_dir / "summary.json").exists())

    def test_rank_zero_writes_shared_artifacts_and_rank_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            self.run_cli(
                "train",
                "--config",
                "configs/train/synthetic_debug.yaml",
                "--run-root",
                str(run_root),
                env_extra={"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"},
            )
            run_dir = run_root / "synthetic_debug"
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

            self.assertTrue((run_dir / "checkpoint.pt").exists())
            self.assertTrue((run_dir / "metrics.rank0.jsonl").exists())
            self.assertEqual(summary["distributed"]["world_size"], 2)
