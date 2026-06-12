import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from neurotwin.runtime.distributed import cleanup_process_group, get_distributed_info, maybe_init_process_group


class DistributedTrainingRuntimeTests(unittest.TestCase):
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

    def test_distributed_info_reads_torchrun_environment(self):
        with mock.patch.dict(
            os.environ,
            {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "4"},
            clear=False,
        ):
            info = get_distributed_info()

        self.assertEqual(info.rank, 1)
        self.assertEqual(info.local_rank, 1)
        self.assertEqual(info.world_size, 4)
        self.assertFalse(info.is_rank_zero)

    def test_process_group_init_skips_without_rendezvous_env(self):
        with mock.patch.dict(os.environ, {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2"}, clear=False):
            initialized, backend = maybe_init_process_group()

        self.assertFalse(initialized)
        self.assertIsNone(backend)

    def test_cleanup_process_group_logs_and_swallows_cleanup_failures(self):
        with (
            mock.patch("neurotwin.runtime.distributed.torch.distributed.is_available", return_value=True),
            mock.patch("neurotwin.runtime.distributed.torch.distributed.is_initialized", return_value=True),
            mock.patch(
                "neurotwin.runtime.distributed.torch.distributed.destroy_process_group",
                side_effect=RuntimeError("nccl cleanup failed"),
            ),
            mock.patch("sys.stderr") as stderr,
        ):
            cleanup_process_group()

        stderr.write.assert_called()
        self.assertIn("distributed_cleanup_failed", str(stderr.write.call_args_list))

    def test_train_resume_and_metrics_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            self.run_cli("train", "--config", "configs/train/synthetic_debug.yaml", "--run-root", str(run_root))
            run_dir = run_root / "synthetic_debug"
            resume = self.run_cli(
                "train",
                "--config",
                "configs/train/synthetic_debug.yaml",
                "--run-root",
                str(run_root),
                "--resume",
                str(run_dir / "checkpoint.pt"),
            )

            self.assertIn("resume=", resume.stdout)
            self.assertTrue((run_dir / "metrics.jsonl").exists())
            self.assertGreaterEqual(len((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()), 2)

    def test_train_from_prepared_split_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prep_dir = tmp_path / "prepared"
            run_root = tmp_path / "runs"
            self.run_cli(
                "data",
                "prepare",
                "--dataset",
                "synthetic",
                "--split",
                "subject",
                "--out-dir",
                str(prep_dir),
            )
            config = tmp_path / "prepared_config.yaml"
            config.write_text(
                "\n".join(
                    [
                        "experiment: prepared_synthetic",
                        "dataset: prepared_manifest",
                        "task: future_state_forecasting",
                        "split: subject",
                        f"split_manifest: {prep_dir / 'split_manifest.json'}",
                        "model:",
                        "  type: NeuralStateSpaceTranslator",
                        "  latent_dim: 16",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = self.run_cli("train", "--config", str(config), "--run-root", str(run_root))

            self.assertIn("run_dir=", result.stdout)
            self.assertTrue((run_root / "prepared_synthetic" / "split_manifest.json").exists())
