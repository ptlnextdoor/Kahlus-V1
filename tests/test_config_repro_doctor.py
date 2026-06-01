import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from neurotwin.config import ConfigError, load_config
from neurotwin.repro import (
    capture_environment,
    capture_run_metadata,
    checkpoint_manifest,
    create_run_dir,
    manifest_hash,
    resolve_source_commit,
    set_global_seed,
    snapshot_config,
    stable_hash,
)


class ConfigReproDoctorTests(unittest.TestCase):
    def test_load_config_and_hash_are_stable(self):
        config = load_config("configs/train/synthetic_debug.yaml")

        self.assertEqual(config["experiment"], "synthetic_debug")
        self.assertEqual(stable_hash(config), stable_hash(load_config("configs/train/synthetic_debug.yaml")))

    def test_load_config_rejects_missing_file(self):
        with self.assertRaises(ConfigError):
            load_config("configs/train/does_not_exist.yaml")

    def test_set_global_seed_controls_numpy_and_torch(self):
        set_global_seed(123)
        np_a = np.random.rand(3)
        torch_a = torch.rand(3)
        set_global_seed(123)
        np_b = np.random.rand(3)
        torch_b = torch.rand(3)

        np.testing.assert_allclose(np_a, np_b)
        self.assertTrue(torch.equal(torch_a, torch_b))

    def test_run_dir_snapshot_and_environment_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_dir(Path(tmp), run_id="unit-test")
            config_path = snapshot_config({"experiment": "unit"}, run_dir)
            env_path = run_dir / "environment.json"
            env_path.write_text(json.dumps(capture_environment(argv=["nt", "train"])), encoding="utf-8")

            self.assertTrue(config_path.exists())
            self.assertTrue(env_path.exists())
            env = json.loads(env_path.read_text())
            self.assertIn("python", env["runtime"])
            self.assertIn("source_commit_missing", env)
            self.assertEqual(env["run"]["argv"], ["nt", "train"])
            self.assertIn("cuda_device_count", env["torch"])

    def test_non_git_runner_uses_commit_hash_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "COMMIT_HASH.txt").write_text("abc123fallback\n", encoding="utf-8")

            commit = resolve_source_commit(root)
            env = capture_environment(repo_root=root, argv=["nt", "train", "--config", "unit.yaml"])

            self.assertEqual(commit["commit"], "abc123fallback")
            self.assertEqual(commit["source"], "COMMIT_HASH.txt")
            self.assertTrue(commit["source_commit_missing"])
            self.assertEqual(env["git"]["commit"], "abc123fallback")
            self.assertTrue(env["source_commit_missing"])

    def test_nested_no_git_runner_ignores_parent_git_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "parent"
            parent.mkdir()
            subprocess.run(["git", "init"], cwd=parent, check=True, text=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "unit@example.com"], cwd=parent, check=True)
            subprocess.run(["git", "config", "user.name", "Unit Test"], cwd=parent, check=True)
            (parent / "tracked.txt").write_text("parent\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=parent, check=True)
            subprocess.run(["git", "commit", "-m", "parent"], cwd=parent, check=True, text=True, capture_output=True)
            parent_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=parent,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
            runner = parent / "runner"
            runner.mkdir()
            (runner / "COMMIT_HASH.txt").write_text("runnerfallback123\n", encoding="utf-8")

            env = capture_environment(repo_root=runner)

            self.assertNotEqual(env["git"]["commit"], parent_commit)
            self.assertEqual(env["git"]["commit"], "runnerfallback123")
            self.assertEqual(env["git"]["source"], "COMMIT_HASH.txt")
            self.assertTrue(env["git"]["source_commit_missing"])
            self.assertTrue(env["source_commit_missing"])

    def test_run_metadata_captures_direct_and_slurm_modes(self):
        direct = capture_run_metadata(argv=["nt", "doctor"], env={})
        slurm = capture_run_metadata(
            argv=["nt", "train", "--api-token", "secret-value"],
            env={
                "SLURM_JOB_ID": "12345",
                "SLURM_NODEID": "2",
                "SLURM_JOB_NODELIST": "gpu-[01-02]",
            },
        )

        self.assertEqual(direct["mode"], "direct")
        self.assertEqual(slurm["mode"], "slurm")
        self.assertEqual(slurm["slurm"]["job_id"], "12345")
        self.assertEqual(slurm["slurm"]["node_id"], "2")
        self.assertEqual(slurm["argv"][-1], "<redacted>")

    def test_environment_captures_docker_and_ddp_metadata(self):
        env = capture_environment(
            argv=["nt", "train"],
            env={
                "CONTAINER": "docker",
                "DOCKER_IMAGE": "neurotwin-a100-runner:local",
                "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5",
                "LOCAL_RANK": "2",
                "RANK": "2",
                "WORLD_SIZE": "6",
                "NCCL_DEBUG": "INFO",
            },
        )

        self.assertEqual(env["run"]["mode"], "container")
        self.assertEqual(env["run"]["container"]["docker_image"], "neurotwin-a100-runner:local")
        self.assertEqual(env["run"]["distributed"]["cuda_visible_devices"], "0,1,2,3,4,5")
        self.assertEqual(env["run"]["distributed"]["local_rank"], "2")
        self.assertEqual(env["run"]["distributed"]["rank"], "2")
        self.assertEqual(env["run"]["distributed"]["world_size"], "6")
        self.assertIn("torch_cuda_version", env["torch"])
        self.assertIn("nccl_version", env["torch"])
        self.assertIn("cuda_device_count", env["torch"])
        self.assertIn("cuda_device_names", env["torch"])

    def test_run_metadata_redacts_underscore_secret_flags(self):
        metadata = capture_run_metadata(
            argv=[
                "nt",
                "train",
                "--api_key=secret-a",
                "--access_token",
                "secret-b",
                "--private_key=secret-c",
                "--wandb_api_key",
                "secret-d",
            ],
            env={},
        )

        self.assertEqual(metadata["argv"][2], "--api_key=<redacted>")
        self.assertEqual(metadata["argv"][3:5], ["--access_token", "<redacted>"])
        self.assertEqual(metadata["argv"][5], "--private_key=<redacted>")
        self.assertEqual(metadata["argv"][6:8], ["--wandb_api_key", "<redacted>"])
        self.assertNotIn("secret-a", metadata["command"])
        self.assertNotIn("secret-b", metadata["command"])
        self.assertNotIn("secret-c", metadata["command"])
        self.assertNotIn("secret-d", metadata["command"])

    def test_checkpoint_manifest_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoint_b.pt").write_bytes(b"bbb")
            (root / "checkpoint.pt").write_bytes(b"aaa")
            (root / "not_a_checkpoint.pt").write_bytes(b"ignored")

            manifest = checkpoint_manifest(root)

            self.assertEqual([entry["filename"] for entry in manifest], ["checkpoint.pt", "checkpoint_b.pt"])
            self.assertEqual([entry["path"] for entry in manifest], ["checkpoint.pt", "checkpoint_b.pt"])
            self.assertEqual([entry["size"] for entry in manifest], [3, 3])
            self.assertTrue(all(len(entry["sha256"]) == 64 for entry in manifest))

    def test_manifest_hash_changes_with_payload(self):
        self.assertNotEqual(manifest_hash([{"a": 1}]), manifest_hash([{"a": 2}]))

    def test_create_run_dir_falls_back_from_default_non_writable_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                local_runs = Path("runs")
                local_runs.mkdir(exist_ok=True)
                local_runs.chmod(0o555)

                run_dir = create_run_dir(run_id="fallback")

                self.assertTrue(run_dir.exists())
                self.assertTrue(run_dir.is_dir())
                self.assertNotEqual(run_dir.parent.name, "runs")
            finally:
                os.chdir(cwd)

    def test_doctor_cli_reports_missing_optional_dependencies(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        result = subprocess.run(
            [sys.executable, "-m", "neurotwin.cli", "doctor"],
            check=True,
            text=True,
            capture_output=True,
            env=env,
        )

        self.assertIn("python", result.stdout)
        self.assertIn("torch", result.stdout)
        self.assertIn("moabb", result.stdout)
        self.assertIn("cuda_available", result.stdout)
