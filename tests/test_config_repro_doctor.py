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
    create_run_dir,
    manifest_hash,
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
            env_path.write_text(json.dumps(capture_environment()), encoding="utf-8")

            self.assertTrue(config_path.exists())
            self.assertTrue(env_path.exists())
            self.assertIn("python", json.loads(env_path.read_text())["runtime"])

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
