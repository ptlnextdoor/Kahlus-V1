import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurotwin.training_v3 import KTMTrainConfig
from neurotwin.training_v3.redteam_runner import run_redteam, write_redteam_report

# Tiny config so the multi-seed + family battery stays fast and deterministic on CPU.
_CFG = KTMTrainConfig(
    mode="cpu_smoke", n_episodes=24, steps=10, eval_every_steps=5,
    checkpoint_every_steps=5, seed=0, recovery_margin=0.05,
)


class RedteamRunnerTests(unittest.TestCase):
    def test_runner_produces_blocked_report_at_tiny_scale(self):
        report = run_redteam(
            _CFG, seeds=(0, 1), families=("linear", "nonlinear"), min_seeds=2
        )
        # A tiny CPU model cannot beat the baselines; recovery must be blocked, with reasons.
        self.assertFalse(report["recovery_allowed"])
        self.assertTrue(report["blocker_reasons"])

        gate = report["redteam_gate"]
        self.assertEqual(gate["seed_summary"]["n_seeds"], 2)
        self.assertEqual(len(gate["per_seed"]), 2)
        self.assertEqual(len(gate["generator_families"]), 2)
        # Symmetric selection is recorded per seed.
        for s in gate["per_seed"]:
            self.assertEqual(s["ktm_checkpoint_policy"], "best_val")
            self.assertEqual(s["baseline_checkpoint_policy"], "best_val")

    def test_diagnostics_shapes_and_finite(self):
        report = run_redteam(_CFG, seeds=(0,), families=("linear",), min_seeds=1)
        diag = report["diagnostics"]
        self.assertEqual(len(diag["per_perturbation_mse"]), _CFG.n_perturbations)
        self.assertEqual(len(diag["per_horizon_mse"]), _CFG.horizon)
        self.assertTrue(diag["finite"])
        self.assertTrue(np.isfinite(diag["response_profile_mse"]))
        self.assertIn("compose", diag["heldout_composition_note"])

    def test_architecture_affinity_summary_present(self):
        report = run_redteam(_CFG, seeds=(0,), families=("linear", "nonlinear"), min_seeds=1)
        aff = report["architecture_affinity"]
        self.assertIn("appears_generator_aligned", aff)
        self.assertIn("linear", aff["by_family_relative_improvement"])
        self.assertIn("nonlinear", aff["by_family_relative_improvement"])

    def test_report_files_written(self):
        report = run_redteam(_CFG, seeds=(0,), families=("linear",), min_seeds=1)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_redteam_report(tmp, report)
            jp = Path(paths["ktm_redteam_report_json"])
            mp = Path(paths["ktm_redteam_report_md"])
            self.assertTrue(jp.exists() and mp.exists())
            loaded = json.loads(jp.read_text(encoding="utf-8"))
            self.assertEqual(loaded["schema"], "kahlus.ktm_recovery_redteam_report.v1")
            self.assertIn("Red-Team", mp.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
