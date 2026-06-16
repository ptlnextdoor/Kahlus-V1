import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurotwin.training_v3 import KTMTrainConfig
from neurotwin.training_v3.failure_analysis import (
    ABLATION_LABELS,
    REPORT_SCHEMA,
    SCHEMA,
    build_ablations,
    run_ablations,
    run_failure_analysis,
    write_failure_analysis,
)

# Tiny config so the autopsy + (subset) ablation runs stay fast and deterministic on CPU.
_CFG = KTMTrainConfig(
    mode="cpu_smoke", n_episodes=24, steps=10, eval_every_steps=5,
    checkpoint_every_steps=5, seed=0,
)

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "train"
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_ktm_failure_analysis.py"
_ABLATION_YAMLS = (
    "ktm_ablation_trajectory_only.yaml",
    "ktm_ablation_traj_profile.yaml",
    "ktm_ablation_traj_nll.yaml",
    "ktm_ablation_full_objective.yaml",
    "ktm_ablation_uncertainty_off.yaml",
    "ktm_ablation_memory_sweep.yaml",
    "ktm_ablation_embed_sweep.yaml",
)


class FailureAnalysisReportTests(unittest.TestCase):
    """Base (no-ablation) failure-analysis report shape, finiteness, and gate discipline."""

    @classmethod
    def setUpClass(cls):
        cls.report = run_failure_analysis(_CFG, run_ablations_flag=False)

    def test_schema_scope_and_recovery_blocked(self):
        self.assertEqual(self.report["schema"], REPORT_SCHEMA)
        self.assertEqual(self.report["result_schema"], SCHEMA)
        self.assertEqual(self.report["claim_scope"], "synthetic_ktm_training_harness")
        self.assertIs(self.report["recovery_claim_allowed"], False)
        self.assertTrue(self.report["synthetic_only"])
        # Default command must not run the sweep.
        self.assertFalse(self.report["ablations_ran"])
        self.assertEqual(self.report["ablations"], [])

    def test_autopsy_slice_lengths_and_finite(self):
        a = self.report["autopsy"]
        for block, expected in (
            ("per_horizon", _CFG.horizon),
            ("per_perturbation", _CFG.n_perturbations),
            ("per_channel", _CFG.eeg_channels),
        ):
            for series in ("ktm_mse", "ssm_mse", "ratio_ktm_over_ssm"):
                vals = a[block][series]
                self.assertEqual(len(vals), expected, f"{block}.{series}")
                self.assertTrue(np.isfinite(vals).all(), f"{block}.{series} not finite")
        self.assertTrue(a["finite"])

    def test_ssm_comparison_nonempty(self):
        a = self.report["autopsy"]
        ov = a["overall"]
        for model in ("ktm", "ssm"):
            for metric in ("mse", "mae", "r2", "pearson_r"):
                self.assertIn(metric, ov[model])
                self.assertTrue(np.isfinite(ov[model][metric]))
        self.assertGreater(ov["ktm"]["mse"], 0.0)
        self.assertGreater(ov["ssm"]["mse"], 0.0)
        # At least one slice row exists -> the SSM comparison table is non-empty.
        self.assertGreaterEqual(len(a["per_horizon"]["ktm_mse"]), 1)
        self.assertEqual(a["ssm_model_id"], "ssm_fallback")

    def test_loss_components_finite(self):
        lc = self.report["loss_components"]
        for key in ("trajectory", "profile", "nll", "total"):
            self.assertTrue(np.isfinite(lc[key]), key)
        self.assertTrue(lc["finite"])
        self.assertIn("empirical_coverage_1sigma", lc["calibration"])

    def test_hypothesis_is_nonempty_string(self):
        self.assertIsInstance(self.report["best_failure_hypothesis"], str)
        self.assertTrue(self.report["best_failure_hypothesis"].strip())

    def test_report_files_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_failure_analysis(tmp, self.report)
            jp = Path(paths["ktm_failure_analysis_json"])
            mp = Path(paths["ktm_failure_analysis_md"])
            cp = Path(paths["ktm_error_by_slice_csv"])
            self.assertTrue(jp.exists() and mp.exists() and cp.exists())
            loaded = json.loads(jp.read_text(encoding="utf-8"))
            self.assertEqual(loaded["schema"], REPORT_SCHEMA)
            self.assertIs(loaded["recovery_claim_allowed"], False)
            csv_lines = cp.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(csv_lines[0], "dimension,index,ktm_mse,ssm_mse,ratio_ktm_over_ssm")
            self.assertGreater(len(csv_lines), 1)


class AblationTests(unittest.TestCase):
    """The ablation matrix loads and smoke-runs, and never earns recovery."""

    def test_ablation_labels_are_well_formed(self):
        # ABLATION_LABELS is derived from build_ablations (single source of truth); assert the
        # resulting matrix is the expected shape and has no duplicate labels.
        labels = tuple(label for label, _ in build_ablations(_CFG))
        self.assertEqual(labels, ABLATION_LABELS)
        self.assertEqual(len(ABLATION_LABELS), 10)
        self.assertEqual(len(set(ABLATION_LABELS)), len(ABLATION_LABELS))
        for required in ("trajectory_only", "full_objective", "uncertainty_off"):
            self.assertIn(required, ABLATION_LABELS)

    def test_ablation_smoke_run_blocks_recovery(self):
        # A 2-entry subset keeps the test fast while exercising the full ablation path.
        subset = build_ablations(_CFG)[:2]
        rows = run_ablations(_CFG, ablations=subset)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertTrue(np.isfinite(row["ktm_test_mse"]), row["label"])
            self.assertIs(row["recovery_allowed"], False)
            for key in ("trajectory", "profile", "nll", "total"):
                self.assertTrue(np.isfinite(row["loss_components"][key]), f"{row['label']}.{key}")


class AblationConfigYamlTests(unittest.TestCase):
    """Each shipped ablation YAML loads into a valid KTMTrainConfig."""

    def test_yaml_configs_load_and_validate(self):
        for name in _ABLATION_YAMLS:
            path = _CONFIG_DIR / name
            self.assertTrue(path.is_file(), path)
            cfg = KTMTrainConfig.from_yaml(path)  # from_yaml validates on load
            self.assertEqual(cfg.mode, "cpu_smoke", name)
            self.assertGreater(cfg.steps, 0, name)


class FailureAnalysisCliContractTests(unittest.TestCase):
    """The public Sprint 3B diagnostic CLI must stay local-only and non-clustered."""

    def test_cli_blocks_distributed_mode_surface(self):
        text = _SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn('choices=["cpu_smoke", "single_gpu"]', text)
        self.assertNotIn('"ddp"', text)
        self.assertIn("distributed/cluster execution is intentionally blocked", text)


if __name__ == "__main__":
    unittest.main()
