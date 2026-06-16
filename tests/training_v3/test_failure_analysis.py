import json
import subprocess
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
    objective_gap_comparison,
    run_multiseed_objective_check,
    run_ablations,
    run_failure_analysis,
    write_multiseed_objective_check,
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
_SPRINT3C_YAMLS = (
    "ktm_recovery_point_objective.yaml",
    "ktm_recovery_capacity_smoke.yaml",
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
        self.assertEqual(len(ABLATION_LABELS), 7)
        self.assertEqual(len(set(ABLATION_LABELS)), len(ABLATION_LABELS))
        for required in (
            "full_objective",
            "point_only",
            "traj_profile",
            "traj_profile_small_nll",
            "uncertainty_off",
            "uncertainty_on",
            "capacity_smoke",
        ):
            self.assertIn(required, ABLATION_LABELS)

    def test_ablation_smoke_run_blocks_recovery(self):
        # A 2-entry subset keeps the test fast while exercising old full vs new point objective.
        subset = build_ablations(_CFG)[:2]
        rows = run_ablations(_CFG, ablations=subset)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertTrue(np.isfinite(row["ktm_test_mse"]), row["label"])
            self.assertTrue(np.isfinite(row["ratio_ktm_over_ssm"]), row["label"])
            self.assertIs(row["recovery_allowed"], False)
            for key in ("trajectory", "profile", "nll", "total"):
                self.assertTrue(np.isfinite(row["loss_components"][key]), f"{row['label']}.{key}")
        summary = objective_gap_comparison(rows)
        self.assertTrue(summary["available"])
        self.assertEqual(summary["reference_label"], "full_objective")
        self.assertIn(summary["candidate_label"], ABLATION_LABELS)
        self.assertIn("gap_narrowed", summary)

    def test_ablation_report_writes_gap_comparison(self):
        rows = run_ablations(_CFG, ablations=build_ablations(_CFG)[:2])
        report = run_failure_analysis(_CFG, run_ablations_flag=False)
        report["ablations_ran"] = True
        report["ablations"] = rows
        report["objective_gap_comparison"] = objective_gap_comparison(rows)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_failure_analysis(tmp, report)
            loaded = json.loads(Path(paths["ktm_failure_analysis_json"]).read_text(encoding="utf-8"))
            self.assertTrue(loaded["objective_gap_comparison"]["available"])
            md = Path(paths["ktm_failure_analysis_md"]).read_text(encoding="utf-8")
            self.assertIn("Objective gap comparison", md)


class MultiSeedCheckTests(unittest.TestCase):
    def test_multiseed_check_writes_requested_artifacts(self):
        report = run_multiseed_objective_check(_CFG, seeds=(0, 1))
        self.assertEqual(report["schema"], "kahlus.ktm_3c_multiseed_check.v1")
        self.assertEqual(report["candidate_label"], "traj_profile")
        self.assertEqual(report["n_seeds"], 2)
        self.assertEqual(len(report["per_seed"]), 2)
        self.assertFalse(report["recovery_claim_allowed"])
        self.assertIn("mean", report["relative_ratio_reduction_summary"])
        for row in report["per_seed"]:
            self.assertIn("full_ratio_ktm_over_ssm", row)
            self.assertIn("candidate_ratio_ktm_over_ssm", row)
            self.assertTrue(np.isfinite(row["full_ratio_ktm_over_ssm"]))
            self.assertTrue(np.isfinite(row["candidate_ratio_ktm_over_ssm"]))
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_multiseed_objective_check(tmp, report)
            jp = Path(paths["ktm_3c_multiseed_check_json"])
            mp = Path(paths["ktm_3c_multiseed_check_md"])
            self.assertTrue(jp.exists() and mp.exists())
            loaded = json.loads(jp.read_text(encoding="utf-8"))
            self.assertEqual(loaded["schema"], report["schema"])
            self.assertIn("Multi-Seed Check", mp.read_text(encoding="utf-8"))


class AblationConfigYamlTests(unittest.TestCase):
    """Each shipped ablation / Sprint 3C YAML loads into a valid KTMTrainConfig."""

    def test_yaml_configs_load_and_validate(self):
        for name in _ABLATION_YAMLS + _SPRINT3C_YAMLS:
            path = _CONFIG_DIR / name
            self.assertTrue(path.is_file(), path)
            cfg = KTMTrainConfig.from_yaml(path)  # from_yaml validates on load
            self.assertEqual(cfg.mode, "cpu_smoke", name)
            self.assertGreater(cfg.steps, 0, name)

    def test_sprint3c_configs_use_point_objective_and_local_capacity(self):
        point = KTMTrainConfig.from_yaml(_CONFIG_DIR / "ktm_recovery_point_objective.yaml")
        capacity = KTMTrainConfig.from_yaml(_CONFIG_DIR / "ktm_recovery_capacity_smoke.yaml")
        self.assertEqual(point.effective_nll_weight(), 0.0)
        self.assertEqual(capacity.effective_nll_weight(), 0.0)
        self.assertEqual(point.baseline_train_steps, 0)
        self.assertEqual(capacity.baseline_train_steps, 0)
        self.assertEqual(capacity.embed_dim, 32)
        self.assertEqual(capacity.memory_dim, 24)
        self.assertEqual(capacity.decoder_hidden_dim, 192)
        self.assertEqual(capacity.steps, 300)


class FailureAnalysisCliContractTests(unittest.TestCase):
    """The public Sprint 3B diagnostic CLI must stay local-only and non-clustered."""

    def test_cli_blocks_distributed_mode_surface(self):
        text = _SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn('choices=["cpu_smoke", "single_gpu"]', text)
        self.assertNotIn('"ddp"', text)
        self.assertIn("distributed/cluster execution is intentionally blocked", text)
        self.assertIn("--seeds", text)
        self.assertIn("run_multiseed_objective_check", text)


class FrozenV1ScopeTests(unittest.TestCase):
    def test_branch_diff_excludes_load_bearing_v1_paths(self):
        repo = Path(__file__).resolve().parents[2]
        changed: set[str] = set()
        for args in (
            ["git", "diff", "--name-only", "origin/main...HEAD"],
            ["git", "diff", "--name-only"],
        ):
            result = subprocess.run(
                args,
                cwd=repo,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                self.skipTest(f"{' '.join(args)} unavailable: {result.stderr.strip()}")
            changed.update(result.stdout.splitlines())
        forbidden = {
            "src/neurotwin/training/prepared.py",
            "src/neurotwin/benchmarks/suite.py",
            "src/neurotwin/data/split_manifest.py",
            "src/neurotwin/reports/evidence_gate.py",
            "src/neurotwin/models/__init__.py",
        }
        self.assertFalse(changed & forbidden)


if __name__ == "__main__":
    unittest.main()
