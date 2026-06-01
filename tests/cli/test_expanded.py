import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class ExpandedCliTests(unittest.TestCase):
    @staticmethod
    def _valid_paper_mode_gate() -> dict[str, object]:
        return {
            "passed": True,
            "require_ci": True,
            "violations": [],
            "required_seeds": [0, 1, 2],
            "observed_seeds": [0, 1, 2],
        }

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

    def run_script(self, script: str, *args: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        return subprocess.run(
            [sys.executable, script, *args],
            check=False,
            text=True,
            capture_output=True,
            env=env,
        )

    def test_estimate_and_train_dry_run(self):
        estimate = self.run_cli("estimate", "--config", "configs/train/synthetic_debug.yaml")
        train = self.run_cli("train", "--dry-run", "--config", "configs/train/synthetic_debug.yaml")
        a100 = self.run_cli("train", "--dry-run", "--config", "configs/train/neurotwin_v1_a100.yaml")

        self.assertIn("estimated_parameters", estimate.stdout)
        self.assertIn("dry_run=True", train.stdout)
        self.assertIn("backbone=ssm_fallback", a100.stdout)

    def test_data_and_split_audits(self):
        data = self.run_cli("data", "audit", "--dataset", "synthetic")
        split = self.run_cli("split", "audit", "--dataset", "synthetic", "--split", "subject")

        self.assertIn("audit_passed=True", data.stdout)
        self.assertIn("leakage_passed=True", split.stdout)

    def test_report_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "metrics.json").write_text('{"mse": 0.1}', encoding="utf-8")
            (run_dir / "summary.json").write_text('{"synthetic_only": false, "status": "completed"}', encoding="utf-8")
            result = self.run_cli("report", "--run-dir", str(run_dir))
            self.assertTrue((run_dir / "tables" / "metrics_flat.csv").exists())
            self.assertTrue((run_dir / "tables" / "baseline_ranking.csv").exists())
            self.assertTrue((run_dir / "tables" / "baseline_failures.csv").exists())
            self.assertTrue((run_dir / "figures" / "metric_summary.json").exists())

        self.assertIn("NeuroTwin Run Report", result.stdout)
        self.assertIn("mse", result.stdout)

    def test_report_run_dir_promotes_claim_from_valid_colocated_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "metrics.json").write_text('{"test_mse": 0.1}', encoding="utf-8")
            (run_dir / "summary.json").write_text(
                '{"synthetic_only": false, "real_data_smoke": false, "status": "completed", "scientific_claim_allowed": false}',
                encoding="utf-8",
            )
            (run_dir / "paper_mode_gate.json").write_text(
                json.dumps(self._valid_paper_mode_gate()),
                encoding="utf-8",
            )

            result = self.run_cli("report", "--run-dir", str(run_dir))
            metric_summary = json.loads((run_dir / "figures" / "metric_summary.json").read_text(encoding="utf-8"))

        self.assertIn("effective_scientific_claim_allowed=True", result.stdout)
        self.assertTrue(metric_summary["scientific_claim_allowed"])

    def test_report_run_dir_rejects_invalid_colocated_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "metrics.json").write_text('{"test_mse": 0.1}', encoding="utf-8")
            (run_dir / "summary.json").write_text(
                '{"synthetic_only": false, "real_data_smoke": false, "status": "completed", "scientific_claim_allowed": false}',
                encoding="utf-8",
            )
            (run_dir / "paper_mode_gate.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "require_ci": False,
                        "violations": [],
                        "required_seeds": [0, 1, 2],
                        "observed_seeds": [0, 1, 2],
                    }
                ),
                encoding="utf-8",
            )

            result = self.run_cli("report", "--run-dir", str(run_dir))
            metric_summary = json.loads((run_dir / "figures" / "metric_summary.json").read_text(encoding="utf-8"))

        self.assertIn("effective_scientific_claim_allowed=False", result.stdout)
        self.assertFalse(metric_summary["scientific_claim_allowed"])

    def test_report_compare_writes_aggregate_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "run_a"
            run_b = root / "run_b"
            out = root / "compare"
            run_a.mkdir()
            run_b.mkdir()
            (run_a / "metrics.json").write_text('{"test_mse": 0.2}', encoding="utf-8")
            (run_a / "summary.json").write_text(
                '{"status": "completed", "synthetic_only": false, "real_data_smoke": false, "scientific_claim_allowed": false, "test_mse": 0.2}',
                encoding="utf-8",
            )
            (run_a / "paper_mode_gate.json").write_text(json.dumps(self._valid_paper_mode_gate()), encoding="utf-8")
            (run_b / "metrics.json").write_text('{"test_mse": 0.3}', encoding="utf-8")
            (run_b / "summary.json").write_text(
                '{"status": "completed", "synthetic_only": false, "real_data_smoke": false, "scientific_claim_allowed": false, "test_mse": 0.3}',
                encoding="utf-8",
            )

            result = self.run_cli("report", "--compare", str(run_a), str(run_b), "--out-dir", str(out))
            compare_rows = json.loads((out / "compare_runs.json").read_text(encoding="utf-8"))

            self.assertTrue((out / "compare_runs.csv").exists())
            self.assertTrue((out / "compare_runs.json").exists())
        self.assertIn("NeuroTwin Run Comparison", result.stdout)
        self.assertEqual(compare_rows[0]["scientific_claim_allowed"], True)
        self.assertEqual(compare_rows[1]["scientific_claim_allowed"], False)

    def test_report_compare_surfaces_malformed_json_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "broken"
            out = root / "compare"
            run_dir.mkdir()
            (run_dir / "metrics.json").write_text("{broken\n", encoding="utf-8")

            result = self.run_cli("report", "--compare", str(run_dir), "--out-dir", str(out))
            compare_rows = json.loads((out / "compare_runs.json").read_text(encoding="utf-8"))

        self.assertIn("artifact_errors", result.stdout)
        self.assertEqual(compare_rows[0]["status"], "artifact_error")
        self.assertIn("invalid_json", compare_rows[0]["artifact_error"])

    def test_make_tables_treats_malformed_gate_as_plumbing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "metrics.json").write_text('{"test_mse": 0.2}', encoding="utf-8")
            (run_dir / "summary.json").write_text(
                '{"synthetic_only": false, "real_data_smoke": false, "scientific_claim_allowed": false}',
                encoding="utf-8",
            )
            (run_dir / "paper_mode_gate.json").write_text("{broken\n", encoding="utf-8")

            result = self.run_script("scripts/make_tables.py", str(run_dir))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("| run | claim_status | plumbing |", result.stdout)

    def test_make_figures_treats_malformed_gate_as_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "metrics.json").write_text('{"test_mse": 0.2}', encoding="utf-8")
            (run_dir / "summary.json").write_text(
                '{"synthetic_only": false, "real_data_smoke": false, "scientific_claim_allowed": false}',
                encoding="utf-8",
            )
            (run_dir / "paper_mode_gate.json").write_text("{broken\n", encoding="utf-8")

            result = self.run_script("scripts/make_figures.py", str(run_dir))
            figure_summary = (run_dir / "figure_summary.txt").read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("scientific_claim_allowed: False", figure_summary)

    def test_make_tables_treats_malformed_summary_as_plumbing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "metrics.json").write_text('{"test_mse": 0.2}', encoding="utf-8")
            (run_dir / "summary.json").write_text("{broken\n", encoding="utf-8")

            result = self.run_script("scripts/make_tables.py", str(run_dir))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("| run | claim_status | plumbing |", result.stdout)

    def test_make_figures_treats_malformed_summary_as_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "metrics.json").write_text('{"test_mse": 0.2}', encoding="utf-8")
            (run_dir / "summary.json").write_text("{broken\n", encoding="utf-8")

            result = self.run_script("scripts/make_figures.py", str(run_dir))
            figure_summary = (run_dir / "figure_summary.txt").read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("scientific_claim_allowed: False", figure_summary)

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
