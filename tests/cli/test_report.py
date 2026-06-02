import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from neurotwin.benchmarks.reports import generate_model_card_report, generate_run_report


class CliReportTests(unittest.TestCase):
    def test_report_mentions_corrected_boss_fight_and_split_rules(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"

        result = subprocess.run(
            [sys.executable, "-m", "neurotwin.cli", "report", "--suite", "translation_smoke"],
            cwd=os.getcwd(),
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("Brain-OF", result.stdout)
        self.assertIn("TRIBE v2", result.stdout)
        self.assertIn("held-out subject/site/dataset", result.stdout)
        self.assertIn("missing-modality reconstruction", result.stdout)

    def test_run_report_does_not_reraise_artifact_read_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text('{"mse": 0.1}', encoding="utf-8")
            original_read_text = Path.read_text

            def fail_metrics_read(path: Path, *args: object, **kwargs: object) -> str:
                if path == metrics_path:
                    raise OSError("read blocked")
                return original_read_text(path, *args, **kwargs)

            with mock.patch.object(Path, "read_text", fail_metrics_read):
                report = generate_run_report(run_dir)

        self.assertIn("artifact_error=read_failed", report)
        self.assertIn("read blocked", report)

    def test_model_card_report_writes_card_and_paper_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "summary.json").write_text(
                json.dumps({"status": "completed_prepared_training", "synthetic_only": False, "real_data_smoke": False}),
                encoding="utf-8",
            )
            (run_dir / "eval_audit.json").write_text(
                json.dumps({"passed": True, "checked": ["split_policy_leakage"], "violations": []}),
                encoding="utf-8",
            )
            (run_dir / "paper_mode_gate.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "required_seeds": [0, 1, 2],
                        "observed_seeds": [0, 1, 2],
                        "require_ci": True,
                        "violations": [],
                        "warnings": [],
                        "checked": ["eval_audit"],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "prepared_baseline_suite.json").write_text(
                json.dumps(
                    {
                        "scope": {"status": "prepared-data"},
                        "seeds": [0, 1, 2],
                        "prepared_data": {
                            "event_manifest": "/tmp/event_manifest.json",
                            "split_manifest": "/tmp/split_manifest.json",
                            "window_length": 8,
                            "stride": 8,
                            "event_summary": {
                                "modalities": ["eeg"],
                                "datasets": ["BNCI2014_001"],
                                "subjects": 9,
                                "synthetic_only": False,
                            },
                        },
                        "aggregate": {
                            "aggregate_rank": [{"model_id": "linear_ridge", "mean_rank": 1.0}],
                        },
                        "tasks": {
                            "future_state_forecasting": {
                                "ranking": [{"model_id": "linear_ridge", "metric": "mse", "value": 0.1, "rank": 1}],
                            }
                        },
                        "baseline_failures": [],
                    }
                ),
                encoding="utf-8",
            )
            out = run_dir / "EEG_MODEL_CARD.md"

            report = generate_model_card_report(run_dir, out)

            self.assertIn("# EEG Model Card", report)
            self.assertTrue(out.exists())
            self.assertTrue((run_dir / "CLAIM_GATE.json").exists())
            self.assertTrue((run_dir / "LEAKAGE_AUDIT.json").exists())
            self.assertTrue((run_dir / "BASELINE_RANKING.csv").exists())
            self.assertIn("linear_ridge", (run_dir / "BASELINE_RANKING.csv").read_text(encoding="utf-8"))

    def test_model_card_report_rejects_missing_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "missing"

            with self.assertRaisesRegex(ValueError, "run-dir does not exist"):
                generate_model_card_report(run_dir)

            self.assertFalse(run_dir.exists())

    def test_model_card_report_rejects_empty_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()

            with self.assertRaisesRegex(ValueError, "lacks required source artifacts"):
                generate_model_card_report(run_dir)

            self.assertFalse((run_dir / "EEG_MODEL_CARD.md").exists())

    def test_model_card_cli(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            out = Path(tmp) / "nested" / "reports" / "EEG_MODEL_CARD.md"
            run_dir.mkdir()
            (run_dir / "summary.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "report",
                    "model-card",
                    "--run-dir",
                    str(run_dir),
                    "--out",
                    str(out),
                ],
                cwd=os.getcwd(),
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )

            self.assertIn("model_card=", result.stdout)
            self.assertTrue(out.exists())
