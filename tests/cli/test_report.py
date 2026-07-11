import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurotwin.eval.forecast_eligibility import write_forecast_eligibility_artifact
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

    def test_report_nfc_synthetic_suite(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"

        result = subprocess.run(
            [sys.executable, "-m", "neurotwin.cli", "report", "--suite", "nfc_synthetic"],
            cwd=os.getcwd(),
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("NeuroTwin NFC Synthetic Suite", result.stdout)
        self.assertIn("Pair-Operator is a baseline/ablation", result.stdout)

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

    def test_run_report_does_not_finalize_or_mutate_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            summary = {"status": "completed", "scientific_claim_allowed": False}
            summary_path = run_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")

            report = generate_run_report(run_dir)
            rewritten = json.loads(summary_path.read_text(encoding="utf-8"))
            evidence_gate_exists = (run_dir / "evidence_gate.json").exists()

        self.assertIn("scientific_claim_allowed=False", report)
        self.assertEqual(rewritten, summary)
        self.assertFalse(evidence_gate_exists)

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
            summary_before = (run_dir / "summary.json").read_text(encoding="utf-8")

            report = generate_model_card_report(run_dir, out)

            self.assertEqual((run_dir / "summary.json").read_text(encoding="utf-8"), summary_before)
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

    def test_evidence_gate_cli_explicitly_writes_sidecar_without_mutating_summary(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            summary = {
                "status": "completed",
                "scientific_claim_allowed": True,
                "quarantined_tasks": [],
                "task_results": [{"task_id": "future_state_forecasting", "best_val_mse": 0.2, "test_mse": 0.3}],
            }
            summary_path = run_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
            (run_dir / "eval_audit.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
            (run_dir / "paper_mode_gate.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "require_ci": True,
                        "violations": [],
                        "required_seeds": [0, 1, 2],
                        "observed_seeds": [0, 1, 2],
                        "forecast_eligibility_required": True,
                        "forecast_eligibility_passed": True,
                    }
                ),
                encoding="utf-8",
            )
            write_forecast_eligibility_artifact(
                run_dir / "forecast_eligibility.json",
                {
                    "protocol": {"protocol_id": "kahlus.forecast.v2_nonoverlap", "schema_version": 2},
                    "source_hashes": ["a" * 64],
                    "source_hash_verification_passed": True,
                    "transform_lineage_hash": "b" * 64,
                    "transform_lineage_complete": True,
                    "split_audit": {
                        "passed": True,
                        "violations": [],
                        "subject_overlap_count": 0,
                        "recording_overlap_count": 0,
                        "session_overlap_count": 0,
                    },
                    "firebreak_audit": {"passed": True, "violations": [], "target_overlaps_context": False},
                    "invalidated_result_ids": [],
                },
            )
            (run_dir / "prepared_baseline_suite.json").write_text(
                json.dumps(
                    {
                        "baseline_catalog": [{"model_id": "linear_ridge", "status": "local_baseline"}],
                        "tasks": {
                            "future_state_forecasting": {
                                "ranking": [{"model_id": "linear_ridge", "metric": "mse", "value": 0.3, "rank": 1}]
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, "-m", "neurotwin.cli", "report", "evidence-gate", "--run-dir", str(run_dir)],
                cwd=os.getcwd(),
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )

            rewritten = json.loads(summary_path.read_text(encoding="utf-8"))
            evidence_gate_exists = (run_dir / "evidence_gate.json").exists()

        self.assertIn("evidence_gate_passed=True", result.stdout)
        self.assertEqual(rewritten, summary)
        self.assertTrue(evidence_gate_exists)
