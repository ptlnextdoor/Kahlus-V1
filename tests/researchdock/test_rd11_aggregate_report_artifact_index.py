import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ResearchDockRD11AggregateReportArtifactIndexTests(unittest.TestCase):
    def test_aggregate_report_indexes_observation_split_audit_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_researchdock_synthetic.py",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--run-observation-model",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = (Path(tmp) / "researchdock_report.md").read_text(encoding="utf-8")

        self.assertIn("## RD-2 Observation Model", report)
        self.assertIn("researchdock_observation_split_audit.json", report)
        self.assertIn("researchdock_observation_report.md", report)
        self.assertIn("## RD-2 Baseline Ladder Summary", report)
        ladder = report.split("## RD-2 Baseline Ladder Summary", 1)[1].split("## RD-2 Observation Model", 1)[0]
        self.assertLess(ladder.index("train_mean"), ladder.index("linear_ridge"))
        self.assertLess(ladder.index("linear_ridge"), ladder.index("researchdock_observation_operator"))
        self.assertIn("- observation_operator_beats_best_baseline:", ladder)
        self.assertIn("## RD-2 Split Audit Summary", report)
        split_summary = report.split("## RD-2 Split Audit Summary", 1)[1].split("## RD-2 Observation Model", 1)[0]
        self.assertIn("- split_type: subject_held_out", split_summary)
        self.assertIn("- train_subjects: 3", split_summary)
        self.assertIn("- test_subjects: 1", split_summary)
        self.assertIn("- subject_overlap: False", split_summary)
        self.assertIn("- leakage_passed: True", split_summary)
        self.assertIn("## RD-2 Missing-Modality Summary", report)
        missing_summary = report.split("## RD-2 Missing-Modality Summary", 1)[1].split("## RD-2 Observation Model", 1)[0]
        self.assertIn("- total_trials: 48", missing_summary)
        self.assertIn("- eligible_trials: 24", missing_summary)
        self.assertIn("- skipped_trials: 24", missing_summary)
        self.assertIn("- skip_reasons: missing_pupil_diameter, missing_behavior_response", missing_summary)
        self.assertIn("| missing_pupil_diameter | 12 |", missing_summary)
        self.assertIn("| missing_behavior_response | 16 |", missing_summary)

    def test_aggregate_report_indexes_optional_researchdock_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_researchdock_synthetic.py",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--write-session-export",
                    "--run-observation-model",
                    "--write-public-dataset-review",
                    "--write-pilot-preflight",
                    "--write-profile-readiness",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = (Path(tmp) / "researchdock_report.md").read_text(encoding="utf-8")

        artifact_index = report.split("## Evidence Artifact Index", 1)[1].split("## Failure Reasons Summary", 1)[0]
        for artifact_name in (
            "researchdock_rd1_protocol.json",
            "researchdock_interface_contract.json",
            "session_export/",
            "researchdock_observation_task.json",
            "researchdock_observation_metrics.json",
            "researchdock_observation_baselines.csv",
            "researchdock_observation_split_audit.json",
            "researchdock_observation_report.md",
            "researchdock_public_dataset_review.json",
            "researchdock_public_dataset_review.md",
            "researchdock_pilot_manifest.json",
            "researchdock_pilot_preflight_gate.json",
            "researchdock_pilot_preflight_report.md",
            "researchdock_profile_readiness.json",
            "researchdock_profile_readiness_report.md",
        ):
            self.assertIn(artifact_name, artifact_index)
        self.assertIn("## RD-5 Readiness Audit Summary", report)
        readiness = report.split("## RD-5 Readiness Audit Summary", 1)[1].split("## RD-5 Response-Profile Readiness", 1)[0]
        self.assertIn("- ready_for_future_clustering: False", readiness)
        self.assertIn("- clustering_performed: False", readiness)
        self.assertIn("- n_metric_rows: 4", readiness)
        self.assertIn("- minimum_sessions_required: 20", readiness)
        self.assertIn("- finite_profile_vectors: True", readiness)
        self.assertIn(
            "- failure_reasons: insufficient_sessions_for_future_clustering, missing_pupil_profiles_present",
            readiness,
        )


if __name__ == "__main__":
    unittest.main()
