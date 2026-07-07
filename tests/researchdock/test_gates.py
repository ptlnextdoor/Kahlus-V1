import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurotwin.researchdock.gates import (
    RESEARCHDOCK_ALLOWED_CLAIM_SCOPE,
    build_researchdock_gate,
)
from neurotwin.researchdock.metrics import compute_researchdock_metrics
from neurotwin.researchdock.synthetic import make_synthetic_researchdock_sessions


class ResearchDockGateTests(unittest.TestCase):
    def test_gate_allows_only_narrow_synthetic_response_profile_claim(self):
        metrics = compute_researchdock_metrics(make_synthetic_researchdock_sessions(seed=0))
        allowed = build_researchdock_gate(
            dataset="researchdock_synthetic_v0",
            metrics=metrics,
            claim_scope=RESEARCHDOCK_ALLOWED_CLAIM_SCOPE,
            data_card_passed=True,
        )
        self.assertTrue(allowed["scientific_claim_allowed"], allowed["failure_reasons"])

        for broad_scope in (
            "diagnosis",
            "treatment",
            "clinical",
            "depression_detection",
            "ptsd_detection",
            "anhedonia_diagnosis",
        ):
            blocked = build_researchdock_gate(
                dataset="researchdock_synthetic_v0",
                metrics=metrics,
                claim_scope=broad_scope,
                data_card_passed=True,
            )
            self.assertFalse(blocked["scientific_claim_allowed"])
            self.assertTrue(blocked["failure_reasons"])

    def test_script_writes_expected_artifacts_and_blocks_clinical_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_researchdock_synthetic.py",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )

            expected = {
                "researchdock_metrics.json",
                "researchdock_data_card.json",
                "researchdock_evidence_gate.json",
                "researchdock_report.md",
            }
            self.assertTrue(expected.issubset({path.name for path in Path(tmp).iterdir()}))
            gate = json.loads((Path(tmp) / "researchdock_evidence_gate.json").read_text(encoding="utf-8"))
            metrics = json.loads((Path(tmp) / "researchdock_metrics.json").read_text(encoding="utf-8"))
            report = (Path(tmp) / "researchdock_report.md").read_text(encoding="utf-8")

            self.assertIn("claim_scope=researchdock_synthetic_response_profile", result.stdout)
            self.assertTrue(gate["scientific_claim_allowed"], gate["failure_reasons"])
            self.assertEqual(len(metrics["session_metrics"]), 4)
            self.assertIn("Clinical Claim Boundary", report)
            self.assertIn("does not diagnose", report)
            self.assertIn("## Data Card Summary", report)
            self.assertIn("- contains_pii: False", report)
            self.assertIn("- contains_real_participant_data: False", report)
            self.assertIn("- contains_clinical_labels: False", report)
            self.assertIn("- contains_stimulation: False", report)
            self.assertIn("- quality_flags: missing_pupil", report)
            self.assertIn("## Quality Flag Counts", report)
            self.assertIn("- missing_pupil: 1", report)
            self.assertIn("- synthetic_high_noise: 1", report)

    def test_script_writes_researchdock_gate_criteria(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_researchdock_synthetic.py",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )

            gate = json.loads((Path(tmp) / "researchdock_evidence_gate.json").read_text(encoding="utf-8"))
            report = (Path(tmp) / "researchdock_report.md").read_text(encoding="utf-8")

        self.assertIn("gate_criteria", gate)
        self.assertEqual(gate["gate_criteria"]["allowed_claim_scope"], RESEARCHDOCK_ALLOWED_CLAIM_SCOPE)
        self.assertEqual(
            gate["gate_criteria"]["blocked_claim_terms"],
            [
                "diagnosis",
                "treatment",
                "clinical",
                "depression_detection",
                "ptsd_detection",
                "anhedonia_diagnosis",
            ],
        )
        self.assertIn("## Evidence Gate Criteria", report)
        self.assertIn(f"- allowed_claim_scope: {RESEARCHDOCK_ALLOWED_CLAIM_SCOPE}", report)
        self.assertIn("- requires_data_card_passed: True", report)
        self.assertIn("- blocked_claim_terms: diagnosis, treatment, clinical, depression_detection, ptsd_detection, anhedonia_diagnosis", report)

    def test_script_writes_researchdock_failure_reasons_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_researchdock_synthetic.py",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("failure_reasons=", result.stdout)
            failures_path = Path(tmp) / "researchdock_failure_reasons.json"
            self.assertTrue(failures_path.exists())
            failures = json.loads(failures_path.read_text(encoding="utf-8"))
            gate = json.loads((Path(tmp) / "researchdock_evidence_gate.json").read_text(encoding="utf-8"))

        self.assertEqual(failures["gate_failures"], gate["failure_reasons"])
        self.assertEqual(failures["data_card_failures"], [])
        self.assertEqual(failures["blocked_claim_terms"], gate["gate_criteria"]["blocked_claim_terms"])

    def test_report_indexes_researchdock_failure_reasons_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_researchdock_synthetic.py",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )

            report = (Path(tmp) / "researchdock_report.md").read_text(encoding="utf-8")

        self.assertIn("## Evidence Artifact Index", report)
        self.assertIn("researchdock_failure_reasons.json", report)
        self.assertIn("researchdock_evidence_gate.json", report)

    def test_report_summarizes_researchdock_failure_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_researchdock_synthetic.py",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )

            report = (Path(tmp) / "researchdock_report.md").read_text(encoding="utf-8")

        self.assertIn("## Failure Reasons Summary", report)
        self.assertIn("- gate_failure_count: 0", report)
        self.assertIn("- data_card_failure_count: 0", report)
        self.assertIn("- blocked_claim_terms_count: 6", report)

    def test_report_lists_researchdock_failure_reason_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_researchdock_synthetic.py",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )

            report = (Path(tmp) / "researchdock_report.md").read_text(encoding="utf-8")

        self.assertIn("## Failure Reasons Details", report)
        self.assertIn("- gate_failures: none", report)
        self.assertIn("- data_card_failures: none", report)
        self.assertIn("- blocked_claim_terms: diagnosis, treatment, clinical, depression_detection, ptsd_detection, anhedonia_diagnosis", report)


if __name__ == "__main__":
    unittest.main()
