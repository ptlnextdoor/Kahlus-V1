import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurotwin.researchdock import build_rd1_session_protocol, make_synthetic_researchdock_sessions
from neurotwin.researchdock.pilot_preflight import (
    build_researchdock_pilot_manifest,
    run_researchdock_pilot_preflight,
    write_researchdock_pilot_preflight_artifacts,
)


class ResearchDockRD4PilotPreflightTests(unittest.TestCase):
    def test_manifest_is_pre_collection_and_names_required_prior_evidence(self):
        sessions = make_synthetic_researchdock_sessions(seed=0)
        protocol = build_rd1_session_protocol(seed=0)
        manifest = build_researchdock_pilot_manifest(sessions=sessions, protocol=protocol)

        self.assertEqual(manifest["sprint"], "RD-4")
        self.assertEqual(manifest["collection_status"], "pre_collection_preflight_only")
        self.assertFalse(manifest["hardware_access_enabled"])
        self.assertFalse(manifest["contains_real_participant_data"])
        self.assertFalse(manifest["contains_pii"])
        self.assertIn("rd2_observation_model_baselines", manifest["required_prior_evidence"])
        self.assertIn("rd3_public_dataset_review", manifest["required_prior_evidence"])
        self.assertIn("reward_anticipation", manifest["task_battery"])
        self.assertIn("no diagnosis claims", manifest["safety_boundaries"])

    def test_preflight_blocks_real_data_hardware_or_clinical_claims(self):
        sessions = make_synthetic_researchdock_sessions(seed=0)
        protocol = build_rd1_session_protocol(seed=0)
        manifest = build_researchdock_pilot_manifest(sessions=sessions, protocol=protocol)
        result = run_researchdock_pilot_preflight(manifest)
        self.assertTrue(result["passed"])
        self.assertEqual(result["failure_reasons"], [])

        unsafe = dict(manifest)
        unsafe["hardware_access_enabled"] = True
        unsafe["contains_real_participant_data"] = True
        unsafe["claimed_use"] = "diagnoses depression"
        unsafe_result = run_researchdock_pilot_preflight(unsafe)
        self.assertFalse(unsafe_result["passed"])
        self.assertIn("hardware access must be disabled for RD-4", unsafe_result["failure_reasons"])
        self.assertIn("real participant data is not allowed in RD-4", unsafe_result["failure_reasons"])
        self.assertIn("clinical or diagnostic claims are blocked", unsafe_result["failure_reasons"])

    def test_writer_outputs_manifest_gate_and_report(self):
        sessions = make_synthetic_researchdock_sessions(seed=0)
        protocol = build_rd1_session_protocol(seed=0)
        manifest = build_researchdock_pilot_manifest(sessions=sessions, protocol=protocol)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_researchdock_pilot_preflight_artifacts(tmp, manifest)
            payload = json.loads(Path(paths["manifest"]).read_text())
            gate = json.loads(Path(paths["gate"]).read_text())
            report = Path(paths["report"]).read_text()

        self.assertEqual(payload["collection_status"], "pre_collection_preflight_only")
        self.assertTrue(gate["passed"])
        self.assertIn("Required Prior Evidence", report)
        self.assertIn("pre-collection only", report.lower())
        self.assertIn("no raw participant data", report.lower())

    def test_script_writes_rd4_pilot_preflight_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_researchdock_synthetic.py",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--write-pilot-preflight",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            self.assertIn("pilot_manifest=", result.stdout)
            self.assertIn("pilot_preflight_gate=", result.stdout)
            manifest = json.loads((Path(tmp) / "researchdock_pilot_manifest.json").read_text())
            gate = json.loads((Path(tmp) / "researchdock_pilot_preflight_gate.json").read_text())

        self.assertEqual(manifest["sprint"], "RD-4")
        self.assertTrue(gate["passed"])
        self.assertFalse(manifest["hardware_access_enabled"])


if __name__ == "__main__":
    unittest.main()
