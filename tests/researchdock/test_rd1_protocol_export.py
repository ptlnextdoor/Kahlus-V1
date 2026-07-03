import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurotwin.researchdock.export import export_researchdock_sessions
from neurotwin.researchdock.protocol import build_rd1_session_protocol, researchdock_interface_contract
from neurotwin.researchdock.quality import assess_trial_quality, summarize_quality_flags
from neurotwin.researchdock.schemas import ResearchDockTrial
from neurotwin.researchdock.synthetic import make_synthetic_researchdock_sessions


class ResearchDockRD1ProtocolExportTests(unittest.TestCase):
    def test_rd1_protocol_is_safe_deterministic_and_covers_task_battery(self):
        left = build_rd1_session_protocol(seed=4)
        right = build_rd1_session_protocol(seed=4)

        self.assertEqual(left, right)
        task_names = {block.task_name for block in left.blocks}
        self.assertEqual(
            task_names,
            {
                "reward_anticipation",
                "probabilistic_reward_learning",
                "effort_for_reward",
                "mild_frustration",
                "recovery_rest",
                "visual_attention",
            },
        )
        self.assertFalse(left.hardware_required)
        self.assertFalse(left.clinical_claim_allowed)
        self.assertTrue(all(block.n_trials > 0 for block in left.blocks))
        joined = " ".join(block.task_name for block in left.blocks)
        self.assertNotIn("trauma", joined)
        self.assertNotIn("treatment", joined)

    def test_interface_contract_describes_inputs_without_requiring_hardware(self):
        contract = researchdock_interface_contract()

        self.assertEqual(contract["mode"], "design_contract_only")
        self.assertFalse(contract["opens_camera"])
        self.assertFalse(contract["requires_ppg_device"])
        self.assertIn("webcam_pupil_gaze", contract["input_channels"])
        self.assertIn("optional_ppg_hrv", contract["input_channels"])
        self.assertIn("csv_session_export", contract["outputs"])

    def test_quality_flags_catch_missing_pupil_and_invalid_reaction_time(self):
        trial = ResearchDockTrial(
            participant_id_hash="sha256:test",
            session_id="ses-quality",
            timestamp=0.0,
            task_name="reward_anticipation",
            event_type="response",
            reward_condition="reward",
            reaction_time_ms=50.0,
            accuracy=1.0,
        )

        flags = assess_trial_quality(trial)
        self.assertIn("missing_pupil", flags)
        self.assertIn("implausible_reaction_time", flags)
        summary = summarize_quality_flags((trial,))
        self.assertEqual(summary["n_trials"], 1)
        self.assertEqual(summary["flag_counts"]["missing_pupil"], 1)

    def test_csv_export_writes_session_tables_without_pii_columns(self):
        sessions = make_synthetic_researchdock_sessions(seed=6)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = export_researchdock_sessions(sessions, tmp)
            root = Path(tmp)

            expected = {
                "researchdock_sessions.csv",
                "researchdock_trials.csv",
                "researchdock_sensor_packets.csv",
                "researchdock_task_events.csv",
                "researchdock_self_reports.csv",
                "researchdock_session_manifest.json",
            }
            self.assertEqual(expected, {path.name for path in root.iterdir()})
            self.assertEqual(manifest["n_sessions"], 4)
            self.assertEqual(manifest["export_format"], "researchdock_csv_v1")
            trials = list(csv.DictReader((root / "researchdock_trials.csv").read_text(encoding="utf-8").splitlines()))
            self.assertGreater(len(trials), 0)
            self.assertIn("participant_id_hash", trials[0])
            self.assertIn("quality_flags", trials[0])
            self.assertNotIn("name", trials[0])
            self.assertNotIn("email", trials[0])

    def test_script_writes_rd1_protocol_and_csv_export_artifacts(self):
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
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )

            root = Path(tmp)
            self.assertIn("session_export=", result.stdout)
            self.assertTrue((root / "researchdock_rd1_protocol.json").exists())
            self.assertTrue((root / "researchdock_interface_contract.json").exists())
            export_dir = root / "session_export"
            self.assertTrue((export_dir / "researchdock_trials.csv").exists())
            protocol = json.loads((root / "researchdock_rd1_protocol.json").read_text(encoding="utf-8"))
            self.assertFalse(protocol["hardware_required"])
            self.assertFalse(protocol["clinical_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
