import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurotwin.neurovisual import build_episode_intake_profile


class NeurovisualIntakeTests(unittest.TestCase):
    def test_intake_profile_generates_clinician_questions_and_red_flags(self):
        profile = build_episode_intake_profile(
            {
                "duration_seconds": 20,
                "awareness_retained": True,
                "visual_field_location": "left_peripheral",
                "motion_or_flicker": True,
                "screen_or_sun_trigger": True,
                "prior_seizure_history": True,
                "no_loss_of_consciousness": True,
                "source_text": "synthetic generic report with no identifying details",
            }
        )
        payload = profile.to_dict()

        self.assertIn("episode_phenotype_profile", payload)
        self.assertIn("missing_clinician_questions", payload)
        self.assertIn("red_flag_checklist", payload)
        self.assertTrue(payload["red_flag_checklist"])
        self.assertIn("claim_gate", payload)
        self.assertIn("not a diagnosis", json.dumps(payload).lower())
        self.assertTrue(payload["claim_gate"]["passed"], payload["claim_gate"])

    def test_intake_smoke_writes_json_and_markdown_without_a100(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_neurovisual_intake_smoke.py",
                    "--out-dir",
                    tmp,
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            payload = json.loads((Path(tmp) / "neurovisual_intake_smoke.json").read_text(encoding="utf-8"))
            report = (Path(tmp) / "neurovisual_intake_smoke.md").read_text(encoding="utf-8")

        self.assertEqual(payload["execution"]["a100_jobs_launched"], False)
        self.assertEqual(payload["execution"]["bulk_dataset_download"], False)
        self.assertIn("not a diagnosis", report.lower())
        self.assertNotIn("safe for unsupervised", report.lower())


if __name__ == "__main__":
    unittest.main()
