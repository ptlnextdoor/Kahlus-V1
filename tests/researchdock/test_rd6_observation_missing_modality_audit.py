import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurotwin.researchdock.observation_model import build_researchdock_observation_task
from neurotwin.researchdock.synthetic import make_synthetic_researchdock_sessions


class ResearchDockRD6ObservationMissingModalityAuditTests(unittest.TestCase):
    def test_observation_task_records_missing_modality_audit(self):
        task = build_researchdock_observation_task(make_synthetic_researchdock_sessions(seed=0), seed=0)
        audit = task.metadata.get("missing_modality_audit")

        self.assertIsInstance(audit, dict)
        self.assertEqual(audit["total_trials"], 48)
        self.assertGreater(audit["eligible_trials"], 0)
        self.assertGreater(audit["skipped_trials"], 0)
        self.assertGreater(audit["missing_pupil_diameter"], 0)
        self.assertGreater(audit["missing_behavior_response"], 0)
        self.assertIn("missing_pupil_diameter", audit["skip_reasons"])
        self.assertIn("missing_behavior_response", audit["skip_reasons"])

    def test_observation_script_writes_missing_modality_audit(self):
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
            task_json = json.loads((Path(tmp) / "researchdock_observation_task.json").read_text(encoding="utf-8"))
            audit = task_json["metadata"]["missing_modality_audit"]
            self.assertGreater(audit["missing_pupil_diameter"], 0)
            self.assertGreater(audit["missing_behavior_response"], 0)


if __name__ == "__main__":
    unittest.main()
