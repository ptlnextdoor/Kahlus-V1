import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ResearchDockRD9ObservationSplitAuditTests(unittest.TestCase):
    def test_observation_script_writes_split_audit_sidecar(self):
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
            self.assertIn("observation_split_audit=", result.stdout)
            audit_path = Path(tmp) / "researchdock_observation_split_audit.json"
            self.assertTrue(audit_path.exists())
            audit = json.loads(audit_path.read_text(encoding="utf-8"))

        self.assertEqual(audit["split_type"], "subject_held_out")
        self.assertTrue(audit["leakage_passed"], audit["failure_reasons"])
        self.assertFalse(audit["subject_overlap"])
        self.assertGreater(audit["n_train_subjects"], 0)
        self.assertGreater(audit["n_test_subjects"], 0)
        self.assertEqual(audit["failure_reasons"], [])


if __name__ == "__main__":
    unittest.main()
