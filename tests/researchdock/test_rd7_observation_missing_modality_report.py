import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ResearchDockRD7ObservationMissingModalityReportTests(unittest.TestCase):
    def test_observation_report_lists_missing_modality_reason_counts(self):
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
            report = (Path(tmp) / "researchdock_observation_report.md").read_text(encoding="utf-8")

        self.assertIn("## Missing-Modality Audit", report)
        self.assertIn("| reason | count |", report)
        self.assertIn("| missing_pupil_diameter |", report)
        self.assertIn("| missing_behavior_response |", report)


if __name__ == "__main__":
    unittest.main()
