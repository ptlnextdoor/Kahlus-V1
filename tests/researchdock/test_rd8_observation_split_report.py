import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ResearchDockRD8ObservationSplitReportTests(unittest.TestCase):
    def test_observation_report_lists_subject_held_out_split_summary(self):
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

        self.assertIn("## Subject-Held-Out Split", report)
        self.assertIn("- split_type: subject_held_out", report)
        self.assertIn("- train_subjects:", report)
        self.assertIn("- test_subjects:", report)
        self.assertIn("- subject_overlap: False", report)


if __name__ == "__main__":
    unittest.main()
