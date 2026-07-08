import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class EEGV1RidgeVisualsTests(unittest.TestCase):
    def test_renderer_writes_labeled_synthetic_fixture_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/render_eeg_v1_ridge_visuals.py",
                    "--out-dir",
                    tmp,
                    "--dataset",
                    "synthetic_fixture",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            out = Path(tmp)
            for name in (
                "eeg_v1_ridge_waveform_window.svg",
                "eeg_v1_ridge_feature_map.svg",
                "eeg_v1_ridge_prediction_overlay.svg",
                "eeg_v1_ridge_visual_analysis.md",
                "eeg_v1_ridge_visual_summary.json",
            ):
                self.assertTrue((out / name).exists(), name)
            analysis = (out / "eeg_v1_ridge_visual_analysis.md").read_text(encoding="utf-8")
            self.assertIn("This is not a new benchmark", analysis)
            self.assertIn("linear_ridge", analysis)
            self.assertIn("autocorrelation", analysis)


if __name__ == "__main__":
    unittest.main()
