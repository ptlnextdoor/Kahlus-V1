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
                "fig01_eeg_window_overlap_diagnostic.png",
                "fig01_eeg_window_overlap_diagnostic.pdf",
                "fig02_ridge_design_matrix_contract.png",
                "fig02_ridge_design_matrix_contract.pdf",
                "fig03_prediction_overlay_and_residuals.png",
                "fig03_prediction_overlay_and_residuals.pdf",
                "fig04_baseline_and_autocorrelation_controls.png",
                "fig04_baseline_and_autocorrelation_controls.pdf",
                "eeg_v1_ridge_visual_analysis.md",
                "eeg_v1_ridge_visual_summary.json",
            ):
                self.assertTrue((out / name).exists(), name)
            analysis = (out / "eeg_v1_ridge_visual_analysis.md").read_text(encoding="utf-8")
            self.assertIn("This is not a new benchmark", analysis)
            self.assertIn("Publication-style diagnostic figures", analysis)
            self.assertIn("linear_ridge", analysis)
            self.assertIn("autocorrelation", analysis)
            self.assertNotIn(".svg", analysis)


if __name__ == "__main__":
    unittest.main()
