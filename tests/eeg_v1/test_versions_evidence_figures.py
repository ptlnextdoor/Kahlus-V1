import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from tests.eeg_v1.test_eeg_v1_ridge_visuals import assert_standard_figure_packet


class VersionsEvidenceFigureTests(unittest.TestCase):
    def test_renderer_uses_real_evidence_zip_and_does_not_emit_synthetic_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "versions"
            zipdir = root / "_organized_index" / "artifact_views" / "evidence_zips"
            zipdir.mkdir(parents=True)
            with zipfile.ZipFile(zipdir / "sample-moabb-evidence.zip", "w") as zf:
                zf.writestr(
                    "sample/run/tables/task_results.csv",
                    "task_id,source_modality,target_modality,eval_mse,eval_mae,eval_pearsonr,eval_r2,test_mse,best_val_mse\n"
                    "future_state_forecasting,eeg,eeg,12.5,2.1,0.81,0.62,12.5,6.0\n"
                    "masked_neural_reconstruction,eeg,eeg,15.0,2.5,0.72,0.50,15.0,8.5\n",
                )
                zf.writestr(
                    "sample/run/tables/baseline_ranking.csv",
                    "task_id,model_id,metric,value,rank\n"
                    "future_state_forecasting,linear_ridge,mse,7.7,1\n"
                    "future_state_forecasting,persistence,mse,8.2,2\n",
                )
                zf.writestr(
                    "sample/prepared/leakage_report.json",
                    json.dumps({"passed": True, "checked_keys": ["subject_id"], "violations": []}),
                )
                zf.writestr(
                    "sample/run/paper_mode_gate.json",
                    json.dumps({"passed": True, "observed_seeds": [0, 1, 2], "violations": []}),
                )

            out = Path(tmp) / "out"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/render_eeg_v1_ridge_visuals.py",
                    "--versions-root",
                    str(root),
                    "--out-dir",
                    str(out),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            summary = json.loads((out / "eeg_v1_ridge_visual_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["source_mode"], "versions_evidence")
            self.assertEqual(summary["task_result_rows"], 2)
            self.assertEqual(summary["baseline_rows"], 2)
            self.assertFalse(summary["raw_tensor_artifacts_found"])
            self.assertIn("benchmark_overview_png", summary["figure_files"])
            self.assertIn("audit_matrix_png", summary["figure_files"])
            self.assertIn("baseline_ranking_png", summary["figure_files"])
            self.assertNotIn("synthetic_fixture", json.dumps(summary))
            self.assertFalse((out / "fig03_prediction_overlay_and_residuals.png").exists())
            packet = Path(tmp) / "eeg_v1_figure_source"
            for name in (
                "data/task_results.csv",
                "data/baseline_ranking.csv",
                "data/audits.csv",
            ):
                self.assertTrue((packet / name).exists(), name)
            assert_standard_figure_packet(self, packet)
            self.assertTrue((out / "eeg_v1_ridge_visual_analysis.md").exists())
            analysis = (out / "eeg_v1_ridge_visual_analysis.md").read_text(encoding="utf-8")
            self.assertIn("Real evidence artifacts", analysis)
            self.assertIn("CEBRA-style figure-source packet", analysis)
            self.assertIn("standard matplotlib/seaborn", analysis)
            self.assertIn("No raw tensor or prediction-array artifact was found", analysis)
            self.assertNotIn("synthetic fixture", analysis.lower())


if __name__ == "__main__":
    unittest.main()
