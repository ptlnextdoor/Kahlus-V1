import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


class EEGV1RidgeVisualsTests(unittest.TestCase):
    def test_renderer_writes_real_artifact_figures_without_synthetic_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            versions = Path(tmp) / "versions"
            zipdir = versions / "_organized_index" / "artifact_views" / "evidence_zips"
            zipdir.mkdir(parents=True)
            with zipfile.ZipFile(zipdir / "sample-evidence.zip", "w") as zf:
                zf.writestr(
                    "sample/run/tables/task_results.csv",
                    "task_id,source_modality,target_modality,eval_mse,eval_mae,eval_pearsonr,eval_r2,test_mse,best_val_mse\n"
                    "future_state_forecasting,eeg,eeg,12.5,2.1,0.81,0.62,12.5,6.0\n",
                )
                zf.writestr(
                    "sample/run/tables/baseline_ranking.csv",
                    "task_id,model_id,metric,value,rank\nfuture_state_forecasting,linear_ridge,mse,7.7,1\n",
                )
                zf.writestr("sample/prepared/leakage_report.json", json.dumps({"passed": True, "violations": []}))

            out = Path(tmp) / "figures"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/render_eeg_v1_ridge_visuals.py",
                    "--versions-root",
                    str(versions),
                    "--out-dir",
                    str(out),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            for name in (
                "fig01_versions_evidence_inventory.png",
                "fig01_versions_evidence_inventory.pdf",
                "fig02_eeg_task_metrics_from_versions.png",
                "fig02_eeg_task_metrics_from_versions.pdf",
                "fig03_real_baseline_ranking.png",
                "fig03_real_baseline_ranking.pdf",
                "fig04_leakage_and_gate_audit.png",
                "fig04_leakage_and_gate_audit.pdf",
                "eeg_v1_ridge_visual_analysis.md",
                "eeg_v1_ridge_visual_summary.json",
            ):
                self.assertTrue((out / name).exists(), name)
            self.assertFalse((out / "fig03_prediction_overlay_and_residuals.png").exists())
            summary = json.loads((out / "eeg_v1_ridge_visual_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["source_mode"], "versions_evidence")
            self.assertFalse(summary["raw_tensor_artifacts_found"])
            analysis = (out / "eeg_v1_ridge_visual_analysis.md").read_text(encoding="utf-8")
            self.assertIn("Real evidence artifacts", analysis)
            self.assertIn("No raw tensor or prediction-array artifact was found", analysis)
            self.assertNotIn("synthetic fixture", analysis.lower())

    def test_restored_schematic_diagnostic_packet_is_kept_and_labeled(self):
        root = Path("docs/research/ridge_eeg_diagnostic_schematics")
        expected = (
            "fig1_ridge_input_target_waveforms.png",
            "fig2_ridge_prediction_overlay.png",
            "fig3_autocorrelation_lag_structure.png",
            "fig4_ridge_coefficient_channel_map.png",
            "fig5_psd_residual_diagnostics.png",
        )
        for name in expected:
            self.assertTrue((root / name).exists(), name)
            self.assertGreater((root / name).stat().st_size, 100_000, name)
        readme = (root / "README.md").read_text(encoding="utf-8")
        self.assertIn("schematic demo figures, not benchmark evidence", readme)
        page = Path("docs/figures/eeg-v1-ridge-visuals.md").read_text(encoding="utf-8")
        self.assertIn("Restored diagnostic schematic packet", page)
        self.assertIn("not benchmark evidence", page)
        self.assertIn("fig2_ridge_prediction_overlay.png", page)


if __name__ == "__main__":
    unittest.main()
