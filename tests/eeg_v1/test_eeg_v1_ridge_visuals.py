import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image


STANDARD_FIGURE_STEMS = (
    "Figure1_eeg_v1_benchmark_overview",
    "Figure2_eeg_v1_audit_matrix",
    "Figure3_eeg_v1_baseline_ranking",
)


def assert_standard_figure_packet(testcase: unittest.TestCase, packet: Path) -> None:
    for stem in STANDARD_FIGURE_STEMS:
        for ext in ("png", "pdf", "svg"):
            testcase.assertTrue((packet / f"figures/{stem}.{ext}").exists(), f"{stem}.{ext}")
        with Image.open(packet / f"figures/{stem}.png") as image:
            testcase.assertGreaterEqual(image.width, 1200, stem)
            testcase.assertGreaterEqual(image.height, 700, stem)
        svg = (packet / f"figures/{stem}.svg").read_text(encoding="utf-8")
        testcase.assertIn("<text", svg, f"{stem} should keep SVG text selectable")

    for source in (packet / "src").glob("Figure*.py"):
        text = source.read_text(encoding="utf-8")
        testcase.assertNotIn("FancyBboxPatch", text, source.name)
        testcase.assertNotIn("FancyArrowPatch", text, source.name)
        testcase.assertIn("constrained", text, source.name)


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
                    "future_state_forecasting,eeg,eeg,3.11607456072082,1.29685807808655,0.9721078350075555,0.9418946133048665,3.11607456072082,2.2286510506462704\n"
                    "masked_neural_reconstruction,eeg,eeg,53.977132,5.690621,-0.012507,-0.006490,53.977132,47.425705\n",
                )
                zf.writestr(
                    "sample/run/tables/baseline_ranking.csv",
                    "task_id,model_id,metric,value,rank\n"
                    "future_state_forecasting,linear_ridge,mse,7.7,1\n"
                    "masked_neural_reconstruction,linear_ridge,mse,7.8,1\n",
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
            for name in ("eeg_v1_ridge_visual_analysis.md", "eeg_v1_ridge_visual_summary.json"):
                self.assertTrue((out / name).exists(), name)
            packet = Path(tmp) / "eeg_v1_figure_source"
            for name in (
                "data/task_results.csv",
                "data/baseline_ranking.csv",
                "data/audits.csv",
                "data/inventory.json",
                "data/provenance.json",
            ):
                self.assertTrue((packet / name).exists(), name)
            assert_standard_figure_packet(self, packet)
            figure3_svg = (packet / "figures/Figure3_eeg_v1_baseline_ranking.svg").read_text(encoding="utf-8")
            self.assertIn("winner: Kahlus v1 recovered", figure3_svg)
            self.assertIn("winner: linear ridge", figure3_svg)
            self.assertNotIn("Kahlus wins this saved comparison", figure3_svg)
            self.assertFalse((out / "fig03_prediction_overlay_and_residuals.png").exists())
            summary = json.loads((out / "eeg_v1_ridge_visual_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["source_mode"], "versions_evidence")
            self.assertFalse(summary["raw_tensor_artifacts_found"])
            comparison = summary["recovered_kahlus_vs_ridge"]
            self.assertAlmostEqual(comparison["future_state_forecasting"]["kahlus_v1_recovered_mse"], 3.11607456072082)
            self.assertAlmostEqual(comparison["future_state_forecasting"]["linear_ridge_mse"], 7.7)
            self.assertEqual(comparison["future_state_forecasting"]["winner"], "kahlus_v1_recovered")
            self.assertEqual(comparison["masked_neural_reconstruction"]["winner"], "linear_ridge")
            self.assertIn("benchmark_overview_png", summary["figure_files"])
            self.assertIn("audit_matrix_png", summary["figure_files"])
            self.assertIn("baseline_ranking_png", summary["figure_files"])
            analysis = (out / "eeg_v1_ridge_visual_analysis.md").read_text(encoding="utf-8")
            self.assertIn("Real evidence artifacts", analysis)
            self.assertIn("CEBRA-style figure-source packet", analysis)
            self.assertIn("standard matplotlib/seaborn", analysis)
            self.assertIn("Kahlus v1 recovered beats linear ridge on future_state_forecasting", analysis)
            self.assertIn("linear ridge beats Kahlus v1 recovered on masked_neural_reconstruction", analysis)
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
        self.assertIn("STANDARD MATPLOTLIB/SEABORN FIGURES", page)
        self.assertIn("Figure1_eeg_v1_benchmark_overview.png", page)
        self.assertIn("Figure2_eeg_v1_audit_matrix.png", page)
        self.assertIn("Figure3_eeg_v1_baseline_ranking.png", page)
        self.assertIn("Restored schematic diagnostic packet", page)
        self.assertIn("not benchmark evidence", page)
        self.assertIn("fig2_ridge_prediction_overlay.png", page)

    def test_ridge_waveform_sanity_diagrams_are_reproducible_and_labeled(self):
        script = Path("docs/research/eeg_v1_ridge_sanity_diagrams/src/render_ridge_waveform_sanity.py")
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        result = subprocess.run([sys.executable, str(script)], check=False, capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

        root = Path("docs/research/eeg_v1_ridge_sanity_diagrams")
        expected = (
            "FigureS6_ridge_future_window_contract",
            "FigureS7_ridge_prediction_overlay",
        )
        for stem in expected:
            for ext in ("png", "pdf", "svg"):
                self.assertTrue((root / f"figures/{stem}.{ext}").exists(), f"{stem}.{ext}")
            with Image.open(root / f"figures/{stem}.png") as image:
                self.assertGreaterEqual(image.width, 1200, stem)
                self.assertGreaterEqual(image.height, 700, stem)
            svg = (root / f"figures/{stem}.svg").read_text(encoding="utf-8")
            self.assertIn("<text", svg)

        summary = json.loads((root / "data/ridge_waveform_sanity_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["contract"]["future_forecasting"], "X = EEG window[:-1], Y = EEG window[1:]")
        self.assertEqual(summary["x_test_shape"], [9, 7, 6])
        self.assertIn("not raw EEG evidence", summary["claim_scope"])

        page = Path("docs/figures/eeg-v1-ridge-visuals.md").read_text(encoding="utf-8")
        self.assertIn("Ridge sanity-check diagrams", page)
        self.assertIn("does **not** contain raw EEG windows", page)
        self.assertIn("FigureS6_ridge_future_window_contract.png", page)
        self.assertIn("FigureS7_ridge_prediction_overlay.png", page)


if __name__ == "__main__":
    unittest.main()
