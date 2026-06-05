import tempfile
import unittest
from pathlib import Path

from neurotwin.benchmarks.nfc_suite import format_nfc_synthetic_report, run_nfc_synthetic_suite


class NFCSyntheticSuiteTests(unittest.TestCase):
    def test_nfc_synthetic_suite_emits_required_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = run_nfc_synthetic_suite(seed=0, train_steps=1, out_dir=tmp)
            out = Path(tmp)

            self.assertTrue((out / "nfc_synthetic_results.json").exists())
            self.assertTrue((out / "nfc_synthetic_results.csv").exists())
            self.assertTrue((out / "nfc_ablation_table.csv").exists())
            self.assertTrue((out / "nfc_falsification_report.md").exists())
            self.assertTrue((out / "uncertainty_calibration.csv").exists())

        self.assertEqual(payload["scope"]["status"], "synthetic-only")
        self.assertIn("nfc_full", payload["models"])
        self.assertIn("pair_operator", payload["models"])
        self.assertEqual(payload["models"]["pair_operator"]["role"], "baseline_ablation")
        self.assertIn("synthetic_latent_field_recovery", payload["tasks"])
        self.assertIn("falsification", payload)

    def test_nfc_synthetic_suite_aggregates_multiple_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = run_nfc_synthetic_suite(seeds=(0, 1), train_steps=1, out_dir=tmp)
            out = Path(tmp)

            self.assertEqual(payload["seeds"], [0, 1])
            self.assertEqual(payload["tasks"]["synthetic_latent_field_recovery"]["metrics_by_model"]["nfc_full"]["n_seeds"], 2)
            self.assertTrue((out / "uncertainty_calibration.csv").exists())
            calibration = (out / "uncertainty_calibration.csv").read_text(encoding="utf-8")

        self.assertIn("finite", calibration)
        self.assertIn("no_nan_metrics", [row["criterion"] for row in payload["falsification"]["criteria"]])

    def test_nfc_report_states_pair_operator_is_not_main_architecture(self):
        report = format_nfc_synthetic_report(run_nfc_synthetic_suite(seed=1, train_steps=1))

        self.assertIn("NeuroTwin NFC Synthetic Suite", report)
        self.assertIn("Pair-Operator is a baseline/ablation", report)
        self.assertIn("synthetic-only", report)


if __name__ == "__main__":
    unittest.main()
