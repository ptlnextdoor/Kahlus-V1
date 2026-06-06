import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurotwin.benchmarks.nfc_suite import (
    NfcPredictionShapeError,
    NfcSyntheticTaskSpec,
    _predict_autoregressive_baseline,
    format_nfc_synthetic_report,
    rank_nfc_task_predictions,
    run_nfc_synthetic_suite,
)


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
            self.assertTrue((out / "evidence_gate.json").exists())
            self.assertTrue((out / "diagnostic_report.md").exists())

        self.assertEqual(payload["scope"]["status"], "synthetic-only")
        self.assertIn("nfc_full", payload["models"])
        self.assertIn("pair_operator", payload["models"])
        self.assertIn("direct_linear", payload["models"])
        self.assertIn("direct_mlp", payload["models"])
        self.assertEqual(payload["models"]["pair_operator"]["role"], "baseline_ablation")
        self.assertIn("synthetic_latent_field_recovery", payload["tasks"])
        self.assertIn("synthetic_latent_observation_recovery", payload["tasks"])
        self.assertIn("synthetic_eeg_to_fmri", payload["tasks"])
        self.assertIn("synthetic_fmri_to_eeg", payload["tasks"])
        self.assertIn("falsification", payload)
        self.assertEqual(payload["task_contracts"]["synthetic_latent_field_recovery"]["target_kind"], "latent_field")

    def test_nfc_synthetic_suite_aggregates_multiple_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = run_nfc_synthetic_suite(seeds=(0, 1), train_steps=1, out_dir=tmp)
            out = Path(tmp)

            self.assertEqual(payload["seeds"], [0, 1])
            self.assertEqual(payload["tasks"]["synthetic_latent_observation_recovery"]["metrics_by_model"]["nfc_full"]["n_seeds"], 2)
            self.assertTrue((out / "uncertainty_calibration.csv").exists())
            calibration = (out / "uncertainty_calibration.csv").read_text(encoding="utf-8")

        self.assertIn("finite", calibration)
        self.assertIn("no_nan_metrics", [row["criterion"] for row in payload["falsification"]["criteria"]])

    def test_nfc_report_states_pair_operator_is_not_main_architecture(self):
        report = format_nfc_synthetic_report(run_nfc_synthetic_suite(seed=1, train_steps=1))

        self.assertIn("NeuroTwin NFC Synthetic Suite", report)
        self.assertIn("Pair-Operator is a baseline/ablation", report)
        self.assertIn("synthetic-only", report)

    def test_prediction_shape_mismatch_fails_hard(self):
        spec = NfcSyntheticTaskSpec(
            task_id="shape_contract",
            train_inputs={"stimulus": np.zeros((2, 3, 1), dtype=np.float32)},
            train_targets=np.zeros((2, 3, 2), dtype=np.float32),
            test_inputs={"stimulus": np.zeros((1, 3, 1), dtype=np.float32)},
            test_targets=np.zeros((1, 3, 2), dtype=np.float32),
            target_kind="fmri_observation",
            expected_prediction_shape=(1, 3, 2),
        )

        with self.assertRaisesRegex(NfcPredictionShapeError, "task=shape_contract model=bad_model"):
            rank_nfc_task_predictions(spec, {"bad_model": np.zeros((1, 3, 1), dtype=np.float32)})

    def test_autoregressive_baseline_does_not_read_test_targets(self):
        history = np.arange(30, dtype=np.float32).reshape(2, 5, 3)
        target_a = np.zeros((2, 5, 3), dtype=np.float32)
        target_b = np.full((2, 5, 3), 1000.0, dtype=np.float32)
        base = {
            "task_id": "synthetic_fmri_forecasting",
            "train_inputs": {"fmri_history": history},
            "train_targets": target_a,
            "test_inputs": {"fmri_history": history},
            "target_kind": "future_fmri_observation",
            "expected_prediction_shape": history.shape,
        }
        spec_a = NfcSyntheticTaskSpec(test_targets=target_a, **base)
        spec_b = NfcSyntheticTaskSpec(test_targets=target_b, **base)

        pred_a = _predict_autoregressive_baseline(spec_a)
        pred_b = _predict_autoregressive_baseline(spec_b)

        self.assertTrue(np.array_equal(pred_a, history))
        self.assertTrue(np.array_equal(pred_a, pred_b))

    def test_eeg_autoregressive_baseline_does_not_read_test_targets(self):
        history = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
        target_a = np.zeros((2, 4, 3), dtype=np.float32)
        target_b = np.full((2, 4, 3), -1000.0, dtype=np.float32)
        base = {
            "task_id": "synthetic_eeg_forecasting",
            "train_inputs": {"eeg_history": history},
            "train_targets": target_a,
            "test_inputs": {"eeg_history": history},
            "target_kind": "future_eeg_observation",
            "expected_prediction_shape": history.shape,
            "target_modality": "eeg",
        }
        spec_a = NfcSyntheticTaskSpec(test_targets=target_a, **base)
        spec_b = NfcSyntheticTaskSpec(test_targets=target_b, **base)

        pred_a = _predict_autoregressive_baseline(spec_a)
        pred_b = _predict_autoregressive_baseline(spec_b)

        self.assertTrue(np.array_equal(pred_a, history))
        self.assertTrue(np.array_equal(pred_a, pred_b))


if __name__ == "__main__":
    unittest.main()
