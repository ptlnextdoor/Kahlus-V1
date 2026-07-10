from __future__ import annotations

import json
import tempfile
import unittest

import numpy as np

from neurotwin.adapters.synthetic import make_synthetic_event_batches, make_synthetic_recordings
from neurotwin.data.split_manifest import build_split_manifest
from neurotwin.eeg_v1.window_evidence import (
    HISTORICAL_OVERLAP_PROTOCOL,
    WindowEvidenceConfig,
    build_historical_window_evidence,
    write_historical_window_evidence,
)


class WindowEvidenceTests(unittest.TestCase):
    def test_export_labels_the_shifted_task_as_overlapping_and_ineligible(self) -> None:
        records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=1, modalities=("eeg",))
        batches = make_synthetic_event_batches(n_subjects=6, sessions_per_subject=1, modalities=("eeg",), n_time=80)
        for batch in batches:
            batch.metadata.update(
                {
                    "sampling_rate": 250.0,
                    "signal_unit": "uV",
                    "channel_names": [f"C{idx}" for idx in range(batch.n_space)],
                }
            )
        split = build_split_manifest(records, policy="subject", seed=0)
        export = build_historical_window_evidence(
            batches,
            split,
            config=WindowEvidenceConfig(
                context_samples=8,
                forecast_horizon_samples=1,
                stride_samples=8,
                max_train_windows=8,
                max_test_windows=2,
            ),
        )

        self.assertEqual(export.manifest["protocol_id"], HISTORICAL_OVERLAP_PROTOCOL)
        self.assertFalse(export.manifest["claim_eligible"])
        self.assertEqual(export.manifest["shared_samples_per_example"], 7)
        self.assertEqual(export.arrays["x_train"].shape[1:], (8, batches[0].n_space))
        self.assertEqual(export.arrays["ridge_prediction_test"].shape, export.arrays["y_test"].shape)
        self.assertTrue(np.array_equal(export.arrays["persistence_prediction_test"], export.arrays["x_test"]))

    def test_export_writes_local_arrays_and_provenance_without_claiming_model_predictions(self) -> None:
        records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=1, modalities=("eeg",))
        batches = make_synthetic_event_batches(n_subjects=6, sessions_per_subject=1, modalities=("eeg",), n_time=80)
        for batch in batches:
            batch.metadata.update(
                {
                    "sampling_rate": 250.0,
                    "signal_unit": "uV",
                    "channel_names": [f"C{idx}" for idx in range(batch.n_space)],
                }
            )
        export = build_historical_window_evidence(
            batches,
            build_split_manifest(records, policy="subject", seed=0),
            config=WindowEvidenceConfig(context_samples=8, forecast_horizon_samples=1, stride_samples=8),
        )

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_historical_window_evidence(export, tmp)
            payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            with np.load(paths["npz"], allow_pickle=False) as arrays:
                self.assertIn("per_channel_lag_correlation", arrays.files)
                self.assertNotIn("kahlus_prediction_test", arrays.files)

        self.assertEqual(payload["model_prediction_status"], "not_available_without_a_provenance_matched_checkpoint_or_prediction_export")
        self.assertTrue(payload["raw_public_data_not_committed"])


if __name__ == "__main__":
    unittest.main()
