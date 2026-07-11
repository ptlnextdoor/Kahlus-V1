from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from neurotwin.adapters.synthetic import make_synthetic_event_batches, make_synthetic_recordings
from neurotwin.data.split_manifest import build_split_manifest
from neurotwin.eeg_v1.window_evidence import (
    HISTORICAL_OVERLAP_PROTOCOL,
    WindowEvidenceConfig,
    build_historical_window_evidence,
    recordwise_autocorrelation,
    write_historical_window_evidence,
)


class WindowEvidenceTests(unittest.TestCase):
    @staticmethod
    def _add_moabb_provenance(batches) -> None:
        for batch in batches:
            batch.metadata.update(
                {
                    "sampling_rate": 250.0,
                    "signal_unit": "uV",
                    "channel_names": [f"C{idx}" for idx in range(batch.n_space)],
                    "signal_source": "MOABB-preprocessed epochs",
                    "moabb_dataset": "FixtureDataset",
                    "moabb_paradigm": "FixtureParadigm",
                    "moabb_version": "1.2.3",
                    "moabb_filters": {"source": "paradigm.filters", "value": [[8.0, 32.0]]},
                    "moabb_preprocessing": {"api": "paradigm.get_data(return_epochs=True)"},
                    "unit_factor_provenance": {
                        "factor": 1e6,
                        "factor_source": "FixtureDataset.unit_factor",
                        "operation": "multiply",
                        "output_unit": "uV",
                    },
                }
            )

    def test_export_labels_the_shifted_task_as_overlapping_and_ineligible(self) -> None:
        records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=1, modalities=("eeg",))
        batches = make_synthetic_event_batches(n_subjects=6, sessions_per_subject=1, modalities=("eeg",), n_time=80)
        self._add_moabb_provenance(batches)
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
        self.assertEqual(export.manifest["signal_source"], "MOABB-preprocessed epochs")
        self.assertEqual(export.manifest["dataset"], "FixtureDataset")
        self.assertEqual(export.manifest["paradigm"], "FixtureParadigm")
        self.assertEqual(export.manifest["moabb_version"], "1.2.3")
        self.assertEqual(export.manifest["filters"]["value"], [[8.0, 32.0]])
        self.assertEqual(export.manifest["unit_factor_provenance"]["factor"], 1e6)
        self.assertEqual(export.arrays["train_record_ids"].shape[0], export.arrays["x_train"].shape[0])
        self.assertEqual(export.arrays["train_subject_ids"].shape[0], export.arrays["x_train"].shape[0])

    def test_export_writes_local_arrays_and_provenance_without_claiming_model_predictions(self) -> None:
        records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=1, modalities=("eeg",))
        batches = make_synthetic_event_batches(n_subjects=6, sessions_per_subject=1, modalities=("eeg",), n_time=80)
        self._add_moabb_provenance(batches)
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

    def test_recordwise_autocorrelation_never_crosses_trial_boundaries(self) -> None:
        windows = np.asarray([[[0.0], [1.0], [2.0]], [[100.0], [101.0], [102.0]]])

        acf = recordwise_autocorrelation(windows, np.asarray(["trial-a", "trial-b"]), max_lag=1)

        self.assertEqual(acf.shape, (2, 1))
        self.assertAlmostEqual(acf[0, 0], 1.0)
        self.assertAlmostEqual(acf[1, 0], 0.0)

    def test_writer_rejects_output_inside_git_repository(self) -> None:
        records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=1, modalities=("eeg",))
        batches = make_synthetic_event_batches(n_subjects=6, sessions_per_subject=1, modalities=("eeg",), n_time=80)
        self._add_moabb_provenance(batches)
        export = build_historical_window_evidence(
            batches,
            build_split_manifest(records, policy="subject", seed=0),
            config=WindowEvidenceConfig(context_samples=8, forecast_horizon_samples=1, stride_samples=8),
        )
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["git", "init", "-q", tmp], check=True)
            with self.assertRaisesRegex(ValueError, "outside every Git"):
                write_historical_window_evidence(export, Path(tmp) / "nested" / "output")

    def test_atomic_writer_removes_stale_output_files(self) -> None:
        records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=1, modalities=("eeg",))
        batches = make_synthetic_event_batches(n_subjects=6, sessions_per_subject=1, modalities=("eeg",), n_time=80)
        self._add_moabb_provenance(batches)
        export = build_historical_window_evidence(
            batches,
            build_split_manifest(records, policy="subject", seed=0),
            config=WindowEvidenceConfig(context_samples=8, forecast_horizon_samples=1, stride_samples=8),
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "output"
            out.mkdir()
            (out / "stale-figure.png").write_bytes(b"stale")

            write_historical_window_evidence(export, out)

            self.assertFalse((out / "stale-figure.png").exists())
            self.assertEqual(
                {path.name for path in out.iterdir()},
                {"moabb_historical_window_evidence.json", "moabb_historical_window_evidence.npz"},
            )

    def test_plotter_rejects_unsupported_kahlus_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npz = root / "unsupported.npz"
            arrays = np.zeros((2, 4, 1), dtype=np.float32)
            np.savez(
                npz,
                x_train=arrays,
                y_train=arrays,
                x_test=arrays,
                y_test=arrays,
                train_record_ids=np.asarray(["a", "b"]),
                channel_names=np.asarray(["Cz"]),
                sfreq=np.asarray(128.0),
                kahlus_prediction_test=arrays,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/analysis/plot_ridge_eeg_diagnostics.py",
                    "--npz",
                    str(npz),
                    "--out",
                    str(root / "figures"),
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("kahlus_prediction_test is unsupported", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
