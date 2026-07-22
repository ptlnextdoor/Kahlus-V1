"""Regression: overlapping forecast windows must not yield a near-zero masked score for a copy predictor."""

from __future__ import annotations

import unittest

import numpy as np

from neurotwin.benchmarks.baseline_suite import _metrics
from neurotwin.data.forecast_contract import strictly_future_metric_mask
from neurotwin.data.prepared_tasks import build_future_forecasting_task_from_windows
from neurotwin.data.schemas import NeuralEventBatch
from neurotwin.eeg_v1.dataset import build_future_forecasting_task, make_synthetic_eeg_v1_dataset


def _window_batch(signal: np.ndarray, *, subject: str = "s0", recording: str = "r0") -> NeuralEventBatch:
    signal = np.asarray(signal, dtype=np.float32)
    return NeuralEventBatch(
        modality="eeg",
        dataset="synthetic",
        subject_id=subject,
        session_id="sess0",
        site_id="site0",
        time=np.arange(signal.shape[0], dtype=np.float32),
        signal=signal,
        mask=np.ones_like(signal, dtype=bool),
        stimulus_embedding=None,
        behavior={},
        space_index=np.arange(signal.shape[-1]),
        uncertainty=np.full_like(signal, 0.05),
        provenance={"source": "unit_test"},
        metadata={"recording_id": recording, "window_start_index": 0},
    )


class ForecastNoOverlapTests(unittest.TestCase):
    def test_strictly_future_mask_marks_only_absent_positions(self):
        y = np.zeros((2, 8, 3), dtype=np.float32)
        mask = strictly_future_metric_mask(y, forecast_horizon=1, input_length=8)
        self.assertEqual(mask.shape, y.shape)
        self.assertTrue(np.all(mask[:, -1, :]))
        self.assertFalse(np.any(mask[:, :-1, :]))

    def test_eeg_v1_forecast_task_carries_strictly_future_mask(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=0, n_subjects=6, n_time=40, n_channels=2)
        task = build_future_forecasting_task(dataset, window_length=8, forecast_horizon=1, stride=2)
        self.assertIsNotNone(task.metric_mask)
        self.assertIsNotNone(task.train_metric_mask)
        self.assertTrue(bool(task.metadata.get("strictly_future_metric_mask")))
        scored = np.where(np.any(task.metric_mask[0], axis=-1))[0]
        self.assertTrue(np.array_equal(scored, np.array([7])))
        for index in scored:
            absolute = 1 + int(index)
            self.assertGreaterEqual(absolute, 8)

    def test_copy_input_shifted_predictor_is_not_near_zero_when_masked(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=1, n_subjects=8, n_time=48, n_channels=3)
        task = build_future_forecasting_task(dataset, window_length=8, forecast_horizon=1, stride=2)
        x = np.asarray(task.x_test, dtype=np.float32)
        y = np.asarray(task.y_test, dtype=np.float32)
        # Trap for equal-length H-shifted windows: copy x[:, H:] into y[:, :-H].
        horizon = int(task.metadata["forecast_horizon"])
        copy_pred = np.empty_like(y)
        copy_pred[:, :-horizon] = x[:, horizon:]
        copy_pred[:, -horizon:] = x[:, -horizon:]

        overlap_mse = float(np.mean((y[:, :-horizon] - copy_pred[:, :-horizon]) ** 2))
        unmasked = _metrics(y, copy_pred, None, source_modality="eeg", target_modality="eeg", seed=0)
        masked = _metrics(
            y,
            copy_pred,
            task.metric_mask,
            source_modality="eeg",
            target_modality="eeg",
            seed=0,
        )
        self.assertLess(overlap_mse, 1e-10, "copy must be exact on overlapping positions")
        self.assertGreater(masked["mse"], unmasked["mse"] * 5.0)
        self.assertGreater(masked["mse"], 0.05, "masked score must not be inflated by the copy")

    def test_prepared_legacy_windows_attach_mask_and_reject_copy_trap(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=2, n_subjects=8, n_time=48, n_channels=2)
        by_id = {batch.recording_id: batch for batch in dataset.batches}
        windows: dict[str, list[NeuralEventBatch]] = {"train": [], "val": [], "test": []}
        for split_name, records in (
            ("train", dataset.split_manifest.train),
            ("val", dataset.split_manifest.val),
            ("test", dataset.split_manifest.test),
        ):
            for record in records:
                batch = by_id[record.record_id]
                signal = batch.signal
                for start in range(0, max(1, signal.shape[0] - 8), 8):
                    windows[split_name].append(
                        _window_batch(
                            signal[start : start + 8],
                            subject=record.subject_id,
                            recording=record.record_id,
                        )
                    )
        task = build_future_forecasting_task_from_windows(windows)
        self.assertIsNotNone(task)
        assert task is not None
        self.assertIsNotNone(task.metric_mask)
        # Legacy path: x=s[:-1], y=s[1:]. Exact overlap copy is pred[:, :-1] = x[:, 1:].
        x = np.asarray(task.x_test, dtype=np.float32)
        y = np.asarray(task.y_test, dtype=np.float32)
        copy_pred = np.empty_like(y)
        copy_pred[:, :-1] = x[:, 1:]
        copy_pred[:, -1:] = x[:, -1:]
        overlap_mse = float(np.mean((y[:, :-1] - copy_pred[:, :-1]) ** 2))
        unmasked = _metrics(
            y,
            copy_pred,
            None,
            source_modality=task.source_modality,
            target_modality=task.target_modality,
            seed=0,
        )
        masked = _metrics(
            y,
            copy_pred,
            task.metric_mask,
            source_modality=task.source_modality,
            target_modality=task.target_modality,
            seed=0,
        )
        self.assertLess(overlap_mse, 1e-10)
        self.assertGreater(masked["mse"], unmasked["mse"] * 5.0)
        self.assertGreater(masked["mse"], 0.05)


if __name__ == "__main__":
    unittest.main()
