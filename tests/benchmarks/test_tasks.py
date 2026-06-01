import unittest

import numpy as np

from neurotwin.benchmarks.tasks import (
    run_cross_modal_translation_task,
    run_future_forecasting_task,
    run_masked_reconstruction_task,
    run_subject_adaptation_task,
)


class BenchmarkTaskTests(unittest.TestCase):
    def test_future_and_masked_tasks_return_metrics(self):
        signal = np.arange(80, dtype=np.float32).reshape(10, 8)

        forecast = run_future_forecasting_task(signal, history=3, horizon=1)
        masked = run_masked_reconstruction_task(signal, mask_fraction=0.25, seed=3)

        self.assertEqual(forecast.status, "completed")
        self.assertIn("mse", forecast.metrics)
        self.assertEqual(masked.status, "completed")
        self.assertIn("masked_mse", masked.metrics)

    def test_cross_modal_translation_skips_without_pairing(self):
        result = run_cross_modal_translation_task({"eeg": np.ones((4, 3))}, source="eeg", target="fmri")

        self.assertEqual(result.status, "skipped")
        self.assertIn("missing paired modalities", result.notes[0])

    def test_subject_adaptation_reports_gain(self):
        support = np.ones((4, 3), dtype=np.float32)
        query = np.ones((4, 3), dtype=np.float32) * 1.1

        result = run_subject_adaptation_task(support, query)

        self.assertEqual(result.status, "completed")
        self.assertIn("adaptation_gain", result.metrics)
