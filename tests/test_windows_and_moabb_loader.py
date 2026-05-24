import unittest
from unittest import mock

import numpy as np

from neurotwin.adapters.moabb import load_moabb_trials
from neurotwin.adapters.synthetic import make_synthetic_event_batch
from neurotwin.data.windows import WindowSpec, batch_to_windows


class WindowsAndMoabbLoaderTests(unittest.TestCase):
    def test_batch_to_windows_preserves_source_metadata(self):
        batch = make_synthetic_event_batch(modality="eeg", n_time=10, n_space=3)

        windows = batch_to_windows(batch, WindowSpec(length=4, stride=3))

        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0].signal.shape, (4, 3))
        self.assertEqual(windows[0].provenance["source_subject_id"], batch.subject_id)
        self.assertEqual(windows[0].metadata["window_start_index"], 0)
        self.assertEqual(windows[1].metadata["window_start_index"], 3)

    def test_load_moabb_trials_uses_paradigm_interface_when_available(self):
        fake_dataset = mock.Mock()
        fake_dataset_cls = mock.Mock(return_value=fake_dataset)
        fake_paradigm = mock.Mock()
        fake_paradigm.get_data.return_value = (
            np.ones((2, 5, 3), dtype=np.float32),
            np.array(["left", "right"]),
            [
                {"subject": 1, "session": "0", "run": "0"},
                {"subject": 2, "session": "0", "run": "1"},
            ],
        )

        with mock.patch("neurotwin.adapters.moabb.require_moabb"), mock.patch(
            "neurotwin.adapters.moabb._resolve_moabb_dataset", return_value=fake_dataset_cls
        ), mock.patch("neurotwin.adapters.moabb._build_moabb_paradigm", return_value=fake_paradigm):
            trials = load_moabb_trials("FakeDataset", subjects=(1, 2), max_trials=1, sampling_rate=256.0)

        self.assertEqual(len(trials), 1)
        self.assertEqual(trials[0]["subject"], 1)
        self.assertEqual(trials[0]["label"], "left")
        self.assertEqual(trials[0]["sampling_rate"], 256.0)
