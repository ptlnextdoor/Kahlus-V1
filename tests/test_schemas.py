import unittest

import numpy as np

from neurotwin.data.schemas import NeuralEventBatch


class NeuralEventBatchTests(unittest.TestCase):
    def test_valid_batch_preserves_core_metadata(self):
        batch = NeuralEventBatch(
            modality="fmri",
            dataset="synthetic_fmri",
            subject_id="sub-001",
            session_id="ses-01",
            site_id="site-a",
            time=np.array([0.0, 1.0, 2.0]),
            signal=np.ones((3, 4), dtype=np.float32),
            mask=np.ones((3, 4), dtype=bool),
            stimulus_embedding=np.zeros((3, 8), dtype=np.float32),
            behavior={"response_time": np.array([0.2, 0.3, 0.4])},
            space_index=np.arange(4),
            uncertainty=np.full((3, 4), 0.1, dtype=np.float32),
            provenance={"split_stage": "recording_manifest"},
        )

        self.assertEqual(batch.n_time, 3)
        self.assertEqual(batch.n_space, 4)
        self.assertEqual(batch.modality, "fmri")
        self.assertEqual(batch.provenance["split_stage"], "recording_manifest")

    def test_rejects_mismatched_time_axis(self):
        with self.assertRaisesRegex(ValueError, "time axis"):
            NeuralEventBatch(
                modality="eeg",
                dataset="synthetic_eeg",
                subject_id="sub-001",
                session_id="ses-01",
                site_id="site-a",
                time=np.array([0.0, 1.0]),
                signal=np.ones((3, 2), dtype=np.float32),
                mask=np.ones((3, 2), dtype=bool),
                stimulus_embedding=None,
                behavior={},
                space_index=np.arange(2),
            )

    def test_rejects_unknown_modality(self):
        with self.assertRaisesRegex(ValueError, "Unsupported modality"):
            NeuralEventBatch(
                modality="bloodwork",
                dataset="synthetic",
                subject_id="sub-001",
                session_id="ses-01",
                site_id="site-a",
                time=np.array([0.0]),
                signal=np.ones((1, 2), dtype=np.float32),
                mask=np.ones((1, 2), dtype=bool),
                stimulus_embedding=None,
                behavior={},
                space_index=np.arange(2),
            )
