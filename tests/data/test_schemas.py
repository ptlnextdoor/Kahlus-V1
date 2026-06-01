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
            metadata={
                "recording_id": "rec-001",
                "dataset_id": "dataset-a",
                "task_id": "movie",
                "sampling_rate": 0.5,
                "source_hash": "raw-hash",
                "preprocessing_hash": "prep-hash",
                "split_assignment": "test",
                "geometry": {"space": "parcel"},
                "stimulus_alignment": {"stimulus_id": "clip-1"},
                "behavior_metadata": {"label_space": "rt"},
            },
        )

        self.assertEqual(batch.n_time, 3)
        self.assertEqual(batch.n_space, 4)
        self.assertEqual(batch.modality, "fmri")
        self.assertEqual(batch.provenance["split_stage"], "recording_manifest")
        self.assertEqual(batch.recording_id, "rec-001")
        self.assertEqual(batch.dataset_id, "dataset-a")
        self.assertEqual(batch.task_id, "movie")
        self.assertEqual(batch.sampling_rate, 0.5)
        self.assertEqual(batch.time_start, 0.0)
        self.assertEqual(batch.time_end, 2.0)
        self.assertEqual(batch.source_hash, "raw-hash")
        self.assertEqual(batch.preprocessing_hash, "prep-hash")
        self.assertEqual(batch.split_assignment, "test")
        self.assertEqual(batch.geometry_metadata["space"], "parcel")
        self.assertEqual(batch.stimulus_alignment_metadata["stimulus_id"], "clip-1")
        self.assertEqual(batch.behavior_metadata["label_space"], "rt")

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
