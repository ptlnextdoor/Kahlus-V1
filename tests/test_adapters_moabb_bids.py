import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurotwin.adapters.bids import bids_manifest_summary, records_to_event_batches, scan_bids_manifest
from neurotwin.adapters.moabb import (
    MissingOptionalDependency,
    balanced_trial_subset,
    moabb_optional_status,
    trials_to_event_batches,
    trials_to_recordings,
)


class AdapterTests(unittest.TestCase):
    def test_moabb_missing_dependency_is_clean(self):
        status = moabb_optional_status()

        self.assertIn("moabb", status)
        if not status["moabb"]:
            with self.assertRaisesRegex(MissingOptionalDependency, "pip install"):
                trials_to_recordings(None, dataset_id="BNCI2014_001")

    def test_mock_moabb_trials_convert_to_records_and_batches(self):
        trials = [
            {
                "signal": np.ones((4, 3), dtype=np.float32),
                "subject": "1",
                "session": "0",
                "run": "0",
                "label": "left_hand",
                "sampling_rate": 128.0,
                "channel_names": ["C3", "Cz", "C4"],
            },
            {
                "signal": np.zeros((4, 3), dtype=np.float32),
                "subject": "2",
                "session": "0",
                "run": "1",
                "label": "right_hand",
                "sampling_rate": 128.0,
                "channel_names": ["C3", "Cz", "C4"],
            },
        ]

        records = trials_to_recordings(trials, dataset_id="mock_moabb")
        batches = trials_to_event_batches(trials, dataset_id="mock_moabb")

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].subject_id, "sub-1")
        self.assertEqual(records[0].record_id, batches[0].metadata["record_id"])
        self.assertEqual(records[0].record_id, batches[0].metadata["source_record_id"])
        self.assertEqual(records[0].metadata["run_id"], "run-0")
        self.assertEqual(records[0].metadata["sampling_rate"], 128.0)
        self.assertEqual(batches[0].modality, "eeg")
        self.assertEqual(batches[0].signal.shape, (4, 3))
        self.assertEqual(batches[0].metadata["channel_names"], ["C3", "Cz", "C4"])
        forbidden = {"label", "target", "target_label", "task_label", "diagnosis"}
        for record in records:
            self.assertFalse(forbidden.intersection(key.lower() for key in record.metadata))
        for batch in batches:
            self.assertFalse(forbidden.intersection(key.lower() for key in batch.metadata))
            self.assertNotIn("label", batch.behavior)

    def test_moabb_channels_first_trials_are_transposed_to_event_schema(self):
        trials = [
            {
                "signal": np.ones((3, 5), dtype=np.float32),
                "subject": "1",
                "session": "0",
                "run": "0",
                "label": "left_hand",
                "sampling_rate": 100.0,
                "channel_names": ["C3", "Cz", "C4"],
            }
        ]

        records = trials_to_recordings(trials, dataset_id="mock_moabb")
        batches = trials_to_event_batches(trials, dataset_id="mock_moabb")

        self.assertEqual(batches[0].signal.shape, (5, 3))
        self.assertEqual(batches[0].time.shape, (5,))
        self.assertAlmostEqual(records[0].end_time, 0.05)

    def test_moabb_balanced_trial_subset_preserves_subject_groups(self):
        trials = []
        for subject in ("1", "2", "3"):
            for idx in range(4):
                trials.append(
                    {
                        "signal": np.ones((4, 3), dtype=np.float32),
                        "subject": subject,
                        "session": "0",
                        "run": str(idx),
                    }
                )

        subset = balanced_trial_subset(trials, split_policy="subject", max_trials=6)

        self.assertEqual(len(subset), 6)
        self.assertEqual({trial["subject"] for trial in subset}, {"1", "2", "3"})

    def test_moabb_balanced_trial_subset_rejects_two_subject_smoke(self):
        trials = [
            {"signal": np.ones((4, 3), dtype=np.float32), "subject": "1"},
            {"signal": np.ones((4, 3), dtype=np.float32), "subject": "2"},
        ]

        with self.assertRaisesRegex(ValueError, "at least 3 subject groups"):
            balanced_trial_subset(trials, split_policy="subject", max_trials=2)

    def test_bids_scanner_parses_participants_events_and_scans(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "participants.tsv").write_text("participant_id\tage\nsub-01\t22\n", encoding="utf-8")
            func = root / "sub-01" / "ses-01" / "func"
            func.mkdir(parents=True)
            bold = func / "sub-01_ses-01_task-rest_run-01_bold.nii.gz"
            bold.write_text("placeholder", encoding="utf-8")
            (func / "sub-01_ses-01_task-rest_run-01_events.tsv").write_text(
                "onset\tduration\ttrial_type\n0\t1\tfixation\n",
                encoding="utf-8",
            )
            (root / "sub-01" / "ses-01" / "sub-01_ses-01_scans.tsv").write_text(
                "filename\tacq_time\nfunc/sub-01_ses-01_task-rest_run-01_bold.nii.gz\t2026-01-01T00:00:00\n",
                encoding="utf-8",
            )

            records = scan_bids_manifest(root, dataset_id="mock_bids", site_id="site-x")
            summary = bids_manifest_summary(records)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].subject_id, "sub-01")
        self.assertEqual(records[0].session_id, "ses-01")
        self.assertEqual(records[0].metadata["task"], "rest")
        self.assertEqual(records[0].metadata["run_id"], "run-01")
        self.assertEqual(records[0].metadata["participants"]["age"], "22")
        self.assertEqual(records[0].metadata["events"][0]["trial_type"], "fixation")
        self.assertEqual(summary["with_timeseries_derivative"], 0)
        self.assertTrue(summary["derivative_only"])

    def test_bids_timeseries_derivative_converts_to_event_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            func = root / "sub-01" / "ses-01" / "func"
            func.mkdir(parents=True)
            bold = func / "sub-01_ses-01_task-rest_run-01_bold.nii.gz"
            bold.write_text("placeholder", encoding="utf-8")
            np.save(func / "sub-01_ses-01_task-rest_run-01_timeseries.npy", np.ones((5, 3), dtype=np.float32))

            records = scan_bids_manifest(root, dataset_id="mock_bids", site_id="site-x")
            batches = records_to_event_batches(records)

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].modality, "fmri")
        self.assertEqual(batches[0].signal.shape, (5, 3))
        self.assertEqual(batches[0].metadata["record_id"], records[0].record_id)
        self.assertIn("timeseries_derivative", records[0].metadata)

    def test_bids_derivative_fixture_parses_site_run_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "participants.tsv").write_text("participant_id\tsite\nsub-02\tparticipant-site\n", encoding="utf-8")
            func = root / "sub-02" / "ses-02" / "func"
            func.mkdir(parents=True)
            bold = func / "sub-02_ses-02_task-memory_run-03_bold.nii.gz"
            bold.write_text("placeholder", encoding="utf-8")
            (root / "sub-02" / "ses-02" / "sub-02_ses-02_scans.tsv").write_text(
                "filename\tsite\nfunc/sub-02_ses-02_task-memory_run-03_bold.nii.gz\tscan-site\n",
                encoding="utf-8",
            )
            derivative_dir = root / "derivatives" / "neurotwin" / "sub-02" / "ses-02" / "func"
            derivative_dir.mkdir(parents=True)
            np.savez(
                derivative_dir / "sub-02_ses-02_task-memory_run-03_timeseries.npz",
                signal=np.ones((6, 4), dtype=np.float32),
                labels=np.array(["r1", "r2", "r3", "r4"]),
            )

            records = scan_bids_manifest(root, dataset_id="mock_bids", site_id="fallback-site")
            batches = records_to_event_batches(records)

        self.assertEqual(records[0].site_id, "scan-site")
        self.assertEqual(records[0].metadata["run_id"], "run-03")
        self.assertEqual(batches[0].metadata["space_labels"], ["r1", "r2", "r3", "r4"])
        self.assertEqual(batches[0].signal.shape, (6, 4))

    def test_bids_timeseries_derivative_rejects_nonfinite_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            func = root / "sub-01" / "ses-01" / "func"
            func.mkdir(parents=True)
            bold = func / "sub-01_ses-01_task-rest_run-01_bold.nii.gz"
            bold.write_text("placeholder", encoding="utf-8")
            signal = np.ones((5, 3), dtype=np.float32)
            signal[0, 0] = np.nan
            np.save(func / "sub-01_ses-01_task-rest_run-01_timeseries.npy", signal)
            records = scan_bids_manifest(root, dataset_id="mock_bids", site_id="site-x")

            with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                records_to_event_batches(records)
