from __future__ import annotations

import os
from unittest import mock
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurotwin.forecastability.m2 import (
    SleepFixture,
    _as_transition_fixture,
    _sleep_edf_record_metadata,
    fetch_sleep_edf_records_index,
    load_sleep_edf_fixture,
    make_synthetic_sleep_fixture,
    run_m2_gate,
)


class ForecastabilityM2Tests(unittest.TestCase):
    def test_synthetic_sleep_fixture_has_transition_events(self) -> None:
        fixture = make_synthetic_sleep_fixture(seed=2)
        self.assertGreater(int(fixture.transition.sum()), 80)
        self.assertGreaterEqual(len(set(fixture.subject[fixture.transition == 1])), 10)

    def test_sleep_edf_filename_metadata_parses_subject_and_night(self) -> None:
        first = _sleep_edf_record_metadata("SC4001E0-PSG.edf")
        second = _sleep_edf_record_metadata("SC4002E0-Hypnogram.edf")
        telemetry = _sleep_edf_record_metadata("ST7022J0-PSG.edf")

        self.assertEqual(first["record_id"], "SC4001E0")
        self.assertEqual(first["dataset_id"], "sleep-cassette")
        self.assertEqual(first["study_code"], "SC")
        self.assertEqual(first["subject_id"], "SC-00")
        self.assertEqual(first["night"], 1)
        self.assertEqual(first["session_id"], "night-1")
        self.assertEqual(second["subject_id"], "SC-00")
        self.assertEqual(second["night"], 2)
        self.assertNotEqual(first["subject_session_id"], second["subject_session_id"])
        self.assertEqual(telemetry["dataset_id"], "sleep-telemetry")
        self.assertEqual(telemetry["subject_id"], "ST-02")
        self.assertEqual(telemetry["night"], 2)

    def test_sleep_edf_filename_metadata_rejects_unrecognized_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "Sleep-EDF filename"):
            _sleep_edf_record_metadata("unknown-PSG.edf")

    def test_load_sleep_edf_fixture_groups_same_subject_across_nights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_sleep_pair(root, "SC4001E0")
            _write_sleep_pair(root, "SC4002E0")

            with mock.patch(
                "neurotwin.forecastability.m2._read_edf_signals",
                return_value={
                    "signals": np.ones((6, 64, 3), dtype=np.float32),
                    "record_duration": 30.0,
                    "labels": ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"],
                },
            ), mock.patch(
                "neurotwin.forecastability.m2._read_sleep_edf_hypnogram",
                return_value=[(0.0, 180.0, 0)],
            ):
                fixture = load_sleep_edf_fixture(root, max_pairs=None)

        self.assertEqual(len(set(fixture.subject.tolist())), 1)
        self.assertEqual(len(set(fixture.session.tolist())), 2)
        self.assertEqual(set(fixture.dataset.tolist()), {0})

    def test_as_transition_fixture_preserves_sleep_session(self) -> None:
        fixture = SleepFixture(
            windows=np.zeros((2, 4, 3), dtype=np.float32),
            stages=np.asarray([0, 1], dtype=np.int64),
            transition=np.asarray([0, 1], dtype=np.int64),
            subject=np.asarray([7, 7], dtype=np.int64),
            session=np.asarray([11, 12], dtype=np.int64),
            dataset=np.asarray([0, 0], dtype=np.int64),
            site=np.asarray([0, 0], dtype=np.int64),
            nuisance=np.zeros((2, 5), dtype=np.float32),
        )

        transition = _as_transition_fixture(fixture)

        self.assertEqual(transition.patient.tolist(), [7, 7])
        self.assertEqual(transition.session.tolist(), [11, 12])

    def test_m2_refuses_real_gate_without_sleep_edf_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = run_m2_gate(Path(tmp), seed=2)
            self.assertTrue(gate["synthetic_sleep_machinery_passed"])
            self.assertFalse(gate["gate_passed"])
            self.assertEqual(gate["real_sleep_edf"]["status"], "not_run_no_local_sleep_edf_root")

    def test_sleep_edf_records_index_parses_response(self) -> None:
        response = mock.MagicMock()
        response.read.return_value = b"sleep-cassette/SC4001E0-PSG.edf\nsleep-cassette/SC4001E0-Hypnogram.edf\n"
        response.__enter__.return_value = response
        with mock.patch("neurotwin.forecastability.m2.urlopen", return_value=response) as urlopen_mock:
            records = fetch_sleep_edf_records_index()

        self.assertIn("sleep-cassette/SC4001E0-PSG.edf", records)
        urlopen_mock.assert_called_once_with("https://physionet.org/files/sleep-edfx/1.0.0/RECORDS", timeout=20)

    @unittest.skipUnless(
        os.environ.get("NEUROTWIN_RUN_NETWORK_TESTS") == "1",
        "real network integration disabled; set NEUROTWIN_RUN_NETWORK_TESTS=1",
    )
    def test_sleep_edf_records_index_network_source(self) -> None:
        records = fetch_sleep_edf_records_index()
        self.assertIn("sleep-cassette/SC4001E0-PSG.edf", records)

def _write_sleep_pair(root: Path, stem: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{stem}-PSG.edf").write_bytes(f"{stem}:psg\n".encode("utf-8"))
    (root / f"{stem}-Hypnogram.edf").write_bytes(f"{stem}:hypnogram\n".encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
