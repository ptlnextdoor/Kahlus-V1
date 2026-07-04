from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from neurotwin.forecastability.m3 import (
    _local_tusz_recordings,
    _m3_gate_failures,
    _run_tusz_external,
    fetch_chbmit_seizure_records,
    parse_chbmit_summary,
    parse_tusz_tse,
    run_m3_gate,
    tusz_source_audit,
)


class ForecastabilityM3Tests(unittest.TestCase):
    def test_chbmit_summary_parser(self) -> None:
        parsed = parse_chbmit_summary(
            "\n".join(
                [
                    "File Name: chb01_03.edf",
                    "Number of Seizures in File: 1",
                    "Seizure Start Time: 2996 seconds",
                    "Seizure End Time: 3036 seconds",
                ]
            )
        )
        self.assertEqual(parsed["chb01_03.edf"], ((2996, 3036),))

    def test_chbmit_records_source_parser_uses_official_records_file(self) -> None:
        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b"chb01/chb01_03.edf\n"

        with patch("neurotwin.forecastability.m3.urlopen", return_value=_Response()):
            self.assertIn("chb01/chb01_03.edf", fetch_chbmit_seizure_records())

    def test_external_dataset_audit_is_honest(self) -> None:
        self.assertEqual(tusz_source_audit()["status"], "not_run_requires_external_tusz_access")

    def test_tusz_tse_parser_keeps_only_seizure_allowlist(self) -> None:
        parsed = parse_tusz_tse(
            "\n".join(
                [
                    "version = tse_v1.0.0",
                    "0.0000 10.0000 bckg 1.0000",
                    "10.0000 14.5000 seiz 0.9000",
                    "20.0000 22.0000 fnsz 0.8000",
                    "30.0000 32.0000 artf 0.7000",
                ]
            )
        )
        self.assertEqual(parsed, ((10.0, 14.5), (20.0, 22.0)))

    def test_tusz_recording_scan_pairs_same_stem_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_tusz_record(root, "s001", "rec1", "10 20 seiz 1.0\n")
            _write_tusz_record(root, "s002", "rec2", "10 20 bckg 1.0\n")
            (root / "s003").mkdir()
            (root / "s003" / "rec3.edf").write_bytes(b"edf")

            recordings, missing = _local_tusz_recordings(root)

            self.assertEqual(len(recordings), 1)
            self.assertEqual(recordings[0].subject, "s001")
            self.assertEqual(recordings[0].site, "local")
            self.assertEqual([path.name for path in missing], ["rec3.edf"])

    def test_tusz_external_fixture_runs_with_local_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for idx in range(2):
                _write_tusz_record(root, f"s{idx + 1:03d}", "rec", "900 940 seiz 1.0\n")

            with patch("neurotwin.forecastability.m3._read_edf_signals", side_effect=_fake_edf):
                payload = _run_tusz_external(root, seed=11)

            self.assertEqual(payload["status"], "completed_external_dataset")
            self.assertEqual(payload["metrics"]["event_patients"], 2)

    def test_m3_gate_reports_tusz_underpowered_external_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            root = Path(tmp)
            for idx in range(2):
                _write_tusz_record(root, f"s{idx + 1:03d}", "rec", "900 940 seiz 1.0\n")

            with patch("neurotwin.forecastability.m3._read_edf_signals", side_effect=_fake_edf):
                gate = run_m3_gate(Path(out), seed=13, tusz_root=root)

            report = (Path(out) / "M3_EVIDENCE_REPORT.md").read_text(encoding="utf-8")
            self.assertEqual(gate["tusz_external"]["status"], "completed_external_dataset")
            self.assertIn("underpowered_event_patients", gate["gate_failures"])
            self.assertIn("## TUSZ External Held-Out", report)
            self.assertIn("No clinical seizure prediction claim is permitted", report)

    def test_committed_m3_artifact_matches_current_gate_logic(self) -> None:
        artifact = Path(__file__).parents[2] / "artifacts" / "forecastability_trial0_m3" / "m3_gate_report.json"
        gate = json.loads(artifact.read_text(encoding="utf-8"))

        self.assertIn("tusz_external", gate)
        self.assertEqual(gate["gate_failures"], _m3_gate_failures(gate["chb_mit_development"], gate["tusz_external"]))

def _write_tusz_record(root: Path, subject: str, stem: str, annotation: str) -> None:
    folder = root / subject
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.edf").write_bytes(b"edf")
    (folder / f"{stem}.tse").write_text(annotation, encoding="utf-8")


def _fake_edf(path: Path, *, preferred_labels: tuple[str, ...]) -> dict[str, object]:
    del path, preferred_labels
    rows = 160
    t = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    signals = []
    for row in range(rows):
        amp = 1.0 + 0.01 * row
        signals.append(np.stack([amp * np.sin(2.0 * np.pi * (idx + 1) * t) for idx in range(4)], axis=1))
    return {"signals": np.asarray(signals, dtype=np.float32), "record_duration": 10.0}


if __name__ == "__main__":
    unittest.main()
