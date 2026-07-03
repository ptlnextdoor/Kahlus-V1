import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from edfio import Edf, EdfSignal

from neurotwin.stf import CHB_MIT_DATASET_ID, parse_chb_mit_summary_text, run_chb_mit_public_smoke


class STFCHBMITSmokeTests(unittest.TestCase):
    def test_chb_mit_public_smoke_reads_edf_fixture_and_reports_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_chb_fixture(Path(tmp))

            payload = run_chb_mit_public_smoke(root, max_records=4, max_samples_per_record=256)

            self.assertTrue(payload["public_smoke_passed"], payload["failure_reasons"])
            self.assertTrue(payload["gate"]["scientific_claim_allowed"], payload["gate"]["failure_reasons"])
            self.assertFalse(payload["a100_jobs_launched"])
            self.assertEqual(payload["sampling_frequencies_hz"], [128.0, 128.0, 128.0, 128.0])
            models = {row["model_id"] for row in payload["baseline_rows"]}
            task_ids = {row["task_id"] for row in payload["baseline_rows"]}
            self.assertIn("persistence", models)
            self.assertIn("ridge_ar", models)
            self.assertIn("tiny_ssm", models)
            self.assertIn("held_out_channel_completion", task_ids)
            self.assertIn("channel_mean", models)
            self.assertIn("logistic_ridge", models)
            self.assertIn("time_shifted_label_control", models)

    def test_chb_mit_public_smoke_cli_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_chb_fixture(Path(tmp))
            out = Path(tmp) / "out"

            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_stf_chb_mit_smoke.py",
                    "--dataset",
                    CHB_MIT_DATASET_ID,
                    "--data-root",
                    str(root),
                    "--out-dir",
                    str(out),
                    "--max-records",
                    "4",
                    "--max-samples-per-record",
                    "256",
                ],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("public_smoke_passed=True", proc.stdout)
            self.assertTrue((out / "chb_mit_public_smoke.json").exists())
            report = (out / "chb_mit_public_smoke_report.md").read_text(encoding="utf-8")
            self.assertIn("a100_jobs_launched: false", report)
            self.assertIn("scientific_claim_allowed: True", report)

    def test_parse_chb_mit_summary_text_extracts_multiple_intervals(self):
        intervals = parse_chb_mit_summary_text(
            """
File Name: chb01_03.edf
Number of Seizures in File: 2
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds
Seizure 2 Start Time: 3100 seconds
Seizure 2 End Time: 3120 seconds
"""
        )

        self.assertEqual(intervals["chb01_03.edf"], [(2996.0, 3036.0), (3100.0, 3120.0)])


def _write_chb_fixture(base: Path) -> Path:
    root = base / "chbmit"
    records = []
    seizure_records = []
    for patient_idx in (1, 2):
        patient = f"chb{patient_idx:02d}"
        patient_dir = root / patient
        patient_dir.mkdir(parents=True)
        for rec_idx in (1, 2):
            record = f"{patient}/{patient}_{rec_idx:02d}.edf"
            records.append(record)
            if rec_idx == 1:
                seizure_records.append(record)
            _write_edf(patient_dir / f"{patient}_{rec_idx:02d}.edf", seed=patient_idx * 10 + rec_idx)
            (patient_dir / f"{patient}_{rec_idx:02d}.edf.seizures").write_text("fixture\n", encoding="utf-8")
        (patient_dir / f"{patient}-summary.txt").write_text(
            f"""Data Sampling Rate: 128 Hz

File Name: {patient}_01.edf
Number of Seizures in File: 1
Seizure Start Time: 1 seconds
Seizure End Time: 2 seconds

File Name: {patient}_02.edf
Number of Seizures in File: 0
""",
            encoding="utf-8",
        )
    (root / "RECORDS").write_text("\n".join(records) + "\n", encoding="utf-8")
    (root / "RECORDS-WITH-SEIZURES").write_text("\n".join(seizure_records) + "\n", encoding="utf-8")
    return root


def _write_edf(path: Path, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 4.0 * np.pi, 384)
    signals = []
    for channel in range(4):
        data = np.sin(t + channel * 0.4) + 0.05 * rng.normal(size=t.shape[0])
        signals.append(EdfSignal(data.astype(np.float64), sampling_frequency=128.0, label=f"EEG {channel}"))
    Edf(signals).write(path)


if __name__ == "__main__":
    unittest.main()
