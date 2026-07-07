from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import edfio
except ImportError:  # pragma: no cover - exercised only when optional EDF dependency is missing.
    edfio = None


@unittest.skipIf(edfio is None, "edfio is required for EDF loader smoke tests")
class MultiDatasetA100Tests(unittest.TestCase):
    def test_prepare_multidataset_evidence_from_tiny_edf_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "roots"
            out = Path(tmp) / "prepared"
            _write_fixture_roots(root)

            from neurotwin.adapters.multidataset import prepare_multidataset_a100_evidence

            result = prepare_multidataset_a100_evidence(
                root,
                out,
                max_records_per_dataset=3,
                max_samples_per_record=256,
                max_channels=4,
                window_length=32,
                stride=32,
            )

            self.assertTrue(result["gate"]["passed"], result["gate"])
            self.assertEqual(
                set(result["gate"]["supported_datasets"]),
                {"sleep_edf_expanded", "chb_mit_physionet", "eegmmi_physionet", "siena_scalp_eeg"},
            )
            self.assertTrue((out / "event_manifest.json").exists())
            self.assertTrue((out / "split_manifest.json").exists())
            self.assertTrue((out / "chbmit_to_siena_transfer_split_manifest.json").exists())
            with (out / "event_manifest.json").open(encoding="utf-8") as handle:
                event_manifest = json.load(handle)
            self.assertGreater(event_manifest["event_count"], 0)
            self.assertEqual(
                event_manifest["metadata"]["claim_scope"],
                "multidataset_eeg_forecasting_completion_benchmark_ready",
            )
            self.assertNotIn("diagnosis", json.dumps(event_manifest).lower())

    def test_cli_prepare_multidataset_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "roots"
            out = Path(tmp) / "prepared"
            _write_fixture_roots(root)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "data",
                    "prepare",
                    "--dataset",
                    "multidataset_a100",
                    "--split",
                    "subject",
                    "--root",
                    str(root),
                    "--out-dir",
                    str(out),
                    "--max-trials",
                    "2",
                    "--window-length",
                    "32",
                    "--stride",
                    "32",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("gate_passed=True", result.stdout)
            self.assertTrue((out / "multidataset_evidence_bundle_manifest.json").exists())
            suite = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "eval",
                    "suite",
                    "--suite",
                    "neural_translation_v1",
                    "--event-manifest",
                    str(out / "event_manifest.json"),
                    "--split-manifest",
                    str(out / "split_manifest.json"),
                    "--window-length",
                    "32",
                    "--stride",
                    "32",
                    "--max-windows-per-split",
                    "4",
                    "--baseline-models",
                    "patient_session_nuisance",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )
            self.assertEqual(suite.returncode, 0, suite.stderr + suite.stdout)
            self.assertIn("patient_session_nuisance", suite.stdout)

    def test_prepare_script_writes_materialized_training_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "roots"
            out = Path(tmp) / "prepared"
            _write_fixture_roots(root)

            result = subprocess.run(
                ["bash", "scripts/prepare_multidataset_a100_evidence.sh", str(out)],
                cwd=Path(__file__).resolve().parents[2],
                env={
                    **os.environ,
                    "PYTHONPATH": "src",
                    "MULTIDATASET_ROOT": str(root),
                    "MAX_RECORDS_PER_DATASET": "2",
                    "MAX_WINDOWS_PER_SPLIT": "4",
                    "WINDOW_LENGTH": "32",
                    "STRIDE": "32",
                },
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            materialized = out / "configs" / "kahlus_multidataset_a100_evidence.materialized.yaml"
            self.assertTrue(materialized.exists())
            dry_run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "train",
                    "--dry-run",
                    "--config",
                    str(materialized),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )
            self.assertEqual(dry_run.returncode, 0, dry_run.stderr + dry_run.stdout)
            self.assertIn("estimated_7xa100_ddp_per_gpu_mb", dry_run.stdout)


def _write_fixture_roots(root: Path) -> None:
    _write_sleep_fixture(root / "sleep-edfx")
    _write_chb_fixture(root / "chbmit")
    _write_eegmmi_fixture(root / "eegmmi")
    _write_siena_fixture(root / "siena")


def _write_sleep_fixture(root: Path) -> None:
    for idx in range(3):
        stem = f"SC40{idx:02d}1E0"
        _write_edf(
            root / f"{stem}-PSG.edf",
            labels=("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"),
            sfreq=64,
            n_samples=320,
        )
        _write_edf(
            root / f"{stem[:6]}EC-Hypnogram.edf",
            labels=("Hypnogram",),
            sfreq=1,
            n_samples=16,
            annotations=[(0.0, 30.0, "Sleep stage W"), (30.0, 30.0, "Sleep stage 1")],
        )


def _write_chb_fixture(root: Path) -> None:
    for patient in ("chb01", "chb02", "chb21"):
        folder = root / patient
        _write_edf(folder / f"{patient}_01.edf", labels=("FP1-F7", "F7-T7", "T7-P7", "P7-O1"), sfreq=128, n_samples=384)
        (folder / f"{patient}-summary.txt").write_text(
            f"File Name: {patient}_01.edf\n"
            "Number of Seizures in File: 1\n"
            "Seizure Start Time: 10 seconds\n"
            "Seizure End Time: 20 seconds\n",
            encoding="utf-8",
        )
    (root / "RECORDS").write_text(
        "\n".join(f"{patient}/{patient}_01.edf" for patient in ("chb01", "chb02", "chb21")) + "\n",
        encoding="utf-8",
    )


def _write_eegmmi_fixture(root: Path) -> None:
    for subject in ("S001", "S002", "S003"):
        _write_edf(
            root / subject / f"{subject}R03.edf",
            labels=("Fc5.", "Fc3.", "Fc1.", "Fcz."),
            sfreq=160,
            n_samples=320,
            annotations=[(0.0, 1.0, "T0"), (1.0, 1.0, "T1")],
        )


def _write_siena_fixture(root: Path) -> None:
    for patient in ("PN00", "PN01", "PN02"):
        _write_edf(
            root / patient / f"{patient}-1.edf",
            labels=("EEG Fp1", "EEG Fp2", "EEG C3", "EKG"),
            sfreq=128,
            n_samples=384,
        )
    (root / "Seizures-list.txt").write_text(
        "File Name: PN00-1.edf\nSeizure Start Time: 10\nSeizure End Time: 20\n",
        encoding="utf-8",
    )


def _write_edf(
    path: Path,
    *,
    labels: tuple[str, ...],
    sfreq: float,
    n_samples: int,
    annotations: list[tuple[float, float, str]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0.0, 4.0 * np.pi, n_samples, dtype=np.float64)
    signals = []
    for idx, label in enumerate(labels):
        data = np.sin(x + idx).astype(np.float64) * 50.0
        signals.append(edfio.EdfSignal(data, sfreq, label=label, physical_range=(-100.0, 100.0)))
    edf_annotations = [edfio.EdfAnnotation(start, duration, text) for start, duration, text in annotations or []]
    edfio.Edf(signals, annotations=edf_annotations).write(path)
