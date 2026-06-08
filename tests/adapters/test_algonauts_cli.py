import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurotwin.data.event_io import load_event_batches
from neurotwin.data.manifest_io import load_split_manifest
from neurotwin.data.prepared_tasks import build_prepared_window_tasks
from neurotwin.adapters.algonauts import (
    _ResponseRecord,
    _canonical_stimulus_id,
    _candidate_feature_files,
    _feature_candidate_sort_key,
    _load_matching_stimulus_features,
    _split_assignment,
)


class AlgonautsCliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        return subprocess.run(
            [sys.executable, "-m", "neurotwin.cli", *args],
            text=True,
            capture_output=True,
            env=env,
        )

    def test_algonauts_prepare_writes_real_stimulus_manifests_and_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            out_dir = Path(tmp) / "prepared"
            _write_tiny_algonauts_fixture(root)

            result = self.run_cli(
                "data",
                "prepare",
                "--dataset",
                "algonauts2025",
                "--root",
                str(root),
                "--out-dir",
                str(out_dir),
                "--split",
                "official",
                "--window-length",
                "16",
                "--stride",
                "16",
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("dataset=algonauts2025", result.stdout)
            self.assertIn("eval_audit_passed=True", result.stdout)
            for name in (
                "event_manifest.json",
                "split_manifest.json",
                "data_manifest.json",
                "feature_manifest.json",
                "stimulus_manifest.json",
                "leakage_report.json",
                "eval_audit.json",
            ):
                self.assertTrue((out_dir / name).exists(), name)

            events = load_event_batches(out_dir / "event_manifest.json")
            self.assertEqual(sorted({event.subject_id for event in events}), ["sub-01", "sub-02", "sub-03", "sub-05"])
            self.assertTrue(all(event.modality == "fmri" for event in events))
            self.assertTrue(all(event.signal.shape[1] == 1000 for event in events))
            self.assertTrue(all(event.stimulus_embedding is not None for event in events))
            self.assertTrue(all(event.stimulus_embedding.shape[0] == event.signal.shape[0] for event in events))

            split = load_split_manifest(out_dir / "split_manifest.json")
            tasks, skipped = build_prepared_window_tasks(events, split, window_length=16, stride=16, seed=0)
            task_ids = {task.task_id for task in tasks}
            self.assertIn("stimulus_to_fmri_response", task_ids)
            self.assertNotIn(
                "stimulus_to_fmri_response",
                {row["task_id"] for row in skipped},
            )
            stimulus_task = next(task for task in tasks if task.task_id == "stimulus_to_fmri_response")
            evidence = stimulus_task.metadata["stimulus_evidence"]
            self.assertEqual(evidence["status"], "real_stimulus_features")
            self.assertTrue(evidence["claim_eligible"])

            feature_manifest = json.loads((out_dir / "feature_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(feature_manifest["claim_eligible"])
            self.assertTrue(feature_manifest["hash_verified"])

    def test_algonauts_stimulus_hash_mismatch_loses_claim_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            out_dir = Path(tmp) / "prepared"
            files = _write_tiny_algonauts_fixture(root)
            result = self.run_cli(
                "data",
                "prepare",
                "--dataset",
                "algonauts2025",
                "--root",
                str(root),
                "--out-dir",
                str(out_dir),
                "--split",
                "official",
                "--window-length",
                "16",
                "--stride",
                "16",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

            files[0].write_bytes(b"tampered stimulus feature bytes\n")
            events = load_event_batches(out_dir / "event_manifest.json")
            split = load_split_manifest(out_dir / "split_manifest.json")
            tasks, _ = build_prepared_window_tasks(events, split, window_length=16, stride=16, seed=0)
            stimulus_task = next(task for task in tasks if task.task_id == "stimulus_to_fmri_response")
            evidence = stimulus_task.metadata["stimulus_evidence"]

            self.assertFalse(evidence["claim_eligible"])
            self.assertIn("stimulus_feature_hash is not verified", evidence["failure_reasons"])

    def test_hdf5_style_task_key_matches_canonical_feature_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "features-v2" / "reduced"
            feature_dir.mkdir(parents=True)
            feature_path = feature_dir / "friends_s01e02a_features.npy"
            np.save(feature_path, np.ones((482, 8), dtype=np.float32))

            key = "ses-001_task-s01e02a"
            self.assertEqual(_canonical_stimulus_id(key), "friends_s01e02a")
            self.assertEqual(_split_assignment("ses-006_task-s06e20b", root / "fmri" / "sub-01" / "func"), "val")
            self.assertEqual(_split_assignment("ses-001_task-bourne01", root / "fmri" / "sub-01" / "func"), "test")
            self.assertEqual(list(_candidate_feature_files(root, key)), [feature_path])

    def test_reduced_feature_candidate_wins_over_nonfinite_raw_feature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reduced_dir = root / "features-v2" / "reduced"
            raw_dir = root / "features-v2" / "raw_official" / "language"
            reduced_dir.mkdir(parents=True)
            raw_dir.mkdir(parents=True)
            reduced_path = reduced_dir / "friends_s01e03a_features.npy"
            raw_path = raw_dir / "friends_s01e03a_features_language.npy"
            np.save(reduced_path, np.ones((472, 8), dtype=np.float32))
            raw = np.ones((472, 8), dtype=np.float32)
            raw[0, 0] = np.nan
            np.save(raw_path, raw)

            source = _ResponseRecord(
                path=root / "fmri" / "sub-01" / "func" / "sub-01_task-friends_bold.h5",
                key="ses-001_task-s01e03a",
                signal=np.ones((472, 1000), dtype=np.float32),
                stimulus_id="ses-001_task-s01e03a",
                session_id="friends_s01e03a",
            )

            candidates = sorted(_candidate_feature_files(root, source.stimulus_id), key=_feature_candidate_sort_key)
            self.assertEqual(candidates[0], reduced_path)
            stimulus = _load_matching_stimulus_features(root, source)
            self.assertEqual(stimulus.path, reduced_path)
            self.assertTrue(np.isfinite(stimulus.array).all())


def _write_tiny_algonauts_fixture(root: Path) -> list[Path]:
    rng = np.random.default_rng(17)
    files: list[Path] = []
    split_to_stimulus = {
        "train": "friends_s01e01a",
        "val": "friends_s06e01a",
        "test": "movie10_hidden_figures_01",
    }
    for subject in ("sub-01", "sub-02", "sub-03", "sub-05"):
        subject_dir = root / "fmri" / subject / "func"
        subject_dir.mkdir(parents=True, exist_ok=True)
        for split, stimulus_id in split_to_stimulus.items():
            signal = rng.normal(size=(48, 1000)).astype(np.float32)
            stimulus = rng.normal(size=(48, 12)).astype(np.float32)
            path = subject_dir / f"{subject}_{split}_{stimulus_id}_fmri.npz"
            np.savez_compressed(path, signal=signal, stimulus_embedding=stimulus)
            files.append(path)
    return files


if __name__ == "__main__":
    unittest.main()
