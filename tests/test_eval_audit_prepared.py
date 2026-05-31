import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurotwin.adapters.synthetic import make_synthetic_event_batches, make_synthetic_recordings
from neurotwin.data.event_io import save_event_batches
from neurotwin.data.manifest_io import save_split_manifest
from neurotwin.data.split_manifest import build_split_manifest
from neurotwin.eval.audit import audit_prepared_eval_inputs


class PreparedEvalAuditTests(unittest.TestCase):
    def _write_prepared(
        self,
        root: Path,
        n_subjects: int = 6,
        modalities: tuple[str, ...] = ("eeg", "fmri"),
        n_time: int = 64,
    ) -> tuple[Path, Path]:
        records = make_synthetic_recordings(n_subjects=n_subjects, sessions_per_subject=1, modalities=modalities)
        batches = make_synthetic_event_batches(
            n_subjects=n_subjects,
            sessions_per_subject=1,
            modalities=modalities,
            n_time=n_time,
        )
        split = build_split_manifest(records, policy="subject", seed=0)
        split_path = save_split_manifest(split, root / "split_manifest.json")
        event_path = save_event_batches(batches, root)
        return event_path, split_path

    def test_prepared_eval_audit_passes_clean_manifest_and_cli_writes_artifact(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event_path, split_path = self._write_prepared(root)
            out_dir = root / "audit"

            report = audit_prepared_eval_inputs(event_path, split_path, out_dir=out_dir)
            self.assertTrue(report.passed)
            self.assertTrue((out_dir / "eval_audit.json").exists())

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "eval",
                    "audit",
                    "--suite",
                    "neural_translation_v1",
                    "--event-manifest",
                    str(event_path),
                    "--split-manifest",
                    str(split_path),
                    "--out-dir",
                    str(out_dir),
                ],
                check=True,
                text=True,
                capture_output=True,
                env=env,
            )

            self.assertIn("eval_audit_prepared=True", result.stdout)
            self.assertIn("eval_audit_passed=True", result.stdout)
            audit_payload = json.loads((out_dir / "eval_audit.json").read_text(encoding="utf-8"))
            self.assertIn("window_counts_by_split", audit_payload)

    def test_moabb_sized_windows_pass_required_window_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event_path, split_path = self._write_prepared(root, modalities=("eeg",), n_time=256)

            report = audit_prepared_eval_inputs(
                event_path,
                split_path,
                window_length=128,
                stride=128,
                require_windows=True,
            )

            self.assertTrue(report.passed)
            self.assertGreater(report.window_count, 0)
            for split_name in ("train", "val", "test"):
                self.assertGreater(report.window_counts_by_split[split_name], 0)

    def test_zero_window_benchmark_fails_required_window_gate(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event_path, split_path = self._write_prepared(root, modalities=("eeg",), n_time=256)

            report = audit_prepared_eval_inputs(
                event_path,
                split_path,
                window_length=1024,
                stride=512,
                require_windows=True,
                out_dir=root / "audit",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "eval",
                    "audit",
                    "--suite",
                    "neural_translation_v1",
                    "--event-manifest",
                    str(event_path),
                    "--split-manifest",
                    str(split_path),
                    "--window-length",
                    "1024",
                    "--stride",
                    "512",
                    "--require-windows",
                ],
                text=True,
                capture_output=True,
                env=env,
            )

            self.assertFalse(report.passed)
            self.assertEqual(report.window_count, 0)
            self.assertTrue(any("zero windows" in violation for violation in report.violations))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("prepared benchmark produced zero windows", result.stdout)
            audit_payload = json.loads((root / "audit" / "eval_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit_payload["window_counts_by_split"], {"train": 0, "val": 0, "test": 0})

    def test_prepared_eval_audit_fails_on_corrupted_event_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event_path, split_path = self._write_prepared(root)
            payload = json.loads(event_path.read_text(encoding="utf-8"))
            first_event = root / payload["events"][0]["path"]
            with first_event.open("ab") as handle:
                handle.write(b"corruption")

            report = audit_prepared_eval_inputs(event_path, split_path)

            self.assertFalse(report.passed)
            self.assertTrue(any("integrity" in violation for violation in report.violations))

    def test_prepared_eval_audit_catches_window_overlap_across_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=1, modalities=("eeg",))
            batches = make_synthetic_event_batches(n_subjects=6, sessions_per_subject=1, modalities=("eeg",))
            split = build_split_manifest(records, policy="subject", seed=0)
            train_id = split.train[0].record_id
            test_ids = {record.record_id for record in split.test}
            for batch in batches:
                if batch.metadata["record_id"] in test_ids:
                    batch.metadata["source_record_id"] = train_id
                    break
            split_path = save_split_manifest(split, root / "split_manifest.json")
            event_path = save_event_batches(batches, root)

            report = audit_prepared_eval_inputs(event_path, split_path, window_length=8, stride=8)

            self.assertFalse(report.passed)
            self.assertTrue(any("prepared window leakage" in violation for violation in report.violations))

    def test_prepared_eval_audit_catches_metadata_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=1, modalities=("eeg",))
            batches = make_synthetic_event_batches(n_subjects=6, sessions_per_subject=1, modalities=("eeg",))
            split = build_split_manifest(records, policy="subject", seed=0)
            split_by_record = {}
            for split_name in ("train", "val", "test"):
                for record in getattr(split, split_name):
                    split_by_record[record.record_id] = split_name
            touched = set()
            for batch in batches:
                split_name = split_by_record.get(batch.metadata["record_id"])
                if split_name == "train" and "train" not in touched:
                    batch.metadata["source_hash"] = "same-raw-hash"
                    batch.metadata["preprocessing_hash"] = "same-prep-hash"
                    batch.metadata["stimulus_segment_id"] = "clip-001:0:10"
                    touched.add("train")
                elif split_name == "val" and "val" not in touched:
                    batch.metadata["target_label"] = "left"
                    touched.add("val")
                elif split_name == "test" and "test" not in touched:
                    batch.metadata["source_hash"] = "same-raw-hash"
                    batch.metadata["preprocessing_hash"] = "same-prep-hash"
                    batch.metadata["stimulus_segment_id"] = "clip-001:0:10"
                    batch.metadata["hidden_subject_id"] = batch.subject_id
                    touched.add("test")
            split_path = save_split_manifest(split, root / "split_manifest.json")
            event_path = save_event_batches(batches, root)

            report = audit_prepared_eval_inputs(event_path, split_path, window_length=8, stride=8)

            self.assertFalse(report.passed)
            self.assertTrue(any("duplicate source_hash across splits" in violation for violation in report.violations))
            self.assertTrue(any("duplicate preprocessing_hash across splits" in violation for violation in report.violations))
            self.assertTrue(any("stimulus segment leakage across splits" in violation for violation in report.violations))
            self.assertTrue(any("forbidden event metadata field 'target_label'" in violation for violation in report.violations))
            self.assertTrue(any("hidden subject metadata field 'hidden_subject_id'" in violation for violation in report.violations))


if __name__ == "__main__":
    unittest.main()
