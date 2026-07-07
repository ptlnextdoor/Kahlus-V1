import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurotwin.stf import (
    CHB_MIT_DATASET_ID,
    audit_chb_mit_root,
    fetch_chb_mit_smoke_subset,
    stf_public_dataset_registry,
)


class STFPublicDataTests(unittest.TestCase):
    def test_registry_names_verified_chb_mit_source_without_raw_data(self):
        registry = stf_public_dataset_registry()

        self.assertIn(CHB_MIT_DATASET_ID, registry)
        self.assertIn("physionet.org/content/chbmit/1.0.0", registry[CHB_MIT_DATASET_ID]["source_url"])
        self.assertIn("raw EDF commit", registry[CHB_MIT_DATASET_ID]["blocked"])

    def test_chb_mit_root_audit_passes_minimal_local_fixture_outside_repo(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as repo:
            root = Path(tmp) / "chbmit"
            (root / "chb01").mkdir(parents=True)
            (root / "RECORDS").write_text("chb01/chb01_01.edf\n", encoding="utf-8")
            (root / "RECORDS-WITH-SEIZURES").write_text("chb01/chb01_01.edf\n", encoding="utf-8")
            (root / "chb01" / "chb01_01.edf.seizures").write_text("fixture\n", encoding="utf-8")

            audit = audit_chb_mit_root(root, repo_root=repo)

            self.assertTrue(audit.passed, audit.failure_reasons)
            self.assertEqual(audit.record_count, 1)
            self.assertEqual(audit.seizure_record_count, 1)
            self.assertEqual(audit.patients, ("chb01",))

    def test_chb_mit_root_audit_rejects_missing_annotations_and_repo_raw_root(self):
        with tempfile.TemporaryDirectory() as repo:
            root = Path(repo) / "raw_chbmit"
            (root / "chb01").mkdir(parents=True)
            (root / "RECORDS").write_text("chb01/chb01_01.edf\n", encoding="utf-8")
            (root / "RECORDS-WITH-SEIZURES").write_text("chb01/chb01_01.edf\n", encoding="utf-8")

            audit = audit_chb_mit_root(root, repo_root=repo)

            self.assertFalse(audit.passed)
            self.assertIn("public raw EEG root is inside the repository; keep raw data outside git", audit.failure_reasons)
            self.assertIn("missing .edf.seizures annotation files for seizure records", audit.failure_reasons)

    def test_public_data_audit_cli_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "chbmit"
            out = Path(tmp) / "out"
            (root / "chb01").mkdir(parents=True)
            (root / "RECORDS").write_text("chb01/chb01_01.edf\n", encoding="utf-8")
            (root / "RECORDS-WITH-SEIZURES").write_text("chb01/chb01_01.edf\n", encoding="utf-8")
            (root / "chb01" / "chb01_01.edf.seizures").write_text("fixture\n", encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_stf_public_data_audit.py",
                    "--dataset",
                    CHB_MIT_DATASET_ID,
                    "--data-root",
                    str(root),
                    "--out-dir",
                    str(out),
                ],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("public_data_audit_passed=True", proc.stdout)
            self.assertTrue((out / "chb_mit_root_audit.json").exists())
            self.assertIn("a100_jobs_launched: false", (out / "stf_public_data_report.md").read_text(encoding="utf-8"))

    def test_fetch_chb_mit_smoke_subset_materializes_auditable_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            remote = Path(tmp) / "remote"
            root = Path(tmp) / "chbmit"
            _write_fake_physionet_chbmit(remote)

            manifest = fetch_chb_mit_smoke_subset(root, base_url=remote.as_uri() + "/")
            audit = audit_chb_mit_root(root, repo_root=Path(tmp) / "repo")

            self.assertTrue(audit.passed, audit.failure_reasons)
            self.assertEqual(len(manifest["patients"]), 2)
            self.assertEqual(len(manifest["selected_records"]), 4)
            self.assertFalse(manifest["a100_jobs_launched"])
            self.assertTrue((root / "chb01" / "chb01-summary.txt").exists())
            self.assertTrue((root / "chb02" / "chb02_03.edf.seizures").exists())

    def test_fetch_chb_mit_smoke_subset_cli_writes_external_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            remote = Path(tmp) / "remote"
            root = Path(tmp) / "chbmit"
            _write_fake_physionet_chbmit(remote)

            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/fetch_chb_mit_smoke_subset.py",
                    "--dataset",
                    CHB_MIT_DATASET_ID,
                    "--out-root",
                    str(root),
                    "--base-url",
                    remote.as_uri() + "/",
                ],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("selected_records=4", proc.stdout)
            self.assertTrue((root / "kahlus_stf_smoke_subset_manifest.json").exists())


def _write_fake_physionet_chbmit(root: Path) -> None:
    records = [
        "chb01/chb01_03.edf",
        "chb01/chb01_01.edf",
        "chb02/chb02_03.edf",
        "chb02/chb02_01.edf",
    ]
    seizure_records = ["chb01/chb01_03.edf", "chb02/chb02_03.edf"]
    root.mkdir(parents=True)
    (root / "RECORDS").write_text("\n".join(records) + "\n", encoding="utf-8")
    (root / "RECORDS-WITH-SEIZURES").write_text(
        "\n".join(seizure_records) + "\n",
        encoding="utf-8",
    )
    for record in records:
        path = root / record
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake edf\n", encoding="utf-8")
    for record in seizure_records:
        (root / f"{record}.seizures").write_text("fake seizure sidecar\n", encoding="utf-8")
    for patient in ("chb01", "chb02"):
        (root / patient / f"{patient}-summary.txt").write_text(
            f"File Name: {patient}_03.edf\nSeizure Start Time: 1 seconds\nSeizure End Time: 2 seconds\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
