import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurotwin.neurovisual import build_seed_dataset_registry, validate_local_split_records


def _manifest_record(
    *,
    record_id: str,
    subject_id: str,
    session_id: str,
    split_name: str,
    signal_path: str,
    event_annotations_path: str,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "subject_id": subject_id,
        "session_id": session_id,
        "dataset_name": "CHB-MIT Scalp EEG Database",
        "signal_path": signal_path,
        "sampling_rate": 256,
        "channel_names": ["FP1-F7", "F7-T7"],
        "event_annotations_path": event_annotations_path,
        "task_label": "retrospective_event_window_research",
        "license_or_access_confirmation": "user_confirmed_access_terms",
        "split_name": split_name,
    }


class NeurovisualSplitAuditTests(unittest.TestCase):
    def test_local_split_audit_accepts_disjoint_subject_heldout_records(self):
        registry = build_seed_dataset_registry(date_checked="2026-06-20")
        records = [
            _manifest_record(
                record_id="synthetic-record-train",
                subject_id="synthetic-subject-train",
                session_id="synthetic-session-train",
                split_name="train",
                signal_path="USER_PROVIDED_SIGNAL_PATH_TRAIN",
                event_annotations_path="USER_PROVIDED_EVENT_PATH_TRAIN",
            ),
            _manifest_record(
                record_id="synthetic-record-validation",
                subject_id="synthetic-subject-validation",
                session_id="synthetic-session-validation",
                split_name="validation",
                signal_path="USER_PROVIDED_SIGNAL_PATH_VALIDATION",
                event_annotations_path="USER_PROVIDED_EVENT_PATH_VALIDATION",
            ),
            _manifest_record(
                record_id="synthetic-record-test",
                subject_id="synthetic-subject-test",
                session_id="synthetic-session-test",
                split_name="test",
                signal_path="USER_PROVIDED_SIGNAL_PATH_TEST",
                event_annotations_path="USER_PROVIDED_EVENT_PATH_TEST",
            ),
        ]

        audit = validate_local_split_records(records, registry=registry)

        self.assertEqual(audit["schema"], "kahlus.nv1.local_split_audit.v1")
        self.assertTrue(audit["passed"], audit)
        self.assertTrue(audit["manifest_contract_passed"], audit)
        self.assertEqual(audit["failures"], [])
        self.assertEqual(audit["records_audited"], 3)
        self.assertEqual(audit["split_counts"], {"train": 1, "validation": 1, "test": 1})
        self.assertTrue(audit["execution"]["split_audit_executed"])
        self.assertFalse(audit["execution"]["adapters_implemented"])
        self.assertFalse(audit["execution"]["bulk_dataset_download"])
        self.assertFalse(audit["execution"]["a100_jobs_launched"])
        self.assertFalse(audit["execution"]["raw_file_existence_checked"])

    def test_local_split_audit_rejects_subject_session_record_and_path_overlap(self):
        registry = build_seed_dataset_registry(date_checked="2026-06-20")
        records = [
            _manifest_record(
                record_id="synthetic-record-leak",
                subject_id="synthetic-subject-leak",
                session_id="synthetic-session-leak",
                split_name="train",
                signal_path="USER_PROVIDED_SIGNAL_PATH_LEAK",
                event_annotations_path="USER_PROVIDED_EVENT_PATH_LEAK",
            ),
            _manifest_record(
                record_id="synthetic-record-leak",
                subject_id="synthetic-subject-leak",
                session_id="synthetic-session-leak",
                split_name="test",
                signal_path="USER_PROVIDED_SIGNAL_PATH_LEAK",
                event_annotations_path="USER_PROVIDED_EVENT_PATH_LEAK",
            ),
        ]

        audit = validate_local_split_records(records, registry=registry)

        self.assertFalse(audit["passed"], audit)
        self.assertIn("subject_overlap:synthetic-subject-leak:test,train", audit["failures"])
        self.assertIn("session_overlap:synthetic-session-leak:test,train", audit["failures"])
        self.assertIn("record_overlap:synthetic-record-leak:test,train", audit["failures"])
        self.assertIn("duplicate_signal_path:USER_PROVIDED_SIGNAL_PATH_LEAK:test,train", audit["failures"])
        self.assertIn("event_annotation_overlap:USER_PROVIDED_EVENT_PATH_LEAK:test,train", audit["failures"])

    def test_local_split_audit_rejects_invalid_manifest_rows_before_split_claim(self):
        registry = build_seed_dataset_registry(date_checked="2026-06-20")
        record = _manifest_record(
            record_id="synthetic-record-openneuro",
            subject_id="synthetic-subject-openneuro",
            session_id="synthetic-session-openneuro",
            split_name="dev",
            signal_path="USER_PROVIDED_SIGNAL_PATH_OPENNEURO",
            event_annotations_path="USER_PROVIDED_EVENT_PATH_OPENNEURO",
        )
        record["dataset_name"] = "OpenNeuro lead: EEG plus pupillometry plus PPG working-memory task"

        audit = validate_local_split_records([record], registry=registry)

        self.assertFalse(audit["passed"], audit)
        self.assertFalse(audit["manifest_contract_passed"], audit)
        self.assertIn(
            "dataset_not_confirmed:0:OpenNeuro lead: EEG plus pupillometry plus PPG working-memory task",
            audit["failures"],
        )
        self.assertIn("invalid_split_name:0:dev", audit["failures"])

    def test_local_split_audit_cli_passes_clean_split_and_fails_leaky_split(self):
        registry = build_seed_dataset_registry(date_checked="2026-06-20")
        clean_manifest = [
            _manifest_record(
                record_id="synthetic-record-cli-train",
                subject_id="synthetic-subject-cli-train",
                session_id="synthetic-session-cli-train",
                split_name="train",
                signal_path="USER_PROVIDED_SIGNAL_PATH_CLI_TRAIN",
                event_annotations_path="USER_PROVIDED_EVENT_PATH_CLI_TRAIN",
            ),
            _manifest_record(
                record_id="synthetic-record-cli-test",
                subject_id="synthetic-subject-cli-test",
                session_id="synthetic-session-cli-test",
                split_name="test",
                signal_path="USER_PROVIDED_SIGNAL_PATH_CLI_TEST",
                event_annotations_path="USER_PROVIDED_EVENT_PATH_CLI_TEST",
            ),
        ]
        leaky_manifest = [
            _manifest_record(
                record_id="synthetic-record-cli-leak",
                subject_id="synthetic-subject-cli-leak",
                session_id="synthetic-session-cli-leak",
                split_name="train",
                signal_path="USER_PROVIDED_SIGNAL_PATH_CLI_LEAK",
                event_annotations_path="USER_PROVIDED_EVENT_PATH_CLI_LEAK",
            ),
            _manifest_record(
                record_id="synthetic-record-cli-leak",
                subject_id="synthetic-subject-cli-leak",
                session_id="synthetic-session-cli-leak",
                split_name="validation",
                signal_path="USER_PROVIDED_SIGNAL_PATH_CLI_LEAK",
                event_annotations_path="USER_PROVIDED_EVENT_PATH_CLI_LEAK",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "registry.json"
            clean_path = root / "clean_split_manifest.json"
            leaky_path = root / "leaky_split_manifest.json"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            clean_path.write_text(json.dumps(clean_manifest), encoding="utf-8")
            leaky_path.write_text(json.dumps(leaky_manifest), encoding="utf-8")

            clean_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_neurovisual_local_split.py",
                    "--manifest",
                    str(clean_path),
                    "--registry",
                    str(registry_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )
            self.assertEqual(clean_result.returncode, 0, clean_result.stderr + clean_result.stdout)
            clean_payload = json.loads(clean_result.stdout)
            self.assertTrue(clean_payload["passed"], clean_payload)
            self.assertEqual(clean_payload["manifest_path"], str(clean_path))
            self.assertEqual(clean_payload["registry_path"], str(registry_path))
            self.assertEqual(clean_payload["records_audited"], 2)
            self.assertFalse(clean_payload["execution"]["raw_file_existence_checked"])

            leaky_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_neurovisual_local_split.py",
                    "--manifest",
                    str(leaky_path),
                    "--registry",
                    str(registry_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(leaky_result.returncode, 0, leaky_result.stdout)
            leaky_payload = json.loads(leaky_result.stdout)
            self.assertFalse(leaky_payload["passed"], leaky_payload)
            self.assertIn("subject_overlap:synthetic-subject-cli-leak:train,validation", leaky_payload["failures"])


if __name__ == "__main__":
    unittest.main()
