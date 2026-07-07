import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurotwin.neurovisual import (
    LOCAL_MANIFEST_REQUIRED_FIELDS,
    build_local_manifest_schema,
    build_seed_dataset_registry,
    validate_local_manifest_records,
)


class NeurovisualManifestTests(unittest.TestCase):
    def test_local_manifest_schema_lists_allowed_and_blocked_dataset_names(self):
        registry = build_seed_dataset_registry(date_checked="2026-06-20")
        schema = build_local_manifest_schema(registry)

        self.assertEqual(schema["schema"], "kahlus.nv1.local_manifest_schema.v1")
        self.assertEqual(schema["required_fields"], list(LOCAL_MANIFEST_REQUIRED_FIELDS))
        self.assertEqual(schema["execution"]["adapters_implemented"], False)
        self.assertIn("CHB-MIT Scalp EEG Database", schema["allowed_confirmed_dataset_names"])
        self.assertIn(
            "OpenNeuro lead: EEG plus pupillometry plus PPG working-memory task",
            schema["blocked_unverified_dataset_names"],
        )
        self.assertIn("Private or unprovided personal medical notes", schema["blocked_rejected_dataset_names"])
        self.assertEqual(schema["field_types"]["channel_names"], "list[str]")
        self.assertEqual(schema["field_types"]["sampling_rate"], "positive_number")

    def test_local_manifest_accepts_confirmed_dataset_records_without_adapter_execution(self):
        registry = build_seed_dataset_registry(date_checked="2026-06-20")
        record = {
            "record_id": "synthetic-record-001",
            "subject_id": "synthetic-subject-001",
            "session_id": "synthetic-session-001",
            "dataset_name": "CHB-MIT Scalp EEG Database",
            "signal_path": "USER_PROVIDED_SIGNAL_PATH",
            "sampling_rate": 256,
            "channel_names": ["FP1-F7", "F7-T7"],
            "event_annotations_path": "USER_PROVIDED_EVENT_ANNOTATIONS_PATH",
            "task_label": "retrospective_event_window_research",
            "license_or_access_confirmation": "user_confirmed_access_terms",
        }

        audit = validate_local_manifest_records([record], registry=registry)

        self.assertEqual(audit["schema"], "kahlus.nv1.local_manifest_contract_audit.v1")
        self.assertTrue(audit["passed"], audit)
        self.assertEqual(audit["records_validated"], 1)
        self.assertEqual(audit["failures"], [])
        self.assertEqual(audit["required_fields"], list(LOCAL_MANIFEST_REQUIRED_FIELDS))
        self.assertFalse(audit["execution"]["adapters_implemented"])
        self.assertFalse(audit["execution"]["bulk_dataset_download"])
        self.assertFalse(audit["execution"]["a100_jobs_launched"])

    def test_local_manifest_rejects_unverified_rejected_and_incomplete_records(self):
        registry = build_seed_dataset_registry(date_checked="2026-06-20")
        rows = [
            {
                "record_id": "synthetic-record-002",
                "subject_id": "synthetic-subject-002",
                "session_id": "synthetic-session-002",
                "dataset_name": "OpenNeuro lead: EEG plus pupillometry plus PPG working-memory task",
                "signal_path": "USER_PROVIDED_SIGNAL_PATH",
                "sampling_rate": 256,
                "channel_names": ["Cz"],
                "event_annotations_path": "USER_PROVIDED_EVENT_ANNOTATIONS_PATH",
                "task_label": "candidate_task",
                "license_or_access_confirmation": "unverified",
            },
            {
                "record_id": "synthetic-record-003",
                "subject_id": "synthetic-subject-003",
                "session_id": "synthetic-session-003",
                "dataset_name": "Private or unprovided personal medical notes",
                "signal_path": "USER_PROVIDED_SIGNAL_PATH",
                "sampling_rate": 256,
                "channel_names": ["Cz"],
                "event_annotations_path": "USER_PROVIDED_EVENT_ANNOTATIONS_PATH",
                "task_label": "private_notes",
                "license_or_access_confirmation": "not_allowed",
            },
            {
                "record_id": "synthetic-record-004",
                "dataset_name": "CHB-MIT Scalp EEG Database",
                "signal_path": "",
                "sampling_rate": 0,
                "channel_names": [],
            },
        ]

        audit = validate_local_manifest_records(rows, registry=registry)

        self.assertFalse(audit["passed"], audit)
        self.assertIn("dataset_not_confirmed:0:OpenNeuro lead: EEG plus pupillometry plus PPG working-memory task", audit["failures"])
        self.assertIn("dataset_rejected:1:Private or unprovided personal medical notes", audit["failures"])
        self.assertIn("missing_field:2:subject_id", audit["failures"])
        self.assertIn("invalid_sampling_rate:2", audit["failures"])
        self.assertIn("empty_channel_names:2", audit["failures"])

    def test_local_manifest_audit_cli_passes_valid_file_and_fails_invalid_file(self):
        registry = build_seed_dataset_registry(date_checked="2026-06-20")
        valid_manifest = [
            {
                "record_id": "synthetic-record-005",
                "subject_id": "synthetic-subject-005",
                "session_id": "synthetic-session-005",
                "dataset_name": "CHB-MIT Scalp EEG Database",
                "signal_path": "USER_PROVIDED_SIGNAL_PATH",
                "sampling_rate": 256,
                "channel_names": ["FP1-F7"],
                "event_annotations_path": "USER_PROVIDED_EVENT_ANNOTATIONS_PATH",
                "task_label": "retrospective_event_window_research",
                "license_or_access_confirmation": "user_confirmed_access_terms",
            }
        ]
        invalid_manifest = [
            {
                "record_id": "synthetic-record-006",
                "subject_id": "synthetic-subject-006",
                "session_id": "synthetic-session-006",
                "dataset_name": "OpenNeuro lead: EEG plus pupillometry plus PPG working-memory task",
                "signal_path": "USER_PROVIDED_SIGNAL_PATH",
                "sampling_rate": 256,
                "channel_names": ["Cz"],
                "event_annotations_path": "USER_PROVIDED_EVENT_ANNOTATIONS_PATH",
                "task_label": "candidate_task",
                "license_or_access_confirmation": "unverified",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "registry.json"
            valid_path = root / "valid_manifest.json"
            invalid_path = root / "invalid_manifest.json"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            valid_path.write_text(json.dumps(valid_manifest), encoding="utf-8")
            invalid_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")

            valid_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_neurovisual_local_manifest.py",
                    "--manifest",
                    str(valid_path),
                    "--registry",
                    str(registry_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )
            self.assertEqual(valid_result.returncode, 0, valid_result.stderr + valid_result.stdout)
            valid_payload = json.loads(valid_result.stdout)
            self.assertTrue(valid_payload["passed"], valid_payload)
            self.assertEqual(valid_payload["manifest_path"], str(valid_path))
            self.assertEqual(valid_payload["registry_path"], str(registry_path))
            self.assertEqual(valid_payload["records_validated"], 1)

            invalid_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_neurovisual_local_manifest.py",
                    "--manifest",
                    str(invalid_path),
                    "--registry",
                    str(registry_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(invalid_result.returncode, 0, invalid_result.stdout)
            invalid_payload = json.loads(invalid_result.stdout)
            self.assertFalse(invalid_payload["passed"], invalid_payload)
            self.assertIn(
                "dataset_not_confirmed:0:OpenNeuro lead: EEG plus pupillometry plus PPG working-memory task",
                invalid_payload["failures"],
            )


if __name__ == "__main__":
    unittest.main()
