from __future__ import annotations

from typing import Any


LOCAL_MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "record_id",
    "subject_id",
    "session_id",
    "dataset_name",
    "signal_path",
    "sampling_rate",
    "channel_names",
    "event_annotations_path",
    "task_label",
    "license_or_access_confirmation",
)


def build_local_manifest_schema(registry: dict[str, Any]) -> dict[str, Any]:
    entries = [entry for entry in registry.get("entries", ()) if isinstance(entry, dict)]
    confirmed_names = sorted(
        str(entry.get("dataset_name")) for entry in entries if entry.get("verification_status") == "confirmed"
    )
    unverified_names = sorted(
        str(entry.get("dataset_name")) for entry in entries if entry.get("verification_status") == "unverified"
    )
    rejected_names = sorted(
        str(entry.get("dataset_name")) for entry in entries if entry.get("verification_status") == "rejected"
    )
    return {
        "schema": "kahlus.nv1.local_manifest_schema.v1",
        "scope": "schema-only local manifest contract; no file existence check and no adapter execution",
        "required_fields": list(LOCAL_MANIFEST_REQUIRED_FIELDS),
        "field_types": {
            "record_id": "non_empty_string",
            "subject_id": "non_empty_string",
            "session_id": "non_empty_string",
            "dataset_name": "confirmed_registry_dataset_name",
            "signal_path": "non_empty_user_provided_path_string",
            "sampling_rate": "positive_number",
            "channel_names": "list[str]",
            "event_annotations_path": "non_empty_user_provided_path_string",
            "task_label": "non_empty_string",
            "license_or_access_confirmation": "non_empty_string",
        },
        "validation_rules": [
            "dataset_name must match a confirmed registry entry",
            "unverified registry leads are rejected until metadata is confirmed",
            "rejected private-data targets are always rejected",
            "sampling_rate must be positive",
            "channel_names must be a non-empty list of non-empty strings",
            "raw file existence is not checked in NV-1 without user-provided local paths",
        ],
        "allowed_confirmed_dataset_names": confirmed_names,
        "blocked_unverified_dataset_names": unverified_names,
        "blocked_rejected_dataset_names": rejected_names,
        "execution": {
            "adapters_implemented": False,
            "bulk_dataset_download": False,
            "a100_jobs_launched": False,
            "cluster_jobs_launched": False,
            "raw_file_existence_checked": False,
            "metadata_only": True,
        },
    }


def validate_local_manifest_records(
    records: list[dict[str, Any]],
    *,
    registry: dict[str, Any],
) -> dict[str, Any]:
    entries_by_name = {
        str(entry.get("dataset_name")): entry for entry in registry.get("entries", ()) if isinstance(entry, dict)
    }
    failures: list[str] = []
    dataset_statuses: dict[str, str] = {}

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            failures.append(f"invalid_record:{index}")
            continue
        for field in LOCAL_MANIFEST_REQUIRED_FIELDS:
            if field not in record:
                failures.append(f"missing_field:{index}:{field}")
        dataset_name = str(record.get("dataset_name", ""))
        registry_entry = entries_by_name.get(dataset_name)
        if registry_entry is None:
            failures.append(f"unknown_dataset:{index}:{dataset_name}")
        else:
            status = str(registry_entry.get("verification_status"))
            dataset_statuses[dataset_name] = status
            if status == "rejected":
                failures.append(f"dataset_rejected:{index}:{dataset_name}")
            elif status != "confirmed":
                failures.append(f"dataset_not_confirmed:{index}:{dataset_name}")
        _validate_record_values(index, record, failures)

    return {
        "schema": "kahlus.nv1.local_manifest_contract_audit.v1",
        "scope": "local manifest contract validation only; no adapter execution",
        "passed": not failures,
        "failures": failures,
        "records_validated": len(records),
        "required_fields": list(LOCAL_MANIFEST_REQUIRED_FIELDS),
        "dataset_statuses": dataset_statuses,
        "execution": {
            "adapters_implemented": False,
            "bulk_dataset_download": False,
            "a100_jobs_launched": False,
            "cluster_jobs_launched": False,
            "metadata_only": True,
        },
    }


def _validate_record_values(index: int, record: dict[str, Any], failures: list[str]) -> None:
    for field in (
        "record_id",
        "subject_id",
        "session_id",
        "dataset_name",
        "signal_path",
        "event_annotations_path",
        "task_label",
        "license_or_access_confirmation",
    ):
        if field in record and not str(record.get(field, "")).strip():
            failures.append(f"empty_field:{index}:{field}")
    if "sampling_rate" in record:
        try:
            sampling_rate = float(record["sampling_rate"])
        except (TypeError, ValueError):
            failures.append(f"invalid_sampling_rate:{index}")
        else:
            if sampling_rate <= 0:
                failures.append(f"invalid_sampling_rate:{index}")
    if "channel_names" in record:
        channel_names = record["channel_names"]
        if not isinstance(channel_names, list) or not channel_names or not all(str(name).strip() for name in channel_names):
            failures.append(f"empty_channel_names:{index}")
