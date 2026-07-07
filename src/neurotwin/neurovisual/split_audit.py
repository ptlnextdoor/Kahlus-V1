from __future__ import annotations

from typing import Any

from neurotwin.neurovisual.manifest import LOCAL_MANIFEST_REQUIRED_FIELDS, validate_local_manifest_records


ALLOWED_SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")
SPLIT_AUDIT_REQUIRED_FIELDS: tuple[str, ...] = (*LOCAL_MANIFEST_REQUIRED_FIELDS, "split_name")
LEAKAGE_CHECKS: tuple[str, ...] = (
    "subject_overlap",
    "session_overlap",
    "record_overlap",
    "duplicate_signal_path",
    "event_annotation_overlap",
)


def validate_local_split_records(
    records: list[dict[str, Any]],
    *,
    registry: dict[str, Any],
) -> dict[str, Any]:
    manifest_audit = validate_local_manifest_records(records, registry=registry)
    failures = list(manifest_audit["failures"])
    split_counts = {split_name: 0 for split_name in ALLOWED_SPLIT_NAMES}
    value_splits: dict[str, dict[str, set[str]]] = {check: {} for check in LEAKAGE_CHECKS}

    if not records:
        failures.append("no_records")

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        if "split_name" not in record:
            failures.append(f"missing_field:{index}:split_name")
            continue
        split_name = str(record.get("split_name", "")).strip()
        if split_name not in ALLOWED_SPLIT_NAMES:
            failures.append(f"invalid_split_name:{index}:{split_name}")
            continue

        split_counts[split_name] += 1
        _track_value_split(value_splits["subject_overlap"], record.get("subject_id"), split_name)
        _track_value_split(value_splits["session_overlap"], record.get("session_id"), split_name)
        _track_value_split(value_splits["record_overlap"], record.get("record_id"), split_name)
        _track_value_split(value_splits["duplicate_signal_path"], record.get("signal_path"), split_name)
        _track_value_split(value_splits["event_annotation_overlap"], record.get("event_annotations_path"), split_name)

    for check_name in LEAKAGE_CHECKS:
        for value, splits in sorted(value_splits[check_name].items()):
            if len(splits) > 1:
                failures.append(f"{check_name}:{value}:{','.join(sorted(splits))}")

    return {
        "schema": "kahlus.nv1.local_split_audit.v1",
        "scope": "local split leakage audit over explicit manifest records; no adapter execution",
        "passed": not failures,
        "failures": failures,
        "records_audited": len(records),
        "required_fields": list(SPLIT_AUDIT_REQUIRED_FIELDS),
        "allowed_split_names": list(ALLOWED_SPLIT_NAMES),
        "split_counts": split_counts,
        "leakage_checks": list(LEAKAGE_CHECKS),
        "manifest_contract_passed": bool(manifest_audit["passed"]),
        "manifest_audit_schema": manifest_audit["schema"],
        "execution": {
            "split_audit_executed": True,
            "adapters_implemented": False,
            "baselines_run": False,
            "models_run": False,
            "bulk_dataset_download": False,
            "a100_jobs_launched": False,
            "cluster_jobs_launched": False,
            "raw_file_existence_checked": False,
        },
    }


def _track_value_split(value_splits: dict[str, set[str]], raw_value: Any, split_name: str) -> None:
    value = str(raw_value or "").strip()
    if value:
        value_splits.setdefault(value, set()).add(split_name)
