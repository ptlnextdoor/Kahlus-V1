from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from neurotwin.data.leakage import check_manifest_leakage
from neurotwin.data.split_manifest import RecordingRecord, SplitManifest


POLICY_KEYS = {
    "subject": ("subject_id",),
    "session": ("session_id",),
    "site": ("site_id",),
    "dataset": ("dataset",),
    "run": ("metadata.run_id",),
}


@dataclass(frozen=True)
class AuditReport:
    passed: bool
    violations: tuple[str, ...]
    checked: tuple[str, ...]


def audit_split_manifest(
    manifest: SplitManifest,
    policy: str,
    forbidden_metadata_fields: tuple[str, ...] = ("label", "target", "target_label", "task_label", "diagnosis"),
) -> AuditReport:
    violations: list[str] = []
    checked: list[str] = ["record_reuse", "group_overlap", "metadata_labels", "window_overlap"]
    keys = tuple(key for key in POLICY_KEYS.get(policy, ()) if not key.startswith("metadata."))
    if keys:
        leakage = check_manifest_leakage(manifest, keys=keys)
        violations.extend(leakage.violations)
    if policy == "run":
        violations.extend(_metadata_group_overlap(manifest, "run_id"))
    violations.extend(_forbidden_metadata(manifest.all_records, forbidden_metadata_fields))
    violations.extend(_window_overlap(manifest))
    return AuditReport(not violations, tuple(violations), tuple(checked))


def _forbidden_metadata(records: list[RecordingRecord], fields: tuple[str, ...]) -> list[str]:
    violations = []
    lowered = tuple(field.lower() for field in fields)
    for record in records:
        for key in record.metadata:
            if key.lower() in lowered:
                violations.append(f"forbidden metadata field {key!r} in record {record.record_id!r}")
    return violations


def _metadata_group_overlap(manifest: SplitManifest, metadata_key: str) -> list[str]:
    split_values = {
        "train": {str(record.metadata.get(metadata_key)) for record in manifest.train if metadata_key in record.metadata},
        "val": {str(record.metadata.get(metadata_key)) for record in manifest.val if metadata_key in record.metadata},
        "test": {str(record.metadata.get(metadata_key)) for record in manifest.test if metadata_key in record.metadata},
    }
    violations = []
    for left, right in combinations(split_values, 2):
        overlap = split_values[left] & split_values[right]
        if overlap:
            violations.append(f"metadata.{metadata_key} leakage between {left} and {right}: {', '.join(sorted(overlap))}")
    return violations


def _window_overlap(manifest: SplitManifest) -> list[str]:
    by_source: dict[str, list[tuple[str, float, float, str]]] = {}
    for split, records in (("train", manifest.train), ("val", manifest.val), ("test", manifest.test)):
        for record in records:
            source = str(record.metadata.get("source_record_id", record.record_id))
            start = float(record.metadata.get("window_start", record.start_time))
            end = float(record.metadata.get("window_end", record.end_time))
            by_source.setdefault(source, []).append((split, start, end, record.record_id))

    violations: list[str] = []
    for source, windows in by_source.items():
        for left, right in combinations(windows, 2):
            left_split, left_start, left_end, left_id = left
            right_split, right_start, right_end, right_id = right
            if left_split == right_split:
                continue
            if max(left_start, right_start) < min(left_end, right_end):
                violations.append(f"window overlap for source {source!r}: {left_id} vs {right_id}")
    return violations
