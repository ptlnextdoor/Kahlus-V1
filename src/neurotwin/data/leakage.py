from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

from neurotwin.data.split_manifest import RecordingRecord, SplitManifest


@dataclass(frozen=True)
class LeakageReport:
    passed: bool
    violations: tuple[str, ...]
    checked_keys: tuple[str, ...]


def check_manifest_leakage(
    manifest: SplitManifest,
    keys: Iterable[str] = ("subject_id", "session_id", "site_id", "dataset"),
) -> LeakageReport:
    """Check record reuse and held-out group overlap across train/val/test."""

    checked_keys = tuple(keys)
    split_records = {
        "train": manifest.train,
        "val": manifest.val,
        "test": manifest.test,
    }
    violations: list[str] = []

    record_locations: dict[str, list[str]] = {}
    for split_name, records in split_records.items():
        for record in records:
            record_locations.setdefault(record.record_id, []).append(split_name)
    for record_id, locations in sorted(record_locations.items()):
        if len(locations) > 1:
            violations.append(f"record_id {record_id!r} appears in multiple splits: {locations}")

    for key in checked_keys:
        values_by_split = {
            split_name: _values(records, key)
            for split_name, records in split_records.items()
        }
        for left, right in combinations(values_by_split, 2):
            overlap = values_by_split[left] & values_by_split[right]
            if overlap:
                sample = ", ".join(sorted(overlap)[:5])
                violations.append(f"{key} leakage between {left} and {right}: {sample}")

    return LeakageReport(
        passed=not violations,
        violations=tuple(violations),
        checked_keys=checked_keys,
    )


def _values(records: list[RecordingRecord], key: str) -> set[str]:
    return {str(getattr(record, key)) for record in records}
