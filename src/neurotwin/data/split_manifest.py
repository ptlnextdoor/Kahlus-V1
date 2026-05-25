from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import random
from typing import Any, Iterable


@dataclass(frozen=True)
class RecordingRecord:
    """Atomic recording-level unit used for splitting before windowing."""

    record_id: str
    modality: str
    dataset: str
    subject_id: str
    session_id: str
    site_id: str
    start_time: float
    end_time: float
    stimulus_id: str | None = None
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def stable_hash(self) -> str:
        payload = asdict(self)
        payload["metadata"] = dict(sorted(self.metadata.items()))
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(encoded).hexdigest()


@dataclass
class SplitManifest:
    """Leakage-aware split manifest built before preprocessing and augmentation."""

    policy: str
    seed: int
    train: list[RecordingRecord]
    val: list[RecordingRecord]
    test: list[RecordingRecord]
    record_hashes: dict[str, str]
    split_stage: str = "recording_manifest"
    notes: list[str] = field(default_factory=list)

    @property
    def all_records(self) -> list[RecordingRecord]:
        return [*self.train, *self.val, *self.test]


POLICY_TO_KEY = {
    "subject": "subject_id",
    "session": "session_id",
    "site": "site_id",
    "dataset": "dataset",
}


def build_split_manifest(
    records: Iterable[RecordingRecord],
    policy: str,
    seed: int = 0,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> SplitManifest:
    records = list(records)
    if not records:
        raise ValueError("Cannot build a split manifest from zero records")
    if policy == "time":
        train, val, test = _time_split(records, val_fraction=val_fraction, test_fraction=test_fraction)
    elif policy in POLICY_TO_KEY:
        train, val, test = _group_split(
            records,
            key=POLICY_TO_KEY[policy],
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
    else:
        supported = ", ".join([*POLICY_TO_KEY.keys(), "time"])
        raise ValueError(f"Unsupported split policy {policy!r}; expected one of: {supported}")

    hashes = {record.record_id: record.stable_hash() for record in records}
    return SplitManifest(
        policy=policy,
        seed=seed,
        train=train,
        val=val,
        test=test,
        record_hashes=hashes,
        notes=[
            "Split was built at recording-manifest stage before preprocessing/windowing.",
            "Raw public data paths are represented but not committed.",
        ],
    )


def _group_split(
    records: list[RecordingRecord],
    key: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> tuple[list[RecordingRecord], list[RecordingRecord], list[RecordingRecord]]:
    grouped: dict[str, list[RecordingRecord]] = {}
    for record in records:
        grouped.setdefault(str(getattr(record, key)), []).append(record)

    groups = list(grouped)
    random.Random(seed).shuffle(groups)
    n_test = _fraction_count(len(groups), test_fraction)
    n_val = _fraction_count(len(groups) - n_test, val_fraction)

    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test : n_test + n_val])

    train: list[RecordingRecord] = []
    val: list[RecordingRecord] = []
    test: list[RecordingRecord] = []
    for record in records:
        group = str(getattr(record, key))
        if group in test_groups:
            test.append(record)
        elif group in val_groups:
            val.append(record)
        else:
            train.append(record)
    return train, val, test


def _time_split(
    records: list[RecordingRecord],
    val_fraction: float,
    test_fraction: float,
) -> tuple[list[RecordingRecord], list[RecordingRecord], list[RecordingRecord]]:
    ordered = sorted(records, key=lambda record: (record.start_time, record.end_time, record.record_id))
    n_test = _fraction_count(len(ordered), test_fraction)
    n_val = _fraction_count(len(ordered) - n_test, val_fraction)
    n_train = len(ordered) - n_val - n_test
    return ordered[:n_train], ordered[n_train : n_train + n_val], ordered[n_train + n_val :]


def _fraction_count(total: int, fraction: float) -> int:
    if total <= 1:
        return 0
    return max(1, min(total - 1, round(total * fraction)))
