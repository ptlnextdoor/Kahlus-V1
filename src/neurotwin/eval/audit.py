from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from neurotwin.data.audit import audit_split_manifest
from neurotwin.data.event_io import event_manifest_summary, load_event_batches
from neurotwin.data.manifest_io import load_split_manifest
from neurotwin.data.schemas import NeuralEventBatch
from neurotwin.data.windows import WindowSpec, batch_to_windows
from neurotwin.repro import write_json


@dataclass(frozen=True)
class PreparedEvalAuditReport:
    passed: bool
    violations: tuple[str, ...]
    warnings: tuple[str, ...]
    checked: tuple[str, ...]
    event_count: int
    window_count: int
    window_counts_by_split: dict[str, int]
    event_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_prepared_eval_inputs(
    event_manifest: str | Path,
    split_manifest: str | Path,
    window_length: int = 8,
    stride: int = 8,
    out_dir: str | Path | None = None,
    require_windows: bool = False,
) -> PreparedEvalAuditReport:
    """Audit prepared eval inputs before any benchmark score is trusted."""

    checked = [
        "event_hashes",
        "event_split_membership",
        "split_policy_leakage",
        "prepared_window_overlap",
        "split_coverage",
        "duplicate_source_hashes",
        "duplicate_preprocessing_hashes",
        "stimulus_segment_leakage",
        "event_metadata_target_leakage",
        "hidden_subject_metadata",
    ]
    violations: list[str] = []
    warnings: list[str] = []
    try:
        batches = load_event_batches(event_manifest)
        summary = event_manifest_summary(event_manifest)
    except Exception as exc:  # noqa: BLE001 - audit must report integrity failures, not hide them.
        report = PreparedEvalAuditReport(
            passed=False,
            violations=(f"event manifest integrity failure: {exc}",),
            warnings=(),
            checked=tuple(checked),
            event_count=0,
            window_count=0,
            window_counts_by_split={"train": 0, "val": 0, "test": 0},
            event_summary={},
        )
        if out_dir is not None:
            write_json(Path(out_dir) / "eval_audit.json", report.to_dict())
        return report

    try:
        split = load_split_manifest(split_manifest)
        split_report = audit_split_manifest(split, policy=split.policy)
    except Exception as exc:  # noqa: BLE001 - audit must report split integrity failures, not leak raw errors.
        report = PreparedEvalAuditReport(
            passed=False,
            violations=(f"split manifest integrity failure: {exc}",),
            warnings=(),
            checked=tuple(checked),
            event_count=len(batches),
            window_count=0,
            window_counts_by_split={"train": 0, "val": 0, "test": 0},
            event_summary=summary,
        )
        if out_dir is not None:
            write_json(Path(out_dir) / "eval_audit.json", report.to_dict())
        return report
    violations.extend(split_report.violations)

    split_by_record = {}
    for split_name in ("train", "val", "test"):
        for record in getattr(split, split_name):
            split_by_record[record.record_id] = split_name
    event_record_ids = {_record_id(batch) for batch in batches}
    missing_from_split = sorted(record_id for record_id in event_record_ids if record_id not in split_by_record)
    if missing_from_split:
        violations.append("events missing from split manifest: " + ", ".join(missing_from_split[:10]))

    missing_events = sorted(record_id for record_id in split_by_record if record_id not in event_record_ids)
    if missing_events:
        warnings.append("split records without prepared events: " + ", ".join(missing_events[:10]))

    split_counts = {split_name: 0 for split_name in ("train", "val", "test")}
    for batch in batches:
        split_name = split_by_record.get(_record_id(batch))
        if split_name:
            split_counts[split_name] += 1
    empty_splits = [split_name for split_name, count in split_counts.items() if count == 0]
    if empty_splits:
        violations.append("prepared events absent for split(s): " + ", ".join(empty_splits))
    violations.extend(_event_metadata_violations(batches, split_by_record))

    windows = _prepared_windows_by_split(batches, split_by_record, WindowSpec(length=window_length, stride=stride))
    window_counts_by_split = {split_name: len(split_windows) for split_name, split_windows in windows.items()}
    window_count = sum(window_counts_by_split.values())
    violations.extend(_window_overlap_violations(windows))
    if require_windows:
        if window_count == 0:
            violations.append(
                f"prepared benchmark produced zero windows for window_length={window_length} stride={stride}"
            )
        empty_window_splits = [
            split_name for split_name in ("train", "val", "test") if window_counts_by_split.get(split_name, 0) == 0
        ]
        if empty_window_splits:
            violations.append("prepared windows absent for split(s): " + ", ".join(empty_window_splits))
    report = PreparedEvalAuditReport(
        passed=not violations,
        violations=tuple(violations),
        warnings=tuple(warnings),
        checked=tuple(checked),
        event_count=len(batches),
        window_count=window_count,
        window_counts_by_split=window_counts_by_split,
        event_summary=summary,
    )
    if out_dir is not None:
        write_json(Path(out_dir) / "eval_audit.json", report.to_dict())
    return report


def format_prepared_eval_audit(report: PreparedEvalAuditReport) -> str:
    lines = [
        "eval_audit_prepared=True",
        f"eval_audit_passed={report.passed}",
        f"event_count={report.event_count}",
        f"window_count={report.window_count}",
        "window_counts_by_split="
        + ",".join(f"{split_name}:{report.window_counts_by_split.get(split_name, 0)}" for split_name in ("train", "val", "test")),
        "checked=" + ",".join(report.checked),
    ]
    for violation in report.violations:
        lines.append(f"violation={violation}")
    for warning in report.warnings:
        lines.append(f"warning={warning}")
    return "\n".join(lines)


def _prepared_windows_by_split(
    batches: list[NeuralEventBatch],
    split_by_record: dict[str, str],
    spec: WindowSpec,
) -> dict[str, list[NeuralEventBatch]]:
    windows = {"train": [], "val": [], "test": []}
    for batch in batches:
        split_name = split_by_record.get(_record_id(batch))
        if split_name is None:
            continue
        windows[split_name].extend(batch_to_windows(batch, spec))
    return windows


def _window_overlap_violations(windows: dict[str, list[NeuralEventBatch]]) -> list[str]:
    seen: dict[tuple[str, str, int, int], str] = {}
    violations: list[str] = []
    for split_name, split_windows in windows.items():
        for window in split_windows:
            key = (
                str(window.metadata.get("source_record_id") or _record_id(window)),
                window.modality,
                int(window.metadata.get("window_start_index", 0)),
                int(window.metadata.get("window_end_index", window.n_time)),
            )
            previous = seen.get(key)
            if previous is not None and previous != split_name:
                violations.append(
                    f"prepared window leakage across splits for source={key[0]} modality={key[1]} "
                    f"indices={key[2]}:{key[3]} ({previous} vs {split_name})"
                )
            else:
                seen[key] = split_name
    return violations


def _record_id(batch: NeuralEventBatch) -> str:
    return str(batch.metadata.get("record_id") or batch.metadata.get("source_record_id"))


def _event_metadata_violations(
    batches: list[NeuralEventBatch],
    split_by_record: dict[str, str],
) -> list[str]:
    violations: list[str] = []
    violations.extend(_duplicate_metadata_value_violations(batches, split_by_record, "source_hash"))
    violations.extend(_duplicate_metadata_value_violations(batches, split_by_record, "preprocessing_hash"))
    violations.extend(_stimulus_segment_violations(batches, split_by_record))
    violations.extend(_forbidden_event_metadata_violations(batches))
    violations.extend(_hidden_subject_metadata_violations(batches))
    return violations


def _duplicate_metadata_value_violations(
    batches: list[NeuralEventBatch],
    split_by_record: dict[str, str],
    field: str,
) -> list[str]:
    seen: dict[str, tuple[str, str]] = {}
    violations: list[str] = []
    for batch in batches:
        value = getattr(batch, field)
        if not value:
            continue
        split_name = split_by_record.get(_record_id(batch), "unknown")
        previous = seen.get(value)
        if previous is not None and previous[0] != split_name:
            violations.append(
                f"duplicate {field} across splits: {value} ({previous[0]}:{previous[1]} vs {split_name}:{_record_id(batch)})"
            )
        else:
            seen[value] = (split_name, _record_id(batch))
    return violations


def _stimulus_segment_violations(
    batches: list[NeuralEventBatch],
    split_by_record: dict[str, str],
) -> list[str]:
    seen: dict[str, tuple[str, str]] = {}
    violations: list[str] = []
    for batch in batches:
        key = _stimulus_segment_key(batch)
        if key is None:
            continue
        split_name = split_by_record.get(_record_id(batch), "unknown")
        previous = seen.get(key)
        if previous is not None and previous[0] != split_name:
            violations.append(
                f"stimulus segment leakage across splits: {key} ({previous[0]}:{previous[1]} vs {split_name}:{_record_id(batch)})"
            )
        else:
            seen[key] = (split_name, _record_id(batch))
    return violations


def _stimulus_segment_key(batch: NeuralEventBatch) -> str | None:
    metadata = batch.metadata
    for key in ("stimulus_segment_id", "stimulus_clip_id"):
        value = metadata.get(key)
        if value is not None:
            return f"{key}:{value}"
    stimulus_id = metadata.get("stimulus_id")
    start = metadata.get("stimulus_start")
    end = metadata.get("stimulus_end")
    if stimulus_id is not None and start is not None and end is not None:
        return f"stimulus_id:{stimulus_id}:{start}:{end}"
    alignment = batch.stimulus_alignment_metadata
    if alignment:
        segment = alignment.get("segment_id") or alignment.get("clip_id")
        if segment is not None:
            return f"stimulus_alignment:{segment}"
    return None


def _forbidden_event_metadata_violations(batches: list[NeuralEventBatch]) -> list[str]:
    forbidden = {"label", "target", "target_label", "task_label", "diagnosis"}
    violations: list[str] = []
    for batch in batches:
        for key in batch.metadata:
            if key.lower() in forbidden:
                violations.append(f"forbidden event metadata field {key!r} in record {_record_id(batch)!r}")
    return violations


def _hidden_subject_metadata_violations(batches: list[NeuralEventBatch]) -> list[str]:
    violations: list[str] = []
    for batch in batches:
        for key, value in batch.metadata.items():
            lowered = key.lower()
            if "subject" not in lowered and "participant" not in lowered:
                continue
            if lowered in {"source_subject_id"}:
                continue
            values = value if isinstance(value, (list, tuple, set)) else (value,)
            if any(str(item) == batch.subject_id for item in values):
                violations.append(f"hidden subject metadata field {key!r} in record {_record_id(batch)!r}")
    return violations
