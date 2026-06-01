from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from neurotwin.adapters.synthetic import make_synthetic_event_batches, make_synthetic_recordings
from neurotwin.data.leakage import check_manifest_leakage
from neurotwin.data.manifest_io import save_data_manifest, save_leakage_report, save_split_manifest
from neurotwin.data.split_manifest import build_split_manifest
from neurotwin.repro import manifest_hash
from neurotwin.runtime.command_result import CommandResult


@dataclass(frozen=True)
class DataPrepareConfig:
    dataset: str
    split: str
    root: str | None = None
    out_dir: str | None = None
    moabb_dataset: str = "BNCI2014_001"
    moabb_paradigm: str = "LeftRightImagery"
    subjects: list[int] | None = None
    max_trials: int | None = None
    sampling_rate: float | None = None


@dataclass(frozen=True)
class DataAuditConfig:
    dataset: str
    root: str | None = None


def run_data_prepare_command(config: DataPrepareConfig) -> CommandResult:
    event_batches = None
    if config.dataset == "synthetic":
        records = make_synthetic_recordings()
        event_batches = make_synthetic_event_batches()
    elif config.dataset == "bids":
        if not config.root:
            return CommandResult("", exit_code=1, error="--root is required for --dataset bids")
        from neurotwin.adapters.bids import records_to_event_batches, scan_bids_manifest

        records = scan_bids_manifest(config.root, dataset_id="bids")
        event_batches = records_to_event_batches(records) or None
    elif config.dataset == "moabb":
        from neurotwin.adapters.moabb import (
            MissingOptionalDependency,
            balanced_trial_subset,
            load_moabb_trials,
            trials_to_event_batches,
            trials_to_recordings,
        )

        try:
            trials = load_moabb_trials(
                config.moabb_dataset,
                subjects=tuple(config.subjects) if config.subjects else None,
                paradigm=config.moabb_paradigm,
                max_trials=None,
                sampling_rate=config.sampling_rate,
            )
            trials = balanced_trial_subset(trials, split_policy=config.split, max_trials=config.max_trials)
        except (MissingOptionalDependency, ValueError) as exc:
            return CommandResult("", exit_code=1, error=str(exc))
        records = trials_to_recordings(trials, dataset_id=config.moabb_dataset)
        event_batches = trials_to_event_batches(trials, dataset_id=config.moabb_dataset)
    else:
        return CommandResult("", exit_code=1, error="Supported datasets: synthetic, bids, moabb")

    manifest = build_split_manifest(records, policy=config.split, seed=0)
    report = check_manifest_leakage(manifest, keys=_leakage_keys(config.split))
    manifest_digest = manifest_hash([record.__dict__ for record in manifest.all_records])
    lines = [
        f"dataset={config.dataset}",
        f"split_policy={manifest.policy}",
        f"split_stage={manifest.split_stage}",
        f"train={len(manifest.train)} val={len(manifest.val)} test={len(manifest.test)}",
        f"leakage_passed={report.passed}",
    ]
    lines.extend(f"violation={violation}" for violation in report.violations)
    lines.append(f"manifest_hash={manifest_digest}")
    if config.out_dir:
        out_dir = Path(config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        split_path = save_split_manifest(manifest, out_dir / "split_manifest.json")
        data_path = save_data_manifest(records, out_dir / "data_manifest.json")
        audit_path = save_leakage_report(report, out_dir / "leakage_report.json")
        lines.extend((f"split_manifest={split_path}", f"data_manifest={data_path}", f"leakage_report={audit_path}"))
        if event_batches is not None:
            from neurotwin.data.event_io import save_event_batches

            event_path = save_event_batches(
                event_batches,
                out_dir,
                manifest_metadata={
                    "dataset": config.dataset,
                    "moabb_dataset": config.moabb_dataset if config.dataset == "moabb" else None,
                    "moabb_paradigm": config.moabb_paradigm if config.dataset == "moabb" else None,
                    "subjects": config.subjects,
                    "split_policy": config.split,
                    "max_trials": config.max_trials,
                    "manifest_hash": manifest_digest,
                    "leakage_report": str(audit_path),
                },
            )
            lines.append(f"event_manifest={event_path}")
    return CommandResult("\n".join(lines))


def run_data_audit_command(config: DataAuditConfig) -> CommandResult:
    if config.dataset == "synthetic":
        records = make_synthetic_recordings()
        return CommandResult("\n".join(("dataset=synthetic", f"records={len(records)}", "audit_passed=True")))
    if config.dataset == "bids":
        if not config.root:
            return CommandResult("", exit_code=1, error="--root is required for BIDS audit")
        from neurotwin.adapters.bids import bids_manifest_summary, scan_bids_manifest

        records = scan_bids_manifest(config.root, dataset_id="bids")
        summary = bids_manifest_summary(records)
        return CommandResult(
            "\n".join(
                (
                    "dataset=bids",
                    f"records={len(records)}",
                    f"subjects={len(summary['subjects'])}",
                    f"sites={len(summary['sites'])}",
                    f"with_timeseries_derivative={summary['with_timeseries_derivative']}",
                    "derivative_only=True",
                    f"audit_passed={len(records) > 0}",
                )
            )
        )
    if config.dataset == "moabb":
        from neurotwin.adapters.moabb import moabb_optional_status

        status = moabb_optional_status()
        return CommandResult("\n".join(("dataset=moabb", f"optional_status={status}", f"audit_passed={all(status.values())}")))
    return CommandResult("", exit_code=1, error="Supported datasets: synthetic, bids, moabb")


def _leakage_keys(split: str) -> tuple[str, ...]:
    return {
        "subject": ("subject_id",),
        "session": ("session_id",),
        "site": ("site_id",),
        "dataset": ("dataset",),
        "time": (),
    }[split]
