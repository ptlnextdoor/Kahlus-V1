from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from neurotwin.benchmarks.prepared_suite import format_prepared_baseline_report, run_prepared_baseline_suite
from neurotwin.data.leakage import check_manifest_leakage
from neurotwin.data.manifest_io import save_data_manifest, save_leakage_report, save_split_manifest
from neurotwin.data.prepared_tasks import PreparedSuiteConfig
from neurotwin.data.split_manifest import build_split_manifest
from neurotwin.eval.audit import audit_prepared_eval_inputs, format_prepared_eval_audit
from neurotwin.repro import manifest_hash
from neurotwin.runtime.command_result import CommandResult


@dataclass(frozen=True)
class DataSmokeConfig:
    dataset: str
    split: str
    out_dir: str
    moabb_dataset: str = "BNCI2014_001"
    moabb_paradigm: str = "LeftRightImagery"
    subjects: list[int] | None = None
    max_trials: int = 6
    sampling_rate: float | None = None
    window_length: int = 8
    stride: int = 8
    train_steps: int = 1
    seed: int = 0
    require_windows: bool = False


def run_data_smoke_command(config: DataSmokeConfig) -> CommandResult:
    if config.dataset != "moabb":
        return CommandResult("", exit_code=1, error="MOABB smoke currently supports --dataset moabb only")
    from neurotwin.adapters.moabb import (
        MissingOptionalDependency,
        balanced_trial_subset,
        load_moabb_trials,
        trials_to_event_batches,
        trials_to_recordings,
    )
    from neurotwin.data.event_io import save_event_batches

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        trials = list(
            load_moabb_trials(
                config.moabb_dataset,
                subjects=tuple(config.subjects) if config.subjects else None,
                paradigm=config.moabb_paradigm,
                max_trials=None,
                sampling_rate=config.sampling_rate,
            )
        )
        trials = balanced_trial_subset(trials, split_policy=config.split, max_trials=config.max_trials)
    except (MissingOptionalDependency, ValueError) as exc:
        return CommandResult("", exit_code=1, error=str(exc))
    if not trials:
        return CommandResult("", exit_code=1, error="MOABB smoke produced no trials; adjust subject/paradigm filters.")

    records = trials_to_recordings(trials, dataset_id=config.moabb_dataset)
    event_batches = trials_to_event_batches(trials, dataset_id=config.moabb_dataset)
    split = build_split_manifest(records, policy=config.split, seed=config.seed)
    report = check_manifest_leakage(split, keys=_leakage_keys(config.split))
    split_path = save_split_manifest(split, out_dir / "split_manifest.json")
    data_path = save_data_manifest(records, out_dir / "data_manifest.json")
    leakage_path = save_leakage_report(report, out_dir / "leakage_report.json")
    manifest_digest = manifest_hash([record.__dict__ for record in split.all_records])
    event_path = save_event_batches(
        event_batches,
        out_dir,
        manifest_metadata={
            "dataset": config.dataset,
            "moabb_dataset": config.moabb_dataset,
            "moabb_paradigm": config.moabb_paradigm,
            "subjects": config.subjects,
            "split_policy": config.split,
            "max_trials": config.max_trials,
            "window_length": config.window_length,
            "stride": config.stride,
            "manifest_hash": manifest_digest,
            "leakage_report": str(leakage_path),
            "real_data_smoke": True,
        },
    )
    lines = [
        f"dataset={config.dataset}",
        f"moabb_dataset={config.moabb_dataset}",
        f"split_policy={split.policy}",
        f"split_stage={split.split_stage}",
        f"train={len(split.train)} val={len(split.val)} test={len(split.test)}",
        f"leakage_passed={report.passed}",
    ]
    lines.extend(f"violation={violation}" for violation in report.violations)
    lines.extend(
        (
            f"manifest_hash={manifest_digest}",
            f"data_manifest={data_path}",
            f"split_manifest={split_path}",
            f"event_manifest={event_path}",
            f"leakage_report={leakage_path}",
        )
    )

    audit = audit_prepared_eval_inputs(
        event_manifest=event_path,
        split_manifest=split_path,
        window_length=config.window_length,
        stride=config.stride,
        out_dir=out_dir,
        require_windows=config.require_windows,
    )
    lines.append(format_prepared_eval_audit(audit))
    if not audit.passed:
        return CommandResult("\n".join(lines), exit_code=1)

    payload = run_prepared_baseline_suite(
        PreparedSuiteConfig(
            event_manifest=event_path,
            split_manifest=split_path,
            window_length=config.window_length,
            stride=config.stride,
            seed=config.seed,
            train_steps=config.train_steps,
        ),
        out_dir=out_dir,
    )
    lines.append("scope=real_data_smoke")
    lines.append(format_prepared_baseline_report(payload))
    return CommandResult("\n".join(lines))


def _leakage_keys(split: str) -> tuple[str, ...]:
    return {
        "subject": ("subject_id",),
        "session": ("session_id",),
        "site": ("site_id",),
        "dataset": ("dataset",),
        "time": (),
    }[split]
