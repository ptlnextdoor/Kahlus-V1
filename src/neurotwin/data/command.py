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
    window_length: int = 128
    stride: int = 128


@dataclass(frozen=True)
class DataAuditConfig:
    dataset: str
    root: str | None = None


def run_data_prepare_command(config: DataPrepareConfig) -> CommandResult:
    event_batches = None
    if config.dataset == "algonauts2025":
        if not config.root:
            return CommandResult("", exit_code=1, error="--root is required for --dataset algonauts2025")
        if not config.out_dir:
            return CommandResult("", exit_code=1, error="--out-dir is required for --dataset algonauts2025")
        try:
            from neurotwin.adapters.algonauts import prepare_algonauts2025
            from neurotwin.data.event_io import save_event_batches
            from neurotwin.eval.audit import audit_prepared_eval_inputs

            prepared = prepare_algonauts2025(config.root, config.out_dir, split=config.split)
            out_dir = Path(config.out_dir)
            split_path = save_split_manifest(prepared.split_manifest, out_dir / "split_manifest.json")
            report = check_manifest_leakage(prepared.split_manifest, keys=_leakage_keys(config.split))
            audit_path = save_leakage_report(report, out_dir / "leakage_report.json")
            manifest_digest = manifest_hash([record.__dict__ for record in prepared.split_manifest.all_records])
            event_path = save_event_batches(
                prepared.event_batches,
                out_dir,
                manifest_metadata={
                    "dataset": config.dataset,
                    "split_policy": config.split,
                    "manifest_hash": manifest_digest,
                    "leakage_report": str(audit_path),
                    "data_manifest": str(prepared.data_manifest),
                    "feature_manifest": str(prepared.feature_manifest),
                    "stimulus_manifest": str(prepared.stimulus_manifest),
                    "window_length": config.window_length,
                    "stride": config.stride,
                },
            )
            eval_audit = audit_prepared_eval_inputs(
                event_path,
                split_path,
                window_length=config.window_length,
                stride=config.stride,
                out_dir=out_dir,
                require_windows=True,
            )
        except Exception as exc:  # noqa: BLE001 - CLI should surface preparation failure without a traceback by default.
            return CommandResult("", exit_code=1, error=str(exc))
        lines = [
            "dataset=algonauts2025",
            f"split_policy={prepared.split_manifest.policy}",
            f"split_stage={prepared.split_manifest.split_stage}",
            f"train={len(prepared.split_manifest.train)} val={len(prepared.split_manifest.val)} test={len(prepared.split_manifest.test)}",
            f"leakage_passed={report.passed}",
            f"eval_audit_passed={eval_audit.passed}",
            f"window_count={eval_audit.window_count}",
            "window_counts_by_split="
            + ",".join(
                f"{split_name}:{eval_audit.window_counts_by_split.get(split_name, 0)}"
                for split_name in ("train", "val", "test")
            ),
            f"manifest_hash={manifest_digest}",
            f"split_manifest={split_path}",
            f"data_manifest={prepared.data_manifest}",
            f"feature_manifest={prepared.feature_manifest}",
            f"stimulus_manifest={prepared.stimulus_manifest}",
            f"leakage_report={audit_path}",
            f"eval_audit={out_dir / 'eval_audit.json'}",
            f"event_manifest={event_path}",
        ]
        lines.extend(f"violation={violation}" for violation in report.violations)
        lines.extend(f"eval_violation={violation}" for violation in eval_audit.violations)
        return CommandResult("\n".join(lines), exit_code=0 if eval_audit.passed and report.passed else 1)
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
        return CommandResult("", exit_code=1, error="Supported datasets: synthetic, bids, moabb, algonauts2025")

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
    if config.dataset == "algonauts2025":
        if not config.root:
            return CommandResult("", exit_code=1, error="--root is required for Algonauts audit")
        try:
            from neurotwin.adapters.algonauts import ALGONAUTS_SUBJECTS, prepare_algonauts2025
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                prepared = prepare_algonauts2025(config.root, tmp, split="official")
        except Exception as exc:  # noqa: BLE001 - audit command returns the exact missing contract.
            return CommandResult("", exit_code=1, error=str(exc))
        subjects = sorted({record.subject_id for record in prepared.records})
        return CommandResult(
            "\n".join(
                (
                    "dataset=algonauts2025",
                    f"records={len(prepared.records)}",
                    "subjects=" + ",".join(subjects),
                    "expected_subjects=" + ",".join(ALGONAUTS_SUBJECTS),
                    "audit_passed=True",
                )
            )
        )
    return CommandResult("", exit_code=1, error="Supported datasets: synthetic, bids, moabb, algonauts2025")


def _leakage_keys(split: str) -> tuple[str, ...]:
    return {
        "subject": ("subject_id",),
        "session": ("session_id",),
        "site": ("site_id",),
        "dataset": ("dataset",),
        "time": (),
        "official": (),
    }[split]
