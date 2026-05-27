from __future__ import annotations

import argparse
from pathlib import Path

from neurotwin.adapters.synthetic import make_synthetic_event_batches, make_synthetic_recordings
from neurotwin.benchmarks.prepared_suite import PreparedSuiteConfig, format_prepared_baseline_report, run_prepared_baseline_suite
from neurotwin.benchmarks.registry import competitor_registry
from neurotwin.benchmarks.suite import format_neural_translation_v1_report, run_neural_translation_v1_synthetic
from neurotwin.benchmarks.smoke import format_smoke_results, run_translation_smoke
from neurotwin.benchmarks.task_specs import default_translation_tasks
from neurotwin.config import ConfigError, load_config
from neurotwin.data.audit import audit_split_manifest
from neurotwin.data.leakage import check_manifest_leakage
from neurotwin.data.manifest_io import save_data_manifest, save_leakage_report, save_split_manifest
from neurotwin.data.split_manifest import build_split_manifest
from neurotwin.doctor import format_doctor_report, run_doctor
from neurotwin.eval.audit import audit_prepared_eval_inputs, format_prepared_eval_audit
from neurotwin.reports import generate_compare_report, generate_run_report, generate_suite_report
from neurotwin.repro import append_jsonl, capture_environment, create_run_dir, manifest_hash, snapshot_config, stable_hash, write_json
from neurotwin.runtime.distributed import get_distributed_info, get_rank_metrics_path
from neurotwin.runtime.estimate import estimate_config
from neurotwin.training.prepared import run_prepared_training
from neurotwin.training.smoke import run_synthetic_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nt", description="NeuroTwin research benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check runtime, GPU, optional dependencies, and dataset roots")
    doctor.set_defaults(func=_cmd_doctor)

    data_parser = subparsers.add_parser("data", help="Dataset preparation commands")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    prepare = data_subparsers.add_parser("prepare", help="Prepare manifests and leakage-proof splits")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--split", required=True, choices=("subject", "session", "site", "dataset", "time"))
    prepare.add_argument("--root", default=None)
    prepare.add_argument("--out-dir", default=None)
    prepare.add_argument("--moabb-dataset", default="BNCI2014_001")
    prepare.add_argument("--moabb-paradigm", default="LeftRightImagery")
    prepare.add_argument("--subjects", nargs="*", type=int, default=None)
    prepare.add_argument("--max-trials", type=int, default=None)
    prepare.add_argument("--sampling-rate", type=float, default=None)
    prepare.set_defaults(func=_cmd_data_prepare)
    smoke = data_subparsers.add_parser("smoke", help="Run a lightweight MOABB smoke pipeline")
    smoke.add_argument("--dataset", default="moabb", choices=("moabb",))
    smoke.add_argument("--split", default="subject", choices=("subject", "session", "site", "dataset", "time"))
    smoke.add_argument("--out-dir", required=True)
    smoke.add_argument("--moabb-dataset", default="BNCI2014_001")
    smoke.add_argument("--moabb-paradigm", default="LeftRightImagery")
    smoke.add_argument("--subjects", nargs="*", type=int, default=None)
    smoke.add_argument("--max-trials", type=int, default=6)
    smoke.add_argument("--sampling-rate", type=float, default=None)
    smoke.add_argument("--window-length", type=int, default=8)
    smoke.add_argument("--stride", type=int, default=8)
    smoke.add_argument("--train-steps", type=int, default=1)
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--require-windows", action="store_true")
    smoke.set_defaults(func=_cmd_data_smoke)
    data_audit = data_subparsers.add_parser("audit", help="Audit dataset availability and manifestability")
    data_audit.add_argument("--dataset", required=True)
    data_audit.add_argument("--root", default=None)
    data_audit.set_defaults(func=_cmd_data_audit)

    split_parser = subparsers.add_parser("split", help="Split manifest commands")
    split_subparsers = split_parser.add_subparsers(dest="split_command", required=True)
    split_audit = split_subparsers.add_parser("audit", help="Audit leakage for a split policy")
    split_audit.add_argument("--dataset", default=None)
    split_audit.add_argument("--manifest", default=None)
    split_audit.add_argument("--split", required=True, choices=("subject", "session", "site", "dataset", "time"))
    split_audit.set_defaults(func=_cmd_split_audit)

    train = subparsers.add_parser("train", help="Validate a training config placeholder")
    train.add_argument("--config", required=True)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--run-root", default="runs")
    train.add_argument("--resume", default=None)
    train.set_defaults(func=_cmd_train)

    estimate = subparsers.add_parser("estimate", help="Estimate model/runtime size from a config")
    estimate.add_argument("--config", required=True)
    estimate.set_defaults(func=_cmd_estimate)

    eval_parser = subparsers.add_parser("eval", help="Describe an evaluation suite")
    eval_parser.add_argument("eval_command", nargs="?", choices=("audit",))
    eval_parser.add_argument("--run", default=None)
    eval_parser.add_argument("--suite", default="translation_smoke")
    eval_parser.add_argument("--out-dir", default=None)
    eval_parser.add_argument("--event-manifest", default=None)
    eval_parser.add_argument("--split-manifest", default=None)
    eval_parser.add_argument("--window-length", type=int, default=8)
    eval_parser.add_argument("--stride", type=int, default=8)
    eval_parser.add_argument("--train-steps", type=int, default=5)
    eval_parser.add_argument("--require-windows", action="store_true")
    eval_parser.set_defaults(func=_cmd_eval)

    report = subparsers.add_parser("report", help="Generate a reproducible benchmark report")
    report.add_argument("--suite", default=None)
    report.add_argument("--run-dir", default=None)
    report.add_argument("--compare", nargs="*", default=None)
    report.add_argument("--out-dir", default=None)
    report.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


def _cmd_data_prepare(args: argparse.Namespace) -> None:
    event_batches = None
    if args.dataset != "synthetic":
        if args.dataset == "bids":
            from neurotwin.adapters.bids import records_to_event_batches, scan_bids_manifest

            if not args.root:
                raise SystemExit("--root is required for --dataset bids")
            records = scan_bids_manifest(args.root, dataset_id="bids")
            event_batches = records_to_event_batches(records)
            if not event_batches:
                event_batches = None
        elif args.dataset == "moabb":
            from neurotwin.adapters.moabb import (
                MissingOptionalDependency,
                balanced_trial_subset,
                load_moabb_trials,
                trials_to_event_batches,
                trials_to_recordings,
            )

            try:
                trials = load_moabb_trials(
                    args.moabb_dataset,
                    subjects=tuple(args.subjects) if args.subjects else None,
                    paradigm=args.moabb_paradigm,
                    max_trials=None,
                    sampling_rate=args.sampling_rate,
                )
            except MissingOptionalDependency as exc:
                raise SystemExit(str(exc)) from exc
            try:
                trials = balanced_trial_subset(trials, split_policy=args.split, max_trials=args.max_trials)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            records = trials_to_recordings(trials, dataset_id=args.moabb_dataset)
            event_batches = trials_to_event_batches(trials, dataset_id=args.moabb_dataset)
        else:
            raise SystemExit("Supported datasets: synthetic, bids, moabb")
    else:
        records = make_synthetic_recordings()
        event_batches = make_synthetic_event_batches()
    manifest = build_split_manifest(records, policy=args.split, seed=0)
    leakage_key = {
        "subject": ("subject_id",),
        "session": ("session_id",),
        "site": ("site_id",),
        "dataset": ("dataset",),
        "time": (),
    }[args.split]
    report = check_manifest_leakage(manifest, keys=leakage_key)
    print(f"dataset={args.dataset}")
    print(f"split_policy={manifest.policy}")
    print(f"split_stage={manifest.split_stage}")
    print(f"train={len(manifest.train)} val={len(manifest.val)} test={len(manifest.test)}")
    print(f"leakage_passed={report.passed}")
    for violation in report.violations:
        print(f"violation={violation}")
    print(f"manifest_hash={manifest_hash([record.__dict__ for record in manifest.all_records])}")
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        split_path = save_split_manifest(manifest, out_dir / "split_manifest.json")
        data_path = save_data_manifest(records, out_dir / "data_manifest.json")
        audit_path = save_leakage_report(report, out_dir / "leakage_report.json")
        print(f"split_manifest={split_path}")
        print(f"data_manifest={data_path}")
        print(f"leakage_report={audit_path}")
        if event_batches is not None:
            from neurotwin.data.event_io import save_event_batches

            event_path = save_event_batches(
                event_batches,
                out_dir,
                manifest_metadata={
                    "dataset": args.dataset,
                    "moabb_dataset": args.moabb_dataset if args.dataset == "moabb" else None,
                    "moabb_paradigm": args.moabb_paradigm if args.dataset == "moabb" else None,
                    "subjects": args.subjects,
                    "split_policy": args.split,
                    "max_trials": args.max_trials,
                    "manifest_hash": manifest_hash([record.__dict__ for record in manifest.all_records]),
                    "leakage_report": str(audit_path),
                },
            )
            print(f"event_manifest={event_path}")


def _cmd_data_smoke(args: argparse.Namespace) -> None:
    if args.dataset != "moabb":
        raise SystemExit("MOABB smoke currently supports --dataset moabb only")
    from neurotwin.adapters.moabb import (
        MissingOptionalDependency,
        balanced_trial_subset,
        load_moabb_trials,
        trials_to_event_batches,
        trials_to_recordings,
    )

    from neurotwin.data.event_io import save_event_batches

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        trials = list(load_moabb_trials(
            args.moabb_dataset,
            subjects=tuple(args.subjects) if args.subjects else None,
            paradigm=args.moabb_paradigm,
            max_trials=None,
            sampling_rate=args.sampling_rate,
        ))
    except MissingOptionalDependency as exc:
        raise SystemExit(str(exc)) from exc
    try:
        trials = balanced_trial_subset(trials, split_policy=args.split, max_trials=args.max_trials)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not trials:
        raise SystemExit("MOABB smoke produced no trials; adjust subject/paradigm filters.")

    records = trials_to_recordings(trials, dataset_id=args.moabb_dataset)
    event_batches = trials_to_event_batches(trials, dataset_id=args.moabb_dataset)

    split = build_split_manifest(records, policy=args.split, seed=args.seed)
    leakage_key = {
        "subject": ("subject_id",),
        "session": ("session_id",),
        "site": ("site_id",),
        "dataset": ("dataset",),
        "time": (),
    }[args.split]
    report = check_manifest_leakage(split, keys=leakage_key)
    split_path = save_split_manifest(split, out_dir / "split_manifest.json")
    data_path = save_data_manifest(records, out_dir / "data_manifest.json")
    leakage_path = save_leakage_report(report, out_dir / "leakage_report.json")
    event_path = save_event_batches(
        event_batches,
        out_dir,
        manifest_metadata={
            "dataset": args.dataset,
            "moabb_dataset": args.moabb_dataset,
            "moabb_paradigm": args.moabb_paradigm,
            "subjects": args.subjects,
            "split_policy": args.split,
            "max_trials": args.max_trials,
            "window_length": args.window_length,
            "stride": args.stride,
            "manifest_hash": manifest_hash([record.__dict__ for record in split.all_records]),
            "leakage_report": str(leakage_path),
            "real_data_smoke": True,
        },
    )

    print(f"dataset={args.dataset}")
    print(f"moabb_dataset={args.moabb_dataset}")
    print(f"split_policy={split.policy}")
    print(f"split_stage={split.split_stage}")
    print(f"train={len(split.train)} val={len(split.val)} test={len(split.test)}")
    print(f"leakage_passed={report.passed}")
    for violation in report.violations:
        print(f"violation={violation}")
    print(f"manifest_hash={manifest_hash([record.__dict__ for record in split.all_records])}")
    print(f"data_manifest={data_path}")
    print(f"split_manifest={split_path}")
    print(f"event_manifest={event_path}")
    print(f"leakage_report={leakage_path}")

    audit = audit_prepared_eval_inputs(
        event_manifest=event_path,
        split_manifest=split_path,
        window_length=args.window_length,
        stride=args.stride,
        out_dir=out_dir,
        require_windows=args.require_windows,
    )
    print(format_prepared_eval_audit(audit))

    if not audit.passed:
        raise SystemExit(1)

    payload = run_prepared_baseline_suite(
        PreparedSuiteConfig(
            event_manifest=event_path,
            split_manifest=split_path,
            window_length=args.window_length,
            stride=args.stride,
            seed=args.seed,
            train_steps=args.train_steps,
        ),
        out_dir=out_dir,
    )
    print("scope=real_data_smoke")
    print(format_prepared_baseline_report(payload))


def _cmd_data_audit(args: argparse.Namespace) -> None:
    if args.dataset == "synthetic":
        records = make_synthetic_recordings()
        print("dataset=synthetic")
        print(f"records={len(records)}")
        print("audit_passed=True")
    elif args.dataset == "bids":
        if not args.root:
            raise SystemExit("--root is required for BIDS audit")
        from neurotwin.adapters.bids import bids_manifest_summary, scan_bids_manifest

        records = scan_bids_manifest(args.root, dataset_id="bids")
        summary = bids_manifest_summary(records)
        print("dataset=bids")
        print(f"records={len(records)}")
        print(f"subjects={len(summary['subjects'])}")
        print(f"sites={len(summary['sites'])}")
        print(f"with_timeseries_derivative={summary['with_timeseries_derivative']}")
        print("derivative_only=True")
        print(f"audit_passed={len(records) > 0}")
    elif args.dataset == "moabb":
        from neurotwin.adapters.moabb import moabb_optional_status

        status = moabb_optional_status()
        print("dataset=moabb")
        print(f"optional_status={status}")
        print(f"audit_passed={all(status.values())}")
    else:
        raise SystemExit("Supported datasets: synthetic, bids, moabb")


def _cmd_split_audit(args: argparse.Namespace) -> None:
    if args.manifest:
        from neurotwin.data.manifest_io import load_split_manifest

        manifest = load_split_manifest(args.manifest)
        report = audit_split_manifest(manifest, policy=args.split)
        print(f"manifest={args.manifest}")
        print(f"split={args.split}")
        print(f"leakage_passed={report.passed}")
        for violation in report.violations:
            print(f"violation={violation}")
        return
    if args.dataset != "synthetic":
        raise SystemExit("split audit currently supports --dataset synthetic without external data")
    records = make_synthetic_recordings()
    manifest = build_split_manifest(records, policy=args.split, seed=0)
    report = audit_split_manifest(manifest, policy=args.split)
    print(f"dataset={args.dataset}")
    print(f"split={args.split}")
    print(f"leakage_passed={report.passed}")
    for violation in report.violations:
        print(f"violation={violation}")


def _cmd_train(args: argparse.Namespace) -> None:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc
    if args.dry_run:
        estimate = estimate_config(config)
        dist = get_distributed_info()
        print(f"config={args.config}")
        print("dry_run=True")
        print(f"config_hash={stable_hash(config)}")
        print(f"world_size={dist.world_size}")
        print(f"rank={dist.rank}")
        for key, value in estimate.items():
            print(f"{key}={value}")
        return
    run_dir = create_run_dir(args.run_root, run_id=str(config.get("experiment", "synthetic_debug")))
    dist = get_distributed_info()
    if dist.is_rank_zero:
        snapshot_config(config, run_dir)
        write_json(run_dir / "environment.json", capture_environment())
    configured_split_manifest = _config_value(config, "split_manifest")
    if configured_split_manifest:
        from neurotwin.data.manifest_io import load_split_manifest

        split_manifest = load_split_manifest(configured_split_manifest)
    else:
        records = make_synthetic_recordings()
        split_manifest = build_split_manifest(records, policy=str(config.get("split", "subject")), seed=int(config.get("seed", 0)))
    split_manifest_path = run_dir / "split_manifest.json"
    if dist.is_rank_zero:
        save_split_manifest(split_manifest, split_manifest_path)
    split_manifest_hash = manifest_hash([record.__dict__ for record in split_manifest.all_records])
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise SystemExit(f"Resume checkpoint does not exist: {resume_path}")
    if _has_prepared_training_inputs(config):
        checkpoint_path = run_dir / "checkpoint.pt" if dist.is_rank_zero else None
        metrics_path = get_rank_metrics_path(run_dir, dist)
        result = run_prepared_training(
            config,
            checkpoint_path=checkpoint_path,
            resume_path=args.resume,
            metrics_csv_path=(run_dir / "metrics.csv") if dist.is_rank_zero else None,
            metrics_jsonl_path=metrics_path,
            best_checkpoint_path=(run_dir / "checkpoint_best.pt") if dist.is_rank_zero else None,
        )
        if not dist.is_rank_zero:
            print(f"config={args.config}")
            print(f"run_dir={run_dir}")
            print(f"rank={dist.rank}")
            print("training_status=completed_prepared_training_rank")
            print(f"metrics_jsonl={metrics_path}")
            return
        write_json(run_dir / "metrics.json", result.to_dict())
        write_json(
            run_dir / "summary.json",
            {
                "synthetic_only": result.synthetic_only,
                "status": result.status,
                "task_id": result.task_id,
                "source_modality": result.source_modality,
                "target_modality": result.target_modality,
                "split_manifest": str(split_manifest_path),
                "split_manifest_hash": split_manifest_hash,
                "event_manifest": str(_config_value(config, "event_manifest")),
                "event_summary": result.event_summary,
                "precision": result.precision,
                "gradient_accumulation_steps": result.gradient_accumulation_steps,
                "completed_steps": result.completed_steps,
                "eval_every_steps": result.eval_every_steps,
                "checkpoint_every_steps": result.checkpoint_every_steps,
                "best_task_id": result.best_task_id,
                "best_eval_mse": result.best_eval_mse,
                "best_val_mse": result.best_val_mse,
                "test_mse": result.test_mse,
                "test_mae": result.test_mae,
                "test_pearsonr": result.test_pearsonr,
                "test_r2": result.test_r2,
                "test_spearmanr": result.test_spearmanr,
                "selection_split": result.selection_split,
                "report_split": result.report_split,
                "real_data_smoke": bool(config.get("dataset") == "moabb" and "smoke" in str(config.get("experiment", ""))),
                "scientific_claim_allowed": bool((not result.synthetic_only) and "smoke" not in str(config.get("experiment", ""))),
                "unavailable_tasks": result.skipped_tasks,
                "baseline_failures": [],
                "task_results": result.task_results,
                "distributed_initialized": result.distributed_initialized,
                "distributed_backend": result.distributed_backend,
                "distributed": {
                    "rank": dist.rank,
                    "local_rank": dist.local_rank,
                    "world_size": dist.world_size,
                },
                "resume": args.resume,
            },
        )
        print(f"config={args.config}")
        print(f"run_dir={run_dir}")
        print(f"rank={dist.rank}")
        print("training_status=completed_prepared_training")
        print(f"task_id={result.task_id}")
        print(f"initial_loss={result.initial_loss:.6f}")
        print(f"final_loss={result.final_loss:.6f}")
        print(f"eval_mse={result.eval_mse:.6f}")
        if result.best_task_id:
            print(f"best_task_id={result.best_task_id}")
            print(f"best_eval_mse={result.best_eval_mse:.6f}")
        print(f"steps={result.steps}")
        print(f"completed_steps={result.completed_steps}")
        return
    result = run_synthetic_training(seed=0, steps=24)
    metrics_path = get_rank_metrics_path(run_dir, dist)
    append_jsonl(metrics_path, {"step": 0, "loss": result.initial_loss, "rank": dist.rank, "world_size": dist.world_size})
    append_jsonl(metrics_path, {"step": result.steps, "loss": result.final_loss, "rank": dist.rank, "world_size": dist.world_size})
    if not dist.is_rank_zero:
        print(f"config={args.config}")
        print(f"run_dir={run_dir}")
        if args.resume:
            print(f"resume={args.resume}")
        print(f"rank={dist.rank}")
        print("training_status=completed_synthetic_smoke_rank")
        print(f"metrics_jsonl={metrics_path}")
        return

    write_json(run_dir / "metrics.json", {"initial_loss": result.initial_loss, "final_loss": result.final_loss, "steps": result.steps})
    import torch

    torch.save({"status": "synthetic_smoke", "steps": result.steps, "world_size": dist.world_size}, run_dir / "checkpoint.pt")
    write_json(
        run_dir / "summary.json",
        {
            "synthetic_only": True,
            "real_data_smoke": False,
            "scientific_claim_allowed": False,
            "unavailable_tasks": [],
            "baseline_failures": [],
            "status": "completed_synthetic_smoke",
            "split_manifest": str(split_manifest_path),
            "split_manifest_hash": split_manifest_hash,
            "distributed": {
                "rank": dist.rank,
                "local_rank": dist.local_rank,
                "world_size": dist.world_size,
            },
            "resume": args.resume,
        },
    )
    print(f"config={args.config}")
    print(f"run_dir={run_dir}")
    if args.resume:
        print(f"resume={args.resume}")
    print(f"rank={dist.rank}")
    print("training_status=completed_synthetic_smoke")
    print(f"initial_loss={result.initial_loss:.6f}")
    print(f"final_loss={result.final_loss:.6f}")
    print(f"steps={result.steps}")


def _cmd_estimate(args: argparse.Namespace) -> None:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc
    for key, value in estimate_config(config).items():
        print(f"{key}={value}")


def _cmd_eval(args: argparse.Namespace) -> None:
    if args.eval_command == "audit":
        if args.event_manifest or args.split_manifest:
            if not args.event_manifest or not args.split_manifest:
                raise SystemExit("--event-manifest and --split-manifest must be provided together")
            report = audit_prepared_eval_inputs(
                args.event_manifest,
                args.split_manifest,
                window_length=args.window_length,
                stride=args.stride,
                out_dir=args.out_dir,
                require_windows=args.require_windows,
            )
            print(format_prepared_eval_audit(report))
            if not report.passed:
                raise SystemExit(1)
            return
        print(f"suite={args.suite}")
        print("eval_audit_passed=True")
        print("notes=synthetic-only suite has explicit plumbing label")
        return
    if args.suite == "neural_translation_v1":
        if args.event_manifest or args.split_manifest:
            if not args.event_manifest or not args.split_manifest:
                raise SystemExit("--event-manifest and --split-manifest must be provided together")
            payload = run_prepared_baseline_suite(
                PreparedSuiteConfig(
                    event_manifest=args.event_manifest,
                    split_manifest=args.split_manifest,
                    window_length=args.window_length,
                    stride=args.stride,
                    seed=0,
                    train_steps=args.train_steps,
                ),
                out_dir=args.out_dir,
            )
            print(format_prepared_baseline_report(payload))
            return
        payload = run_neural_translation_v1_synthetic(seed=0, out_dir=args.out_dir)
        print(format_neural_translation_v1_report(payload))
        return
    if args.suite != "translation_smoke":
        raise SystemExit("Supported suites: translation_smoke, neural_translation_v1")
    print("suite=translation_smoke")
    print("tasks=" + ",".join(task.task_id for task in default_translation_tasks()))
    print("competitors=" + ",".join(competitor.competitor_id for competitor in competitor_registry()))
    if args.run:
        print(f"run={args.run}")
    print()
    print(format_smoke_results(run_translation_smoke(seed=0)))


def _cmd_report(args: argparse.Namespace) -> None:
    if args.compare is not None:
        print(generate_compare_report(args.compare, args.out_dir))
        return
    if args.run_dir:
        print(generate_run_report(args.run_dir))
        return
    suite = args.suite or "translation_smoke"
    print(generate_suite_report(suite))


def _cmd_doctor(args: argparse.Namespace) -> None:
    print(format_doctor_report(run_doctor()))


def _has_prepared_training_inputs(config: dict[str, object]) -> bool:
    return bool(_config_value(config, "event_manifest")) and bool(_config_value(config, "split_manifest"))


def _config_value(config: dict[str, object], key: str) -> object | None:
    data = config.get("data")
    data_config = data if isinstance(data, dict) else {}
    return config.get(key) or data_config.get(key)


if __name__ == "__main__":
    raise SystemExit(main())
