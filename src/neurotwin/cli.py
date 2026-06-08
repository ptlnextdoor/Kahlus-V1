from __future__ import annotations

import argparse
from pathlib import Path
import sys

from neurotwin.adapters.synthetic import make_synthetic_recordings
from neurotwin.config import ConfigError, load_config
from neurotwin.config_types import as_prepared_training_config_input
from neurotwin.data.audit import audit_split_manifest
from neurotwin.data.command import (
    DataAuditConfig,
    DataPrepareConfig,
    run_data_audit_command,
    run_data_prepare_command,
)
from neurotwin.data.split_manifest import build_split_manifest
from neurotwin.doctor import format_doctor_report, run_doctor
from neurotwin.eval.command import EvalCommandConfig, run_eval_command
from neurotwin.benchmarks.reports import generate_compare_report, generate_model_card_report, generate_run_report, generate_suite_report
from neurotwin.runtime.command_result import CommandResult
from neurotwin.runtime.estimate import estimate_config
from neurotwin.runtime.preflight import (
    format_cluster_materialize_config,
    format_cluster_preflight,
    materialize_cluster_config,
    run_cluster_preflight,
)
from neurotwin.runtime.smoke_command import DataSmokeConfig, run_data_smoke_command
from neurotwin.training.command import TrainingCommandConfig, run_training_command


def main(argv: list[str] | None = None) -> int:
    parse_argv = _normalize_eval_argv(list(sys.argv[1:] if argv is None else argv))
    parser = argparse.ArgumentParser(prog="nt", description="NeuroTwin research benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check runtime, GPU, optional dependencies, and dataset roots")
    doctor.set_defaults(func=_cmd_doctor)

    data_parser = subparsers.add_parser("data", help="Dataset preparation commands")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    prepare = data_subparsers.add_parser("prepare", help="Prepare manifests and leakage-proof splits")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--split", required=True, choices=("subject", "session", "site", "dataset", "time", "official"))
    prepare.add_argument("--root", default=None)
    prepare.add_argument("--out-dir", default=None)
    prepare.add_argument("--moabb-dataset", default="BNCI2014_001")
    prepare.add_argument("--moabb-paradigm", default="LeftRightImagery")
    prepare.add_argument("--subjects", nargs="*", type=int, default=None)
    prepare.add_argument("--max-trials", type=int, default=None)
    prepare.add_argument("--sampling-rate", type=float, default=None)
    prepare.add_argument("--window-length", type=int, default=128)
    prepare.add_argument("--stride", type=int, default=128)
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

    run = subparsers.add_parser("run", help="Run lifecycle commands")
    run_subparsers = run.add_subparsers(dest="run_command", required=True)
    finalize = run_subparsers.add_parser("finalize", help="Finalize a completed run evidence bundle")
    finalize.add_argument("--run-dir", required=True)
    finalize.add_argument("--paper-mode-dir", default=None)
    finalize.add_argument("--event-manifest", default=None)
    finalize.add_argument("--split-manifest", default=None)
    finalize.add_argument("--window-length", type=int, default=128)
    finalize.add_argument("--stride", type=int, default=128)
    finalize.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    finalize.set_defaults(func=_cmd_run)

    eval_parser = subparsers.add_parser("eval", help="Run evaluation suites and paper diagnostics")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_suite = eval_subparsers.add_parser("suite", help="Run an evaluation suite")
    _add_eval_suite_args(eval_suite)
    eval_audit = eval_subparsers.add_parser("audit", help="Audit prepared evaluation inputs")
    _add_eval_audit_args(eval_audit)
    eval_leakage = eval_subparsers.add_parser("leakage-demo", help="Compare bad segment splits against held-out subject splits")
    _add_eval_demo_args(eval_leakage)
    eval_identity = eval_subparsers.add_parser("identity-probe", help="Quantify subject identity leakage risk from windows")
    _add_eval_demo_args(eval_identity)
    eval_parser.set_defaults(func=_cmd_eval)

    report = subparsers.add_parser("report", help="Generate a reproducible benchmark report")
    report.add_argument("report_command", nargs="?", choices=("model-card", "evidence-gate"))
    report.add_argument("--suite", default=None)
    report.add_argument("--run-dir", default=None)
    report.add_argument("--compare", nargs="*", default=None)
    report.add_argument("--out-dir", default=None)
    report.add_argument("--out", default=None)
    report.set_defaults(func=_cmd_report)

    cluster = subparsers.add_parser("cluster", help="Cluster launch safety checks")
    cluster_subparsers = cluster.add_subparsers(dest="cluster_command", required=True)
    preflight = cluster_subparsers.add_parser("preflight", help="Validate cluster launch inputs")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--run-root", required=True)
    preflight.add_argument("--require-cuda", action="store_true")
    preflight.add_argument("--require-prepared-windows", action="store_true")
    preflight.add_argument("--expect-window-count", type=int, default=None)
    preflight.add_argument("--expect-split-windows", default=None)
    preflight.set_defaults(func=_cmd_cluster_preflight)
    materialize_config = cluster_subparsers.add_parser(
        "materialize-config",
        help="Write a cluster config with absolute prepared-manifest paths",
    )
    materialize_config.add_argument("--template", required=True)
    materialize_config.add_argument("--prepared-root", required=True)
    materialize_config.add_argument("--out", required=True)
    materialize_config.add_argument("--allow-tracked-output", action="store_true")
    materialize_config.set_defaults(func=_cmd_cluster_materialize_config)

    args = parser.parse_args(parse_argv)
    args._argv = list(sys.argv if argv is None else ["nt", *parse_argv])
    args.func(args)
    return 0


def _normalize_eval_argv(argv: list[str]) -> list[str]:
    if not argv or argv[0] != "eval":
        return argv
    eval_subcommands = {"suite", "audit", "leakage-demo", "identity-probe"}
    rest = argv[1:]
    if not rest:
        return ["eval", "suite"]
    if rest[0] in eval_subcommands:
        return argv
    if any(token in eval_subcommands for token in rest):
        return argv
    return ["eval", "suite", *rest]


def _emit_command_result(result: CommandResult) -> None:
    if result.output:
        print(result.output)
    if result.exit_code:
        if result.error:
            print(result.error, file=sys.stderr)
        raise SystemExit(result.exit_code)


def _add_eval_manifest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--event-manifest", default=None)
    parser.add_argument("--split-manifest", default=None)


def _add_eval_window_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--window-length", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)


def _add_eval_suite_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", default=None)
    parser.add_argument("--suite", default="translation_smoke")
    parser.add_argument("--out-dir", default=None)
    _add_eval_manifest_args(parser)
    _add_eval_window_args(parser)
    parser.add_argument("--train-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--require-windows", action="store_true")
    parser.add_argument("--paper-mode", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--gate-mode", choices=("evidence", "strict"), default="evidence")
    parser.set_defaults(
        dataset="synthetic",
        eval_command=None,
    )


def _add_eval_audit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", default="translation_smoke")
    parser.add_argument("--out-dir", default=None)
    _add_eval_manifest_args(parser)
    _add_eval_window_args(parser)
    parser.add_argument("--require-windows", action="store_true")
    parser.set_defaults(
        dataset="synthetic",
        run=None,
        train_steps=5,
        seed=0,
        seeds=None,
        paper_mode=False,
        require_pass=False,
        gate_mode="evidence",
    )


def _add_eval_demo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--task", default="future_state_forecasting")
    parser.add_argument("--out-dir", default=None)
    _add_eval_manifest_args(parser)
    _add_eval_window_args(parser)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.set_defaults(
        suite="translation_smoke",
        run=None,
        require_windows=False,
        paper_mode=False,
    )


def _cmd_data_prepare(args: argparse.Namespace) -> None:
    _emit_command_result(
        run_data_prepare_command(
            DataPrepareConfig(
                dataset=args.dataset,
                split=args.split,
                root=args.root,
                out_dir=args.out_dir,
                moabb_dataset=args.moabb_dataset,
                moabb_paradigm=args.moabb_paradigm,
                subjects=args.subjects,
                max_trials=args.max_trials,
                sampling_rate=args.sampling_rate,
                window_length=args.window_length,
                stride=args.stride,
            )
        )
    )


def _cmd_data_smoke(args: argparse.Namespace) -> None:
    _emit_command_result(
        run_data_smoke_command(
            DataSmokeConfig(
                dataset=args.dataset,
                split=args.split,
                out_dir=args.out_dir,
                moabb_dataset=args.moabb_dataset,
                moabb_paradigm=args.moabb_paradigm,
                subjects=args.subjects,
                max_trials=args.max_trials,
                sampling_rate=args.sampling_rate,
                window_length=args.window_length,
                stride=args.stride,
                train_steps=args.train_steps,
                seed=args.seed,
                require_windows=args.require_windows,
            )
        )
    )


def _cmd_data_audit(args: argparse.Namespace) -> None:
    _emit_command_result(run_data_audit_command(DataAuditConfig(dataset=args.dataset, root=args.root)))


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
    _emit_command_result(
        run_training_command(
            TrainingCommandConfig(
                config_path=args.config,
                dry_run=args.dry_run,
                run_root=args.run_root,
                resume=args.resume,
                argv=getattr(args, "_argv", None),
            )
        )
    )


def _cmd_estimate(args: argparse.Namespace) -> None:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc
    for key, value in estimate_config(as_prepared_training_config_input(config)).items():
        print(f"{key}={value}")


def _cmd_eval(args: argparse.Namespace) -> None:
    eval_command = None if getattr(args, "eval_command", None) == "suite" else getattr(args, "eval_command", None)
    result = run_eval_command(
        EvalCommandConfig(
            eval_command=eval_command,
            suite=getattr(args, "suite", "translation_smoke"),
            dataset=getattr(args, "dataset", "synthetic"),
            task=getattr(args, "task", "future_state_forecasting"),
            run=getattr(args, "run", None),
            out_dir=getattr(args, "out_dir", None),
            event_manifest=getattr(args, "event_manifest", None),
            split_manifest=getattr(args, "split_manifest", None),
            window_length=getattr(args, "window_length", 8),
            stride=getattr(args, "stride", 8),
            train_steps=getattr(args, "train_steps", 5),
            seed=getattr(args, "seed", 0),
            seeds=tuple(args.seeds) if getattr(args, "seeds", None) is not None else None,
            require_windows=getattr(args, "require_windows", False),
            paper_mode=getattr(args, "paper_mode", False),
            require_pass=getattr(args, "require_pass", False),
            gate_mode=getattr(args, "gate_mode", "evidence"),
        )
    )
    if result.output:
        print(result.output)
    if result.exit_code:
        if result.error:
            print(result.error, file=sys.stderr)
        raise SystemExit(result.exit_code)


def _cmd_run(args: argparse.Namespace) -> None:
    if args.run_command != "finalize":
        raise SystemExit("supported run commands: finalize")
    from neurotwin.reports.finalize import RunFinalizeConfig, finalize_run

    try:
        payload = finalize_run(
            RunFinalizeConfig(
                run_dir=args.run_dir,
                paper_mode_dir=args.paper_mode_dir,
                event_manifest=args.event_manifest,
                split_manifest=args.split_manifest,
                window_length=args.window_length,
                stride=args.stride,
                seeds=tuple(args.seeds or (0, 1, 2)),
            )
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"finalize_run_dir={payload.get('run_dir')}")
    print(f"paper_mode_artifacts_copied={payload.get('paper_mode_artifacts_copied')}")
    print(f"eval_audit_copied={payload.get('eval_audit_copied')}")
    diagnostics = payload.get("diagnostics", {})
    if isinstance(diagnostics, dict):
        for name, ran in diagnostics.items():
            print(f"{name}_ran={ran}")
    print(f"run_report={payload.get('run_report')}")
    print(f"evidence_gate={payload.get('evidence_gate')}")
    print(f"evidence_gate_passed={payload.get('evidence_gate_passed')}")
    print(f"model_card={payload.get('model_card')}")


def _cmd_report(args: argparse.Namespace) -> None:
    if args.report_command == "model-card":
        if not args.run_dir:
            raise SystemExit("report model-card requires --run-dir")
        try:
            print(generate_model_card_report(args.run_dir, args.out))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        return
    if args.report_command == "evidence-gate":
        if not args.run_dir:
            raise SystemExit("report evidence-gate requires --run-dir")
        from neurotwin.reports.evidence_gate import write_final_prepared_evidence_gate

        try:
            gate = write_final_prepared_evidence_gate(args.run_dir)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"evidence_gate={Path(args.run_dir) / 'evidence_gate.json'}")
        print(f"evidence_gate_passed={gate.get('passed', False)}")
        print(f"evidence_gate_stage={gate.get('stage', 'unknown')}")
        failures = gate.get("failures", [])
        if isinstance(failures, list):
            for failure in failures:
                print(f"evidence_gate_failure={failure}")
        return
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


def _cmd_cluster_preflight(args: argparse.Namespace) -> None:
    try:
        report = run_cluster_preflight(
            args.config,
            args.run_root,
            require_cuda=args.require_cuda,
            require_prepared_windows=args.require_prepared_windows,
            expect_window_count=args.expect_window_count,
            expect_split_windows=_parse_split_windows(args.expect_split_windows),
        )
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc
    print(format_cluster_preflight(report))
    if not report.passed:
        raise SystemExit(1)


def _cmd_cluster_materialize_config(args: argparse.Namespace) -> None:
    try:
        report = materialize_cluster_config(
            args.template,
            args.prepared_root,
            args.out,
            allow_tracked_output=args.allow_tracked_output,
        )
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc
    print(format_cluster_materialize_config(report))
    if not report.passed:
        raise SystemExit(1)


def _parse_split_windows(value: str | None) -> dict[str, int] | None:
    if value is None:
        return None
    parsed: dict[str, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise SystemExit("--expect-split-windows must use split:count entries")
        split_name, raw_count = item.split(":", 1)
        split_name = split_name.strip()
        if split_name not in {"train", "val", "test"}:
            raise SystemExit("--expect-split-windows only supports train, val, and test")
        try:
            parsed[split_name] = int(raw_count)
        except ValueError as exc:
            raise SystemExit(f"invalid window count for split {split_name}: {raw_count}") from exc
    if not parsed:
        raise SystemExit("--expect-split-windows must include at least one split:count entry")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
