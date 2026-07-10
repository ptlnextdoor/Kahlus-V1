from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurotwin.benchmarks.prepared_suite import format_prepared_baseline_report, run_prepared_baseline_suite
from neurotwin.benchmarks.nfc_suite import format_nfc_synthetic_report, nfc_falsification_status, run_nfc_synthetic_suite
from neurotwin.benchmarks.registry import competitor_registry
from neurotwin.benchmarks.smoke import format_smoke_results, run_translation_smoke
from neurotwin.benchmarks.suite import format_neural_translation_v1_report, run_neural_translation_v1_synthetic
from neurotwin.benchmarks.task_specs import default_translation_tasks
from neurotwin.eval.audit import audit_prepared_eval_inputs, format_prepared_eval_audit
from neurotwin.eval.paper_gate import format_paper_mode_gate, validate_paper_mode_payload
from neurotwin.eval.paper_demos import (
    PaperDemoConfig,
    format_identity_probe,
    format_leakage_demo,
    run_identity_probe,
    run_leakage_demo,
)
from neurotwin.eval.prepared_paper_mode import run_prepared_baseline_suite_multi_seed, write_prepared_paper_mode_artifacts
from neurotwin.data.prepared_tasks import PreparedSuiteConfig
from neurotwin.data.forecast_contract import ForecastTaskSpec
from neurotwin.repro import write_json


MANIFEST_PAIR_ERROR = "--event-manifest and --split-manifest must be provided together"


@dataclass(frozen=True)
class EvalCommandConfig:
    eval_command: str | None = None
    suite: str = "translation_smoke"
    dataset: str = "synthetic"
    task: str = "future_state_forecasting"
    run: str | None = None
    out_dir: str | Path | None = None
    event_manifest: str | Path | None = None
    split_manifest: str | Path | None = None
    window_length: int = 8
    stride: int = 8
    train_steps: int = 5
    seed: int = 0
    seeds: tuple[int, ...] | None = None
    max_windows_per_split: int | None = None
    baseline_model_ids: tuple[str, ...] | None = None
    require_windows: bool = False
    paper_mode: bool = False
    require_pass: bool = False
    gate_mode: str = "evidence"
    forecast_task: ForecastTaskSpec | None = None


@dataclass(frozen=True)
class EvalCommandResult:
    output: str
    exit_code: int = 0
    payload: dict[str, Any] | None = None
    error: str | None = None


def run_eval_command(config: EvalCommandConfig) -> EvalCommandResult:
    if config.eval_command == "audit":
        return _run_audit_command(config)
    if config.eval_command == "leakage-demo":
        return _run_leakage_demo_command(config)
    if config.eval_command == "identity-probe":
        return _run_identity_probe_command(config)
    if config.suite == "neural_translation_v1":
        return _run_neural_translation_v1_command(config)
    if config.suite == "nfc_synthetic":
        return _run_nfc_synthetic_command(config)
    if config.suite != "translation_smoke":
        return EvalCommandResult(
            output="",
            exit_code=1,
            error="Supported suites: translation_smoke, neural_translation_v1, nfc_synthetic",
        )
    return _run_translation_smoke_command(config)


def _run_leakage_demo_command(config: EvalCommandConfig) -> EvalCommandResult:
    try:
        payload = run_leakage_demo(_paper_demo_config(config))
    except ValueError as exc:
        return EvalCommandResult(output="", exit_code=1, error=str(exc))
    return EvalCommandResult(
        output=format_leakage_demo(payload),
        exit_code=_paper_demo_exit_code(payload),
        error=_paper_demo_error(payload),
        payload=payload,
    )


def _run_identity_probe_command(config: EvalCommandConfig) -> EvalCommandResult:
    try:
        payload = run_identity_probe(_paper_demo_config(config))
    except ValueError as exc:
        return EvalCommandResult(output="", exit_code=1, error=str(exc))
    return EvalCommandResult(
        output=format_identity_probe(payload),
        exit_code=_paper_demo_exit_code(payload),
        error=_paper_demo_error(payload),
        payload=payload,
    )


def _run_audit_command(config: EvalCommandConfig) -> EvalCommandResult:
    if config.event_manifest or config.split_manifest:
        manifest_paths = _manifest_paths(config)
        if manifest_paths is None:
            return EvalCommandResult(output="", exit_code=1, error=MANIFEST_PAIR_ERROR)
        event_manifest, split_manifest = manifest_paths
        report = audit_prepared_eval_inputs(
            event_manifest,
            split_manifest,
            window_length=config.window_length,
            stride=config.stride,
            out_dir=config.out_dir,
            require_windows=config.require_windows,
        )
        return EvalCommandResult(
            output=format_prepared_eval_audit(report),
            exit_code=0 if report.passed else 1,
            payload={"eval_audit": report.to_dict()},
        )
    return EvalCommandResult(
        output="\n".join(
            (
                f"suite={config.suite}",
                "eval_audit_passed=True",
                "notes=synthetic-only suite has explicit plumbing label",
            )
        )
    )


def _run_neural_translation_v1_command(config: EvalCommandConfig) -> EvalCommandResult:
    if config.event_manifest or config.split_manifest:
        return run_prepared_eval_command(config)
    payload = run_neural_translation_v1_synthetic(seed=config.seed, out_dir=config.out_dir)
    return EvalCommandResult(output=format_neural_translation_v1_report(payload), payload=payload)


def _run_nfc_synthetic_command(config: EvalCommandConfig) -> EvalCommandResult:
    if config.event_manifest or config.split_manifest:
        return EvalCommandResult(output="", exit_code=1, error="nfc_synthetic is synthetic-only and does not accept prepared manifests")
    payload = run_nfc_synthetic_suite(seed=config.seed, seeds=config.seeds, train_steps=config.train_steps, out_dir=config.out_dir)
    status = nfc_falsification_status(payload)
    if config.require_pass or config.gate_mode == "strict":
        if status != "passed":
            return EvalCommandResult(
                output=format_nfc_synthetic_report(payload),
                exit_code=2,
                error=f"nfc_synthetic falsification status={status}",
                payload=payload,
            )
    return EvalCommandResult(output=format_nfc_synthetic_report(payload), payload=payload)


def run_prepared_eval_command(config: EvalCommandConfig) -> EvalCommandResult:
    manifest_paths = _manifest_paths(config)
    if manifest_paths is None:
        return EvalCommandResult(output="", exit_code=1, error=MANIFEST_PAIR_ERROR)
    event_manifest, split_manifest = manifest_paths

    audit = None
    if config.paper_mode:
        audit = audit_prepared_eval_inputs(
            event_manifest,
            split_manifest,
            window_length=config.window_length,
            stride=config.stride,
            out_dir=config.out_dir,
            require_windows=True,
        )
        if not audit.passed:
            payload = {
                "eval_audit": audit.to_dict(),
                "aggregate": {"selection_metric": "mse", "higher_is_better": False, "aggregate_rank": []},
                "tasks": {},
                "seed": config.seed,
                "seeds": [config.seed],
            }
            gate = validate_paper_mode_payload(payload, audit_report=audit, require_ci=True)
            payload["paper_mode_gate"] = gate.to_dict()
            payload["paper_mode_contract"] = {
                "required_seeds": list(gate.required_seeds),
                "observed_seeds": list(gate.observed_seeds),
                "require_ci": gate.require_ci,
                "gate_status": "failed",
            }
            if config.out_dir:
                out_dir = Path(config.out_dir)
                write_json(out_dir / "prepared_baseline_suite.json", payload)
                write_json(out_dir / "paper_mode_gate.json", gate.to_dict())
            return EvalCommandResult(
                output="\n".join((format_prepared_eval_audit(audit), format_paper_mode_gate(gate))),
                exit_code=1,
                payload=payload,
            )
        if config.seeds:
            payload = run_prepared_baseline_suite_multi_seed(
                _prepared_suite_config(
                    config,
                    seed=config.seeds[0],
                    event_manifest=event_manifest,
                    split_manifest=split_manifest,
                ),
                seeds=config.seeds,
                out_dir=None,
            )
            payload["eval_audit"] = audit.to_dict()
            gate = validate_paper_mode_payload(payload, audit_report=audit, require_ci=True)
            payload["paper_mode_gate"] = gate.to_dict()
            payload["paper_mode_contract"] = {
                "required_seeds": list(gate.required_seeds),
                "observed_seeds": list(gate.observed_seeds),
                "require_ci": gate.require_ci,
                "gate_status": "passed" if gate.passed else "failed",
            }
            if config.out_dir:
                write_prepared_paper_mode_artifacts(payload, config.out_dir, gate)
            gate_payload = payload.get("paper_mode_gate", {})
            gate_passed = bool(gate_payload.get("passed")) if isinstance(gate_payload, dict) else False
            return EvalCommandResult(
                output="\n".join((format_prepared_eval_audit(audit), format_prepared_baseline_report(payload), format_paper_mode_gate(gate))),
                exit_code=0 if gate_passed else 1,
                payload=payload,
            )

    payload = run_prepared_baseline_suite(
        _prepared_suite_config(
            config,
            seed=config.seed,
            event_manifest=event_manifest,
            split_manifest=split_manifest,
        ),
        out_dir=config.out_dir,
    )
    if config.paper_mode:
        if audit is None:
            return EvalCommandResult(output="", exit_code=1, error="paper mode audit did not run")
        payload["eval_audit"] = audit.to_dict()
        gate = validate_paper_mode_payload(
            payload,
            audit_report=audit,
            require_ci=True,
        )
        payload["paper_mode_gate"] = gate.to_dict()
        payload["paper_mode_contract"] = {
            "required_seeds": list(gate.required_seeds),
            "observed_seeds": list(gate.observed_seeds),
            "require_ci": gate.require_ci,
            "gate_status": "passed" if gate.passed else "failed",
        }
        if config.out_dir:
            out_dir = Path(config.out_dir)
            write_json(out_dir / "prepared_baseline_suite.json", payload)
            write_json(out_dir / "paper_mode_gate.json", gate.to_dict())
        return EvalCommandResult(
            output="\n".join((format_prepared_eval_audit(audit), format_prepared_baseline_report(payload), format_paper_mode_gate(gate))),
            exit_code=0 if gate.passed else 1,
            payload=payload,
        )
    return EvalCommandResult(output=format_prepared_baseline_report(payload), payload=payload)


def _run_translation_smoke_command(config: EvalCommandConfig) -> EvalCommandResult:
    lines = [
        "suite=translation_smoke",
        "tasks=" + ",".join(task.task_id for task in default_translation_tasks()),
        "competitors=" + ",".join(competitor.competitor_id for competitor in competitor_registry()),
    ]
    if config.run:
        lines.append(f"run={config.run}")
    lines.append("")
    lines.append(format_smoke_results(run_translation_smoke(seed=config.seed)))
    return EvalCommandResult(output="\n".join(lines))


def _prepared_suite_config(
    config: EvalCommandConfig,
    seed: int,
    event_manifest: str | Path,
    split_manifest: str | Path,
) -> PreparedSuiteConfig:
    return PreparedSuiteConfig(
        event_manifest=event_manifest,
        split_manifest=split_manifest,
        window_length=config.window_length,
        stride=config.stride,
        seed=seed,
        train_steps=config.train_steps,
        max_windows_per_split=config.max_windows_per_split,
        model_ids=config.baseline_model_ids,
        forecast_task=config.forecast_task,
    )


def _paper_demo_config(config: EvalCommandConfig) -> PaperDemoConfig:
    return PaperDemoConfig(
        dataset=config.dataset,
        task=config.task,
        event_manifest=config.event_manifest,
        split_manifest=config.split_manifest,
        out_dir=config.out_dir,
        window_length=config.window_length,
        stride=config.stride,
        seed=config.seed,
        seeds=config.seeds,
        train_steps=config.train_steps,
    )


def _paper_demo_exit_code(payload: dict[str, Any]) -> int:
    return 1 if _paper_demo_failures(payload) else 0


def _paper_demo_error(payload: dict[str, Any]) -> str | None:
    failures = _paper_demo_failures(payload)
    if not failures:
        return None
    return "; ".join(f"seed {failure.get('seed')}: {failure.get('error')}" for failure in failures)


def _paper_demo_failures(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("seed_results", [])
    if not isinstance(results, list):
        return []
    return [result for result in results if isinstance(result, dict) and result.get("status") != "completed"]


def _manifest_paths(config: EvalCommandConfig) -> tuple[str | Path, str | Path] | None:
    if config.event_manifest is None or config.split_manifest is None:
        return None
    return config.event_manifest, config.split_manifest
