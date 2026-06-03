from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurotwin.adapters.synthetic import make_synthetic_recordings
from neurotwin.config import ConfigError, load_config
from neurotwin.config_types import PreparedTrainingConfigInput, as_prepared_training_config_input
from neurotwin.data.manifest_io import save_split_manifest
from neurotwin.data.split_manifest import build_split_manifest
from neurotwin.eval.audit import audit_prepared_eval_inputs
from neurotwin.repro import (
    append_jsonl,
    capture_environment,
    checkpoint_manifest,
    create_run_dir,
    manifest_hash,
    snapshot_config,
    stable_hash,
    write_json,
)
from neurotwin.runtime.command_result import CommandResult
from neurotwin.runtime.distributed import get_distributed_info, get_rank_metrics_path
from neurotwin.runtime.estimate import estimate_config
from neurotwin.training.prepared import PreparedTrainingConfig, PreparedTrainingRunPaths, run_prepared_training
from neurotwin.training.smoke import run_synthetic_training


@dataclass(frozen=True)
class TrainingCommandConfig:
    config_path: str
    dry_run: bool = False
    run_root: str = "runs"
    resume: str | None = None
    argv: list[str] | None = None


def run_training_command(command: TrainingCommandConfig) -> CommandResult:
    try:
        config = load_config(command.config_path)
    except ConfigError as exc:
        return CommandResult("", exit_code=1, error=str(exc))
    typed_config = as_prepared_training_config_input(config)
    if command.dry_run:
        return _dry_run_result(command.config_path, typed_config)

    run_dir = create_run_dir(command.run_root, run_id=str(config.get("experiment", "synthetic_debug")))
    dist = get_distributed_info()
    environment = capture_environment(argv=command.argv)
    if dist.is_rank_zero:
        snapshot_config(config, run_dir)
        write_json(run_dir / "environment.json", environment)
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
    if command.resume:
        resume_path = Path(command.resume)
        if not resume_path.exists():
            return CommandResult("", exit_code=1, error=f"Resume checkpoint does not exist: {resume_path}")
    if _has_prepared_training_inputs(typed_config):
        return _run_prepared_training_command(
            command,
            typed_config,
            run_dir,
            dist,
            environment,
            split_manifest_path,
            split_manifest_hash,
        )
    return _run_synthetic_training_command(command, typed_config, run_dir, dist, environment, split_manifest_path, split_manifest_hash)


def _dry_run_result(config_path: str, config: PreparedTrainingConfigInput) -> CommandResult:
    estimate = estimate_config(config)
    dist = get_distributed_info()
    lines = [
        f"config={config_path}",
        "dry_run=True",
        f"config_hash={stable_hash(config)}",
        f"world_size={dist.world_size}",
        f"rank={dist.rank}",
    ]
    lines.extend(f"{key}={value}" for key, value in estimate.items())
    return CommandResult("\n".join(lines))


def _run_prepared_training_command(
    command: TrainingCommandConfig,
    config: PreparedTrainingConfigInput,
    run_dir: Path,
    dist: object,
    environment: dict[str, object],
    split_manifest_path: Path,
    split_manifest_hash: str,
) -> CommandResult:
    checkpoint_path = run_dir / "checkpoint.pt" if dist.is_rank_zero else None
    metrics_path = get_rank_metrics_path(run_dir, dist)
    prepared_config = PreparedTrainingConfig.from_mapping(config)
    result = run_prepared_training(
        prepared_config,
        paths=PreparedTrainingRunPaths(
            checkpoint_path=checkpoint_path,
            resume_path=command.resume,
            metrics_csv_path=(run_dir / "metrics.csv") if dist.is_rank_zero else None,
            metrics_jsonl_path=metrics_path,
            best_checkpoint_path=(run_dir / "checkpoint_best.pt") if dist.is_rank_zero else None,
        ),
    )
    if not dist.is_rank_zero:
        return CommandResult(
            "\n".join(
                (
                    f"config={command.config_path}",
                    f"run_dir={run_dir}",
                    f"rank={dist.rank}",
                    "training_status=completed_prepared_training_rank",
                    f"metrics_jsonl={metrics_path}",
                )
            )
        )

    checkpoints = checkpoint_manifest(run_dir)
    write_json(run_dir / "metrics.json", result.to_dict())
    write_json(run_dir / "checkpoint_manifest.json", checkpoints)
    summary = {
        "synthetic_only": result.synthetic_only,
        "status": result.status,
        "task_id": result.task_id,
        "source_modality": result.source_modality,
        "target_modality": result.target_modality,
        "split_manifest": str(split_manifest_path),
        "split_manifest_hash": split_manifest_hash,
        "event_manifest": str(_config_value(config, "event_manifest")),
        "window_length": result.event_summary.get("metadata", {}).get("window_length", prepared_config.window_length)
        if isinstance(result.event_summary.get("metadata"), dict)
        else prepared_config.window_length,
        "stride": prepared_config.stride,
        "event_summary": result.event_summary,
        "precision": result.precision,
        "gradient_accumulation_steps": result.gradient_accumulation_steps,
        "max_grad_norm": result.max_grad_norm,
        "completed_steps": result.completed_steps,
        "eval_every_steps": result.eval_every_steps,
        "checkpoint_every_steps": result.checkpoint_every_steps,
        "best_task_id": result.best_task_id,
        "best_eval_mse": result.best_eval_mse,
        "best_val_mse": result.best_val_mse,
        "final_val_mse": result.final_val_mse,
        "best_step": result.best_step,
        "best_checkpoint_path": result.best_checkpoint_path,
        "final_checkpoint_path": result.final_checkpoint_path,
        "checkpoint_selection_metric": result.checkpoint_selection_metric,
        "checkpoint_selection_mode": result.checkpoint_selection_mode,
        "test_mse": result.test_mse,
        "test_mae": result.test_mae,
        "test_pearsonr": result.test_pearsonr,
        "test_r2": result.test_r2,
        "test_spearmanr": result.test_spearmanr,
        "selection_split": result.selection_split,
        "report_split": result.report_split,
        "real_data_smoke": bool(config.get("dataset") == "moabb" and "smoke" in str(config.get("experiment", ""))),
        "scientific_claim_allowed": False,
        "unavailable_tasks": result.skipped_tasks,
        "quarantined_tasks": result.quarantined_tasks,
        "stimulus_evidence": result.stimulus_evidence,
        "baseline_failures": [],
        "task_results": result.task_results,
        "distributed_initialized": result.distributed_initialized,
        "distributed_backend": result.distributed_backend,
        "distributed": {"rank": dist.rank, "local_rank": dist.local_rank, "world_size": dist.world_size},
        "git": environment["git"],
        "source_commit_missing": environment["source_commit_missing"],
        "run": environment["run"],
        "checkpoint_manifest": checkpoints,
        "resume": command.resume,
    }
    write_json(run_dir / "summary.json", summary)
    _write_prepared_diagnostic_artifacts(
        run_dir,
        summary,
        event_manifest=str(_config_value(config, "event_manifest")),
        split_manifest=split_manifest_path,
        window_length=prepared_config.window_length,
        stride=prepared_config.stride,
    )
    lines = [
        f"config={command.config_path}",
        f"run_dir={run_dir}",
        f"rank={dist.rank}",
        "training_status=completed_prepared_training",
        f"task_id={result.task_id}",
        f"initial_loss={result.initial_loss:.6f}",
        f"final_loss={result.final_loss:.6f}",
        f"eval_mse={result.eval_mse:.6f}",
    ]
    if result.best_task_id:
        lines.extend((f"best_task_id={result.best_task_id}", f"best_eval_mse={result.best_eval_mse:.6f}"))
    lines.extend((f"steps={result.steps}", f"completed_steps={result.completed_steps}"))
    return CommandResult("\n".join(lines))


def _run_synthetic_training_command(
    command: TrainingCommandConfig,
    config: PreparedTrainingConfigInput,
    run_dir: Path,
    dist: object,
    environment: dict[str, object],
    split_manifest_path: Path,
    split_manifest_hash: str,
) -> CommandResult:
    result = run_synthetic_training(seed=int(config.get("seed", 0)), steps=int(config.get("steps", 24)))
    metrics_path = get_rank_metrics_path(run_dir, dist)
    append_jsonl(metrics_path, {"step": 0, "loss": result.initial_loss, "rank": dist.rank, "world_size": dist.world_size})
    append_jsonl(metrics_path, {"step": result.steps, "loss": result.final_loss, "rank": dist.rank, "world_size": dist.world_size})
    if not dist.is_rank_zero:
        lines = [
            f"config={command.config_path}",
            f"run_dir={run_dir}",
        ]
        if command.resume:
            lines.append(f"resume={command.resume}")
        lines.extend((f"rank={dist.rank}", "training_status=completed_synthetic_smoke_rank", f"metrics_jsonl={metrics_path}"))
        return CommandResult("\n".join(lines))

    write_json(run_dir / "metrics.json", {"initial_loss": result.initial_loss, "final_loss": result.final_loss, "steps": result.steps})
    import torch

    torch.save({"status": "synthetic_smoke", "steps": result.steps, "world_size": dist.world_size}, run_dir / "checkpoint.pt")
    checkpoints = checkpoint_manifest(run_dir)
    write_json(run_dir / "checkpoint_manifest.json", checkpoints)
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
            "distributed": {"rank": dist.rank, "local_rank": dist.local_rank, "world_size": dist.world_size},
            "git": environment["git"],
            "source_commit_missing": environment["source_commit_missing"],
            "run": environment["run"],
            "checkpoint_manifest": checkpoints,
            "resume": command.resume,
        },
    )
    lines = [f"config={command.config_path}", f"run_dir={run_dir}"]
    if command.resume:
        lines.append(f"resume={command.resume}")
    lines.extend(
        (
            f"rank={dist.rank}",
            "training_status=completed_synthetic_smoke",
            f"initial_loss={result.initial_loss:.6f}",
            f"final_loss={result.final_loss:.6f}",
            f"steps={result.steps}",
        )
    )
    return CommandResult("\n".join(lines))


def _has_prepared_training_inputs(config: PreparedTrainingConfigInput) -> bool:
    return bool(_config_value(config, "event_manifest")) and bool(_config_value(config, "split_manifest"))


def _config_value(config: PreparedTrainingConfigInput, key: str) -> object | None:
    data = config.get("data")
    data_config = data if isinstance(data, dict) else {}
    return config.get(key) or data_config.get(key)


def _write_prepared_diagnostic_artifacts(
    run_dir: Path,
    summary: dict[str, Any],
    *,
    event_manifest: str | Path,
    split_manifest: str | Path,
    window_length: int,
    stride: int,
) -> None:
    eval_audit = _write_eval_audit(run_dir, event_manifest, split_manifest, window_length, stride)
    evidence_gate = _prepared_evidence_gate(summary, eval_audit=eval_audit)
    write_json(run_dir / "evidence_gate.json", evidence_gate)
    (run_dir / "diagnostic_report.md").write_text(_diagnostic_report(summary, evidence_gate), encoding="utf-8")
    (run_dir / "pair_operator_ablation.csv").write_text(_pair_operator_ablation_csv(summary), encoding="utf-8")
    (run_dir / "uncertainty_calibration.csv").write_text(_uncertainty_calibration_csv(summary), encoding="utf-8")


def _write_eval_audit(
    run_dir: Path,
    event_manifest: str | Path,
    split_manifest: str | Path,
    window_length: int,
    stride: int,
) -> dict[str, Any]:
    try:
        report = audit_prepared_eval_inputs(
            event_manifest,
            split_manifest,
            window_length=window_length,
            stride=stride,
            out_dir=run_dir,
            require_windows=True,
        )
        return report.to_dict()
    except Exception as exc:  # noqa: BLE001 - diagnostic artifact should fail closed.
        payload = {
            "passed": False,
            "violations": [f"eval audit failed unexpectedly: {exc}"],
            "warnings": [],
            "checked": [],
            "event_count": 0,
            "window_count": 0,
            "window_counts_by_split": {"train": 0, "val": 0, "test": 0},
            "event_summary": {},
        }
        write_json(run_dir / "eval_audit.json", payload)
        return payload


def _prepared_evidence_gate(summary: dict[str, Any], eval_audit: dict[str, Any] | None = None) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(eval_audit, dict) or not eval_audit:
        failures.append("eval_audit.json missing")
    elif not bool(eval_audit.get("passed")):
        failures.append("eval audit did not pass")
    quarantined = summary.get("quarantined_tasks")
    if isinstance(quarantined, (list, tuple)) and quarantined:
        failed_tasks = ",".join(str(row.get("task_id", "unknown")) for row in quarantined if isinstance(row, dict))
        failures.append(f"required task quarantined: {failed_tasks or 'unknown'}")
    for row in _task_result_rows(summary):
        if row.get("test_mse") is None or row.get("best_val_mse") is None:
            failures.append(f"required task has missing/non-finite selected metric: {row.get('task_id', 'unknown')}")
    stimulus = summary.get("stimulus_evidence")
    if isinstance(stimulus, dict) and stimulus and not bool(stimulus.get("claim_eligible")):
        failures.append("stimulus-to-fMRI evidence is not claim eligible")
    failures.append("baseline ranking artifact not generated by training command")
    failures.append("exact competitor reproduction status requires prepared baseline suite artifacts")
    summary_claim = bool(summary.get("scientific_claim_allowed"))
    if not summary_claim:
        failures.append("summary.json scientific_claim_allowed is false")
    return {
        "schema": "neurotwin.prepared_evidence_gate.v1",
        "passed": False if failures else summary_claim,
        "scientific_claim_allowed": summary_claim,
        "summary_is_source_of_truth": True,
        "failures": failures,
        "checks": {
            "quarantined_tasks": quarantined or [],
            "eval_audit": eval_audit or {},
            "stimulus_evidence": stimulus or {},
            "baseline_ranking_present": False,
            "eval_audit_present": bool(eval_audit),
            "eval_audit_passed": bool(eval_audit.get("passed")) if isinstance(eval_audit, dict) else False,
            "competitor_reproduction_status_present": False,
        },
    }


def _diagnostic_report(summary: dict[str, Any], evidence_gate: dict[str, Any]) -> str:
    lines = [
        "# NeuroTwin Prepared Run Diagnostic Report",
        "",
        f"- status: {summary.get('status', 'unknown')}",
        f"- task_id: {summary.get('task_id', 'unknown')}",
        f"- source_modality: {summary.get('source_modality', 'unknown')}",
        f"- target_modality: {summary.get('target_modality', 'unknown')}",
        f"- scientific_claim_allowed: {summary.get('scientific_claim_allowed', False)}",
        f"- evidence_gate_passed: {evidence_gate.get('passed', False)}",
        "",
        "## Gate Failures",
        "",
    ]
    failures = evidence_gate.get("failures")
    if isinstance(failures, list) and failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- none")
    lines.extend(["", "## Stimulus Evidence", ""])
    stimulus = summary.get("stimulus_evidence")
    if isinstance(stimulus, dict) and stimulus:
        for key in ("status", "claim_eligible", "require_real_stimulus", "hash_verified", "claim_note"):
            lines.append(f"- {key}: {stimulus.get(key, 'unknown')}")
    else:
        lines.append("- missing")
    lines.extend(["", "## Quarantined Tasks", ""])
    quarantined = summary.get("quarantined_tasks")
    if isinstance(quarantined, (list, tuple)) and quarantined:
        for row in quarantined:
            if isinstance(row, dict):
                lines.append(f"- {row.get('task_id', 'unknown')}: {row.get('reason', 'unknown')}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _pair_operator_ablation_csv(summary: dict[str, Any]) -> str:
    rows: list[tuple[Any, ...]] = []
    for row in _task_result_rows(summary):
        model_config = row.get("model_config") if isinstance(row.get("model_config"), dict) else {}
        if str(model_config.get("type", "")) != "NeuroTwinPairOperator":
            continue
        rows.append(
            (
                row.get("task_id", ""),
                _pair_operator_variant(model_config),
                model_config.get("use_pair_state", ""),
                model_config.get("use_uncertainty_head", ""),
                model_config.get("refinement_steps", ""),
                row.get("status", ""),
                row.get("test_mse", ""),
                row.get("test_pearsonr", ""),
                row.get("test_r2", ""),
            )
        )
    if not rows:
        rows.append(("unavailable", "not_pair_operator_run", "", "", "", "not_applicable", "", "", ""))
    return _csv_rows(
        ("task_id", "ablation", "use_pair_state", "use_uncertainty_head", "refinement_steps", "status", "test_mse", "test_pearsonr", "test_r2"),
        rows,
    )


def _uncertainty_calibration_csv(summary: dict[str, Any]) -> str:
    rows: list[tuple[Any, ...]] = []
    for row in _task_result_rows(summary):
        model_config = row.get("model_config") if isinstance(row.get("model_config"), dict) else {}
        if str(model_config.get("type", "")) == "NeuroTwinPairOperator" and bool(model_config.get("use_uncertainty_head")):
            rows.append((row.get("task_id", ""), "unavailable", "training command does not persist uncertainty predictions", "", ""))
    if not rows:
        rows.append(("unavailable", "unavailable", "uncertainty head disabled or not a Pair-Operator run", "", ""))
    return _csv_rows(("task_id", "status", "reason", "mean_uncertainty", "error_uncertainty_correlation"), rows)


def _task_result_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("task_results")
    if not isinstance(rows, (list, tuple)):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _pair_operator_variant(model_config: dict[str, Any]) -> str:
    use_pair = bool(model_config.get("use_pair_state"))
    use_uncertainty = bool(model_config.get("use_uncertainty_head"))
    refinement_steps = int(model_config.get("refinement_steps") or 0)
    if not use_pair:
        return "pair_operator_no_pair_state"
    if not use_uncertainty:
        return "pair_operator_no_uncertainty"
    if refinement_steps <= 0:
        return "pair_operator_no_refinement"
    return "pair_operator_full"


def _csv_rows(header: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = [",".join(header)]
    lines.extend(",".join(_csv_cell(value) for value in row) for row in rows)
    return "\n".join(lines) + "\n"


def _csv_cell(value: Any) -> str:
    text = str(value)
    if any(char in text for char in (",", "\"", "\n")):
        return "\"" + text.replace("\"", "\"\"") + "\""
    return text
