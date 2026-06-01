from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from neurotwin.adapters.synthetic import make_synthetic_recordings
from neurotwin.config import ConfigError, load_config
from neurotwin.config_types import PreparedTrainingConfigInput, as_prepared_training_config_input
from neurotwin.data.manifest_io import save_split_manifest
from neurotwin.data.split_manifest import build_split_manifest
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
            "scientific_claim_allowed": False,
            "unavailable_tasks": result.skipped_tasks,
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
        },
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
