from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neurotwin.training.prepared_types import PreparedTrainingRunPaths


def resolve_run_paths(
    paths: PreparedTrainingRunPaths | None,
    *,
    checkpoint_path: str | Path | None,
    resume_path: str | Path | None,
    metrics_csv_path: str | Path | None,
    metrics_jsonl_path: str | Path | None,
    best_checkpoint_path: str | Path | None,
) -> PreparedTrainingRunPaths:
    if paths is None:
        return PreparedTrainingRunPaths(
            checkpoint_path=checkpoint_path,
            resume_path=resume_path,
            metrics_csv_path=metrics_csv_path,
            metrics_jsonl_path=metrics_jsonl_path,
            best_checkpoint_path=best_checkpoint_path,
        )
    if any(value is not None for value in (checkpoint_path, resume_path, metrics_csv_path, metrics_jsonl_path, best_checkpoint_path)):
        raise ValueError("pass either paths=PreparedTrainingRunPaths(...) or legacy path keywords, not both")
    return paths


def load_resume_checkpoint(path: str | Path | None, device: torch.device) -> dict[str, Any] | None:
    return torch.load(Path(path), map_location=device, weights_only=True) if path is not None else None


def load_task_resume(model: torch.nn.Module, optimizer: torch.optim.Optimizer, task_id: str, checkpoint: dict[str, Any] | None) -> None:
    if checkpoint is None:
        return
    task_states = checkpoint.get("task_states")
    if isinstance(task_states, dict) and task_id in task_states:
        task_state = task_states[task_id]
        model.load_state_dict(task_state["model_state_dict"])
        if "optimizer_state_dict" in task_state:
            optimizer.load_state_dict(task_state["optimizer_state_dict"])
        return
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def resume_start_step(checkpoint: dict[str, Any] | None) -> int:
    if checkpoint is None:
        return 0
    return int(checkpoint.get("completed_steps", checkpoint.get("metrics", {}).get("completed_steps", 0)))


def save_task_checkpoint(
    path: str | Path,
    status: str,
    task_result: dict[str, Any],
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "status": status,
            "task": {key: task_result.get(key) for key in ("task_id", "source_modality", "target_modality")},
            "metrics": task_result,
            "model_config": task_result.get("model_config"),
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "completed_steps": task_result.get("completed_steps"),
            "best_step": task_result.get("best_step"),
            "checkpoint_selection_metric": task_result.get("checkpoint_selection_metric"),
            "checkpoint_selection_mode": task_result.get("checkpoint_selection_mode"),
        },
        out,
    )
