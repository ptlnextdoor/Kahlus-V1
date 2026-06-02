from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from neurotwin.config_types import (
    ConfigPath,
    PreparedTrainingConfigInput,
    ResolvedPreparedModelConfig,
    ResolvedPreparedRuntimeConfig,
    resolve_prepared_config,
)
from neurotwin.data.prepared_tasks import PreparedSuiteConfig
from neurotwin.runtime.distributed import DistributedInfo


@dataclass(frozen=True)
class PreparedTrainingConfig:
    event_manifest: ConfigPath
    split_manifest: ConfigPath
    seed: int
    window_length: int
    stride: int
    steps: int
    requested_task: str
    model: ResolvedPreparedModelConfig
    runtime: ResolvedPreparedRuntimeConfig

    @classmethod
    def from_mapping(cls, config: PreparedTrainingConfigInput) -> "PreparedTrainingConfig":
        resolved = resolve_prepared_config(
            config,
            require_manifests=True,
            latent_dim_default=64,
            n_layers_default=1,
            projection_dim_default=32,
        )
        return cls(
            event_manifest=resolved.event_manifest or "",
            split_manifest=resolved.split_manifest or "",
            seed=resolved.seed,
            window_length=resolved.window_length,
            stride=resolved.stride,
            steps=resolved.steps,
            requested_task=resolved.requested_task,
            model=resolved.model,
            runtime=resolved.runtime,
        )

    @property
    def objective_weights(self) -> dict[str, float]:
        return self.runtime.objective_weights

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.runtime.gradient_accumulation_steps

    @property
    def precision(self) -> str:
        return self.runtime.precision

    @property
    def eval_every_steps(self) -> int:
        return self.runtime.eval_every_steps

    @property
    def checkpoint_every_steps(self) -> int:
        return self.runtime.checkpoint_every_steps

    @property
    def compile(self) -> bool:
        return self.runtime.compile

    @property
    def learning_rate(self) -> float:
        return self.runtime.learning_rate

    @property
    def batch_size(self) -> int | None:
        return self.runtime.batch_size

    @property
    def eval_batch_size(self) -> int | None:
        return self.runtime.eval_batch_size

    @property
    def gradient_checkpointing(self) -> bool:
        return self.model.gradient_checkpointing

    def suite_config(self) -> PreparedSuiteConfig:
        return PreparedSuiteConfig(
            event_manifest=self.event_manifest,
            split_manifest=self.split_manifest,
            window_length=self.window_length,
            stride=self.stride,
            seed=self.seed,
            train_steps=self.steps,
        )


@dataclass(frozen=True)
class PreparedTrainingRunPaths:
    checkpoint_path: str | Path | None = None
    resume_path: str | Path | None = None
    metrics_csv_path: str | Path | None = None
    metrics_jsonl_path: str | Path | None = None
    best_checkpoint_path: str | Path | None = None


@dataclass(frozen=True)
class PreparedTrainingResult:
    status: str
    task_id: str
    source_modality: str
    target_modality: str
    initial_loss: float
    final_loss: float
    eval_mse: float
    eval_mae: float
    eval_pearsonr: float
    eval_r2: float
    eval_spearmanr: float
    best_val_mse: float | None
    test_mse: float
    test_mae: float
    test_pearsonr: float
    test_r2: float
    test_spearmanr: float
    selection_split: str
    report_split: str
    steps: int
    start_step: int
    completed_steps: int
    gradient_accumulation_steps: int
    precision: str
    device: str
    resumed_from: str | None
    distributed_initialized: bool
    distributed_backend: str | None
    rank: int
    world_size: int
    train_samples: int
    test_samples: int
    synthetic_only: bool
    skipped_tasks: tuple[dict[str, str], ...]
    event_summary: dict[str, Any]
    task_results: tuple[dict[str, Any], ...] = ()
    best_task_id: str | None = None
    best_eval_mse: float | None = None
    eval_every_steps: int = 0
    checkpoint_every_steps: int = 0
    final_val_mse: float | None = None
    best_step: int | None = None
    best_checkpoint_path: str | None = None
    final_checkpoint_path: str | None = None
    checkpoint_selection_metric: str = "val_mse"
    checkpoint_selection_mode: str = "min"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedTaskSelection:
    batches: list[Any]
    selected_tasks: tuple[Any, ...]
    skipped_tasks: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class PreparedRuntimeContext:
    device: torch.device
    dist_info: DistributedInfo
    resume_checkpoint: dict[str, Any] | None
    paths: PreparedTrainingRunPaths
    start_step: int
    checkpoint_dir: Path | None


@dataclass(frozen=True)
class PreparedTaskTrainingState:
    task_results: tuple[dict[str, Any], ...]
    task_states: dict[str, dict[str, Any]]
    best_task_id: str | None
    best_eval_mse: float | None
    first_model_state: dict[str, Any] | None
    first_optimizer_state: dict[str, Any] | None
    best_model_state: dict[str, Any] | None
    best_optimizer_state: dict[str, Any] | None
    start_step: int


@dataclass(frozen=True)
class TaskTrainingTensors:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    selection_split: str
    batch_size: int
    eval_batch_size: int
    train_sampler: "PreparedBatchSampler"


@dataclass(frozen=True)
class PreparedBatchSampler:
    num_samples: int
    batch_size: int
    gradient_accumulation_steps: int
    rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.world_size < 1:
            raise ValueError("world_size must be positive")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}), got {self.rank}")

    def indices(self, step: int, micro_step: int) -> list[int]:
        global_micro_step = step * self.gradient_accumulation_steps + micro_step
        if self.world_size == 1:
            start = (global_micro_step * self.batch_size) % self.num_samples
            end = min(start + self.batch_size, self.num_samples)
            return list(range(start, end))

        start = ((global_micro_step * self.world_size + self.rank) * self.batch_size) % self.num_samples
        return [(start + offset) % self.num_samples for offset in range(self.batch_size)]


@dataclass(frozen=True)
class PreparedTaskRunResult:
    status: str
    task_id: str
    source_modality: str
    target_modality: str
    initial_loss: float
    final_loss: float
    steps: int
    start_step: int
    completed_steps: int
    train_samples: int
    val_samples: int
    test_samples: int
    resumed_from: str | None
    objective_weight: float
    compile: bool
    selection_split: str
    report_split: str
    model_config: dict[str, Any]
    best_val_mse: float | None
    final_val_mse: float | None
    best_step: int | None
    best_checkpoint_path: str | None
    final_checkpoint_path: str | None
    checkpoint_selection_metric: str
    checkpoint_selection_mode: str
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "task_id": self.task_id,
            "source_modality": self.source_modality,
            "target_modality": self.target_modality,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "steps": self.steps,
            "start_step": self.start_step,
            "completed_steps": self.completed_steps,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "test_samples": self.test_samples,
            "resumed_from": self.resumed_from,
            "objective_weight": self.objective_weight,
            "compile": self.compile,
            "selection_split": self.selection_split,
            "report_split": self.report_split,
            "model_config": self.model_config,
            "best_val_mse": self.best_val_mse,
            "final_val_mse": self.final_val_mse,
            "best_step": self.best_step,
            "best_checkpoint_path": self.best_checkpoint_path,
            "final_checkpoint_path": self.final_checkpoint_path,
            "checkpoint_selection_metric": self.checkpoint_selection_metric,
            "checkpoint_selection_mode": self.checkpoint_selection_mode,
            "eval_mse": self.test_metrics["test_mse"],
            "eval_mae": self.test_metrics["test_mae"],
            "eval_pearsonr": self.test_metrics["test_pearsonr"],
            "eval_r2": self.test_metrics["test_r2"],
            "eval_spearmanr": self.test_metrics["test_spearmanr"],
        }
        payload.update(self.val_metrics)
        payload.update(self.test_metrics)
        return payload


@dataclass(frozen=True)
class PreparedEvalSnapshot:
    step: int
    metrics: dict[str, float]
    selection_metric: str = "val_mse"

    @property
    def selection_value(self) -> float:
        return float(self.metrics[self.selection_metric])


@dataclass(frozen=True)
class PreparedCheckpointInfo:
    best_step: int | None
    best_val_mse: float | None
    final_val_mse: float | None
    best_checkpoint_path: str | None
    final_checkpoint_path: str | None
    checkpoint_selection_metric: str = "val_mse"
    checkpoint_selection_mode: str = "min"


@dataclass(frozen=True)
class PreparedRuntimeInfo:
    device: str
    rank: int
    world_size: int
    distributed_backend: str | None
