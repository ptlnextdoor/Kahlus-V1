"""Config for the v3 KTM training harness (PROPOSED / SYNTHETIC ONLY).

A single frozen dataclass that carries seed/optimization knobs, the synthetic-world knobs, and
the model dims. Loadable from a YAML mapping via :func:`neurotwin.config.load_config`. No A100,
no real data — these knobs only ever drive a synthetic Transition Gym smoke / micro run.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from neurotwin.config import load_config
from neurotwin.models.ktm import TorchKTMConfig
from neurotwin.transition_gym import SyntheticWorldConfig

VALID_MODES: frozenset[str] = frozenset({"cpu_smoke", "single_gpu", "ddp"})
VALID_PRECISIONS: frozenset[str] = frozenset({"fp32", "bf16", "fp16"})


@dataclass(frozen=True)
class KTMTrainConfig:
    seed: int = 0
    mode: str = "cpu_smoke"
    steps: int = 150
    batch_size: int = 16
    lr: float = 1e-2
    weight_decay: float = 0.0
    precision: str = "fp32"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    checkpoint_every_steps: int = 50
    eval_every_steps: int = 50
    w_traj: float = 1.0
    w_profile: float = 0.5
    w_nll: float = 0.1
    loss_explosion_factor: float = 8.0
    resume_path: str | None = None

    # Fairness knobs for the KTM-vs-baselines recovery comparison. ``baseline_train_steps == 0``
    # means "match the KTM optimizer-step budget" (resolved against ``steps`` at use time), so the
    # baselines are not handed an unfairly short schedule. ``recovery_margin`` is the relative MSE
    # improvement the trained KTM must clear over the strongest baseline to earn the recovery scope.
    baseline_train_steps: int = 0
    recovery_margin: float = 0.05

    # Synthetic-world knobs (kept tiny by default so the CPU smoke is fast + deterministic).
    n_episodes: int = 48
    n_subjects: int = 4
    state_dim: int = 6
    n_perturbations: int = 4
    horizon: int = 5
    history_len: int = 6
    eeg_channels: int = 5
    behavior_dim: int = 2

    # Model dims.
    embed_dim: int = 16
    memory_dim: int = 12
    memory_rho: float = 0.8
    uncertainty_floor: float = 1e-3

    def validate(self) -> "KTMTrainConfig":
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {self.mode!r}")
        if self.precision not in VALID_PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(VALID_PRECISIONS)}, got {self.precision!r}")
        for name in ("steps", "batch_size", "gradient_accumulation_steps",
                     "checkpoint_every_steps", "eval_every_steps"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.loss_explosion_factor <= 1.0:
            raise ValueError("loss_explosion_factor must be > 1.0")
        if int(self.baseline_train_steps) < 0:
            raise ValueError(f"baseline_train_steps must be >= 0, got {self.baseline_train_steps}")
        if not 0.0 <= float(self.recovery_margin) < 1.0:
            raise ValueError(f"recovery_margin must be in [0.0, 1.0), got {self.recovery_margin}")
        # Constructing the sub-configs validates the world/model knobs.
        self.to_world_config()
        self.to_model_config()
        return self

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "KTMTrainConfig":
        known = {f.name for f in fields(cls)}
        kwargs = {key: value for key, value in payload.items() if key in known}
        return cls(**kwargs).validate()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "KTMTrainConfig":
        return cls.from_mapping(load_config(path))

    def to_world_config(self) -> SyntheticWorldConfig:
        return SyntheticWorldConfig(
            seed=self.seed,
            n_episodes=self.n_episodes,
            n_subjects=self.n_subjects,
            state_dim=self.state_dim,
            n_perturbations=self.n_perturbations,
            horizon=self.horizon,
            history_len=self.history_len,
            eeg_channels=self.eeg_channels,
            behavior_dim=self.behavior_dim,
        ).validate()

    def to_model_config(self) -> TorchKTMConfig:
        return TorchKTMConfig(
            seed=self.seed,
            history_len=self.history_len,
            eeg_channels=self.eeg_channels,
            n_perturbations=self.n_perturbations,
            horizon=self.horizon,
            embed_dim=self.embed_dim,
            memory_dim=self.memory_dim,
            memory_rho=self.memory_rho,
            uncertainty_floor=self.uncertainty_floor,
        ).validate()

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
