"""Kahlus v3 KTM training harness (PROPOSED / SYNTHETIC ONLY).

A standalone, cluster-ready training package for the trainable ``TorchKTM`` on the synthetic
Transition Gym. Isolated from the frozen v1 ``training/`` package; reuses the repo's DDP helpers,
repro utilities, shared baseline runner, and unified evidence gate. Earns only the narrow
``synthetic_ktm_training_harness`` scope — never a recovery/superiority claim. No A100 launched
here; the harness only *builds* the future micro-sweep command.
"""

from __future__ import annotations

from neurotwin.training_v3.bundle import write_training_bundle
from neurotwin.training_v3.config import KTMTrainConfig
from neurotwin.training_v3.dataset import TransitionGymDataset, make_dataloaders
from neurotwin.training_v3.metrics_eval import evaluate_ktm, ktm_vs_baselines
from neurotwin.training_v3.objective import LossExplosionGuard, is_finite_loss, ktm_loss
from neurotwin.training_v3.trainer import build_torchrun_command, resolve_device, train_ktm

__all__ = [
    "KTMTrainConfig",
    "TransitionGymDataset",
    "make_dataloaders",
    "ktm_loss",
    "LossExplosionGuard",
    "is_finite_loss",
    "evaluate_ktm",
    "ktm_vs_baselines",
    "train_ktm",
    "resolve_device",
    "build_torchrun_command",
    "write_training_bundle",
]
