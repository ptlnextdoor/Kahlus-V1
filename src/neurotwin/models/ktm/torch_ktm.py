"""Trainable PyTorch KTM (Kahlus Transition Model) for the v3 training harness.

PROPOSED / SYNTHETIC ONLY. A minimal, trainable ``nn.Module`` sibling of the numpy KTM
scaffold (``neurotwin.models.ktm.ktm.KTM``). It mirrors the scaffold's interface and tensor
shapes so the Transition Gym stays the authoritative grader, but adds autograd parameters so
the Sprint 2A harness can actually train it. The numpy scaffold is left untouched.

Pipeline: history EEG window -> linear encoder (tanh) -> leaky memory projection ->
per-perturbation response-profile head + per-perturbation uncertainty head. Heavier KTM modules
(lie generators, NeuroExperts, active experiment designer) are intentionally out of scope here.

This is not a built model and not a scientific claim. A decreasing loss earns only the narrow
``synthetic_ktm_training_harness`` scope, never ``synthetic_ktm_recovery``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TorchKTMConfig:
    """Config for :class:`TorchKTM`; field names/shapes mirror the numpy ``KTMConfig``."""

    seed: int = 0
    history_len: int = 6  # L
    eeg_channels: int = 5  # C
    n_perturbations: int = 4  # K
    horizon: int = 5  # H
    embed_dim: int = 16
    memory_dim: int = 12
    memory_rho: float = 0.8
    uncertainty_floor: float = 1e-3

    def validate(self) -> "TorchKTMConfig":
        positives = {
            "history_len": self.history_len,
            "eeg_channels": self.eeg_channels,
            "n_perturbations": self.n_perturbations,
            "horizon": self.horizon,
            "embed_dim": self.embed_dim,
            "memory_dim": self.memory_dim,
        }
        for name, value in positives.items():
            if int(value) < 1:
                raise ValueError(f"TorchKTMConfig.{name} must be >= 1, got {value}")
        if not 0.0 <= self.memory_rho < 1.0:
            raise ValueError("TorchKTMConfig.memory_rho must be in [0, 1)")
        if self.uncertainty_floor <= 0.0:
            raise ValueError("TorchKTMConfig.uncertainty_floor must be positive")
        return self


class TorchKTM(nn.Module):
    """Trainable KTM: ``(history, perturbation_index) -> (response_pred, log_var)``."""

    def __init__(self, config: TorchKTMConfig) -> None:
        super().__init__()
        self.config = config.validate()
        c = self.config
        in_dim = c.history_len * c.eeg_channels
        out_dim = c.horizon * c.eeg_channels

        self.encoder = nn.Linear(in_dim, c.embed_dim)
        self.memory_proj = nn.Linear(c.embed_dim, c.memory_dim)
        # One linear response head per perturbation a_k: memory -> flattened (H*C) trajectory.
        self.response_weight = nn.Parameter(torch.empty(c.n_perturbations, c.memory_dim, out_dim))
        self.response_bias = nn.Parameter(torch.zeros(c.n_perturbations, out_dim))
        self.uncertainty = nn.Linear(c.memory_dim, c.n_perturbations)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Deterministic init for reproducible smoke runs (seed via cfg, isolated generator).
        gen = torch.Generator().manual_seed(int(self.config.seed))
        for module in (self.encoder, self.memory_proj, self.uncertainty):
            bound = 1.0 / max(1, module.in_features) ** 0.5
            module.weight.data.uniform_(-bound, bound, generator=gen)
            if module.bias is not None:
                module.bias.data.zero_()
        bound = 1.0 / max(1, self.config.memory_dim) ** 0.5
        self.response_weight.data.uniform_(-bound, bound, generator=gen)
        self.response_bias.data.zero_()

    def _memory(self, history: torch.Tensor) -> torch.Tensor:
        flat = history.reshape(history.shape[0], -1)
        embed = torch.tanh(self.encoder(flat))
        # Leaky one-shot memory (matches the numpy scaffold's project = (1-rho)*tanh(...)).
        return (1.0 - self.config.memory_rho) * torch.tanh(self.memory_proj(embed))

    def forward(
        self, history: torch.Tensor, perturbation_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.config
        mem = self._memory(history)  # (B, memory_dim)
        idx = perturbation_index.long()
        w = self.response_weight[idx]  # (B, memory_dim, H*C)
        b = self.response_bias[idx]  # (B, H*C)
        flat_pred = torch.einsum("bm,bmo->bo", mem, w) + b  # (B, H*C)
        pred = flat_pred.reshape(history.shape[0], c.horizon, c.eeg_channels)
        var_all = torch.nn.functional.softplus(self.uncertainty(mem)) + c.uncertainty_floor
        var = var_all.gather(1, idx.unsqueeze(1)).squeeze(1)  # (B,)
        return pred, torch.log(var)

    def predict_response_profile(self, history: torch.Tensor) -> torch.Tensor:
        """Finite response profile C_K(h): ``(B, K, H, C)`` over the locked battery."""

        c = self.config
        mem = self._memory(history)
        flat = torch.einsum("bm,kmo->bko", mem, self.response_weight) + self.response_bias
        return flat.reshape(history.shape[0], c.n_perturbations, c.horizon, c.eeg_channels)

    def predict_uncertainty(self, history: torch.Tensor) -> torch.Tensor:
        """Per-perturbation positive variances ``(B, K)``."""

        mem = self._memory(history)
        return torch.nn.functional.softplus(self.uncertainty(mem)) + self.config.uncertainty_floor

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))
