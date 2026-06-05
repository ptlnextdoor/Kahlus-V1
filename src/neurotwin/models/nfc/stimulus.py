from __future__ import annotations

import torch
from torch import nn


class StimulusConditioningOperator(nn.Module):
    """Causal stimulus adapter for NFC field updates."""

    def __init__(self, stimulus_dim: int, latent_dim: int, lag_steps: int = 1) -> None:
        super().__init__()
        if stimulus_dim < 1:
            raise ValueError("stimulus_dim must be positive")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        self.stimulus_dim = int(stimulus_dim)
        self.latent_dim = int(latent_dim)
        self.lag_steps = max(0, int(lag_steps))
        self.adapter = nn.Sequential(nn.Linear(self.stimulus_dim, self.latent_dim), nn.GELU(), nn.LayerNorm(self.latent_dim))

    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        if stimulus.ndim != 3 or stimulus.shape[-1] != self.stimulus_dim:
            raise ValueError("stimulus must have shape [batch, time, stimulus_dim]")
        causal: list[torch.Tensor] = []
        for step in range(stimulus.shape[1]):
            start = max(0, step - self.lag_steps)
            causal.append(stimulus[:, start : step + 1].mean(dim=1))
        return self.adapter(torch.stack(causal, dim=1).float())
