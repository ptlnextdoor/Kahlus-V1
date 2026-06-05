from __future__ import annotations

import torch
from torch import nn


class LatentNeuralField(nn.Module):
    """Infer a hidden neural field with shape [batch, time, nodes, latent_dim]."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        subject_dim: int = 0,
        stimulus_dim: int = 0,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.subject_dim = max(0, int(subject_dim))
        self.stimulus_dim = max(0, int(stimulus_dim))
        self.observation_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim),
            nn.GELU(),
            nn.LayerNorm(self.latent_dim),
        )
        self.subject_encoder = nn.Linear(self.subject_dim, self.latent_dim) if self.subject_dim else None
        self.stimulus_encoder = nn.Linear(self.stimulus_dim, self.latent_dim) if self.stimulus_dim else None

    def forward(
        self,
        observations: torch.Tensor,
        *,
        subject_state: torch.Tensor | None = None,
        stimulus_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if observations.ndim == 3:
            observations = observations.unsqueeze(-1)
        if observations.ndim != 4:
            raise ValueError("observations must have shape [batch, time, nodes, features]")
        if observations.shape[-1] != self.input_dim:
            raise ValueError(f"observation feature dim {observations.shape[-1]} does not match {self.input_dim}")
        latent = self.observation_encoder(observations.float())
        batch, time, nodes, _ = latent.shape
        if self.subject_encoder is not None and subject_state is not None:
            if subject_state.shape[0] != batch:
                raise ValueError("subject_state batch does not match observations")
            latent = latent + self.subject_encoder(subject_state.float()).view(batch, 1, 1, self.latent_dim)
        if self.stimulus_encoder is not None and stimulus_state is not None:
            if stimulus_state.shape[:2] != (batch, time):
                raise ValueError("stimulus_state must have shape [batch, time, features]")
            drive = self.stimulus_encoder(stimulus_state.float()).view(batch, time, 1, self.latent_dim)
            latent = latent + drive.expand(batch, time, nodes, self.latent_dim)
        return latent
