from __future__ import annotations

from math import sqrt

import torch
from torch import nn


class LowRankPairKernel(nn.Module):
    """Single-head low-rank scaled dot-product attention over sensor nodes."""

    def __init__(self, latent_dim: int, rank: int = 8, use_pair_state: bool = True) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if rank < 1:
            raise ValueError("rank must be positive")
        self.latent_dim = int(latent_dim)
        self.rank = int(rank)
        self.use_pair_state = bool(use_pair_state)
        self.left = nn.Linear(self.latent_dim, self.rank, bias=False)
        self.right = nn.Linear(self.latent_dim, self.rank, bias=False)
        self.value = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.update = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.GELU())

    def pair_weights(self, z: torch.Tensor, structural_prior: torch.Tensor | None = None) -> torch.Tensor:
        return self.attention_weights(z, structural_prior=structural_prior)

    def attention_weights(self, z: torch.Tensor, structural_prior: torch.Tensor | None = None) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError("z must have shape [batch, nodes, latent_dim]")
        scores = torch.matmul(self.left(z), self.right(z).transpose(-1, -2)) / sqrt(float(self.rank))
        if structural_prior is not None:
            if structural_prior.shape != scores.shape[-2:]:
                raise ValueError("structural_prior must have shape [nodes, nodes]")
            scores = scores + structural_prior.to(device=z.device, dtype=z.dtype).unsqueeze(0)
        return torch.softmax(scores, dim=-1)

    def forward(self, z: torch.Tensor, structural_prior: torch.Tensor | None = None) -> torch.Tensor:
        if not self.use_pair_state:
            return z
        weights = self.attention_weights(z, structural_prior=structural_prior)
        message = torch.matmul(weights, self.value(z))
        return z + self.update(message)
