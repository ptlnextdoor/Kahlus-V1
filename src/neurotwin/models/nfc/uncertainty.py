from __future__ import annotations

import torch
from torch import nn


class UncertaintyMapHead(nn.Module):
    """NFC uncertainty maps over regions, time, and optionally pairs."""

    def __init__(self, latent_dim: int, pair_uncertainty: bool = False) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.pair_uncertainty = bool(pair_uncertainty)
        self.region_head = nn.Linear(self.latent_dim, 1)

    def forward(self, latent_field: torch.Tensor) -> dict[str, torch.Tensor]:
        if latent_field.ndim != 4:
            raise ValueError("latent_field must have shape [batch, time, nodes, latent_dim]")
        region = torch.nn.functional.softplus(self.region_head(latent_field).squeeze(-1)) + 1e-6
        output = {
            "region_uncertainty": region,
            "time_uncertainty": region.mean(dim=-1),
        }
        if self.pair_uncertainty:
            node_score = region.mean(dim=1)
            output["pair_uncertainty"] = 0.5 * (node_score.unsqueeze(-1) + node_score.unsqueeze(-2))
        return output
