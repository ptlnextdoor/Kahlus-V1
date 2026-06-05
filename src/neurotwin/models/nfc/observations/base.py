from __future__ import annotations

import torch
from torch import nn


class BaseObservationOperator(nn.Module):
    """Base class for latent-field-to-observation operators."""

    modality: str = "generic"

    def _check_field(self, latent_field: torch.Tensor, latent_dim: int) -> None:
        if latent_field.ndim != 4:
            raise ValueError("latent_field must have shape [batch, time, nodes, latent_dim]")
        if latent_field.shape[-1] != latent_dim:
            raise ValueError(f"latent dim {latent_field.shape[-1]} does not match {latent_dim}")
