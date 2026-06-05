from __future__ import annotations

import torch
from torch import nn

from neurotwin.models.nfc.observations.base import BaseObservationOperator


class EEGObservationOperator(BaseObservationOperator):
    """Compile a latent neural field into EEG sensor or spectral-proxy outputs."""

    modality = "eeg"

    def __init__(self, latent_dim: int, output_dim: int) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.readout = nn.Linear(self.latent_dim, self.output_dim)

    def forward(self, latent_field: torch.Tensor) -> torch.Tensor:
        self._check_field(latent_field, self.latent_dim)
        return self.readout(latent_field.mean(dim=2))
