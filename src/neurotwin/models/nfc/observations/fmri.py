from __future__ import annotations

import torch
from torch import nn

from neurotwin.models.causal import CausalHRFAdapter
from neurotwin.models.nfc.observations.base import BaseObservationOperator


class FMRIObservationOperator(BaseObservationOperator):
    """Compile a latent neural field into parcel BOLD predictions."""

    modality = "fmri"

    def __init__(self, latent_dim: int, output_dim: int, hrf_delay_steps: int = 1) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.hrf_delay_steps = max(0, int(hrf_delay_steps))
        self.hrf = CausalHRFAdapter(self.latent_dim, delay_steps=self.hrf_delay_steps, kernel_size=3)
        self.readout = nn.Linear(self.latent_dim, 1)

    def forward(self, latent_field: torch.Tensor) -> torch.Tensor:
        self._check_field(latent_field, self.latent_dim)
        if latent_field.shape[2] != self.output_dim:
            raise ValueError("fMRI node count must match output_dim")
        batch, time, nodes, latent_dim = latent_field.shape
        flat = latent_field.permute(0, 2, 3, 1).reshape(batch * nodes, latent_dim, time)
        context = self.hrf(flat)
        context = context.reshape(batch, nodes, latent_dim, time).permute(0, 3, 1, 2)
        return self.readout(context).squeeze(-1)
