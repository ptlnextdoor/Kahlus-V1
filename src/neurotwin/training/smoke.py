from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from neurotwin.models.torch_models import NeuralStateSpaceTranslator


@dataclass(frozen=True)
class TrainingSmokeResult:
    initial_loss: float
    final_loss: float
    steps: int


def run_synthetic_training(seed: int = 0, steps: int = 24) -> TrainingSmokeResult:
    """Run a tiny deterministic CPU training loop for CLI and CI smoke tests."""

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    fmri = torch.randn(8, 10, 6, generator=generator)
    eeg = fmri[..., :4] + 0.05 * torch.randn(8, 10, 4, generator=generator)
    target_future_fmri = torch.roll(fmri, shifts=-1, dims=1)
    target_future_fmri[:, -1] = fmri[:, -1]

    model = NeuralStateSpaceTranslator(
        input_dims={"fmri": 6, "eeg": 4},
        output_dims={"fmri": 6},
        latent_dim=32,
        n_layers=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2)
    loss_fn = nn.MSELoss()
    batch = {"fmri": fmri, "eeg": eeg}

    with torch.no_grad():
        initial_loss = float(loss_fn(model(batch, target_modality="fmri"), target_future_fmri))

    final_loss = initial_loss
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch, target_modality="fmri")
        loss = loss_fn(prediction, target_future_fmri)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    return TrainingSmokeResult(initial_loss=initial_loss, final_loss=final_loss, steps=steps)
