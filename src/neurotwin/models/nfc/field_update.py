from __future__ import annotations

import torch
from torch import nn

from neurotwin.models.nfc.pair_kernel import LowRankPairKernel


class FieldUpdateOperator(nn.Module):
    """Causal latent-field evolution operator."""

    def __init__(
        self,
        latent_dim: int,
        stimulus_dim: int = 0,
        anatomy_dim: int = 0,
        subject_dim: int = 0,
        backend: str = "gru",
        pair_rank: int = 8,
        use_pair_kernel: bool = False,
    ) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.stimulus_dim = max(0, int(stimulus_dim))
        self.anatomy_dim = max(0, int(anatomy_dim))
        self.subject_dim = max(0, int(subject_dim))
        self.backend = backend
        self.pair_kernel = LowRankPairKernel(self.latent_dim, rank=pair_rank, use_pair_state=use_pair_kernel)
        condition_dim = self.latent_dim + self.stimulus_dim + self.subject_dim + 1
        self.delta = nn.Sequential(
            nn.Linear(condition_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.state_gate = nn.Linear(self.latent_dim * 2, self.latent_dim)

    def forward(
        self,
        field: torch.Tensor,
        *,
        stimulus_state: torch.Tensor | None = None,
        anatomy: torch.Tensor | None = None,
        subject_state: torch.Tensor | None = None,
        dt: float = 1.0,
    ) -> torch.Tensor:
        if field.ndim != 4:
            raise ValueError("field must have shape [batch, time, nodes, latent_dim]")
        batch, time, nodes, latent_dim = field.shape
        if latent_dim != self.latent_dim:
            raise ValueError(f"field latent dim {latent_dim} does not match {self.latent_dim}")
        structural_prior = anatomy if anatomy is not None and anatomy.ndim == 2 else None
        outputs: list[torch.Tensor] = []
        previous = field[:, 0]
        for step in range(time):
            current = field[:, step]
            pieces = [current]
            if self.stimulus_dim:
                stimulus_t = _condition_at_step(stimulus_state, batch, time, self.stimulus_dim, step, field)
                pieces.append(stimulus_t.view(batch, 1, self.stimulus_dim).expand(batch, nodes, self.stimulus_dim))
            if self.subject_dim:
                subject = _subject_condition(subject_state, batch, self.subject_dim, field)
                pieces.append(subject.view(batch, 1, self.subject_dim).expand(batch, nodes, self.subject_dim))
            dt_column = torch.full((batch, nodes, 1), float(dt), dtype=field.dtype, device=field.device)
            pieces.append(dt_column)
            delta = self.delta(torch.cat(pieces, dim=-1))
            if step:
                gate = torch.sigmoid(self.state_gate(torch.cat((current, previous), dim=-1)))
                current = gate * current + (1.0 - gate) * previous
            updated = current + float(dt) * delta
            updated = self.pair_kernel(updated, structural_prior=structural_prior)
            outputs.append(updated)
            previous = updated
        return torch.stack(outputs, dim=1)


def _condition_at_step(
    stimulus_state: torch.Tensor | None,
    batch: int,
    time: int,
    dim: int,
    step: int,
    like: torch.Tensor,
) -> torch.Tensor:
    if stimulus_state is None:
        return torch.zeros(batch, dim, dtype=like.dtype, device=like.device)
    if stimulus_state.shape[:2] != (batch, time) or stimulus_state.shape[-1] != dim:
        raise ValueError("stimulus_state shape does not match field and stimulus_dim")
    return stimulus_state[:, step].to(device=like.device, dtype=like.dtype)


def _subject_condition(
    subject_state: torch.Tensor | None,
    batch: int,
    dim: int,
    like: torch.Tensor,
) -> torch.Tensor:
    if subject_state is None:
        return torch.zeros(batch, dim, dtype=like.dtype, device=like.device)
    if subject_state.shape != (batch, dim):
        raise ValueError("subject_state shape does not match field and subject_dim")
    return subject_state.to(device=like.device, dtype=like.dtype)
