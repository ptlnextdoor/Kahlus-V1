from __future__ import annotations

import torch
from torch import nn

from neurotwin.models.causal import CausalTransformerEncoder
from neurotwin.models.nfc.pair_kernel import LowRankPairKernel


class FieldUpdateOperator(nn.Module):
    """Causal temporal evolution followed by same-time sensor attention."""

    def __init__(
        self,
        latent_dim: int,
        stimulus_dim: int = 0,
        subject_dim: int = 0,
        backend: str = "gru",
        n_layers: int = 1,
        n_heads: int = 4,
        pair_rank: int = 8,
        use_pair_kernel: bool = False,
    ) -> None:
        super().__init__()
        if latent_dim < 1 or n_layers < 1:
            raise ValueError("latent_dim and n_layers must be positive")
        self.latent_dim = int(latent_dim)
        self.stimulus_dim = max(0, int(stimulus_dim))
        self.subject_dim = max(0, int(subject_dim))
        self.backend = str(backend).lower()
        condition_dim = self.latent_dim + self.stimulus_dim + self.subject_dim
        self.condition_projection = nn.Linear(condition_dim, self.latent_dim)
        if self.backend == "gru":
            self.temporal: nn.Module = nn.GRU(
                self.latent_dim,
                self.latent_dim,
                num_layers=int(n_layers),
                batch_first=True,
            )
        elif self.backend == "transformer":
            self.temporal = CausalTransformerEncoder(
                self.latent_dim,
                n_heads=int(n_heads),
                n_layers=int(n_layers),
            )
        else:
            raise ValueError(f"Unknown NFC field-update backend {backend!r}; expected 'gru' or 'transformer'")
        self.pair_kernel = LowRankPairKernel(
            self.latent_dim,
            rank=pair_rank,
            use_pair_state=use_pair_kernel,
        )

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
        if not float(dt) > 0.0:
            raise ValueError("dt must be positive")
        structural_prior = _structural_prior(anatomy, nodes)
        conditioned = self.condition_projection(
            torch.cat(
                (
                    field,
                    _stimulus_condition(stimulus_state, field, self.stimulus_dim),
                    _subject_condition(subject_state, field, self.subject_dim),
                ),
                dim=-1,
            )
        )
        flattened = conditioned.permute(0, 2, 1, 3).reshape(batch * nodes, time, latent_dim)
        if isinstance(self.temporal, nn.GRU):
            evolved, _ = self.temporal(flattened)
        else:
            evolved = self.temporal(flattened)
        evolved = evolved.reshape(batch, nodes, time, latent_dim).permute(0, 2, 1, 3)
        if float(dt) != 1.0:
            evolved = field + float(dt) * (evolved - field)
        mixed = self.pair_kernel(
            evolved.reshape(batch * time, nodes, latent_dim),
            structural_prior=structural_prior,
        )
        return mixed.reshape(batch, time, nodes, latent_dim)


def _stimulus_condition(
    stimulus_state: torch.Tensor | None,
    field: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    batch, time, nodes, _ = field.shape
    if dim == 0:
        return field.new_zeros(batch, time, nodes, 0)
    if stimulus_state is None:
        return field.new_zeros(batch, time, nodes, dim)
    if stimulus_state.shape != (batch, time, dim):
        raise ValueError("stimulus_state shape does not match field and stimulus_dim")
    return stimulus_state.to(device=field.device, dtype=field.dtype).unsqueeze(2).expand(batch, time, nodes, dim)


def _subject_condition(
    subject_state: torch.Tensor | None,
    field: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    batch, time, nodes, _ = field.shape
    if dim == 0:
        return field.new_zeros(batch, time, nodes, 0)
    if subject_state is None:
        return field.new_zeros(batch, time, nodes, dim)
    if subject_state.shape != (batch, dim):
        raise ValueError("subject_state shape does not match field and subject_dim")
    return subject_state.to(device=field.device, dtype=field.dtype).view(batch, 1, 1, dim).expand(batch, time, nodes, dim)


def _structural_prior(anatomy: torch.Tensor | None, nodes: int) -> torch.Tensor | None:
    if anatomy is None:
        return None
    if anatomy.ndim != 2 or anatomy.shape != (nodes, nodes):
        raise ValueError(f"structural_prior must have shape [{nodes}, {nodes}]")
    return anatomy
