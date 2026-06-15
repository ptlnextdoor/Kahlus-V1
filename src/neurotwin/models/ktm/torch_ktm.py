"""Trainable PyTorch KTM (Kahlus Transition Model) for the v3 training harness.

PROPOSED / SYNTHETIC ONLY. A minimal, trainable ``nn.Module`` sibling of the numpy KTM
scaffold (``neurotwin.models.ktm.ktm.KTM``). It mirrors the scaffold's interface and tensor
shapes so the Transition Gym stays the authoritative grader, but adds autograd parameters so
the Sprint 2A harness can actually train it. The numpy scaffold is left untouched.

Pipeline: history EEG window -> linear encoder (tanh) -> leaky memory projection ->
perturbation-conditioned latent operators + response-profile decoder + uncertainty head. Heavier
KTM modules (lie generators, NeuroExperts, active experiment designer) are intentionally out of
scope here, but the trainable path now models perturbation-specific latent transitions rather than
only a flat per-perturbation linear readout.

This is not a built model and not a scientific claim. A decreasing loss earns only the narrow
``synthetic_ktm_training_harness`` scope, never ``synthetic_ktm_recovery``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TorchKTMConfig:
    """Config for :class:`TorchKTM`; field names/shapes mirror the numpy ``KTMConfig``."""

    seed: int = 0
    history_len: int = 6  # L
    eeg_channels: int = 5  # C
    n_perturbations: int = 4  # K
    horizon: int = 5  # H
    embed_dim: int = 16
    memory_dim: int = 12
    memory_rho: float = 0.8
    uncertainty_floor: float = 1e-3
    use_operator_path: bool = True
    operator_init_scale: float = 0.02
    decoder_hidden_dim: int = 128
    use_sequence_encoder: bool = True
    use_profile_decoder: bool = True

    def validate(self) -> "TorchKTMConfig":
        positives = {
            "history_len": self.history_len,
            "eeg_channels": self.eeg_channels,
            "n_perturbations": self.n_perturbations,
            "horizon": self.horizon,
            "embed_dim": self.embed_dim,
            "memory_dim": self.memory_dim,
            "decoder_hidden_dim": self.decoder_hidden_dim,
        }
        for name, value in positives.items():
            if int(value) < 1:
                raise ValueError(f"TorchKTMConfig.{name} must be >= 1, got {value}")
        if not 0.0 <= self.memory_rho < 1.0:
            raise ValueError("TorchKTMConfig.memory_rho must be in [0, 1)")
        if self.uncertainty_floor <= 0.0:
            raise ValueError("TorchKTMConfig.uncertainty_floor must be positive")
        if self.operator_init_scale < 0.0:
            raise ValueError("TorchKTMConfig.operator_init_scale must be non-negative")
        return self


class TorchKTM(nn.Module):
    """Trainable KTM: ``(history, perturbation_index) -> (response_pred, log_var)``."""

    def __init__(self, config: TorchKTMConfig) -> None:
        super().__init__()
        self.config = config.validate()
        c = self.config
        in_dim = c.history_len * c.eeg_channels
        out_dim = c.horizon * c.eeg_channels

        self.encoder = nn.Linear(in_dim, c.embed_dim)
        self.memory_proj = nn.Linear(c.embed_dim, c.memory_dim)
        if c.use_sequence_encoder:
            self.sequence_input = nn.Linear(c.eeg_channels, c.embed_dim)
            self.sequence_core = nn.GRU(c.embed_dim, c.memory_dim, batch_first=True)
        else:
            self.sequence_input = None
            self.sequence_core = None
        # One linear response head per perturbation a_k: memory -> flattened (H*C) trajectory.
        self.response_weight = nn.Parameter(torch.empty(c.n_perturbations, c.memory_dim, out_dim))
        self.response_bias = nn.Parameter(torch.zeros(c.n_perturbations, out_dim))
        if c.use_operator_path:
            self.operator_weight = nn.Parameter(
                torch.empty(c.n_perturbations, c.memory_dim, c.memory_dim)
            )
            self.operator_bias = nn.Parameter(torch.zeros(c.n_perturbations, c.memory_dim))
            self.operator_decoder = nn.Sequential(
                nn.Linear(c.memory_dim, c.decoder_hidden_dim),
                nn.GELU(),
                nn.Linear(c.decoder_hidden_dim, out_dim),
            )
            self.operator_response_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_parameter("operator_weight", None)
            self.register_parameter("operator_bias", None)
            self.operator_decoder = None
            self.register_parameter("operator_response_scale", None)
        if c.use_profile_decoder:
            self.profile_decoder = nn.Sequential(
                nn.Linear(c.memory_dim, c.decoder_hidden_dim),
                nn.GELU(),
                nn.Linear(c.decoder_hidden_dim, c.n_perturbations * out_dim),
            )
            self.profile_decoder_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.profile_decoder = None
            self.register_parameter("profile_decoder_scale", None)
        self.uncertainty = nn.Linear(c.memory_dim, c.n_perturbations)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Deterministic init for reproducible smoke runs (seed via cfg, isolated generator).
        gen = torch.Generator().manual_seed(int(self.config.seed))
        for module in (self.encoder, self.memory_proj, self.uncertainty):
            bound = 1.0 / max(1, module.in_features) ** 0.5
            module.weight.data.uniform_(-bound, bound, generator=gen)
            if module.bias is not None:
                module.bias.data.zero_()
        if self.config.use_sequence_encoder:
            assert self.sequence_input is not None
            assert self.sequence_core is not None
            bound = 1.0 / max(1, self.sequence_input.in_features) ** 0.5
            self.sequence_input.weight.data.uniform_(-bound, bound, generator=gen)
            self.sequence_input.bias.data.zero_()
            for name, param in self.sequence_core.named_parameters():
                if "weight" in name:
                    gru_bound = 1.0 / max(1, param.shape[1]) ** 0.5
                    param.data.uniform_(-gru_bound, gru_bound, generator=gen)
                else:
                    param.data.zero_()
        bound = 1.0 / max(1, self.config.memory_dim) ** 0.5
        self.response_weight.data.uniform_(-bound, bound, generator=gen)
        self.response_bias.data.zero_()
        if self.config.use_operator_path:
            eye = torch.eye(self.config.memory_dim, dtype=self.operator_weight.dtype)
            eye = eye.unsqueeze(0).expand(self.config.n_perturbations, -1, -1)
            noise = torch.empty_like(self.operator_weight)
            noise.normal_(mean=0.0, std=float(self.config.operator_init_scale), generator=gen)
            self.operator_weight.data.copy_(eye + noise)
            self.operator_bias.data.zero_()
            assert self.operator_decoder is not None
            for module in self.operator_decoder:
                if isinstance(module, nn.Linear):
                    op_bound = 1.0 / max(1, module.in_features) ** 0.5
                    module.weight.data.uniform_(-op_bound, op_bound, generator=gen)
                    if module.bias is not None:
                        module.bias.data.zero_()
            assert self.operator_response_scale is not None
            self.operator_response_scale.data.fill_(1.0)
        if self.config.use_profile_decoder:
            assert self.profile_decoder is not None
            for module in self.profile_decoder:
                if isinstance(module, nn.Linear):
                    profile_bound = 1.0 / max(1, module.in_features) ** 0.5
                    module.weight.data.uniform_(-profile_bound, profile_bound, generator=gen)
                    if module.bias is not None:
                        module.bias.data.zero_()
            assert self.profile_decoder_scale is not None
            self.profile_decoder_scale.data.fill_(1.0)

    def _memory(self, history: torch.Tensor) -> torch.Tensor:
        if self.config.use_sequence_encoder:
            assert self.sequence_input is not None
            assert self.sequence_core is not None
            token = torch.tanh(self.sequence_input(history))
            _seq, hidden = self.sequence_core(token)
            return (1.0 - self.config.memory_rho) * torch.tanh(hidden[-1])
        flat = history.reshape(history.shape[0], -1)
        embed = torch.tanh(self.encoder(flat))
        # Leaky one-shot memory (matches the numpy scaffold's project = (1-rho)*tanh(...)).
        return (1.0 - self.config.memory_rho) * torch.tanh(self.memory_proj(embed))

    def _flat_response_profile(self, mem: torch.Tensor) -> torch.Tensor:
        c = self.config
        base = torch.einsum("bm,kmo->bko", mem, self.response_weight) + self.response_bias
        if not self.config.use_operator_path:
            combined = base
        else:
            assert self.operator_weight is not None
            assert self.operator_bias is not None
            assert self.operator_decoder is not None
            assert self.operator_response_scale is not None
            operated = torch.einsum("bm,kmd->bkd", mem, self.operator_weight) + self.operator_bias
            operator_response = self.operator_decoder(operated)
            combined = base + self.operator_response_scale * operator_response
        if self.config.use_profile_decoder:
            assert self.profile_decoder is not None
            assert self.profile_decoder_scale is not None
            direct = self.profile_decoder(mem).reshape(
                mem.shape[0], c.n_perturbations, c.horizon * c.eeg_channels
            )
            combined = combined + self.profile_decoder_scale * direct
        return combined

    def forward(
        self,
        history: torch.Tensor,
        perturbation_index: torch.Tensor,
        *,
        return_profile: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.config
        mem = self._memory(history)  # (B, memory_dim)
        idx = perturbation_index.long()
        profile = self._flat_response_profile(mem)
        gather_idx = idx.reshape(-1, 1, 1).expand(-1, 1, c.horizon * c.eeg_channels)
        flat_pred = profile.gather(1, gather_idx).squeeze(1)  # (B, H*C)
        pred = flat_pred.reshape(history.shape[0], c.horizon, c.eeg_channels)
        var_all = torch.nn.functional.softplus(self.uncertainty(mem)) + c.uncertainty_floor
        var = var_all.gather(1, idx.unsqueeze(1)).squeeze(1)  # (B,)
        if return_profile:
            shaped_profile = profile.reshape(
                history.shape[0], c.n_perturbations, c.horizon, c.eeg_channels
            )
            return pred, torch.log(var), shaped_profile
        return pred, torch.log(var)

    def predict_response_profile(self, history: torch.Tensor) -> torch.Tensor:
        """Finite response profile C_K(h): ``(B, K, H, C)`` over the locked battery."""

        c = self.config
        mem = self._memory(history)
        flat = self._flat_response_profile(mem)
        return flat.reshape(history.shape[0], c.n_perturbations, c.horizon, c.eeg_channels)

    def predict_uncertainty(self, history: torch.Tensor) -> torch.Tensor:
        """Per-perturbation positive variances ``(B, K)``."""

        mem = self._memory(history)
        return torch.nn.functional.softplus(self.uncertainty(mem)) + self.config.uncertainty_floor

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))
