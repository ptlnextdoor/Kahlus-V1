from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import sqrt
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class NeuroTwinPairOperatorConfig:
    """Small fMRI-first pairwise neural operator config."""

    latent_dim: int = 64
    n_layers: int = 1
    dropout: float = 0.0
    pair_rank: int = 8
    pair_top_k: int = 0
    network_blocks: int = 1
    pair_confidence_max_parcels: int = 256
    backbone: str = "gru"
    n_heads: int = 4
    projection_dim: int = 32
    use_pair_state: bool = True
    use_uncertainty_head: bool = True
    use_pair_uncertainty: bool = False
    refinement_steps: int = 1
    hrf_delay_steps: int = 1

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], *, strict: bool = False) -> "NeuroTwinPairOperatorConfig":
        field_names = {field.name for field in fields(cls)}
        ignored = {"type", "input_dims", "output_dims", "encoder", "metadata_dim", "geometry_dim", "subject_adapter_dim",
                   "subject_vocab_size", "use_subject_embeddings", "adapter_mode", "gradient_checkpointing"}
        unknown = sorted(set(values) - field_names - ignored)
        if strict and unknown:
            raise TypeError(f"Unknown NeuroTwinPairOperatorConfig option(s): {', '.join(unknown)}")
        selected = {key: values[key] for key in field_names if key in values}
        return cls(
            latent_dim=int(selected.get("latent_dim", cls.latent_dim)),
            n_layers=int(selected.get("n_layers", cls.n_layers)),
            dropout=float(selected.get("dropout", cls.dropout)),
            pair_rank=int(selected.get("pair_rank", cls.pair_rank)),
            pair_top_k=max(0, int(selected.get("pair_top_k", cls.pair_top_k))),
            network_blocks=max(1, int(selected.get("network_blocks", cls.network_blocks))),
            pair_confidence_max_parcels=max(0, int(selected.get("pair_confidence_max_parcels", cls.pair_confidence_max_parcels))),
            backbone=str(selected.get("backbone", cls.backbone)),
            n_heads=int(selected.get("n_heads", cls.n_heads)),
            projection_dim=int(selected.get("projection_dim", cls.projection_dim)),
            use_pair_state=bool(selected.get("use_pair_state", cls.use_pair_state)),
            use_uncertainty_head=bool(selected.get("use_uncertainty_head", cls.use_uncertainty_head)),
            use_pair_uncertainty=bool(selected.get("use_pair_uncertainty", cls.use_pair_uncertainty)),
            refinement_steps=max(0, int(selected.get("refinement_steps", cls.refinement_steps))),
            hrf_delay_steps=max(0, int(selected.get("hrf_delay_steps", cls.hrf_delay_steps))),
        )


class NeuroTwinPairOperator(nn.Module):
    """Pairwise brain-region operator for fMRI prepared-window tasks.

    The first implementation is intentionally small and clean-room: it treats
    the target feature axis as fMRI parcels, builds low-rank parcel-pair mixing,
    and keeps the same forward_task contract as NeuralStateSpaceTranslator.
    """

    def __init__(
        self,
        input_dims: dict[str, int],
        output_dims: dict[str, int],
        config: NeuroTwinPairOperatorConfig | Mapping[str, Any] | None = None,
        **options: Any,
    ) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("input_dims cannot be empty")
        if not output_dims:
            raise ValueError("output_dims cannot be empty")
        resolved = _resolve_pair_operator_config(config, options)
        if resolved.latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if resolved.pair_rank < 1:
            raise ValueError("pair_rank must be positive")
        self.input_dims = dict(input_dims)
        self.output_dims = dict(output_dims)
        self.config = resolved
        self.latent_dim = resolved.latent_dim
        self.pair_rank = resolved.pair_rank
        self.pair_top_k = resolved.pair_top_k
        self.network_blocks = resolved.network_blocks
        self.pair_confidence_max_parcels = resolved.pair_confidence_max_parcels
        self.use_pair_state = resolved.use_pair_state
        self.use_uncertainty_head = resolved.use_uncertainty_head
        self.use_pair_uncertainty = resolved.use_pair_uncertainty
        self.refinement_steps = resolved.refinement_steps
        self.hrf_delay_steps = resolved.hrf_delay_steps

        self.value_projectors = nn.ModuleDict(
            {
                _pair_key(source, target): _build_value_projector(source_dim, target_dim)
                for source, source_dim in sorted(input_dims.items())
                for target, target_dim in sorted(output_dims.items())
            }
        )
        self.node_encoder = nn.Sequential(nn.Linear(1, resolved.latent_dim), nn.GELU(), nn.LayerNorm(resolved.latent_dim))
        self.stimulus_encoders = nn.ModuleDict(
            {
                modality: nn.Linear(dim, resolved.latent_dim)
                for modality, dim in sorted(input_dims.items())
                if modality == "stimulus"
            }
        )
        self.pair_left = nn.ParameterDict(
            {modality: nn.Parameter(torch.randn(dim, resolved.pair_rank) * 0.02) for modality, dim in sorted(output_dims.items())}
        )
        self.pair_right = nn.ParameterDict(
            {modality: nn.Parameter(torch.randn(dim, resolved.pair_rank) * 0.02) for modality, dim in sorted(output_dims.items())}
        )
        self.pair_block_gates = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.ones(resolved.network_blocks, resolved.pair_rank))
                for modality in sorted(output_dims)
            }
        )
        self.pair_update = nn.Sequential(nn.Linear(resolved.latent_dim, resolved.latent_dim), nn.GELU())
        self.hrf_adapter = nn.Sequential(
            nn.Conv1d(resolved.latent_dim, resolved.latent_dim, kernel_size=3, padding=resolved.hrf_delay_steps),
            nn.GELU(),
            nn.Conv1d(resolved.latent_dim, resolved.latent_dim, kernel_size=1),
        )
        self.temporal = _build_temporal_core(resolved)
        self.forecast_heads = nn.ModuleDict({modality: nn.Linear(resolved.latent_dim, 1) for modality in sorted(output_dims)})
        self.reconstruction_heads = nn.ModuleDict({modality: nn.Linear(resolved.latent_dim, 1) for modality in sorted(output_dims)})
        self.readouts = nn.ModuleDict({modality: nn.Linear(resolved.latent_dim, 1) for modality in sorted(output_dims)})
        self.refinement = nn.Sequential(nn.Linear(2, resolved.latent_dim), nn.GELU(), nn.Linear(resolved.latent_dim, 1))
        self.uncertainty_head = nn.Linear(resolved.latent_dim, 1)
        self.pair_uncertainty_head = nn.Linear(resolved.pair_rank, 1)
        self.projection_head = nn.Sequential(
            nn.Linear(resolved.latent_dim, resolved.latent_dim),
            nn.GELU(),
            nn.Linear(resolved.latent_dim, resolved.projection_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor], target_modality: str, **_: Any) -> torch.Tensor:
        return self.forward_task(batch, target_modality=target_modality, task="readout")["prediction"]

    def forward_task(
        self,
        batch: dict[str, torch.Tensor],
        target_modality: str,
        task: str = "reconstruction",
        metadata: torch.Tensor | None = None,
        geometry: dict[str, torch.Tensor] | None = None,
        subject_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del metadata, geometry, subject_ids
        if target_modality not in self.output_dims:
            raise ValueError(f"Unknown target modality {target_modality!r}")
        if not batch:
            raise ValueError("batch cannot be empty")
        source_modality, x = _select_source(batch, self.input_dims)
        _check_sequence_tensor(x)
        if x.shape[-1] != self.input_dims[source_modality]:
            raise ValueError(f"{source_modality} feature dim {x.shape[-1]} does not match expected {self.input_dims[source_modality]}")

        node_values = self.value_projectors[_pair_key(source_modality, target_modality)](x.float())
        node_tokens = self.node_encoder(node_values.unsqueeze(-1))
        if source_modality == "stimulus" and source_modality in self.stimulus_encoders:
            drive = self.stimulus_encoders[source_modality](x.float()).unsqueeze(2)
            node_tokens = node_tokens + drive
        if self.use_pair_state:
            pair_context = self._pair_context(target_modality, node_tokens)
            node_tokens = node_tokens + self.pair_update(pair_context)
        if target_modality == "fmri":
            node_tokens = node_tokens + self._hrf_context(node_tokens)
        latent = self._temporal_operator(node_tokens)

        if task == "forecast":
            prediction = self.forecast_heads[target_modality](latent).squeeze(-1)
        elif task == "reconstruction":
            prediction = self.reconstruction_heads[target_modality](latent).squeeze(-1)
        elif task == "readout":
            prediction = self.readouts[target_modality](latent).squeeze(-1)
        else:
            raise ValueError("task must be one of 'forecast', 'reconstruction', or 'readout'")
        if self.refinement_steps:
            prediction = self._refine_prediction(prediction, latent)
        latent_summary = latent.mean(dim=2)
        output = {
            "prediction": prediction,
            "latent": latent_summary,
            "projection": self.projection_head(latent_summary.mean(dim=1)),
            "pair_confidence": self._pair_confidence(target_modality, dtype=prediction.dtype, device=prediction.device),
            "expert_utilization": self._expert_utilization(source_modality, target_modality, task, prediction.device),
        }
        if self.use_uncertainty_head:
            output["uncertainty"] = torch.nn.functional.softplus(self.uncertainty_head(latent).squeeze(-1)) + 1e-6
        if self.use_pair_uncertainty:
            output["pair_uncertainty"] = self._pair_uncertainty(target_modality, dtype=prediction.dtype, device=prediction.device)
        return output

    def _pair_factors(self, target_modality: str, *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        left = self.pair_left[target_modality].to(device=device, dtype=dtype)
        right = self.pair_right[target_modality].to(device=device, dtype=dtype)
        if self.network_blocks > 1:
            block_ids = torch.arange(left.shape[0], device=device) % self.network_blocks
            block_gates = torch.sigmoid(self.pair_block_gates[target_modality].to(device=device, dtype=dtype))
            left = left * block_gates.index_select(0, block_ids)
        if 0 < self.pair_top_k < right.shape[0]:
            mask = torch.zeros_like(right)
            topk = torch.topk(right.abs(), k=self.pair_top_k, dim=0).indices
            mask.scatter_(0, topk, 1.0)
            right = right * mask
        return torch.tanh(left), torch.tanh(right)

    def _pair_context(self, target_modality: str, node_tokens: torch.Tensor) -> torch.Tensor:
        left, right = self._pair_factors(target_modality, dtype=node_tokens.dtype, device=node_tokens.device)
        source_summary = torch.einsum("nr,btnl->btrl", right, node_tokens)
        source_summary = source_summary / sqrt(float(max(1, right.shape[0])))
        context = torch.einsum("nr,btrl->btnl", left, source_summary)
        return context / sqrt(float(max(1, self.pair_rank)))

    def _pair_weights(self, target_modality: str, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        left, right = self._pair_factors(target_modality, dtype=dtype, device=device)
        weights = (left @ right.T) / sqrt(float(max(1, self.pair_rank)))
        return torch.softmax(weights, dim=-1)

    def _pair_confidence(self, target_modality: str, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if not self.use_pair_state:
            dim = self.output_dims[target_modality]
            if dim <= self.pair_confidence_max_parcels:
                return torch.eye(dim, dtype=dtype, device=device)
            return torch.zeros((2, dim, self.pair_rank), dtype=dtype, device=device)
        dim = self.output_dims[target_modality]
        left, right = self._pair_factors(target_modality, dtype=dtype, device=device)
        if dim <= self.pair_confidence_max_parcels:
            return torch.softmax((left @ right.T) / sqrt(float(max(1, self.pair_rank))), dim=-1)
        return torch.stack((left, right), dim=0)

    def _pair_uncertainty(self, target_modality: str, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        left, right = self._pair_factors(target_modality, dtype=dtype, device=device)
        compact = 0.5 * (left.abs() + right.abs())
        values = torch.nn.functional.softplus(self.pair_uncertainty_head(compact).squeeze(-1)) + 1e-6
        dim = self.output_dims[target_modality]
        if dim <= self.pair_confidence_max_parcels:
            return values[:, None].expand(dim, dim)
        return values

    def _hrf_context(self, node_tokens: torch.Tensor) -> torch.Tensor:
        batch, time, parcels, latent_dim = node_tokens.shape
        flat = node_tokens.permute(0, 2, 3, 1).reshape(batch * parcels, latent_dim, time)
        context = self.hrf_adapter(flat)
        if context.shape[-1] > time:
            context = context[..., :time]
        return context.reshape(batch, parcels, latent_dim, time).permute(0, 3, 1, 2)

    def _temporal_operator(self, node_tokens: torch.Tensor) -> torch.Tensor:
        batch, time, parcels, latent_dim = node_tokens.shape
        flat = node_tokens.permute(0, 2, 1, 3).reshape(batch * parcels, time, latent_dim)
        if isinstance(self.temporal, nn.GRU):
            evolved, _ = self.temporal(flat)
        else:
            evolved = self.temporal(flat)
        return evolved.reshape(batch, parcels, time, latent_dim).permute(0, 2, 1, 3)

    def _refine_prediction(self, prediction: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        current = prediction
        for _ in range(self.refinement_steps):
            uncertainty = torch.nn.functional.softplus(self.uncertainty_head(latent).squeeze(-1))
            current = current + self.refinement(torch.stack((current, uncertainty), dim=-1)).squeeze(-1)
        return current

    def _expert_utilization(self, source_modality: str, target_modality: str, task: str, device: torch.device) -> torch.Tensor:
        values = torch.zeros(6, dtype=torch.float32, device=device)
        values[0] = 1.0 if target_modality == "fmri" else 0.0
        values[1] = 1.0 if source_modality == "stimulus" else 0.0
        values[2] = 1.0 if self.use_pair_state else 0.0
        values[3] = 1.0 if task == "reconstruction" else 0.0
        values[4] = 1.0 if task == "forecast" else 0.0
        values[5] = 1.0 if self.use_uncertainty_head else 0.0
        return values / values.sum().clamp_min(1.0)


def _resolve_pair_operator_config(
    config: NeuroTwinPairOperatorConfig | Mapping[str, Any] | None,
    options: Mapping[str, Any],
) -> NeuroTwinPairOperatorConfig:
    if config is None:
        return NeuroTwinPairOperatorConfig.from_mapping(options, strict=True)
    if options:
        raise TypeError("Pass pair-operator architecture options either through config or keyword options, not both")
    if isinstance(config, NeuroTwinPairOperatorConfig):
        return config
    if isinstance(config, Mapping):
        return NeuroTwinPairOperatorConfig.from_mapping(config, strict=True)
    raise TypeError("config must be a NeuroTwinPairOperatorConfig or mapping")


def _build_temporal_core(config: NeuroTwinPairOperatorConfig) -> nn.Module:
    backbone = config.backbone.lower()
    if backbone in {"gru", "ssm", "ssm_fallback"}:
        return nn.GRU(
            config.latent_dim,
            config.latent_dim,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
        )
    if backbone == "transformer":
        if config.latent_dim % config.n_heads != 0:
            raise ValueError("latent_dim must be divisible by n_heads for transformer pair-operator backbone")
        layer = nn.TransformerEncoderLayer(
            d_model=config.latent_dim,
            nhead=config.n_heads,
            dim_feedforward=config.latent_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        return nn.TransformerEncoder(layer, num_layers=config.n_layers)
    raise ValueError(f"Unknown pair-operator backbone {config.backbone!r}")


def _build_value_projector(input_dim: int, output_dim: int) -> nn.Module:
    if input_dim == output_dim:
        return nn.Identity()
    return nn.Linear(input_dim, output_dim)


def _select_source(batch: dict[str, torch.Tensor], input_dims: dict[str, int]) -> tuple[str, torch.Tensor]:
    for preferred in ("stimulus", "fmri", "eeg", "meg"):
        if preferred in batch:
            return preferred, batch[preferred]
    for modality, value in sorted(batch.items()):
        if modality in input_dims:
            return modality, value
    raise ValueError("batch contains no known input modality")


def _pair_key(source: str, target: str) -> str:
    return f"{source}__to__{target}"


def _check_sequence_tensor(x: torch.Tensor) -> None:
    if x.ndim != 3:
        raise ValueError("Expected tensor with shape [batch, time, features]")
