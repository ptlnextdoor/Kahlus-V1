from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import nn

from neurotwin.models.causal import CausalConv1d, CausalTransformerEncoder


class TinyTransformerBaseline(nn.Module):
    """Small Transformer baseline for CPU shape and smoke tests."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, latent_dim)
        self.encoder = CausalTransformerEncoder(
            latent_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.output = nn.Linear(latent_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_sequence_tensor(x)
        return self.output(self.encoder(self.input(x)))


class TinySSMBaseline(nn.Module):
    """Small stable diagonal state-space baseline for causal sequence prediction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        self.input = nn.Linear(input_dim, latent_dim)
        self.layers = nn.ModuleList(
            _DiagonalStateSpaceLayer(latent_dim, dropout=dropout if layer < n_layers - 1 else 0.0)
            for layer in range(n_layers)
        )
        self.output = nn.Linear(latent_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_sequence_tensor(x)
        latent = self.input(x)
        for layer in self.layers:
            latent = layer(latent)
        return self.output(latent)


class _DiagonalStateSpaceLayer(nn.Module):
    def __init__(self, latent_dim: int, *, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(latent_dim, latent_dim)
        self.logit_decay = nn.Parameter(torch.zeros(latent_dim))
        self.norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        decay = torch.sigmoid(self.logit_decay).view(1, -1)
        state = values.new_zeros(values.shape[0], values.shape[2])
        outputs: list[torch.Tensor] = []
        projected = self.input_projection(values)
        for step in range(values.shape[1]):
            state = decay * state + (1.0 - decay) * projected[:, step]
            outputs.append(state)
        return self.dropout(torch.nn.functional.gelu(self.norm(torch.stack(outputs, dim=1))))


class TinyGRUBaseline(nn.Module):
    """Legacy GRU sequence baseline retained for the historical ssm_fallback ID."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, latent_dim)
        self.core = nn.GRU(
            latent_dim,
            latent_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output = nn.Linear(latent_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_sequence_tensor(x)
        latent, _ = self.core(self.input(x))
        return self.output(latent)


@dataclass(frozen=True)
class NeuralStateSpaceTranslatorConfig:
    """Architecture and adaptation options for the torch translator."""

    latent_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.0
    subject_adapter_dim: int = 0
    projection_dim: int = 64
    metadata_dim: int = 0
    geometry_dim: int = 0
    backbone: str = "ssm_fallback"
    encoder: str = "auto"
    n_heads: int = 4
    subject_vocab_size: int = 0
    use_subject_embeddings: bool = False
    adapter_mode: str = "disabled"
    gradient_checkpointing: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], *, strict: bool = False) -> "NeuralStateSpaceTranslatorConfig":
        field_names = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - field_names)
        if strict and unknown:
            raise TypeError(f"Unknown NeuralStateSpaceTranslatorConfig option(s): {', '.join(unknown)}")
        selected = {key: values[key] for key in field_names if key in values}
        return cls(
            latent_dim=int(selected.get("latent_dim", cls.latent_dim)),
            n_layers=int(selected.get("n_layers", cls.n_layers)),
            dropout=float(selected.get("dropout", cls.dropout)),
            subject_adapter_dim=int(selected.get("subject_adapter_dim", cls.subject_adapter_dim)),
            projection_dim=int(selected.get("projection_dim", cls.projection_dim)),
            metadata_dim=int(selected.get("metadata_dim", cls.metadata_dim)),
            geometry_dim=int(selected.get("geometry_dim", cls.geometry_dim)),
            backbone=str(selected.get("backbone", cls.backbone)),
            encoder=str(selected.get("encoder", cls.encoder)),
            n_heads=int(selected.get("n_heads", cls.n_heads)),
            subject_vocab_size=int(selected.get("subject_vocab_size", cls.subject_vocab_size)),
            use_subject_embeddings=bool(selected.get("use_subject_embeddings", cls.use_subject_embeddings)),
            adapter_mode=str(selected.get("adapter_mode", cls.adapter_mode)),
            gradient_checkpointing=bool(selected.get("gradient_checkpointing", cls.gradient_checkpointing)),
        )

    def describe(self) -> str:
        return (
            f"NeuralStateSpaceTranslator(latent_dim={self.latent_dim}, "
            f"n_layers={self.n_layers}, backbone={self.backbone}, "
            f"encoder={self.encoder}, adapter_mode={self.adapter_mode})"
        )


class NeuralStateSpaceTranslator(nn.Module):
    """Modality encoders + shared latent dynamics + modality readouts."""

    def __init__(
        self,
        input_dims: dict[str, int],
        output_dims: dict[str, int],
        config: NeuralStateSpaceTranslatorConfig | Mapping[str, Any] | None = None,
        **options: Any,
    ) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("input_dims cannot be empty")
        if not output_dims:
            raise ValueError("output_dims cannot be empty")
        resolved = _resolve_translator_config(config, options)
        latent_dim = resolved.latent_dim
        self.input_dims = dict(input_dims)
        self.output_dims = dict(output_dims)
        self.config = resolved
        self.latent_dim = latent_dim
        self.subject_adapter_dim = resolved.subject_adapter_dim
        self.metadata_dim = resolved.metadata_dim
        self.geometry_dim = resolved.geometry_dim
        self.backbone_type = resolved.backbone
        self.encoder_type = resolved.encoder
        self.adapter_mode = resolved.adapter_mode
        self.gradient_checkpointing = resolved.gradient_checkpointing
        self.use_subject_embeddings = resolved.use_subject_embeddings and self.adapter_mode in {"few_shot", "enabled", "subject"}
        self.tokenizers = nn.ModuleDict(
            {
                modality: _build_modality_encoder(
                    modality,
                    dim,
                    latent_dim,
                    encoder=self.encoder_type,
                    dropout=resolved.dropout,
                )
                for modality, dim in sorted(input_dims.items())
            }
        )
        self.geometry_encoders = nn.ModuleDict(
            {modality: nn.Linear(resolved.geometry_dim, latent_dim) for modality in sorted(input_dims)}
        ) if resolved.geometry_dim > 0 else nn.ModuleDict()
        self.metadata_encoder = nn.Linear(resolved.metadata_dim, latent_dim) if resolved.metadata_dim > 0 else None
        self.subject_embedding = (
            nn.Embedding(resolved.subject_vocab_size, latent_dim)
            if self.use_subject_embeddings and resolved.subject_vocab_size > 0
            else None
        )
        self.modality_embeddings = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.zeros(1, 1, latent_dim))
                for modality in sorted(input_dims)
            }
        )
        self.core = _build_backbone(
            backbone=self.backbone_type,
            latent_dim=latent_dim,
            n_layers=resolved.n_layers,
            n_heads=resolved.n_heads,
            dropout=resolved.dropout,
        )
        self.readouts = nn.ModuleDict(
            {modality: nn.Linear(latent_dim, dim) for modality, dim in sorted(output_dims.items())}
        )
        self.reconstruction_heads = nn.ModuleDict(
            {modality: nn.Linear(latent_dim, dim) for modality, dim in sorted(output_dims.items())}
        )
        self.forecast_heads = nn.ModuleDict(
            {modality: nn.Linear(latent_dim, dim) for modality, dim in sorted(output_dims.items())}
        )
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, resolved.projection_dim),
        )
        self.subject_adapter = (
            nn.Sequential(
                nn.Linear(latent_dim, resolved.subject_adapter_dim),
                nn.GELU(),
                nn.Linear(resolved.subject_adapter_dim, latent_dim),
            )
            if resolved.subject_adapter_dim > 0
            else None
        )
        self._reset_parameters()

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        target_modality: str,
        metadata: torch.Tensor | None = None,
        geometry: dict[str, torch.Tensor] | None = None,
        subject_ids: torch.Tensor | None = None,
        task: str = "readout",
    ) -> torch.Tensor:
        return self.forward_task(
            batch,
            target_modality=target_modality,
            task=task,
            metadata=metadata,
            geometry=geometry,
            subject_ids=subject_ids,
        )["prediction"]

    def encode(
        self,
        batch: dict[str, torch.Tensor],
        metadata: torch.Tensor | None = None,
        geometry: dict[str, torch.Tensor] | None = None,
        subject_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode observed modalities into shared latent neural tokens."""

        if not batch:
            raise ValueError("batch cannot be empty")

        encoded: list[torch.Tensor] = []
        batch_shape: tuple[int, int] | None = None
        for modality, x in sorted(batch.items()):
            if modality not in self.tokenizers:
                raise ValueError(f"Unknown input modality {modality!r}")
            _check_sequence_tensor(x)
            expected_dim = self.input_dims[modality]
            if x.shape[-1] != expected_dim:
                raise ValueError(f"{modality} feature dim {x.shape[-1]} does not match expected {expected_dim}")
            current_shape = (int(x.shape[0]), int(x.shape[1]))
            if batch_shape is None:
                batch_shape = current_shape
            elif current_shape != batch_shape:
                raise ValueError("All modalities in a fused batch must share batch and time axes")
            token = self.tokenizers[modality](x) + self.modality_embeddings[modality]
            if geometry is not None and modality in geometry and self.geometry_dim > 0:
                geom = geometry[modality]
                _check_sequence_tensor(geom)
                if geom.shape[:2] != x.shape[:2] or geom.shape[-1] != self.geometry_dim:
                    raise ValueError(f"{modality} geometry must have shape [batch, time, {self.geometry_dim}]")
                token = token + self.geometry_encoders[modality](geom)
            encoded.append(token)

        fused = torch.stack(encoded, dim=0).mean(dim=0)
        if metadata is not None:
            if self.metadata_encoder is None:
                raise ValueError("metadata was provided but metadata_dim is 0")
            _check_sequence_tensor(metadata)
            if metadata.shape[:2] != fused.shape[:2] or metadata.shape[-1] != self.metadata_dim:
                raise ValueError(f"metadata must have shape [batch, time, {self.metadata_dim}]")
            fused = fused + self.metadata_encoder(metadata)
        if subject_ids is not None and self.subject_embedding is not None:
            if subject_ids.ndim != 1 or subject_ids.shape[0] != fused.shape[0]:
                raise ValueError("subject_ids must have shape [batch]")
            fused = fused + self.subject_embedding(subject_ids).unsqueeze(1)
        latent = self.core(fused)
        if self.subject_adapter is not None and self.adapter_mode in {"few_shot", "enabled", "subject"}:
            latent = latent + self.subject_adapter(latent)
        return latent

    def forward_task(
        self,
        batch: dict[str, torch.Tensor],
        target_modality: str,
        task: str = "reconstruction",
        metadata: torch.Tensor | None = None,
        geometry: dict[str, torch.Tensor] | None = None,
        subject_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if target_modality not in self.readouts:
            raise ValueError(f"Unknown target modality {target_modality!r}")
        latent = self.encode(batch, metadata=metadata, geometry=geometry, subject_ids=subject_ids)
        if task == "forecast":
            prediction = self.forecast_heads[target_modality](latent)
        elif task == "reconstruction":
            prediction = self.reconstruction_heads[target_modality](latent)
        elif task == "readout":
            prediction = self.readouts[target_modality](latent)
        else:
            raise ValueError("task must be one of 'forecast', 'reconstruction', or 'readout'")
        return {
            "prediction": prediction,
            "latent": latent,
            "projection": self.projection_head(latent.mean(dim=1)),
        }

    def _reset_parameters(self) -> None:
        for embedding in self.modality_embeddings.values():
            nn.init.normal_(embedding, mean=0.0, std=0.02)


def _check_sequence_tensor(x: torch.Tensor) -> None:
    if x.ndim != 3:
        raise ValueError("Expected tensor with shape [batch, time, features]")


def _resolve_translator_config(
    config: NeuralStateSpaceTranslatorConfig | Mapping[str, Any] | None,
    options: Mapping[str, Any],
) -> NeuralStateSpaceTranslatorConfig:
    if config is None:
        return NeuralStateSpaceTranslatorConfig.from_mapping(options, strict=True)
    if options:
        raise TypeError("Pass translator architecture options either through config or keyword options, not both")
    if isinstance(config, NeuralStateSpaceTranslatorConfig):
        return config
    if isinstance(config, Mapping):
        return NeuralStateSpaceTranslatorConfig.from_mapping(config, strict=True)
    raise TypeError("config must be a NeuralStateSpaceTranslatorConfig or mapping")


class _LinearModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_sequence_tensor(x)
        return self.net(x)


class _TemporalConvModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, latent_dim)
        self.temporal = nn.Sequential(
            CausalConv1d(latent_dim, latent_dim, kernel_size=3),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(latent_dim, latent_dim, kernel_size=1),
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_sequence_tensor(x)
        token = self.input(x)
        temporal = self.temporal(token.transpose(1, 2)).transpose(1, 2)
        return self.norm(token + temporal)


class _TransformerBackbone(nn.Module):
    def __init__(self, latent_dim: int, n_layers: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.net = CausalTransformerEncoder(
            latent_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SSMFallbackBackbone(nn.Module):
    def __init__(self, latent_dim: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.GRU(
            latent_dim,
            latent_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, _ = self.net(x)
        return latent


def _build_modality_encoder(modality: str, input_dim: int, latent_dim: int, encoder: str, dropout: float) -> nn.Module:
    encoder = encoder.lower()
    if encoder == "linear":
        return _LinearModalityEncoder(input_dim, latent_dim, dropout)
    if encoder in {"temporal_conv", "conv"}:
        return _TemporalConvModalityEncoder(input_dim, latent_dim, dropout)
    if encoder != "auto":
        raise ValueError(f"Unknown encoder {encoder!r}")
    if modality in {"eeg", "meg", "spikes", "generic"}:
        return _TemporalConvModalityEncoder(input_dim, latent_dim, dropout)
    return _LinearModalityEncoder(input_dim, latent_dim, dropout)


def _build_backbone(backbone: str, latent_dim: int, n_layers: int, n_heads: int, dropout: float) -> nn.Module:
    backbone = backbone.lower()
    if backbone == "transformer":
        return _TransformerBackbone(latent_dim, n_layers, n_heads, dropout)
    if backbone == "mamba":
        raise ValueError('backbone="mamba" is not wired in v1; use backbone="ssm_fallback" until Mamba support is pinned')
    if backbone == "ssm":
        raise ValueError('backbone="ssm" is ambiguous; use TinySSMBaseline or explicitly select "gru"')
    if backbone in {"ssm_fallback", "gru"}:
        return _SSMFallbackBackbone(latent_dim, n_layers, dropout)
    raise ValueError(f"Unknown backbone {backbone!r}")
