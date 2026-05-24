from __future__ import annotations

import torch
from torch import nn


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
        if latent_dim % n_heads != 0:
            raise ValueError("latent_dim must be divisible by n_heads")
        self.input = nn.Linear(input_dim, latent_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output = nn.Linear(latent_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_sequence_tensor(x)
        return self.output(self.encoder(self.input(x)))


class TinySSMBaseline(nn.Module):
    """CPU debug stand-in for long-sequence SSM baselines until Mamba is pinned."""

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


class NeuralStateSpaceTranslator(nn.Module):
    """Modality-tokenizer + shared latent dynamics + modality-readout scaffold."""

    def __init__(
        self,
        input_dims: dict[str, int],
        output_dims: dict[str, int],
        latent_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.0,
        subject_adapter_dim: int = 0,
        projection_dim: int = 64,
        metadata_dim: int = 0,
        geometry_dim: int = 0,
    ) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("input_dims cannot be empty")
        if not output_dims:
            raise ValueError("output_dims cannot be empty")
        self.input_dims = dict(input_dims)
        self.output_dims = dict(output_dims)
        self.latent_dim = latent_dim
        self.subject_adapter_dim = int(subject_adapter_dim)
        self.metadata_dim = int(metadata_dim)
        self.geometry_dim = int(geometry_dim)
        self.tokenizers = nn.ModuleDict(
            {modality: nn.Linear(dim, latent_dim) for modality, dim in sorted(input_dims.items())}
        )
        self.geometry_encoders = nn.ModuleDict(
            {modality: nn.Linear(geometry_dim, latent_dim) for modality in sorted(input_dims)}
        ) if geometry_dim > 0 else nn.ModuleDict()
        self.metadata_encoder = nn.Linear(metadata_dim, latent_dim) if metadata_dim > 0 else None
        self.modality_embeddings = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.zeros(1, 1, latent_dim))
                for modality in sorted(input_dims)
            }
        )
        self.core = nn.GRU(
            latent_dim,
            latent_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
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
            nn.Linear(latent_dim, projection_dim),
        )
        self.subject_adapter = (
            nn.Sequential(
                nn.Linear(latent_dim, subject_adapter_dim),
                nn.GELU(),
                nn.Linear(subject_adapter_dim, latent_dim),
            )
            if subject_adapter_dim > 0
            else None
        )
        self._reset_parameters()

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        target_modality: str,
        metadata: torch.Tensor | None = None,
        geometry: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if target_modality not in self.readouts:
            raise ValueError(f"Unknown target modality {target_modality!r}")
        latent = self.encode(batch, metadata=metadata, geometry=geometry)
        return self.readouts[target_modality](latent)

    def encode(
        self,
        batch: dict[str, torch.Tensor],
        metadata: torch.Tensor | None = None,
        geometry: dict[str, torch.Tensor] | None = None,
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
        latent, _ = self.core(fused)
        if self.subject_adapter is not None:
            latent = latent + self.subject_adapter(latent)
        return latent

    def forward_task(
        self,
        batch: dict[str, torch.Tensor],
        target_modality: str,
        task: str = "reconstruction",
        metadata: torch.Tensor | None = None,
        geometry: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if target_modality not in self.readouts:
            raise ValueError(f"Unknown target modality {target_modality!r}")
        latent = self.encode(batch, metadata=metadata, geometry=geometry)
        if task == "forecast":
            prediction = self.forecast_heads[target_modality](latent)
        elif task == "reconstruction":
            prediction = self.reconstruction_heads[target_modality](latent)
        else:
            prediction = self.readouts[target_modality](latent)
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
