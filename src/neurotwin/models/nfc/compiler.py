from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import nn

from neurotwin.models.nfc.field_update import FieldUpdateOperator
from neurotwin.models.nfc.latent_field import LatentNeuralField
from neurotwin.models.nfc.observations import BehaviorObservationOperator, EEGObservationOperator, FMRIObservationOperator
from neurotwin.models.nfc.stimulus import StimulusConditioningOperator
from neurotwin.models.nfc.uncertainty import UncertaintyMapHead


@dataclass(frozen=True)
class NeuralFieldCompilerConfig:
    latent_dim: int = 64
    n_layers: int = 1
    n_heads: int = 4
    pair_rank: int = 8
    projection_dim: int = 32
    backbone: str = "gru"
    use_pair_kernel: bool = True
    use_observation_operator: bool = True
    use_uncertainty: bool = True
    stimulus_lag_steps: int = 1
    hrf_delay_steps: int = 1
    subject_state_dim: int = 0
    geometry_dim: int = 0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], *, strict: bool = False) -> "NeuralFieldCompilerConfig":
        field_names = {field.name for field in fields(cls)}
        aliases = {
            "use_pair_state": "use_pair_kernel",
            "use_uncertainty_head": "use_uncertainty",
        }
        ignored = {
            "type",
            "input_dims",
            "output_dims",
            "encoder",
            "metadata_dim",
            "subject_adapter_dim",
            "subject_vocab_size",
            "use_subject_embeddings",
            "adapter_mode",
            "gradient_checkpointing",
            "refinement_steps",
        }
        normalized = dict(values)
        for source, target in aliases.items():
            if source in normalized and target not in normalized:
                normalized[target] = normalized[source]
        unknown = sorted(set(normalized) - field_names - ignored - set(aliases))
        if strict and unknown:
            raise TypeError(f"Unknown NeuralFieldCompilerConfig option(s): {', '.join(unknown)}")
        selected = {key: normalized[key] for key in field_names if key in normalized}
        return cls(
            latent_dim=int(selected.get("latent_dim", cls.latent_dim)),
            n_layers=max(1, int(selected.get("n_layers", cls.n_layers))),
            n_heads=max(1, int(selected.get("n_heads", cls.n_heads))),
            pair_rank=int(selected.get("pair_rank", cls.pair_rank)),
            projection_dim=int(selected.get("projection_dim", cls.projection_dim)),
            backbone=str(selected.get("backbone", cls.backbone)),
            use_pair_kernel=bool(selected.get("use_pair_kernel", cls.use_pair_kernel)),
            use_observation_operator=bool(selected.get("use_observation_operator", cls.use_observation_operator)),
            use_uncertainty=bool(selected.get("use_uncertainty", cls.use_uncertainty)),
            stimulus_lag_steps=max(0, int(selected.get("stimulus_lag_steps", cls.stimulus_lag_steps))),
            hrf_delay_steps=max(0, int(selected.get("hrf_delay_steps", cls.hrf_delay_steps))),
            subject_state_dim=max(0, int(selected.get("subject_state_dim", cls.subject_state_dim))),
            geometry_dim=max(0, int(selected.get("geometry_dim", cls.geometry_dim))),
        )


class NeuralFieldCompiler(nn.Module):
    """Experimental Neural Field Compiler architecture."""

    model_id = "neurotwin_nfc"
    model_status = "experimental_architecture"

    def __init__(
        self,
        input_dims: dict[str, int],
        output_dims: dict[str, int],
        config: NeuralFieldCompilerConfig | Mapping[str, Any] | None = None,
        **options: Any,
    ) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("input_dims cannot be empty")
        if not output_dims:
            raise ValueError("output_dims cannot be empty")
        resolved = _resolve_config(config, options)
        self.input_dims = dict(input_dims)
        self.output_dims = dict(output_dims)
        self.config = resolved
        self.latent_dim = resolved.latent_dim
        self.modality_logits = nn.ParameterDict(
            {modality: nn.Parameter(torch.zeros(())) for modality in sorted(input_dims)}
        )
        self.value_projectors = nn.ModuleDict(
            {
                _pair_key(source, target): nn.Linear(source_dim, target_dim)
                for source, source_dim in sorted(input_dims.items())
                for target, target_dim in sorted(output_dims.items())
            }
        )
        stimulus_dim = input_dims.get("stimulus", 0)
        self.stimulus_conditioner = (
            StimulusConditioningOperator(stimulus_dim, resolved.latent_dim, lag_steps=resolved.stimulus_lag_steps)
            if stimulus_dim
            else None
        )
        self.latent_field = LatentNeuralField(
            input_dim=1,
            latent_dim=resolved.latent_dim,
            subject_dim=resolved.subject_state_dim,
            stimulus_dim=resolved.latent_dim if self.stimulus_conditioner is not None else 0,
        )
        self.field_update = FieldUpdateOperator(
            latent_dim=resolved.latent_dim,
            stimulus_dim=resolved.latent_dim if self.stimulus_conditioner is not None else 0,
            subject_dim=resolved.subject_state_dim,
            backend=resolved.backbone,
            n_layers=resolved.n_layers,
            n_heads=resolved.n_heads,
            pair_rank=resolved.pair_rank,
            use_pair_kernel=resolved.use_pair_kernel,
        )
        self.coordinate_encoder = nn.Linear(resolved.geometry_dim, resolved.latent_dim) if resolved.geometry_dim else None
        self.task_embedding = nn.Embedding(4, resolved.latent_dim)
        self.observation_operators = nn.ModuleDict(
            {
                modality: _observation_operator(modality, resolved.latent_dim, dim, resolved.hrf_delay_steps)
                for modality, dim in sorted(output_dims.items())
            }
            if resolved.use_observation_operator
            else {}
        )
        self.direct_region_heads = nn.ModuleDict(
            {modality: nn.Linear(resolved.latent_dim, 1) for modality in sorted(output_dims)}
            if not resolved.use_observation_operator
            else {}
        )
        self.direct_pooled_heads = nn.ModuleDict(
            {modality: nn.Linear(resolved.latent_dim, dim) for modality, dim in sorted(output_dims.items())}
            if not resolved.use_observation_operator
            else {}
        )
        self.projection_head = nn.Sequential(
            nn.Linear(resolved.latent_dim, resolved.latent_dim),
            nn.GELU(),
            nn.Linear(resolved.latent_dim, resolved.projection_dim),
        )
        self.uncertainty_head = UncertaintyMapHead(resolved.latent_dim, pair_uncertainty=resolved.use_pair_kernel)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        target_modality: str,
        return_output: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        output = self.forward_task(batch, target_modality=target_modality, **kwargs)
        return output if return_output else output["prediction"]

    def forward_task(
        self,
        batch: dict[str, torch.Tensor],
        target_modality: str,
        task: str = "masked_neural_reconstruction",
        metadata: torch.Tensor | None = None,
        geometry: dict[str, torch.Tensor] | None = None,
        subject_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if subject_ids is not None:
            raise ValueError("subject_ids are forbidden in claim-bearing NFC tensors")
        if target_modality not in self.output_dims:
            raise ValueError(f"Unknown target modality {target_modality!r}")
        if not batch:
            raise ValueError("batch cannot be empty")
        node_values, modality_weights = self._fuse_modalities(batch, target_modality)
        stimulus_state = None
        if self.stimulus_conditioner is not None:
            stimulus_source = batch.get("stimulus")
            if stimulus_source is not None:
                stimulus_state = self.stimulus_conditioner(stimulus_source.float())
        subject_state = metadata if metadata is not None and self.config.subject_state_dim else None
        initial = self.latent_field(node_values, subject_state=subject_state, stimulus_state=stimulus_state)
        initial = initial + self.task_embedding.weight[_task_index(task)].view(1, 1, 1, self.latent_dim)
        coordinates, anatomy = _geometry_inputs(
            geometry,
            nodes=self.output_dims[target_modality],
            coordinate_dim=self.config.geometry_dim,
            device=initial.device,
            dtype=initial.dtype,
        )
        if coordinates is not None:
            if self.coordinate_encoder is None:
                raise RuntimeError("coordinate encoder is unavailable")
            initial = initial + self.coordinate_encoder(coordinates).view(1, 1, coordinates.shape[0], self.latent_dim)
        latent = self.field_update(initial, stimulus_state=stimulus_state, anatomy=anatomy, subject_state=subject_state)
        prediction = self._predict_from_latent(latent, target_modality)
        pooled = latent.mean(dim=(1, 2))
        output = {
            "prediction": prediction,
            "latent_field": latent,
            "latent": latent.mean(dim=2),
            "projection": self.projection_head(pooled),
            "modality_weights": modality_weights,
        }
        if self.config.use_uncertainty:
            maps = self.uncertainty_head(latent)
            output["uncertainty"] = maps["region_uncertainty"]
            output.update(maps)
        return output

    def _fuse_modalities(
        self,
        batch: dict[str, torch.Tensor],
        target_modality: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        present = [modality for modality in sorted(self.input_dims) if modality in batch]
        if not present:
            available = ", ".join(sorted(batch))
            expected = ", ".join(sorted(self.input_dims))
            raise ValueError(f"batch modalities {available} do not include expected inputs {expected}")
        reference_shape: tuple[int, int] | None = None
        projected: list[torch.Tensor] = []
        for modality in present:
            source = batch[modality]
            if source.ndim != 3:
                raise ValueError(f"{modality} tensor must have shape [batch, time, features]")
            if source.shape[-1] != self.input_dims[modality]:
                raise ValueError(
                    f"{modality} feature dim {source.shape[-1]} does not match expected {self.input_dims[modality]}"
                )
            if reference_shape is None:
                reference_shape = source.shape[:2]
            elif source.shape[:2] != reference_shape:
                raise ValueError("all fused modalities must share batch and time dimensions")
            projected.append(self.value_projectors[_pair_key(modality, target_modality)](source.float()))
        logits = torch.stack([self.modality_logits[modality] for modality in present])
        weights = torch.softmax(logits, dim=0)
        fused = torch.sum(weights.view(-1, 1, 1, 1) * torch.stack(projected, dim=0), dim=0)
        return fused.unsqueeze(-1), weights

    def _predict_from_latent(self, latent: torch.Tensor, target_modality: str) -> torch.Tensor:
        if self.config.use_observation_operator:
            return self.observation_operators[target_modality](latent)
        if target_modality == "fmri":
            return self.direct_region_heads[target_modality](latent).squeeze(-1)
        return self.direct_pooled_heads[target_modality](latent.mean(dim=2))


def _resolve_config(
    config: NeuralFieldCompilerConfig | Mapping[str, Any] | None,
    options: Mapping[str, Any],
) -> NeuralFieldCompilerConfig:
    if config is None:
        return NeuralFieldCompilerConfig.from_mapping(options, strict=True)
    if options:
        raise TypeError("Pass NFC architecture options either through config or keyword options, not both")
    if isinstance(config, NeuralFieldCompilerConfig):
        return config
    if isinstance(config, Mapping):
        return NeuralFieldCompilerConfig.from_mapping(config, strict=True)
    raise TypeError("config must be a NeuralFieldCompilerConfig or mapping")


def _observation_operator(modality: str, latent_dim: int, output_dim: int, hrf_delay_steps: int) -> nn.Module:
    if modality == "fmri":
        return FMRIObservationOperator(latent_dim, output_dim, hrf_delay_steps=hrf_delay_steps)
    if modality in {"eeg", "meg", "latent_field", "latent_observation"}:
        return EEGObservationOperator(latent_dim, output_dim)
    if modality == "behavior":
        return BehaviorObservationOperator(latent_dim, output_dim)
    raise ValueError(f"No NFC observation operator is registered for modality {modality!r}")


def _task_index(task: str) -> int:
    aliases = {
        "forecast": 0,
        "future_state_forecasting": 0,
        "reconstruction": 1,
        "masked_neural_reconstruction": 1,
        "readout": 2,
        "stimulus_to_fmri_response": 2,
        "translation": 3,
        "cross_modal_translation": 3,
    }
    try:
        return aliases[task]
    except KeyError as exc:
        raise ValueError(f"Unknown NFC task {task!r}") from exc


def _geometry_inputs(
    geometry: dict[str, torch.Tensor] | None,
    *,
    nodes: int,
    coordinate_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    geometry = geometry or {}
    coordinates = geometry.get("coordinates")
    if coordinate_dim:
        if coordinates is None:
            raise ValueError("geometry coordinates are required when geometry_dim is configured")
        if coordinates.ndim != 2 or coordinates.shape != (nodes, coordinate_dim):
            raise ValueError(f"coordinates must have shape [{nodes}, {coordinate_dim}]")
        if not torch.isfinite(coordinates).all():
            raise ValueError("coordinates must be finite")
        coordinates = coordinates.to(device=device, dtype=dtype)
    elif coordinates is not None:
        raise ValueError("coordinates were provided but geometry_dim is zero")
    anatomy = geometry.get("structural_prior")
    if anatomy is None:
        anatomy = geometry.get("anatomy")
    return coordinates, anatomy


def _pair_key(source: str, target: str) -> str:
    return f"{source}__to__{target}"
