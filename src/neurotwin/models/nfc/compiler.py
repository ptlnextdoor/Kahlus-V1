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
    pair_rank: int = 8
    projection_dim: int = 32
    backbone: str = "gru"
    use_pair_kernel: bool = True
    use_observation_operator: bool = True
    use_uncertainty: bool = True
    stimulus_lag_steps: int = 1
    hrf_delay_steps: int = 1
    subject_state_dim: int = 0

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
            "geometry_dim",
            "subject_adapter_dim",
            "subject_vocab_size",
            "use_subject_embeddings",
            "adapter_mode",
            "gradient_checkpointing",
            "n_heads",
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
            n_layers=int(selected.get("n_layers", cls.n_layers)),
            pair_rank=int(selected.get("pair_rank", cls.pair_rank)),
            projection_dim=int(selected.get("projection_dim", cls.projection_dim)),
            backbone=str(selected.get("backbone", cls.backbone)),
            use_pair_kernel=bool(selected.get("use_pair_kernel", cls.use_pair_kernel)),
            use_observation_operator=bool(selected.get("use_observation_operator", cls.use_observation_operator)),
            use_uncertainty=bool(selected.get("use_uncertainty", cls.use_uncertainty)),
            stimulus_lag_steps=max(0, int(selected.get("stimulus_lag_steps", cls.stimulus_lag_steps))),
            hrf_delay_steps=max(0, int(selected.get("hrf_delay_steps", cls.hrf_delay_steps))),
            subject_state_dim=max(0, int(selected.get("subject_state_dim", cls.subject_state_dim))),
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
            pair_rank=resolved.pair_rank,
            use_pair_kernel=resolved.use_pair_kernel,
        )
        self.observation_operators = nn.ModuleDict(
            {modality: _observation_operator(modality, resolved.latent_dim, dim, resolved.hrf_delay_steps) for modality, dim in sorted(output_dims.items())}
        )
        self.direct_region_heads = nn.ModuleDict({modality: nn.Linear(resolved.latent_dim, 1) for modality in sorted(output_dims)})
        self.direct_pooled_heads = nn.ModuleDict({modality: nn.Linear(resolved.latent_dim, dim) for modality, dim in sorted(output_dims.items())})
        self.projection_head = nn.Sequential(
            nn.Linear(resolved.latent_dim, resolved.latent_dim),
            nn.GELU(),
            nn.Linear(resolved.latent_dim, resolved.projection_dim),
        )
        self.uncertainty_head = UncertaintyMapHead(resolved.latent_dim, pair_uncertainty=resolved.use_pair_kernel)

    def forward(self, batch: dict[str, torch.Tensor], target_modality: str, **kwargs: Any) -> torch.Tensor:
        return self.forward_task(batch, target_modality=target_modality, **kwargs)["prediction"]

    def forward_task(
        self,
        batch: dict[str, torch.Tensor],
        target_modality: str,
        task: str = "masked_neural_reconstruction",
        metadata: torch.Tensor | None = None,
        geometry: dict[str, torch.Tensor] | None = None,
        subject_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del task, subject_ids
        if target_modality not in self.output_dims:
            raise ValueError(f"Unknown target modality {target_modality!r}")
        if not batch:
            raise ValueError("batch cannot be empty")
        source_modality, source = _select_source(batch, self.input_dims)
        if source.ndim != 3:
            raise ValueError("source tensor must have shape [batch, time, features]")
        if source.shape[-1] != self.input_dims[source_modality]:
            raise ValueError(f"{source_modality} feature dim {source.shape[-1]} does not match expected {self.input_dims[source_modality]}")
        node_values = self.value_projectors[_pair_key(source_modality, target_modality)](source.float()).unsqueeze(-1)
        stimulus_state = None
        if self.stimulus_conditioner is not None:
            stimulus_source = batch.get("stimulus")
            if stimulus_source is not None:
                stimulus_state = self.stimulus_conditioner(stimulus_source.float())
        subject_state = metadata if metadata is not None and self.config.subject_state_dim else None
        initial = self.latent_field(node_values, subject_state=subject_state, stimulus_state=stimulus_state)
        anatomy = None
        if geometry is not None:
            if "structural_prior" in geometry:
                candidate = geometry["structural_prior"]
            elif "anatomy" in geometry:
                candidate = geometry["anatomy"]
            else:
                candidate = None
            if isinstance(candidate, torch.Tensor):
                anatomy = candidate
        latent = self.field_update(initial, stimulus_state=stimulus_state, anatomy=anatomy, subject_state=subject_state)
        prediction = self._predict_from_latent(latent, target_modality)
        pooled = latent.mean(dim=(1, 2))
        output = {
            "prediction": prediction,
            "latent_field": latent,
            "latent": latent.mean(dim=2),
            "projection": self.projection_head(pooled),
            "expert_utilization": _expert_utilization(source_modality, target_modality, self.config, prediction.device),
        }
        if self.config.use_uncertainty:
            maps = self.uncertainty_head(latent)
            output["uncertainty"] = maps["region_uncertainty"]
            output.update(maps)
        return output

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
    if modality == "eeg":
        return EEGObservationOperator(latent_dim, output_dim)
    if modality == "behavior":
        return BehaviorObservationOperator(latent_dim, output_dim)
    return EEGObservationOperator(latent_dim, output_dim)


def _select_source(batch: dict[str, torch.Tensor], input_dims: dict[str, int]) -> tuple[str, torch.Tensor]:
    for modality in sorted(input_dims):
        if modality in batch:
            return modality, batch[modality]
    available = ", ".join(sorted(batch))
    expected = ", ".join(sorted(input_dims))
    raise ValueError(f"batch modalities {available} do not include expected inputs {expected}")


def _pair_key(source: str, target: str) -> str:
    return f"{source}__to__{target}"


def _expert_utilization(
    source_modality: str,
    target_modality: str,
    config: NeuralFieldCompilerConfig,
    device: torch.device,
) -> torch.Tensor:
    values = torch.zeros(6, dtype=torch.float32, device=device)
    values[0] = 1.0 if target_modality == "fmri" else 0.0
    values[1] = 1.0 if target_modality == "eeg" else 0.0
    values[2] = 1.0 if source_modality == "stimulus" else 0.0
    values[3] = 1.0 if config.use_pair_kernel else 0.0
    values[4] = 1.0 if config.use_observation_operator else 0.0
    values[5] = 1.0 if config.use_uncertainty else 0.0
    return values
