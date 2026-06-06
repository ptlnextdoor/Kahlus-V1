from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NotRequired, TypedDict, cast


ConfigPath = str | Path
ConfigScalar = str | int | float | bool
SUPPORTED_CONFIG_MODALITIES = {
    "generic",
    "fmri",
    "eeg",
    "meg",
    "spikes",
    "calcium",
    "behavior",
    "stimulus",
    "anatomy",
    "clinical",
}


class PreparedDataConfig(TypedDict, total=False):
    event_manifest: ConfigPath
    split_manifest: ConfigPath


class PreparedModelConfig(TypedDict, total=False):
    type: str
    latent_dim: int
    n_layers: int
    input_dim: int
    output_dim: int
    modalities: list[str]
    backbone: str
    encoder: str
    n_heads: int
    subject_adapter_dim: int
    projection_dim: int
    metadata_dim: int
    geometry_dim: int
    subject_vocab_size: int
    use_subject_embeddings: bool
    adapter_mode: str
    gradient_checkpointing: bool
    pair_rank: int
    pair_top_k: int
    network_blocks: int
    pair_confidence_max_parcels: int
    use_pair_state: bool
    use_pair_kernel: bool
    use_observation_operator: bool
    use_uncertainty: bool
    use_uncertainty_head: bool
    use_pair_uncertainty: bool
    refinement_steps: int
    hrf_delay_steps: int
    stimulus_lag_steps: int
    subject_state_dim: int


class PreparedTrainingSectionConfig(TypedDict, total=False):
    batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    max_grad_norm: float
    gradient_clip_norm: float
    precision: str
    compile: bool
    eval_every_steps: int
    checkpoint_every_steps: int
    gradient_checkpointing: bool
    objective_weights: dict[str, int | float]
    adapter_mode: str


class PreparedTrainingConfigInput(TypedDict, total=False):
    experiment: str
    dataset: str
    task: str
    split: str
    seed: int
    data: PreparedDataConfig
    model: PreparedModelConfig
    training: PreparedTrainingSectionConfig
    event_manifest: ConfigPath
    split_manifest: ConfigPath
    window_size: int
    window_length: int
    stride: int
    steps: int
    batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    max_grad_norm: float
    gradient_clip_norm: float
    precision: str
    compile: bool
    learning_rate: float
    lr: float
    eval_every_steps: int
    checkpoint_every_steps: int
    objective_weights: dict[str, int | float]
    gradient_checkpointing: bool
    extra: NotRequired[dict[str, Any]]


def as_prepared_training_config_input(config: Mapping[str, Any]) -> PreparedTrainingConfigInput:
    """Narrow loaded YAML at the command boundary after load_config validation."""

    return cast(PreparedTrainingConfigInput, config)


@dataclass(frozen=True)
class ResolvedPreparedModelConfig:
    type: str
    latent_dim: int
    n_layers: int
    input_dim: int
    output_dim: int
    modalities: tuple[str, ...]
    backbone: str
    encoder: str
    n_heads: int
    subject_adapter_dim: int
    projection_dim: int
    metadata_dim: int
    geometry_dim: int
    subject_vocab_size: int
    use_subject_embeddings: bool
    adapter_mode: str
    gradient_checkpointing: bool
    pair_rank: int
    pair_top_k: int
    network_blocks: int
    pair_confidence_max_parcels: int
    use_pair_state: bool
    use_pair_kernel: bool
    use_observation_operator: bool
    use_uncertainty: bool
    use_uncertainty_head: bool
    use_pair_uncertainty: bool
    refinement_steps: int
    hrf_delay_steps: int
    stimulus_lag_steps: int
    subject_state_dim: int


@dataclass(frozen=True)
class ResolvedPreparedRuntimeConfig:
    batch_size: int | None
    eval_batch_size: int | None
    gradient_accumulation_steps: int
    max_grad_norm: float | None
    precision: str
    compile: bool
    eval_every_steps: int
    checkpoint_every_steps: int
    learning_rate: float
    objective_weights: dict[str, float]


@dataclass(frozen=True)
class ResolvedPreparedConfig:
    event_manifest: ConfigPath | None
    split_manifest: ConfigPath | None
    seed: int
    window_length: int
    stride: int
    steps: int
    requested_task: str
    model: ResolvedPreparedModelConfig
    runtime: ResolvedPreparedRuntimeConfig


def resolve_prepared_config(
    config: PreparedTrainingConfigInput,
    *,
    require_manifests: bool = False,
    latent_dim_default: int = 128,
    n_layers_default: int = 2,
    projection_dim_default: int = 64,
    window_length_default: int = 8,
) -> ResolvedPreparedConfig:
    data_config = _mapping(config.get("data"))
    model_config = _mapping(config.get("model"))
    training_config = _mapping(config.get("training"))
    event_manifest = config.get("event_manifest") or data_config.get("event_manifest")
    split_manifest = config.get("split_manifest") or data_config.get("split_manifest")
    if require_manifests and (not event_manifest or not split_manifest):
        raise ValueError("prepared training requires event_manifest and split_manifest")

    window_length = int(config.get("window_size", config.get("window_length", window_length_default)))
    batch_size = config.get("batch_size", training_config.get("batch_size"))
    eval_batch_size = config.get("eval_batch_size", training_config.get("eval_batch_size"))
    objective_weights = training_config.get("objective_weights", config.get("objective_weights", {}))
    max_grad_norm = config.get(
        "max_grad_norm",
        config.get(
            "gradient_clip_norm",
            training_config.get("max_grad_norm", training_config.get("gradient_clip_norm", 1.0)),
        ),
    )
    model = ResolvedPreparedModelConfig(
        type=str(model_config.get("type", "NeuralStateSpaceTranslator")),
        latent_dim=int(model_config.get("latent_dim", latent_dim_default)),
        n_layers=int(model_config.get("n_layers", n_layers_default)),
        input_dim=int(model_config.get("input_dim", 16)),
        output_dim=int(model_config.get("output_dim", model_config.get("input_dim", 16))),
        modalities=_resolve_modalities(model_config),
        backbone=str(model_config.get("backbone", "ssm_fallback")),
        encoder=str(model_config.get("encoder", "auto")),
        n_heads=int(model_config.get("n_heads", 4)),
        subject_adapter_dim=int(model_config.get("subject_adapter_dim", 0)),
        projection_dim=int(model_config.get("projection_dim", projection_dim_default)),
        metadata_dim=int(model_config.get("metadata_dim", 0)),
        geometry_dim=int(model_config.get("geometry_dim", 0)),
        subject_vocab_size=int(model_config.get("subject_vocab_size", 0)),
        use_subject_embeddings=bool(model_config.get("use_subject_embeddings", False)),
        adapter_mode=str(model_config.get("adapter_mode", training_config.get("adapter_mode", "disabled"))),
        gradient_checkpointing=bool(
            config.get(
                "gradient_checkpointing",
                model_config.get("gradient_checkpointing", training_config.get("gradient_checkpointing", False)),
            )
        ),
        pair_rank=int(model_config.get("pair_rank", 8)),
        pair_top_k=max(0, int(model_config.get("pair_top_k", 0))),
        network_blocks=max(1, int(model_config.get("network_blocks", 1))),
        pair_confidence_max_parcels=max(0, int(model_config.get("pair_confidence_max_parcels", 256))),
        use_pair_state=bool(model_config.get("use_pair_state", True)),
        use_pair_kernel=bool(model_config.get("use_pair_kernel", model_config.get("use_pair_state", True))),
        use_observation_operator=bool(model_config.get("use_observation_operator", True)),
        use_uncertainty=bool(model_config.get("use_uncertainty", model_config.get("use_uncertainty_head", True))),
        use_uncertainty_head=bool(model_config.get("use_uncertainty_head", True)),
        use_pair_uncertainty=bool(model_config.get("use_pair_uncertainty", False)),
        refinement_steps=max(0, int(model_config.get("refinement_steps", 1))),
        hrf_delay_steps=max(0, int(model_config.get("hrf_delay_steps", 1))),
        stimulus_lag_steps=max(0, int(model_config.get("stimulus_lag_steps", model_config.get("hrf_delay_steps", 1)))),
        subject_state_dim=max(0, int(model_config.get("subject_state_dim", 0))),
    )
    runtime = ResolvedPreparedRuntimeConfig(
        batch_size=_optional_int(batch_size),
        eval_batch_size=_optional_int(eval_batch_size),
        gradient_accumulation_steps=max(
            1,
            int(config.get("gradient_accumulation_steps", training_config.get("gradient_accumulation_steps", 1))),
        ),
        max_grad_norm=_optional_nonnegative_float(max_grad_norm),
        precision=str(config.get("precision", training_config.get("precision", "fp32"))).lower(),
        compile=bool(config.get("compile", training_config.get("compile", False))),
        eval_every_steps=max(0, int(config.get("eval_every_steps", training_config.get("eval_every_steps", 0)))),
        checkpoint_every_steps=max(
            0,
            int(config.get("checkpoint_every_steps", training_config.get("checkpoint_every_steps", 0))),
        ),
        learning_rate=float(config.get("lr", config.get("learning_rate", 2e-3))),
        objective_weights={str(key): float(value) for key, value in _mapping(objective_weights).items()},
    )
    return ResolvedPreparedConfig(
        event_manifest=event_manifest,
        split_manifest=split_manifest,
        seed=int(config.get("seed", 0)),
        window_length=window_length,
        stride=int(config.get("stride", window_length)),
        steps=int(config.get("steps", 24)),
        requested_task=str(config.get("task", "future_state_forecasting")),
        model=model,
        runtime=runtime,
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _optional_nonnegative_float(value: object) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if parsed > 0.0 else None


def _resolve_modalities(model_config: Mapping[str, Any]) -> tuple[str, ...]:
    raw = model_config.get("modalities")
    if raw is None:
        return ("generic",)
    if not isinstance(raw, list) or not raw:
        raise ValueError("model.modalities must be a non-empty list of modality names")
    modalities = []
    for value in raw:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("model.modalities entries must be non-empty strings")
        modality = value.strip().lower()
        if modality not in SUPPORTED_CONFIG_MODALITIES:
            supported = ", ".join(sorted(SUPPORTED_CONFIG_MODALITIES))
            raise ValueError(f"unsupported model modality {value!r}; expected one of: {supported}")
        modalities.append(modality)
    return tuple(modalities)
