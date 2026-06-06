from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from torch import nn

from neurotwin.models.nfc import NeuralFieldCompiler, NeuralFieldCompilerConfig
from neurotwin.models.pair_operator import NeuroTwinPairOperator, NeuroTwinPairOperatorConfig
from neurotwin.models.torch_models import NeuralStateSpaceTranslator, NeuralStateSpaceTranslatorConfig


ConfigParser = Callable[[Mapping[str, Any]], Any]
ModelFactory = Callable[[dict[str, int], dict[str, int], Mapping[str, Any]], nn.Module]
EstimateHook = Callable[[Any], int]


@dataclass(frozen=True)
class ArchitectureSpec:
    canonical_type: str
    aliases: tuple[str, ...]
    model_status: str
    config_parser: ConfigParser
    factory: ModelFactory
    estimate_extra_parameters: EstimateHook
    supported_tasks: tuple[str, ...]


def architecture_registry() -> tuple[ArchitectureSpec, ...]:
    return _REGISTRY


def architecture_spec(value: str) -> ArchitectureSpec:
    normalized = _normalize_key(value)
    for spec in _REGISTRY:
        if normalized == _normalize_key(spec.canonical_type) or normalized in {_normalize_key(alias) for alias in spec.aliases}:
            return spec
    raise ValueError(f"Unknown prepared model type {value!r}")


def normalize_architecture_type(value: str) -> str:
    return architecture_spec(value).canonical_type


def architecture_status(value: str) -> str:
    return architecture_spec(value).model_status


def build_architecture_model(model_config: Mapping[str, Any]) -> nn.Module:
    spec = architecture_spec(str(model_config.get("type", "NeuralStateSpaceTranslator")))
    input_dims = {str(key): int(value) for key, value in dict(model_config["input_dims"]).items()}
    output_dims = {str(key): int(value) for key, value in dict(model_config["output_dims"]).items()}
    return spec.factory(input_dims, output_dims, model_config)


def estimate_architecture_extra_parameters(model_config: Any) -> int:
    return architecture_spec(str(model_config.type)).estimate_extra_parameters(model_config)


def _translator_factory(
    input_dims: dict[str, int],
    output_dims: dict[str, int],
    model_config: Mapping[str, Any],
) -> NeuralStateSpaceTranslator:
    return NeuralStateSpaceTranslator(
        input_dims=input_dims,
        output_dims=output_dims,
        config=NeuralStateSpaceTranslatorConfig.from_mapping(model_config),
    )


def _pair_operator_factory(
    input_dims: dict[str, int],
    output_dims: dict[str, int],
    model_config: Mapping[str, Any],
) -> NeuroTwinPairOperator:
    return NeuroTwinPairOperator(
        input_dims=input_dims,
        output_dims=output_dims,
        config=NeuroTwinPairOperatorConfig.from_mapping(model_config),
    )


def _nfc_factory(
    input_dims: dict[str, int],
    output_dims: dict[str, int],
    model_config: Mapping[str, Any],
) -> NeuralFieldCompiler:
    return NeuralFieldCompiler(
        input_dims=input_dims,
        output_dims=output_dims,
        config=NeuralFieldCompilerConfig.from_mapping(model_config),
    )


def _no_extra_parameters(model_config: Any) -> int:
    del model_config
    return 0


def _pair_operator_extra_parameters(model_config: Any) -> int:
    pair_state_factor_values = model_config.output_dim * model_config.pair_rank * 2
    network_block_values = getattr(model_config, "network_blocks", 1) * model_config.pair_rank
    return int(pair_state_factor_values + network_block_values + model_config.latent_dim * model_config.latent_dim)


def _nfc_extra_parameters(model_config: Any) -> int:
    return int(model_config.output_dim * model_config.pair_rank * 2 + model_config.latent_dim * model_config.latent_dim * 3)


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_")


_COMMON_TASKS = (
    "future_state_forecasting",
    "masked_neural_reconstruction",
    "stimulus_to_fmri_response",
    "synthetic_latent_observation_recovery",
)


_REGISTRY = (
    ArchitectureSpec(
        canonical_type="NeuralStateSpaceTranslator",
        aliases=("neuralstatespacetranslator", "neural_state_space_translator", "translator", "current_neurotwin", "neurotwin_v1"),
        model_status="local_baseline",
        config_parser=NeuralStateSpaceTranslatorConfig.from_mapping,
        factory=_translator_factory,
        estimate_extra_parameters=_no_extra_parameters,
        supported_tasks=_COMMON_TASKS,
    ),
    ArchitectureSpec(
        canonical_type="NeuroTwinPairOperator",
        aliases=("neurotwinpairoperator", "neurotwin_pair_operator", "pair_operator", "ntp_o"),
        model_status="local_baseline",
        config_parser=NeuroTwinPairOperatorConfig.from_mapping,
        factory=_pair_operator_factory,
        estimate_extra_parameters=_pair_operator_extra_parameters,
        supported_tasks=_COMMON_TASKS,
    ),
    ArchitectureSpec(
        canonical_type="NeuralFieldCompiler",
        aliases=("neuralfieldcompiler", "neural_field_compiler", "neurotwin_nfc", "nfc", "field_compiler"),
        model_status="experimental_architecture",
        config_parser=NeuralFieldCompilerConfig.from_mapping,
        factory=_nfc_factory,
        estimate_extra_parameters=_nfc_extra_parameters,
        supported_tasks=_COMMON_TASKS,
    ),
)
