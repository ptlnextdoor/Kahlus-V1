"""Baseline and NeuroTwin model specifications."""

from neurotwin.models.architecture_registry import (
    ArchitectureSpec,
    architecture_registry,
    architecture_spec,
    architecture_status,
    build_architecture_model,
    estimate_architecture_extra_parameters,
    normalize_architecture_type,
)
from neurotwin.models.baselines import BaselineSpec, NumpyRidgeBaseline, TorchMLPBaseline, TorchTCNBaseline, baseline_registry
from neurotwin.models.nfc import NeuralFieldCompiler, NeuralFieldCompilerConfig
from neurotwin.models.pair_operator import NeuroTwinPairOperator, NeuroTwinPairOperatorConfig
from neurotwin.models.torch_models import (
    NeuralStateSpaceTranslator,
    NeuralStateSpaceTranslatorConfig,
    TinySSMBaseline,
    TinyTransformerBaseline,
)
from neurotwin.models.tribe_style import TribeStyleModel, TribeStyleStimulusEncoder, TribeStyleStimulusInput

__all__ = [
    "ArchitectureSpec",
    "BaselineSpec",
    "NeuralStateSpaceTranslator",
    "NeuralStateSpaceTranslatorConfig",
    "NeuralFieldCompiler",
    "NeuralFieldCompilerConfig",
    "NeuroTwinPairOperator",
    "NeuroTwinPairOperatorConfig",
    "NumpyRidgeBaseline",
    "TinySSMBaseline",
    "TinyTransformerBaseline",
    "TorchMLPBaseline",
    "TorchTCNBaseline",
    "TribeStyleModel",
    "TribeStyleStimulusEncoder",
    "TribeStyleStimulusInput",
    "architecture_registry",
    "architecture_spec",
    "architecture_status",
    "baseline_registry",
    "build_architecture_model",
    "estimate_architecture_extra_parameters",
    "normalize_architecture_type",
]
