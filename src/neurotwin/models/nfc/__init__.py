"""NeuroTwin NFC experimental architecture."""

from neurotwin.models.nfc.compiler import NeuralFieldCompiler, NeuralFieldCompilerConfig
from neurotwin.models.nfc.field_update import FieldUpdateOperator
from neurotwin.models.nfc.latent_field import LatentNeuralField
from neurotwin.models.nfc.observations.behavior import BehaviorObservationOperator
from neurotwin.models.nfc.observations.eeg import EEGObservationOperator
from neurotwin.models.nfc.observations.fmri import FMRIObservationOperator
from neurotwin.models.nfc.pair_kernel import LowRankPairKernel
from neurotwin.models.nfc.stimulus import StimulusConditioningOperator
from neurotwin.models.nfc.uncertainty import UncertaintyMapHead

__all__ = [
    "BehaviorObservationOperator",
    "EEGObservationOperator",
    "FMRIObservationOperator",
    "FieldUpdateOperator",
    "LatentNeuralField",
    "LowRankPairKernel",
    "NeuralFieldCompiler",
    "NeuralFieldCompilerConfig",
    "StimulusConditioningOperator",
    "UncertaintyMapHead",
]
