"""Structured NFC observation operators."""

from neurotwin.models.nfc.observations.base import BaseObservationOperator
from neurotwin.models.nfc.observations.behavior import BehaviorObservationOperator
from neurotwin.models.nfc.observations.eeg import EEGObservationOperator
from neurotwin.models.nfc.observations.fmri import FMRIObservationOperator

__all__ = [
    "BaseObservationOperator",
    "BehaviorObservationOperator",
    "EEGObservationOperator",
    "FMRIObservationOperator",
]
