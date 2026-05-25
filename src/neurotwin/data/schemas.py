from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


SUPPORTED_MODALITIES = {
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


@dataclass
class NeuralEventBatch:
    """Common event batch contract for heterogeneous neural recordings."""

    modality: str
    dataset: str
    subject_id: str
    session_id: str
    site_id: str
    time: np.ndarray
    signal: np.ndarray
    mask: np.ndarray
    stimulus_embedding: np.ndarray | None
    behavior: dict[str, Any]
    space_index: np.ndarray
    uncertainty: np.ndarray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.modality not in SUPPORTED_MODALITIES:
            supported = ", ".join(sorted(SUPPORTED_MODALITIES))
            raise ValueError(f"Unsupported modality {self.modality!r}; expected one of: {supported}")

        self.time = np.asarray(self.time)
        self.signal = np.asarray(self.signal)
        self.mask = np.asarray(self.mask)
        self.space_index = np.asarray(self.space_index)

        if self.time.ndim != 1:
            raise ValueError("time must be a 1D array")
        if self.signal.ndim < 2:
            raise ValueError("signal must have at least a time axis and a space/channel axis")
        if self.signal.shape[0] != self.time.shape[0]:
            raise ValueError("signal time axis must match time length")
        if self.mask.shape != self.signal.shape:
            raise ValueError("mask must have the same shape as signal")
        if self.space_index.ndim != 1:
            raise ValueError("space_index must be a 1D array")
        if self.space_index.shape[0] != self.signal.shape[1]:
            raise ValueError("space_index length must match signal space/channel axis")

        if self.stimulus_embedding is not None:
            self.stimulus_embedding = np.asarray(self.stimulus_embedding)
            if self.stimulus_embedding.shape[0] != self.time.shape[0]:
                raise ValueError("stimulus_embedding time axis must match time length")

        if self.uncertainty is not None:
            self.uncertainty = np.asarray(self.uncertainty)
            if self.uncertainty.shape != self.signal.shape:
                raise ValueError("uncertainty must have the same shape as signal")

        for name, value in self.behavior.items():
            if hasattr(value, "shape"):
                arr = np.asarray(value)
                if arr.ndim > 0 and arr.shape[0] not in (1, self.time.shape[0]):
                    raise ValueError(f"behavior[{name!r}] has incompatible time axis")

    @property
    def n_time(self) -> int:
        return int(self.signal.shape[0])

    @property
    def n_space(self) -> int:
        return int(self.signal.shape[1])
