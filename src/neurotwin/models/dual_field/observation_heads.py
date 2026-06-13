"""Observation operators for the dual-field scaffold.

EEG-like output is read mostly from the fast neural field N; BOLD/fNIRS-like output is read
mostly from the slow hemodynamic field H. Both heads are linear projections plus optional
measurement noise — the v2 stand-ins for O_EEG and O_BOLD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservationHeads:
    eeg_w: np.ndarray  # (neural_dim, eeg_channels)
    bold_w: np.ndarray  # (hemo_dim, bold_channels)

    def observe_eeg(self, neural_state: np.ndarray) -> np.ndarray:
        """O_EEG(N): linear projection of the fast field to EEG channels."""

        return np.asarray(neural_state, dtype=np.float64) @ self.eeg_w

    def observe_bold(self, hemo_state: np.ndarray) -> np.ndarray:
        """O_BOLD(H): linear projection of the slow field to BOLD/fNIRS channels."""

        return np.asarray(hemo_state, dtype=np.float64) @ self.bold_w
