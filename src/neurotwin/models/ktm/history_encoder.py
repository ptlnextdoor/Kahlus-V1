"""History encoder: maps a window of EEG-like observations to an embedding h_t."""

from __future__ import annotations

import numpy as np


class HistoryEncoder:
    def __init__(self, rng: np.random.Generator, history_len: int, eeg_channels: int, embed_dim: int) -> None:
        self.history_len = int(history_len)
        self.eeg_channels = int(eeg_channels)
        self.embed_dim = int(embed_dim)
        in_dim = self.history_len * self.eeg_channels
        self.weight = rng.normal(scale=1.0 / np.sqrt(in_dim), size=(in_dim, self.embed_dim))
        self.bias = np.zeros(self.embed_dim, dtype=np.float64)

    def encode(self, history: np.ndarray) -> np.ndarray:
        """``history`` is ``(batch, history_len, eeg_channels)`` -> ``(batch, embed_dim)``."""

        history = np.asarray(history, dtype=np.float64)
        if history.ndim != 3:
            raise ValueError("history must be (batch, history_len, eeg_channels)")
        flat = np.ascontiguousarray(history.reshape(history.shape[0], -1))
        if flat.shape[1] != self.weight.shape[0]:
            raise ValueError("history feature dimension does not match encoder")
        return np.tanh(flat @ self.weight + self.bias)
