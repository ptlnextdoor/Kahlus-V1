"""Finite response profile head C_K(h_t).

For each perturbation ``a_k`` a linear head maps the memory state to a predicted future
trajectory ``pθ(τ | h_t, a_k)`` of shape ``(horizon, eeg_channels)``. Stacking over the K
perturbations yields the finite response profile.
"""

from __future__ import annotations

import numpy as np


class ResponseProfileHead:
    def __init__(
        self,
        rng: np.random.Generator,
        memory_dim: int,
        n_perturbations: int,
        horizon: int,
        eeg_channels: int,
    ) -> None:
        self.memory_dim = int(memory_dim)
        self.n_perturbations = int(n_perturbations)
        self.horizon = int(horizon)
        self.eeg_channels = int(eeg_channels)
        out_dim = self.horizon * self.eeg_channels
        # One linear head per perturbation: (K, memory_dim, horizon*eeg_channels).
        self.weights = rng.normal(
            scale=1.0 / np.sqrt(self.memory_dim),
            size=(self.n_perturbations, self.memory_dim, out_dim),
        )

    def predict(self, memory: np.ndarray) -> np.ndarray:
        """``memory`` is ``(batch, memory_dim)`` -> ``(batch, K, horizon, eeg_channels)``."""

        memory = np.asarray(memory, dtype=np.float64)
        flat = np.einsum("bm,kmo->bko", memory, self.weights)
        return flat.reshape(flat.shape[0], self.n_perturbations, self.horizon, self.eeg_channels)
