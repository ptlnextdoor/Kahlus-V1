"""Recurrent memory state for the KTM scaffold.

A leaky update ``m_t = ρ m_{t-1} + (1-ρ) tanh(embed Wm)`` projecting the encoded history
into the memory space the perturbation operators act on.
"""

from __future__ import annotations

import numpy as np


class Memory:
    def __init__(self, rng: np.random.Generator, embed_dim: int, memory_dim: int, rho: float) -> None:
        self.embed_dim = int(embed_dim)
        self.memory_dim = int(memory_dim)
        self.rho = float(rho)
        self.weight = rng.normal(scale=1.0 / np.sqrt(self.embed_dim), size=(self.embed_dim, self.memory_dim))

    def initial(self, batch: int) -> np.ndarray:
        return np.zeros((batch, self.memory_dim), dtype=np.float64)

    def update(self, prev: np.ndarray, embed: np.ndarray) -> np.ndarray:
        prev = np.asarray(prev, dtype=np.float64)
        drive = np.tanh(np.asarray(embed, dtype=np.float64) @ self.weight)
        return self.rho * prev + (1.0 - self.rho) * drive

    def project(self, embed: np.ndarray) -> np.ndarray:
        """One-shot projection from an embedding (memory from a single h_t)."""

        batch = np.asarray(embed).shape[0]
        return self.update(self.initial(batch), embed)
