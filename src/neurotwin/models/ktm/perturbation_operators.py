"""Latent perturbation operators T_a acting on the KTM memory state.

Each perturbation ``a_k`` has a learned linear operator ``P_k`` on memory space, the model's
counterpart to the gym's hidden transition ``T_a``. They are initialized non-commutatively so
that ``T_b∘a`` differs from ``T_a∘b`` (see :mod:`lie_generators`).
"""

from __future__ import annotations

import numpy as np


class PerturbationOperators:
    def __init__(self, rng: np.random.Generator, memory_dim: int, n_perturbations: int, scale: float = 0.3) -> None:
        self.memory_dim = int(memory_dim)
        self.n_perturbations = int(n_perturbations)
        eye = np.eye(self.memory_dim)
        self._operators = [eye + scale * rng.normal(size=(self.memory_dim, self.memory_dim)) for _ in range(self.n_perturbations)]

    def operators(self) -> list[np.ndarray]:
        return [op.copy() for op in self._operators]

    def operator(self, index: int) -> np.ndarray:
        return self._operators[index]

    def apply(self, memory: np.ndarray, index: int) -> np.ndarray:
        """Apply T_{a_index} to a ``(batch, memory_dim)`` memory state."""

        return np.asarray(memory, dtype=np.float64) @ self._operators[index].T

    def compose(self, memory: np.ndarray, first: int, second: int) -> np.ndarray:
        """Apply ``first`` then ``second`` (T_second ∘ T_first)."""

        return self.apply(self.apply(memory, first), second)
