"""Locked perturbation battery for the Transition Gym.

A perturbation ``a_k`` is a *known* affine operator ``T_a: z -> M_k z + b_k`` on the hidden
state. The battery is generated from a seed and then frozen ("locked"), and is deliberately
non-commutative so that AB vs BA perturbation sequences differ. The hidden operators are
known by construction, which is what makes the gym a falsification benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Perturbation:
    name: str
    matrix: np.ndarray  # (state_dim, state_dim) M_k
    bias: np.ndarray  # (state_dim,) b_k


class PerturbationLibrary:
    """A fixed, ordered battery ``A_K = {a_1, ..., a_K}`` of affine operators."""

    def __init__(self, perturbations: list[Perturbation]) -> None:
        if not perturbations:
            raise ValueError("PerturbationLibrary requires at least one perturbation")
        self._perturbations = list(perturbations)
        self._by_name = {p.name: p for p in self._perturbations}

    @classmethod
    def locked(cls, seed: int, state_dim: int, n_perturbations: int, scale: float = 0.6) -> "PerturbationLibrary":
        rng = np.random.default_rng(seed)
        perturbations: list[Perturbation] = []
        for k in range(n_perturbations):
            matrix = np.eye(state_dim) + scale * rng.normal(size=(state_dim, state_dim))
            bias = scale * rng.normal(size=state_dim)
            perturbations.append(Perturbation(name=f"a{k + 1}", matrix=matrix, bias=bias))
        return cls(perturbations)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self._perturbations]

    def __len__(self) -> int:
        return len(self._perturbations)

    def get(self, name: str) -> Perturbation:
        return self._by_name[name]

    def apply(self, name: str, state: np.ndarray) -> np.ndarray:
        """Apply a single perturbation operator to a (batch, state_dim) array."""

        pert = self._by_name[name]
        return np.asarray(state, dtype=np.float64) @ pert.matrix.T + pert.bias[None, :]

    def compose(self, first: str, second: str, state: np.ndarray) -> np.ndarray:
        """Apply ``first`` then ``second`` (i.e. T_second ∘ T_first)."""

        return self.apply(second, self.apply(first, state))

    def commutator_gap(self, name_a: str, name_b: str, state: np.ndarray) -> float:
        """Mean magnitude of ``T_b(T_a z) - T_a(T_b z)`` over the batch (AB vs BA)."""

        ab = self.compose(name_a, name_b, state)
        ba = self.compose(name_b, name_a, state)
        return float(np.mean(np.abs(ab - ba)))

    def operator_matrix(self, name: str) -> np.ndarray:
        return self._by_name[name].matrix
