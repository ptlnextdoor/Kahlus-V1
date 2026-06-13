"""Commutators of the latent perturbation operators: [Ta, Tb] = Ta Tb - Tb Ta.

A non-zero commutator is the formal statement that perturbation order matters. These helpers
quantify the non-commutativity of the KTM operator set.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def commutator(operator_a: np.ndarray, operator_b: np.ndarray) -> np.ndarray:
    """Matrix commutator ``A B - B A``."""

    a = np.asarray(operator_a, dtype=np.float64)
    b = np.asarray(operator_b, dtype=np.float64)
    return a @ b - b @ a


def commutator_norm(operator_a: np.ndarray, operator_b: np.ndarray) -> float:
    """Frobenius norm of the commutator (0 iff the operators commute)."""

    return float(np.linalg.norm(commutator(operator_a, operator_b)))


def commutator_matrix(operators: Sequence[np.ndarray]) -> np.ndarray:
    """Pairwise commutator-norm matrix for a set of operators (``(K, K)``)."""

    n = len(operators)
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            out[i, j] = commutator_norm(operators[i], operators[j])
    return out
