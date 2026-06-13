"""Per-perturbation uncertainty head for the KTM scaffold.

Produces a strictly positive variance estimate per perturbation via softplus + a floor, so
the response profile can be reported with a confidence (the dossier requires uncertainty as a
scientific object, not a decoration).
"""

from __future__ import annotations

import numpy as np


def _softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.logaddexp(0.0, x)


class UncertaintyHead:
    def __init__(self, rng: np.random.Generator, memory_dim: int, n_perturbations: int, floor: float) -> None:
        self.memory_dim = int(memory_dim)
        self.n_perturbations = int(n_perturbations)
        self.floor = float(floor)
        self.weight = rng.normal(scale=1.0 / np.sqrt(self.memory_dim), size=(self.memory_dim, self.n_perturbations))

    def predict(self, memory: np.ndarray) -> np.ndarray:
        """``memory`` is ``(batch, memory_dim)`` -> ``(batch, n_perturbations)`` variances > 0."""

        logits = np.asarray(memory, dtype=np.float64) @ self.weight
        return _softplus(logits) + self.floor
