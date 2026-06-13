"""Stability and NaN-hygiene helpers for the dual-field scaffold.

Stability is treated as part of the scientific contract: non-finite states invalidate any
downstream metric, so the rollout clamps states and asserts finiteness explicitly.
"""

from __future__ import annotations

import numpy as np


def spectral_radius(matrix: np.ndarray) -> float:
    """Largest absolute eigenvalue of a square matrix."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("spectral_radius expects a square matrix")
    eigvals = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eigvals)))


def is_stable(matrix: np.ndarray, threshold: float = 1.0) -> bool:
    """True when the spectral radius is strictly below ``threshold``."""

    return spectral_radius(matrix) < float(threshold)


def stabilize_coupling(matrix: np.ndarray, max_radius: float) -> np.ndarray:
    """Rescale a matrix so its spectral radius does not exceed ``max_radius``."""

    matrix = np.asarray(matrix, dtype=np.float64)
    radius = spectral_radius(matrix)
    if radius <= 0.0 or radius <= max_radius:
        return matrix
    return matrix * (float(max_radius) / radius)


def clamp_state(state: np.ndarray, limit: float) -> np.ndarray:
    """Clip a state array into ``[-limit, limit]`` to prevent blow-up."""

    return np.clip(np.asarray(state, dtype=np.float64), -abs(float(limit)), abs(float(limit)))


def assert_finite(state: np.ndarray, name: str) -> None:
    """Raise if a state/observation array contains NaN or inf."""

    if not np.isfinite(np.asarray(state)).all():
        raise FloatingPointError(f"non-finite values detected in {name}")
