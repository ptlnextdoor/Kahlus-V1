"""Coupling operators for the dual-field scaffold.

Holds the low-rank neural coupling kernel ``K`` (the v2 stand-in for the pair-operator
field-update kernel) and the neural->hemodynamic readout ``gθ`` with an HRF-like lag.
"""

from __future__ import annotations

import numpy as np

from neurotwin.models.dual_field.stability import stabilize_coupling


def low_rank_coupling(
    rng: np.random.Generator,
    dim: int,
    rank: int,
    radius: float,
    scale: float = 0.4,
) -> np.ndarray:
    """Build a symmetric low-rank coupling matrix ``K = U Uᵀ`` (spectral-normalized).

    Symmetric + spectrally bounded keeps ``σ(N) @ K`` contraction-friendly so the fast
    field stays finite.
    """

    factor = rng.normal(scale=scale, size=(dim, max(1, rank))).astype(np.float64)
    kernel = factor @ factor.T
    return stabilize_coupling(kernel, radius)


def hrf_lag_weights(lag: int) -> np.ndarray:
    """Causal HRF-like weights over a window of ``lag + 1`` past neural states.

    Uses a simple gamma-shaped bump, peaking a few steps back, normalized to sum to 1.
    """

    width = int(lag) + 1
    taps = np.arange(width, dtype=np.float64)
    peak = max(1.0, (width - 1) / 2.0)
    shape = (taps ** 2) * np.exp(-taps / peak)
    total = float(shape.sum())
    if total <= 0.0:
        return np.full(width, 1.0 / width, dtype=np.float64)
    return shape / total


def neural_to_hemo(
    neural_window: np.ndarray,
    readout: np.ndarray,
    lag_weights: np.ndarray,
) -> np.ndarray:
    """gθ: lag-weighted average of a neural window, then a linear hemo readout.

    Args:
        neural_window: ``(batch, window, neural_dim)`` recent fast-field states.
        readout: ``(neural_dim, hemo_dim)`` linear map.
        lag_weights: ``(window,)`` causal weights (see :func:`hrf_lag_weights`).
    """

    neural_window = np.asarray(neural_window, dtype=np.float64)
    if neural_window.ndim != 3:
        raise ValueError("neural_window must be (batch, window, neural_dim)")
    window = neural_window.shape[1]
    weights = np.asarray(lag_weights, dtype=np.float64)
    if weights.shape[0] != window:
        raise ValueError("lag_weights length must match the neural window length")
    pooled = np.einsum("w,bwd->bd", weights, neural_window)
    return np.tanh(pooled @ np.asarray(readout, dtype=np.float64))
