"""Slow hemodynamic/measurement field update for the dual-field scaffold.

Implements::

    H_{t+1} = ρ H_t + (1-ρ) gθ(N_{t-L:t}) + noise

The slow field integrates a lag-weighted readout of the fast neural field, giving the
BOLD/fNIRS-like channel its delayed, smoothed character.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurotwin.models.dual_field.coupling import neural_to_hemo
from neurotwin.models.dual_field.stability import clamp_state


@dataclass(frozen=True)
class SlowFieldParams:
    rho: float
    readout: np.ndarray  # (neural_dim, hemo_dim) gθ readout
    lag_weights: np.ndarray  # (hemo_lag + 1,)
    state_clip: float


def slow_field_step(
    hemo_state: np.ndarray,
    neural_window: np.ndarray,
    noise: np.ndarray,
    params: SlowFieldParams,
) -> np.ndarray:
    """Advance the slow field one step.

    Args:
        hemo_state: ``(batch, hemo_dim)`` current H_t.
        neural_window: ``(batch, hemo_lag + 1, neural_dim)`` recent fast-field states.
        noise: ``(batch, hemo_dim)`` additive measurement noise for this step.
        params: fixed field parameters.
    """

    hemo_state = np.asarray(hemo_state, dtype=np.float64)
    drive = neural_to_hemo(neural_window, params.readout, params.lag_weights)
    next_state = params.rho * hemo_state + (1.0 - params.rho) * drive + np.asarray(noise, dtype=np.float64)
    return clamp_state(next_state, params.state_clip)
