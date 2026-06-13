"""Fast neural field update for the dual-field scaffold.

Implements the discrete update::

    N_{t+1} = N_t + dt[-Λ N_t + K σ(N_t) + B U_t + Rθ(N_t, U_t)]

with σ = tanh and a small nonlinear residual Rθ. The state is clamped each step for NaN
hygiene.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurotwin.models.dual_field.stability import clamp_state


@dataclass(frozen=True)
class FastFieldParams:
    decay: np.ndarray  # (neural_dim,) elementwise Λ
    coupling: np.ndarray  # (neural_dim, neural_dim) K
    stimulus_in: np.ndarray  # (stimulus_dim, neural_dim) B
    residual_w: np.ndarray  # (neural_dim, neural_dim) Rθ weight
    residual_scale: float
    dt: float
    state_clip: float


def fast_field_step(
    neural_state: np.ndarray,
    stimulus: np.ndarray,
    params: FastFieldParams,
) -> np.ndarray:
    """Advance the fast neural field one step.

    Args:
        neural_state: ``(batch, neural_dim)`` current N_t.
        stimulus: ``(batch, stimulus_dim)`` current U_t.
        params: fixed field parameters.
    """

    neural_state = np.asarray(neural_state, dtype=np.float64)
    stimulus = np.asarray(stimulus, dtype=np.float64)
    decay_term = -params.decay[None, :] * neural_state
    coupling_term = np.tanh(neural_state) @ params.coupling
    stimulus_term = stimulus @ params.stimulus_in
    residual_term = params.residual_scale * np.tanh(neural_state @ params.residual_w)
    delta = decay_term + coupling_term + stimulus_term + residual_term
    next_state = neural_state + params.dt * delta
    return clamp_state(next_state, params.state_clip)
