"""Observation compilers: hidden states -> EEG-like and behavior-like observations.

These are the gym's observation operators. EEG-like output is a (tanh) linear projection of
the hidden state through the per-subject adapter; behavior/performance-like output is a
linear readout. Both add small measurement noise.
"""

from __future__ import annotations

import numpy as np

from neurotwin.transition_gym.synthetic_worlds import SyntheticWorld


def _apply_subject_adapter(world: SyntheticWorld, states: np.ndarray) -> np.ndarray:
    """Apply each episode's subject adapter ``A_s`` to its states.

    ``states`` is ``(n_episodes, steps, state_dim)``.
    """

    states = np.asarray(states, dtype=np.float64)
    adapters = world.subject_adapters[world.subject_ids]  # (n_episodes, state_dim, state_dim)
    return np.einsum("eij,esj->esi", adapters, states)


def compile_eeg(world: SyntheticWorld, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    adapted = _apply_subject_adapter(world, states)
    clean = np.tanh(adapted @ world.eeg_readout)
    noise = world.config.noise_scale * rng.normal(size=clean.shape)
    return clean + noise


def compile_behavior(world: SyntheticWorld, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    adapted = _apply_subject_adapter(world, states)
    clean = adapted @ world.behavior_readout
    noise = world.config.noise_scale * rng.normal(size=clean.shape)
    return clean + noise
