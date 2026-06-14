"""Synthetic worlds with known hidden operators for the Transition Gym.

Each world is a small affine hidden dynamical system::

    z_{t+1} = A z_t           (base autonomous dynamics, spectrally bounded)

with per-subject low-rank adapters ``A_s = I + U_s V_sᵀ`` applied to observations. Episodes
roll out a history, then for each locked perturbation ``a_k`` the perturbed future trajectory
is generated. EEG-like and behavior/performance-like observations are produced downstream by
``observation_compilers``. The operators are known, so recovery can be checked exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


VALID_DYNAMICS_FAMILIES: frozenset[str] = frozenset({"linear", "nonlinear", "quadratic"})


@dataclass(frozen=True)
class SyntheticWorldConfig:
    seed: int = 0
    n_episodes: int = 48
    n_subjects: int = 4
    state_dim: int = 6
    n_perturbations: int = 4  # K
    horizon: int = 5  # H
    history_len: int = 6
    eeg_channels: int = 5
    behavior_dim: int = 2
    adapter_rank: int = 1
    adapter_scale: float = 0.1
    base_radius: float = 0.9
    noise_scale: float = 0.02
    # Generator family for the hidden dynamics. ``linear`` is the original gym whose operator
    # structure the KTM's operator path may be aligned to; ``nonlinear``/``quadratic`` bend the
    # generator away from that structure so architecture-affinity can be falsified. ``nonlinear``
    # squashes each step (tanh); ``quadratic`` adds a mild element-wise quadratic coupling.
    dynamics_family: str = "linear"
    nonlinearity_scale: float = 0.5

    def validate(self) -> "SyntheticWorldConfig":
        positives = {
            "n_episodes": self.n_episodes,
            "n_subjects": self.n_subjects,
            "state_dim": self.state_dim,
            "n_perturbations": self.n_perturbations,
            "horizon": self.horizon,
            "history_len": self.history_len,
            "eeg_channels": self.eeg_channels,
            "behavior_dim": self.behavior_dim,
            "adapter_rank": self.adapter_rank,
        }
        for name, value in positives.items():
            if int(value) < 1:
                raise ValueError(f"SyntheticWorldConfig.{name} must be >= 1, got {value}")
        if self.n_episodes < self.n_subjects:
            raise ValueError("n_episodes must be >= n_subjects")
        if not 0.0 < self.base_radius < 1.0:
            raise ValueError("base_radius must be in (0, 1)")
        if self.dynamics_family not in VALID_DYNAMICS_FAMILIES:
            raise ValueError(
                f"dynamics_family must be one of {sorted(VALID_DYNAMICS_FAMILIES)}, "
                f"got {self.dynamics_family!r}"
            )
        return self


@dataclass(frozen=True)
class SyntheticWorld:
    config: SyntheticWorldConfig
    base_dynamics: np.ndarray  # (state_dim, state_dim) known A
    eeg_readout: np.ndarray  # (state_dim, eeg_channels)
    behavior_readout: np.ndarray  # (state_dim, behavior_dim)
    subject_adapters: np.ndarray  # (n_subjects, state_dim, state_dim)
    init_states: np.ndarray  # (n_episodes, state_dim)
    subject_ids: np.ndarray  # (n_episodes,)


def _spectral_normalize(matrix: np.ndarray, radius: float) -> np.ndarray:
    eigvals = np.linalg.eigvals(matrix)
    current = float(np.max(np.abs(eigvals)))
    if current <= 0.0:
        return matrix
    return matrix * (radius / current)


def generate_world(config: SyntheticWorldConfig) -> SyntheticWorld:
    cfg = config.validate()
    rng = np.random.default_rng(cfg.seed)
    base = _spectral_normalize(rng.normal(scale=0.5, size=(cfg.state_dim, cfg.state_dim)), cfg.base_radius)
    eeg_readout = rng.normal(scale=0.5, size=(cfg.state_dim, cfg.eeg_channels))
    behavior_readout = rng.normal(scale=0.5, size=(cfg.state_dim, cfg.behavior_dim))

    adapters = np.zeros((cfg.n_subjects, cfg.state_dim, cfg.state_dim), dtype=np.float64)
    for s in range(cfg.n_subjects):
        u = rng.normal(scale=cfg.adapter_scale, size=(cfg.state_dim, cfg.adapter_rank))
        v = rng.normal(scale=cfg.adapter_scale, size=(cfg.state_dim, cfg.adapter_rank))
        adapters[s] = np.eye(cfg.state_dim) + u @ v.T

    init_states = rng.normal(scale=0.5, size=(cfg.n_episodes, cfg.state_dim))
    subject_ids = rng.integers(0, cfg.n_subjects, size=cfg.n_episodes)
    return SyntheticWorld(
        config=cfg,
        base_dynamics=base,
        eeg_readout=eeg_readout,
        behavior_readout=behavior_readout,
        subject_adapters=adapters,
        init_states=init_states,
        subject_ids=subject_ids,
    )


def _dynamics_step(z: np.ndarray, base: np.ndarray, cfg: SyntheticWorldConfig) -> np.ndarray:
    """One hidden-dynamics step under the world's generator family.

    ``linear`` is ``z A^T`` (the original gym). ``nonlinear`` squashes that step with ``tanh``;
    ``quadratic`` adds a mild element-wise quadratic coupling. The non-linear families keep the
    base operator but bend the trajectory away from pure linear-operator structure, so a model
    whose inductive bias is exactly linear operators cannot trivially match them.
    """

    linear = z @ base.T
    family = cfg.dynamics_family
    if family == "linear":
        return linear
    scale = float(cfg.nonlinearity_scale)
    if family == "nonlinear":
        return linear + scale * (np.tanh(linear) - linear)
    # quadratic: a small odd-free element-wise term, kept bounded by the spectral radius.
    return linear + scale * np.tanh(linear * linear) * np.sign(linear)


def roll_history(world: SyntheticWorld) -> np.ndarray:
    """Roll the autonomous base dynamics to produce per-episode history states.

    Returns ``(n_episodes, history_len, state_dim)``.
    """

    cfg = world.config
    states = np.zeros((cfg.n_episodes, cfg.history_len, cfg.state_dim), dtype=np.float64)
    z = world.init_states.copy()
    for t in range(cfg.history_len):
        states[:, t] = z
        z = _dynamics_step(z, world.base_dynamics, cfg)
    return states


def roll_future(world: SyntheticWorld, perturbed_state: np.ndarray) -> np.ndarray:
    """Roll the base dynamics forward ``horizon`` steps from a perturbed state.

    Returns ``(n_episodes, horizon, state_dim)``.
    """

    cfg = world.config
    future = np.zeros((cfg.n_episodes, cfg.horizon, cfg.state_dim), dtype=np.float64)
    z = np.asarray(perturbed_state, dtype=np.float64).copy()
    for t in range(cfg.horizon):
        z = _dynamics_step(z, world.base_dynamics, cfg)
        future[:, t] = z
    return future
