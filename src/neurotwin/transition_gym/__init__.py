"""Kahlus v3 synthetic Transition Gym (PROPOSED / SYNTHETIC ONLY).

The first Transition Gym instance: known hidden operators, a locked non-commutative
perturbation battery ``A_K``, EEG-like observations, behavior/performance-like outputs,
subject adapters, train/val/test splits, and a held-out perturbation-composition split.

Core objects (per the unified dossier):
    history h_t, battery A_K = {a_1,...,a_K}, future τ_{t:t+H},
    finite response profile C_K(h_t) = [pθ(τ|h_t,a_1), ..., pθ(τ|h_t,a_K)].

This is a local falsification benchmark, not a built model and not a scientific claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neurotwin.numerics import ignore_spurious_matmul_warnings
from neurotwin.transition_gym.data_cards import build_data_card
from neurotwin.transition_gym.metrics import mean_commutator_gap, trajectory_metrics
from neurotwin.transition_gym.observation_compilers import compile_behavior, compile_eeg
from neurotwin.transition_gym.perturbation_library import Perturbation, PerturbationLibrary
from neurotwin.transition_gym.splits import TransitionGymSplits, build_splits
from neurotwin.transition_gym.synthetic_worlds import (
    SyntheticWorld,
    SyntheticWorldConfig,
    generate_world,
    roll_future,
    roll_history,
)

__all__ = [
    "SyntheticWorldConfig",
    "SyntheticWorld",
    "PerturbationLibrary",
    "Perturbation",
    "TransitionGymSplits",
    "TransitionGymBundle",
    "build_transition_gym",
    "build_splits",
    "build_data_card",
    "trajectory_metrics",
    "mean_commutator_gap",
    "run_v3_benchmark",
    "write_v3_report",
]


def __getattr__(name):  # lazy to avoid a circular import with the benchmark module
    if name in {"run_v3_benchmark", "write_v3_report"}:
        from neurotwin.transition_gym import benchmark

        return getattr(benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass(frozen=True)
class TransitionGymBundle:
    world: SyntheticWorld
    library: PerturbationLibrary
    splits: TransitionGymSplits
    data_card: dict[str, Any]
    history_states: np.ndarray  # (E, L, state_dim)
    history_eeg: np.ndarray  # (E, L, eeg_channels)
    response_states: np.ndarray  # (E, K, H, state_dim)
    response_eeg: np.ndarray  # (E, K, H, eeg_channels)
    response_behavior: np.ndarray  # (E, K, H, behavior_dim)
    metadata: dict[str, Any]


def build_transition_gym(
    config: SyntheticWorldConfig | None = None,
    *,
    split_seed: int | None = None,
) -> TransitionGymBundle:
    cfg = (config or SyntheticWorldConfig()).validate()
    world = generate_world(cfg)
    library = PerturbationLibrary.locked(cfg.seed, cfg.state_dim, cfg.n_perturbations)
    obs_rng = np.random.default_rng(cfg.seed + 1000)

    response_states = np.zeros((cfg.n_episodes, cfg.n_perturbations, cfg.horizon, cfg.state_dim), dtype=np.float64)
    response_eeg = np.zeros((cfg.n_episodes, cfg.n_perturbations, cfg.horizon, cfg.eeg_channels), dtype=np.float64)
    response_behavior = np.zeros((cfg.n_episodes, cfg.n_perturbations, cfg.horizon, cfg.behavior_dim), dtype=np.float64)
    with ignore_spurious_matmul_warnings():
        history_states = roll_history(world)  # (E, L, Dz)
        history_eeg = compile_eeg(world, history_states, obs_rng)
        last_state = history_states[:, -1]  # (E, Dz)
        for k, name in enumerate(library.names):
            perturbed = library.apply(name, last_state)
            future = roll_future(world, perturbed)  # (E, H, Dz)
            response_states[:, k] = future
            response_eeg[:, k] = compile_eeg(world, future, obs_rng)
            response_behavior[:, k] = compile_behavior(world, future, obs_rng)

    splits = build_splits(
        cfg.n_episodes,
        library.names,
        seed=cfg.seed if split_seed is None else split_seed,
    )
    with ignore_spurious_matmul_warnings():
        gap = mean_commutator_gap(library, splits.heldout_compositions, last_state)
    data_card = build_data_card(world, library, splits, gap)

    metadata = {
        "branch": "v3",
        "claim_status": "synthetic_scaffold_only",
        "seed": int(cfg.seed),
        "history_eeg_shape": list(history_eeg.shape),
        "response_eeg_shape": list(response_eeg.shape),
        "mean_commutator_gap": float(gap),
    }
    return TransitionGymBundle(
        world=world,
        library=library,
        splits=splits,
        data_card=data_card,
        history_states=history_states.astype(np.float32),
        history_eeg=history_eeg.astype(np.float32),
        response_states=response_states.astype(np.float32),
        response_eeg=response_eeg.astype(np.float32),
        response_behavior=response_behavior.astype(np.float32),
        metadata=metadata,
    )
