"""Data card generation for a Transition Gym instance.

The data card is a machine-readable summary of the benchmark instance: shapes, the locked
perturbation battery, split sizes, the held-out composition split, subject-adapter presence,
and the measured non-commutativity. It carries explicit synthetic claim status.
"""

from __future__ import annotations

from typing import Any

from neurotwin.transition_gym.perturbation_library import PerturbationLibrary
from neurotwin.transition_gym.splits import TransitionGymSplits
from neurotwin.transition_gym.synthetic_worlds import SyntheticWorld


def build_data_card(
    world: SyntheticWorld,
    library: PerturbationLibrary,
    splits: TransitionGymSplits,
    mean_commutator_gap: float,
) -> dict[str, Any]:
    cfg = world.config
    return {
        "schema": "kahlus.transition_gym_data_card.v1",
        "branch": "v3",
        "claim_status": "synthetic_scaffold_only",
        "seed": int(cfg.seed),
        "n_episodes": int(cfg.n_episodes),
        "n_subjects": int(cfg.n_subjects),
        "state_dim": int(cfg.state_dim),
        "perturbation_battery_K": len(library),
        "perturbation_names": library.names,
        "horizon_H": int(cfg.horizon),
        "history_len": int(cfg.history_len),
        "eeg_channels": int(cfg.eeg_channels),
        "behavior_dim": int(cfg.behavior_dim),
        "has_subject_adapters": True,
        "splits": {
            "train_episodes": len(splits.train_episodes),
            "val_episodes": len(splits.val_episodes),
            "test_episodes": len(splits.test_episodes),
            "train_compositions": len(splits.train_compositions),
            "heldout_compositions": len(splits.heldout_compositions),
        },
        "heldout_compositions": [list(pair) for pair in splits.heldout_compositions],
        "non_commutative": bool(mean_commutator_gap > 1e-8),
        "mean_commutator_gap": float(mean_commutator_gap),
    }
