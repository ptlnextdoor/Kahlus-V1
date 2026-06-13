"""Train/val/test splits for the Transition Gym, including a held-out composition split.

Two kinds of generalization are tested:

1. Episode splits: disjoint train/val/test episode indices (no episode appears twice).
2. Held-out perturbation composition split: a subset of ordered perturbation pairs
   ``(a_i, a_j)`` is reserved for test only, so the model is evaluated on perturbation
   *sequences* it never saw composed during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np


@dataclass(frozen=True)
class TransitionGymSplits:
    train_episodes: list[int]
    val_episodes: list[int]
    test_episodes: list[int]
    train_compositions: list[tuple[str, str]]
    heldout_compositions: list[tuple[str, str]]

    def assert_no_episode_leakage(self) -> None:
        sets = [set(self.train_episodes), set(self.val_episodes), set(self.test_episodes)]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("episode index overlap detected across splits")

    def assert_no_composition_leakage(self) -> None:
        if set(self.train_compositions) & set(self.heldout_compositions):
            raise ValueError("perturbation composition overlap between train and held-out")


def build_splits(
    n_episodes: int,
    perturbation_names: list[str],
    seed: int = 0,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    heldout_composition_fraction: float = 0.3,
) -> TransitionGymSplits:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_episodes)
    n_test = max(1, int(round(n_episodes * test_fraction)))
    n_val = max(1, int(round(n_episodes * val_fraction)))
    test = sorted(int(i) for i in order[:n_test])
    val = sorted(int(i) for i in order[n_test : n_test + n_val])
    train = sorted(int(i) for i in order[n_test + n_val :])

    pairs = [pair for pair in permutations(perturbation_names, 2)]
    pair_order = rng.permutation(len(pairs))
    n_heldout = max(1, int(round(len(pairs) * heldout_composition_fraction)))
    heldout_idx = set(int(i) for i in pair_order[:n_heldout])
    heldout = [pairs[i] for i in sorted(heldout_idx)]
    train_comp = [pairs[i] for i in range(len(pairs)) if i not in heldout_idx]

    splits = TransitionGymSplits(
        train_episodes=train,
        val_episodes=val,
        test_episodes=test,
        train_compositions=train_comp,
        heldout_compositions=heldout,
    )
    splits.assert_no_episode_leakage()
    splits.assert_no_composition_leakage()
    return splits
