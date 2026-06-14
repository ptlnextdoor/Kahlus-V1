"""Torch Dataset / DataLoaders over a synthetic Transition Gym bundle.

PROPOSED / SYNTHETIC ONLY. Wraps a :class:`TransitionGymBundle` (numpy float32 arrays) as a
``torch.utils.data.Dataset`` indexed by ``(episode, perturbation)``. Honors the gym's own
leakage-checked episode splits and uses a ``DistributedSampler`` under DDP. No real data.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler

from neurotwin.runtime.distributed import DistributedInfo
from neurotwin.transition_gym import TransitionGymBundle


class TransitionGymDataset(Dataset):
    """One sample per ``(episode, perturbation)``: ``(history_eeg, k, response_eeg)``."""

    def __init__(self, bundle: TransitionGymBundle, episodes: Sequence[int]) -> None:
        self.history = np.asarray(bundle.history_eeg, dtype=np.float32)  # (E, L, C)
        self.response = np.asarray(bundle.response_eeg, dtype=np.float32)  # (E, K, H, C)
        self.episodes = [int(e) for e in episodes]
        self.n_perturbations = int(self.response.shape[1])
        self._index = [
            (episode, k) for episode in self.episodes for k in range(self.n_perturbations)
        ]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        episode, k = self._index[i]
        history = torch.from_numpy(self.history[episode])  # (L, C)
        target = torch.from_numpy(self.response[episode, k])  # (H, C)
        return history, torch.tensor(k, dtype=torch.long), target


def make_dataloaders(
    bundle: TransitionGymBundle,
    *,
    batch_size: int,
    seed: int,
    dist_info: DistributedInfo | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build (train, val) loaders. Train uses a DistributedSampler under DDP, else a seeded
    RandomSampler; val is sequential (no sampler) so every rank evaluates the full split."""

    train_ds = TransitionGymDataset(bundle, bundle.splits.train_episodes)
    val_ds = TransitionGymDataset(bundle, bundle.splits.val_episodes)

    if dist_info is not None and dist_info.is_distributed:
        train_sampler: object = DistributedSampler(
            train_ds,
            num_replicas=dist_info.world_size,
            rank=dist_info.rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
    else:
        generator = torch.Generator().manual_seed(int(seed))
        train_sampler = RandomSampler(train_ds, generator=generator)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
