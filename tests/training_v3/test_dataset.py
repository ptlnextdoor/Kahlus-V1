import unittest

import numpy as np
import torch
from torch.utils.data import RandomSampler

from neurotwin.runtime.distributed import DistributedInfo
from neurotwin.training_v3 import KTMTrainConfig, TransitionGymDataset, make_dataloaders
from neurotwin.transition_gym import build_transition_gym

_CFG = KTMTrainConfig(mode="cpu_smoke", n_episodes=32, seed=0)


class TransitionGymDatasetTests(unittest.TestCase):
    def setUp(self):
        self.bundle = build_transition_gym(_CFG.to_world_config())

    def test_length_is_episodes_times_perturbations(self):
        ds = TransitionGymDataset(self.bundle, self.bundle.splits.train_episodes)
        self.assertEqual(len(ds), len(self.bundle.splits.train_episodes) * _CFG.n_perturbations)

    def test_item_shapes(self):
        ds = TransitionGymDataset(self.bundle, self.bundle.splits.train_episodes)
        history, k, target, profile = ds[0]
        self.assertEqual(tuple(history.shape), (_CFG.history_len, _CFG.eeg_channels))
        self.assertEqual(tuple(target.shape), (_CFG.horizon, _CFG.eeg_channels))
        self.assertEqual(
            tuple(profile.shape), (_CFG.n_perturbations, _CFG.horizon, _CFG.eeg_channels)
        )
        self.assertEqual(k.dtype, torch.long)
        self.assertTrue(0 <= int(k) < _CFG.n_perturbations)

    def test_no_episode_leakage(self):
        self.bundle.splits.assert_no_episode_leakage()
        train = set(self.bundle.splits.train_episodes)
        val = set(self.bundle.splits.val_episodes)
        test = set(self.bundle.splits.test_episodes)
        self.assertEqual(train & val, set())
        self.assertEqual(train & test, set())
        self.assertEqual(val & test, set())

    def test_deterministic_per_seed(self):
        other = build_transition_gym(_CFG.to_world_config())
        ds_a = TransitionGymDataset(self.bundle, self.bundle.splits.train_episodes)
        ds_b = TransitionGymDataset(other, other.splits.train_episodes)
        np.testing.assert_array_equal(ds_a[0][0].numpy(), ds_b[0][0].numpy())

    def test_make_dataloaders_single_process_uses_random_sampler(self):
        info = DistributedInfo(rank=0, local_rank=0, world_size=1)
        train_loader, val_loader = make_dataloaders(
            self.bundle, batch_size=8, seed=0, dist_info=info
        )
        self.assertIsInstance(train_loader.sampler, RandomSampler)
        history, k, target, profile = next(iter(train_loader))
        self.assertEqual(history.shape[0], 8)
        self.assertEqual(history.shape[1:], (_CFG.history_len, _CFG.eeg_channels))
        self.assertEqual(target.shape[1:], (_CFG.horizon, _CFG.eeg_channels))
        self.assertEqual(
            profile.shape[1:], (_CFG.n_perturbations, _CFG.horizon, _CFG.eeg_channels)
        )
        self.assertGreater(len(val_loader.dataset), 0)


if __name__ == "__main__":
    unittest.main()
