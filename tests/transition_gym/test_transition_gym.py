import unittest

import numpy as np

from neurotwin.transition_gym import (
    SyntheticWorldConfig,
    build_transition_gym,
    trajectory_metrics,
)


class TransitionGymTests(unittest.TestCase):
    def _bundle(self, **overrides):
        cfg = SyntheticWorldConfig(
            seed=0,
            n_episodes=40,
            n_subjects=4,
            state_dim=6,
            n_perturbations=4,
            horizon=5,
            history_len=6,
            eeg_channels=5,
            behavior_dim=2,
        )
        if overrides:
            cfg = SyntheticWorldConfig(**{**cfg.__dict__, **overrides})
        return build_transition_gym(cfg)

    def test_observation_and_profile_shapes(self):
        b = self._bundle()
        self.assertEqual(b.history_eeg.shape, (40, 6, 5))
        self.assertEqual(b.response_eeg.shape, (40, 4, 5, 5))  # (E, K, H, C)
        self.assertEqual(b.response_behavior.shape, (40, 4, 5, 2))

    def test_all_outputs_finite(self):
        b = self._bundle()
        for array in (b.history_eeg, b.response_eeg, b.response_behavior, b.response_states):
            self.assertTrue(np.isfinite(array).all())

    def test_split_integrity_no_leakage(self):
        b = self._bundle()
        # Constructor asserts internally; re-check here for the contract.
        b.splits.assert_no_episode_leakage()
        b.splits.assert_no_composition_leakage()
        all_eps = set(b.splits.train_episodes) | set(b.splits.val_episodes) | set(b.splits.test_episodes)
        self.assertEqual(all_eps, set(range(40)))

    def test_held_out_composition_split_present(self):
        b = self._bundle()
        self.assertGreater(len(b.splits.heldout_compositions), 0)
        self.assertGreater(len(b.splits.train_compositions), 0)

    def test_perturbations_are_non_commutative(self):
        b = self._bundle()
        self.assertGreater(b.metadata["mean_commutator_gap"], 1e-6)
        self.assertTrue(b.data_card["non_commutative"])

    def test_data_card_matches_config(self):
        b = self._bundle()
        self.assertEqual(b.data_card["perturbation_battery_K"], 4)
        self.assertEqual(b.data_card["horizon_H"], 5)
        self.assertTrue(b.data_card["has_subject_adapters"])
        self.assertEqual(b.data_card["claim_status"], "synthetic_scaffold_only")

    def test_trajectory_metrics_finite(self):
        b = self._bundle()
        m = trajectory_metrics(b.response_eeg, b.response_eeg * 0.9)
        self.assertTrue(np.isfinite(m["mse"]))
        self.assertTrue(np.isfinite(m["pearson_r"]))
        self.assertTrue(m["finite"])

    def test_reproducible(self):
        a = self._bundle()
        b = self._bundle()
        np.testing.assert_array_equal(a.response_eeg, b.response_eeg)


if __name__ == "__main__":
    unittest.main()
