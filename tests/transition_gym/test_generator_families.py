import unittest

import numpy as np

from neurotwin.transition_gym import build_transition_gym
from neurotwin.transition_gym.synthetic_worlds import SyntheticWorldConfig


def _world(family):
    return SyntheticWorldConfig(seed=0, n_episodes=24, dynamics_family=family).validate()


class GeneratorFamilyTests(unittest.TestCase):
    def test_invalid_family_rejected(self):
        with self.assertRaises(ValueError):
            SyntheticWorldConfig(dynamics_family="bogus").validate()

    def test_nonlinear_family_changes_trajectories(self):
        linear = build_transition_gym(_world("linear"))
        nonlinear = build_transition_gym(_world("nonlinear"))
        # Same seed/world params, but the nonlinear generator bends the response trajectories.
        self.assertEqual(linear.response_eeg.shape, nonlinear.response_eeg.shape)
        self.assertFalse(np.allclose(linear.response_eeg, nonlinear.response_eeg))

    def test_quadratic_family_changes_trajectories(self):
        linear = build_transition_gym(_world("linear"))
        quad = build_transition_gym(_world("quadratic"))
        self.assertFalse(np.allclose(linear.response_eeg, quad.response_eeg))

    def test_all_families_finite_and_leakage_safe(self):
        for family in ("linear", "nonlinear", "quadratic"):
            bundle = build_transition_gym(_world(family))
            self.assertTrue(np.isfinite(bundle.response_eeg).all(), family)
            bundle.splits.assert_no_episode_leakage()


if __name__ == "__main__":
    unittest.main()
