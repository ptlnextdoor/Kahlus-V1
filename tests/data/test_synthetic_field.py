import unittest

import numpy as np

from neurotwin.data.synthetic_field import generate_synthetic_latent_field


class SyntheticLatentFieldTests(unittest.TestCase):
    def test_synthetic_field_has_modal_observations_and_ground_truth(self):
        sample = generate_synthetic_latent_field(
            seed=11,
            n_samples=12,
            time_steps=6,
            n_nodes=5,
            latent_dim=4,
            eeg_channels=3,
            stimulus_dim=2,
        )

        self.assertEqual(sample.latent_field.shape, (12, 6, 5, 4))
        self.assertEqual(sample.fmri.shape, (12, 6, 5))
        self.assertEqual(sample.eeg.shape, (12, 6, 3))
        self.assertEqual(sample.stimulus.shape, (12, 6, 2))
        self.assertTrue(np.isfinite(sample.latent_field).all())
        self.assertTrue(np.isfinite(sample.fmri).all())
        self.assertTrue(np.isfinite(sample.eeg).all())
        self.assertEqual(sample.metadata["claim_status"], "synthetic_plumbing_only")

    def test_synthetic_field_is_reproducible_by_seed(self):
        first = generate_synthetic_latent_field(seed=5)
        second = generate_synthetic_latent_field(seed=5)
        third = generate_synthetic_latent_field(seed=6)

        self.assertTrue(np.allclose(first.latent_field, second.latent_field))
        self.assertFalse(np.allclose(first.latent_field, third.latent_field))


if __name__ == "__main__":
    unittest.main()
