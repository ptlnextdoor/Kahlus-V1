import unittest

import numpy as np

from neurotwin.models.baselines import NumpyRidgeBaseline
from neurotwin.models.dual_field import DualFieldCompiler, DualFieldConfig, simulate_dual_field
from neurotwin.models.dual_field.stability import is_stable, spectral_radius


class DualFieldTests(unittest.TestCase):
    def test_rollout_shapes(self):
        cfg = DualFieldConfig(n_samples=6, time_steps=20, neural_dim=8, hemo_dim=5, eeg_channels=4, bold_channels=3)
        out = DualFieldCompiler(cfg).rollout()
        self.assertEqual(out.neural.shape, (6, 20, 8))
        self.assertEqual(out.hemo.shape, (6, 20, 5))
        self.assertEqual(out.eeg.shape, (6, 20, 4))
        self.assertEqual(out.bold.shape, (6, 20, 3))
        self.assertEqual(out.stimulus.shape, (6, 20, cfg.stimulus_dim))

    def test_outputs_are_finite(self):
        out = simulate_dual_field(DualFieldConfig(seed=3))
        for array in (out.neural, out.hemo, out.eeg, out.bold, out.stimulus):
            self.assertTrue(np.isfinite(array).all())

    def test_reproducible_for_same_seed(self):
        a = simulate_dual_field(DualFieldConfig(seed=7))
        b = simulate_dual_field(DualFieldConfig(seed=7))
        np.testing.assert_array_equal(a.eeg, b.eeg)
        np.testing.assert_array_equal(a.bold, b.bold)
        np.testing.assert_array_equal(a.neural, b.neural)

    def test_different_seeds_differ(self):
        a = simulate_dual_field(DualFieldConfig(seed=1))
        b = simulate_dual_field(DualFieldConfig(seed=2))
        self.assertFalse(np.array_equal(a.eeg, b.eeg))

    def test_fast_coupling_is_stable(self):
        compiler = DualFieldCompiler(DualFieldConfig(seed=0, coupling_radius=0.9))
        self.assertTrue(is_stable(compiler.fast_params.coupling))
        self.assertLess(spectral_radius(compiler.fast_params.coupling), 1.0)

    def test_states_stay_bounded(self):
        cfg = DualFieldConfig(time_steps=64, state_clip=50.0)
        out = DualFieldCompiler(cfg).rollout()
        self.assertLessEqual(float(np.max(np.abs(out.neural))), cfg.state_clip)
        self.assertLessEqual(float(np.max(np.abs(out.hemo))), cfg.state_clip)

    def test_ridge_baseline_forecasts_eeg(self):
        # Sanity: a ridge baseline can be fit on the synthetic EEG to forecast next step,
        # producing finite predictions. This is a plumbing check, not a superiority claim.
        out = simulate_dual_field(DualFieldConfig(seed=5, n_samples=12, time_steps=20))
        eeg = out.eeg.astype(np.float64)
        x = eeg[:, :-1].reshape(-1, eeg.shape[-1])
        y = eeg[:, 1:].reshape(-1, eeg.shape[-1])
        model = NumpyRidgeBaseline(alpha=1.0).fit(x, y)
        pred = model.predict(x)
        self.assertEqual(pred.shape, y.shape)
        self.assertTrue(np.isfinite(pred).all())


if __name__ == "__main__":
    unittest.main()
