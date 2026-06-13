import unittest

import numpy as np

from neurotwin.models.ktm import KTM, KTMConfig


class KTMTests(unittest.TestCase):
    def _model(self, **overrides):
        cfg = KTMConfig(seed=0, history_len=6, eeg_channels=5, n_perturbations=4, horizon=5)
        if overrides:
            cfg = KTMConfig(**{**cfg.__dict__, **overrides})
        return KTM(cfg)

    def _history(self, model, batch=8):
        rng = np.random.default_rng(123)
        return rng.normal(size=(batch, model.config.history_len, model.config.eeg_channels))

    def test_response_profile_shape_and_length_k(self):
        model = self._model()
        history = self._history(model)
        profile = model.predict_response_profile(history)
        self.assertEqual(profile.shape, (8, 4, 5, 5))  # (batch, K, H, C)
        self.assertEqual(profile.shape[1], model.config.n_perturbations)

    def test_outputs_finite(self):
        model = self._model()
        history = self._history(model)
        self.assertTrue(np.isfinite(model.predict_response_profile(history)).all())
        self.assertTrue(np.isfinite(model.predict_aggregate(history)).all())

    def test_predict_future_shape(self):
        model = self._model()
        history = self._history(model)
        future = model.predict_future(history, perturbation_index=2)
        self.assertEqual(future.shape, (8, 5, 5))

    def test_aggregate_shape(self):
        model = self._model()
        history = self._history(model)
        self.assertEqual(model.predict_aggregate(history).shape, (8, 5, 5))

    def test_uncertainty_positive_and_finite(self):
        model = self._model()
        history = self._history(model)
        unc = model.predict_uncertainty(history)
        self.assertEqual(unc.shape, (8, 4))
        self.assertTrue(np.isfinite(unc).all())
        self.assertTrue((unc > 0.0).all())

    def test_operators_are_non_commutative(self):
        model = self._model()
        comm = model.commutators()
        self.assertEqual(comm.shape, (4, 4))
        self.assertGreater(float(np.max(comm)), 1e-6)
        self.assertTrue(model.metadata()["non_commutative"])

    def test_reproducible(self):
        a = self._model()
        b = self._model()
        history = self._history(a)
        np.testing.assert_array_equal(a.predict_response_profile(history), b.predict_response_profile(history))


if __name__ == "__main__":
    unittest.main()
