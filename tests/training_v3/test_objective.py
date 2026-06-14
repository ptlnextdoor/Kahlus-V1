import math
import unittest

import torch

from neurotwin.models.ktm import TorchKTM
from neurotwin.training_v3 import KTMTrainConfig
from neurotwin.training_v3.objective import LossExplosionGuard, is_finite_loss, ktm_loss

_CFG = KTMTrainConfig(mode="cpu_smoke", n_episodes=16, seed=0)


class KTMObjectiveTests(unittest.TestCase):
    def _batch(self):
        b, l, c, h = 4, _CFG.history_len, _CFG.eeg_channels, _CFG.horizon
        history = torch.randn(b, l, c)
        k = torch.zeros(b, dtype=torch.long)
        target = torch.randn(b, h, c)
        return history, k, target

    def test_loss_finite_with_components(self):
        model = TorchKTM(_CFG.to_model_config())
        history, k, target = self._batch()
        pred, log_var = model(history, k)
        loss, components = ktm_loss(pred, log_var, target, _CFG)
        self.assertTrue(torch.isfinite(loss).all())
        for key in ("trajectory", "profile", "nll", "total"):
            self.assertIn(key, components)
            self.assertTrue(math.isfinite(components[key]))

    def test_gradients_flow(self):
        model = TorchKTM(_CFG.to_model_config())
        history, k, target = self._batch()
        pred, log_var = model(history, k)
        loss, _ = ktm_loss(pred, log_var, target, _CFG)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_is_finite_loss(self):
        self.assertTrue(is_finite_loss(1.23))
        self.assertFalse(is_finite_loss(float("nan")))
        self.assertFalse(is_finite_loss(float("inf")))

    def test_loss_explosion_guard_triggers(self):
        guard = LossExplosionGuard(factor=8.0, window=8, warmup=8)
        for _ in range(8):
            self.assertFalse(guard.update(1.0))
        self.assertTrue(guard.update(1000.0))

    def test_loss_explosion_guard_flags_nonfinite(self):
        guard = LossExplosionGuard(factor=8.0)
        self.assertTrue(guard.update(float("nan")))


if __name__ == "__main__":
    unittest.main()
