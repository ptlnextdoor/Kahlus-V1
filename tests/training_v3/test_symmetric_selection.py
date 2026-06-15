import unittest

import numpy as np

from neurotwin.baseline_runner import run_baselines, transition_gym_regression_task
from neurotwin.training_v3 import KTMTrainConfig

_CFG = KTMTrainConfig(mode="cpu_smoke", n_episodes=32, seed=0)
_LEARNED = {"mlp", "transformer", "ssm_fallback"}
_FITTED = {"ridge", "autoregressive_ridge"}


class SymmetricSelectionTests(unittest.TestCase):
    def setUp(self):
        self.task = transition_gym_regression_task(_CFG.to_world_config())

    def test_task_has_validation_split(self):
        self.assertTrue(self.task.has_val())
        self.assertEqual(self.task.x_val.shape[1:], self.task.x_train.shape[1:])

    def test_best_val_selection_policies(self):
        result = run_baselines(self.task, train_steps=8, seed=0, select_best_val=True)
        self.assertEqual(result.selection_policy, "symmetric_best_val")
        for model_id, policy in result.checkpoint_policy_by_model.items():
            if model_id in _LEARNED:
                self.assertEqual(policy, "best_val", model_id)
            elif model_id in _FITTED:
                self.assertEqual(policy, "fitted", model_id)
        # Both selected (best-val) and final-step metrics are recorded.
        self.assertTrue(result.metrics_by_model)
        self.assertEqual(set(result.metrics_by_model), set(result.final_metrics_by_model))
        for m in result.metrics_by_model.values():
            self.assertTrue(all(np.isfinite(v) for v in m.values()))

    def test_final_step_default_is_backward_compatible(self):
        result = run_baselines(self.task, train_steps=8, seed=0)  # select_best_val=False
        self.assertEqual(result.selection_policy, "final_step")
        for model_id, policy in result.checkpoint_policy_by_model.items():
            if model_id in _LEARNED:
                self.assertEqual(policy, "final_step", model_id)
            elif model_id in _FITTED:
                self.assertEqual(policy, "fitted", model_id)
        # metrics_by_model and final_metrics_by_model coincide when no best-val selection happens.
        for model_id, m in result.metrics_by_model.items():
            self.assertEqual(m, result.final_metrics_by_model[model_id])


if __name__ == "__main__":
    unittest.main()
