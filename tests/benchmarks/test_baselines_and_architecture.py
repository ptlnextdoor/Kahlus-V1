import unittest

import numpy as np
import torch

from neurotwin.models.baselines import NumpyRidgeBaseline, TorchMLPBaseline, TorchTCNBaseline
from neurotwin.models.torch_models import NeuralStateSpaceTranslator, NeuralStateSpaceTranslatorConfig


class BaselinesAndArchitectureTests(unittest.TestCase):
    def test_numpy_ridge_baseline_predicts_shape(self):
        x = np.arange(20, dtype=np.float32).reshape(10, 2)
        y = x[:, :1] * 2.0
        model = NumpyRidgeBaseline(alpha=1e-3).fit(x, y)

        pred = model.predict(x)

        self.assertEqual(pred.shape, y.shape)

    def test_mlp_and_tcn_shape_contracts(self):
        x = torch.randn(2, 9, 4)

        self.assertEqual(TorchMLPBaseline(4, 3, hidden_dim=8)(x).shape, (2, 9, 3))
        self.assertEqual(TorchTCNBaseline(4, 3, hidden_dim=8)(x).shape, (2, 9, 3))

    def test_neurotwin_exposes_reconstruction_forecast_projection_heads(self):
        model = NeuralStateSpaceTranslator(
            input_dims={"eeg": 4, "fmri": 6},
            output_dims={"eeg": 4, "fmri": 6},
            config=NeuralStateSpaceTranslatorConfig(latent_dim=16, n_layers=1, subject_adapter_dim=4),
        )
        batch = {
            "eeg": torch.randn(2, 5, 4),
            "fmri": torch.randn(2, 5, 6),
        }
        output = model.forward_task(batch, target_modality="eeg", task="forecast")

        self.assertEqual(output["prediction"].shape, (2, 5, 4))
        self.assertEqual(output["latent"].shape, (2, 5, 16))
        self.assertEqual(output["projection"].shape[0], 2)
