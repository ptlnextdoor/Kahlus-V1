import unittest

import torch

from neurotwin.models.torch_models import (
    NeuralStateSpaceTranslator,
    TinySSMBaseline,
    TinyTransformerBaseline,
)


class ModelShapeTests(unittest.TestCase):
    def test_baseline_shape_contracts(self):
        x = torch.randn(2, 7, 5)

        transformer = TinyTransformerBaseline(input_dim=5, output_dim=3, latent_dim=16, n_heads=4)
        ssm = TinySSMBaseline(input_dim=5, output_dim=3, latent_dim=16)

        self.assertEqual(transformer(x).shape, (2, 7, 3))
        self.assertEqual(ssm(x).shape, (2, 7, 3))

    def test_neurotwin_translator_multimodal_shape_contract(self):
        model = NeuralStateSpaceTranslator(
            input_dims={"fmri": 6, "eeg": 4},
            output_dims={"fmri": 6, "eeg": 4},
            latent_dim=24,
            n_layers=2,
        )
        batch = {
            "fmri": torch.randn(3, 5, 6),
            "eeg": torch.randn(3, 5, 4),
        }

        output = model(batch, target_modality="fmri")

        self.assertEqual(output.shape, (3, 5, 6))

    def test_neurotwin_rejects_unknown_target(self):
        model = NeuralStateSpaceTranslator(input_dims={"fmri": 3}, output_dims={"fmri": 3})

        with self.assertRaisesRegex(ValueError, "Unknown target modality"):
            model({"fmri": torch.randn(1, 2, 3)}, target_modality="meg")
