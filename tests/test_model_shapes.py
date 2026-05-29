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

    def test_neurotwin_transformer_backbone_and_subject_controls(self):
        model = NeuralStateSpaceTranslator(
            input_dims={"eeg": 4},
            output_dims={"eeg": 4},
            latent_dim=16,
            n_layers=1,
            backbone="transformer",
            encoder="temporal_conv",
            n_heads=4,
            subject_adapter_dim=4,
            subject_vocab_size=8,
            use_subject_embeddings=True,
            adapter_mode="few_shot",
        )
        batch = {"eeg": torch.randn(2, 6, 4)}
        subject_ids = torch.tensor([0, 1], dtype=torch.long)

        output = model.forward_task(batch, target_modality="eeg", subject_ids=subject_ids)

        self.assertEqual(output["prediction"].shape, (2, 6, 4))
        self.assertEqual(output["projection"].shape[0], 2)

    def test_subject_embedding_is_disabled_without_adapter_mode(self):
        model = NeuralStateSpaceTranslator(
            input_dims={"eeg": 4},
            output_dims={"eeg": 4},
            latent_dim=16,
            subject_vocab_size=8,
            use_subject_embeddings=True,
            adapter_mode="disabled",
        )

        self.assertIsNone(model.subject_embedding)

    def test_neurotwin_rejects_unknown_target(self):
        model = NeuralStateSpaceTranslator(input_dims={"fmri": 3}, output_dims={"fmri": 3})

        with self.assertRaisesRegex(ValueError, "Unknown target modality"):
            model({"fmri": torch.randn(1, 2, 3)}, target_modality="meg")
