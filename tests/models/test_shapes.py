import unittest

import torch

import neurotwin.models as models
from neurotwin.models.pair_operator import NeuroTwinPairOperator, NeuroTwinPairOperatorConfig
from neurotwin.models.torch_models import (
    NeuralStateSpaceTranslator,
    NeuralStateSpaceTranslatorConfig,
    TinySSMBaseline,
    TinyTransformerBaseline,
)


class ModelShapeTests(unittest.TestCase):
    def test_models_package_exports_active_translator_and_local_baselines(self):
        self.assertIs(models.NeuralStateSpaceTranslator, NeuralStateSpaceTranslator)
        self.assertIs(models.NeuralStateSpaceTranslatorConfig, NeuralStateSpaceTranslatorConfig)
        self.assertIs(models.NeuroTwinPairOperator, NeuroTwinPairOperator)
        self.assertIs(models.NeuroTwinPairOperatorConfig, NeuroTwinPairOperatorConfig)
        self.assertIs(models.TinySSMBaseline, TinySSMBaseline)

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
            config=NeuralStateSpaceTranslatorConfig(latent_dim=24, n_layers=2),
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
            config=NeuralStateSpaceTranslatorConfig(
                latent_dim=16,
                n_layers=1,
                backbone="transformer",
                encoder="temporal_conv",
                n_heads=4,
                subject_adapter_dim=4,
                subject_vocab_size=8,
                use_subject_embeddings=True,
                adapter_mode="few_shot",
            ),
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
            config=NeuralStateSpaceTranslatorConfig(
                latent_dim=16,
                subject_vocab_size=8,
                use_subject_embeddings=True,
                adapter_mode="disabled",
            ),
        )

        self.assertIsNone(model.subject_embedding)

    def test_neurotwin_rejects_unknown_config_options(self):
        with self.assertRaisesRegex(TypeError, "Unknown NeuralStateSpaceTranslatorConfig option"):
            NeuralStateSpaceTranslator(
                input_dims={"fmri": 3},
                output_dims={"fmri": 3},
                latnt_dim=16,
            )

    def test_neurotwin_rejects_unwired_mamba_backbone(self):
        with self.assertRaisesRegex(ValueError, "backbone=\"mamba\" is not wired"):
            NeuralStateSpaceTranslator(
                input_dims={"eeg": 4},
                output_dims={"eeg": 4},
                config=NeuralStateSpaceTranslatorConfig(latent_dim=16, backbone="mamba"),
            )

    def test_neurotwin_rejects_unknown_target(self):
        model = NeuralStateSpaceTranslator(input_dims={"fmri": 3}, output_dims={"fmri": 3})

        with self.assertRaisesRegex(ValueError, "Unknown target modality"):
            model({"fmri": torch.randn(1, 2, 3)}, target_modality="meg")

    def test_neurotwin_rejects_unknown_task_mode(self):
        model = NeuralStateSpaceTranslator(input_dims={"fmri": 3}, output_dims={"fmri": 3})

        with self.assertRaisesRegex(ValueError, "task must be one of"):
            model.forward_task({"fmri": torch.randn(1, 2, 3)}, target_modality="fmri", task="forcast")

    def test_pair_operator_fmri_shape_contract_and_uncertainty(self):
        model = NeuroTwinPairOperator(
            input_dims={"stimulus": 5},
            output_dims={"fmri": 7},
            config=NeuroTwinPairOperatorConfig(
                latent_dim=16,
                n_layers=1,
                pair_rank=3,
                projection_dim=8,
                use_pair_uncertainty=True,
            ),
        )

        output = model.forward_task(
            {"stimulus": torch.randn(2, 6, 5)},
            target_modality="fmri",
            task="reconstruction",
        )

        self.assertEqual(output["prediction"].shape, (2, 6, 7))
        self.assertEqual(output["uncertainty"].shape, (2, 6, 7))
        self.assertEqual(output["pair_confidence"].shape, (7, 7))
        self.assertEqual(output["pair_uncertainty"].shape, (7, 7))
        self.assertEqual(output["projection"].shape, (2, 8))
        self.assertTrue(torch.isfinite(output["prediction"]).all())

    def test_pair_operator_pair_state_can_be_disabled(self):
        model = NeuroTwinPairOperator(
            input_dims={"fmri": 4},
            output_dims={"fmri": 4},
            config=NeuroTwinPairOperatorConfig(latent_dim=12, use_pair_state=False, use_uncertainty_head=False),
        )

        output = model.forward_task({"fmri": torch.randn(2, 5, 4)}, target_modality="fmri", task="forecast")

        self.assertEqual(output["prediction"].shape, (2, 5, 4))
        self.assertNotIn("uncertainty", output)
        self.assertTrue(torch.allclose(output["pair_confidence"], torch.eye(4)))

    def test_pair_operator_uses_compact_pair_state_for_1000_fmri_parcels(self):
        model = NeuroTwinPairOperator(
            input_dims={"fmri": 1000},
            output_dims={"fmri": 1000},
            config=NeuroTwinPairOperatorConfig(
                latent_dim=8,
                n_layers=1,
                pair_rank=4,
                pair_confidence_max_parcels=32,
                use_uncertainty_head=True,
            ),
        )

        output = model.forward_task({"fmri": torch.randn(1, 2, 1000)}, target_modality="fmri", task="forecast")

        self.assertEqual(output["prediction"].shape, (1, 2, 1000))
        self.assertEqual(output["uncertainty"].shape, (1, 2, 1000))
        self.assertEqual(output["pair_confidence"].shape, (2, 1000, 4))
        self.assertTrue(torch.isfinite(output["prediction"]).all())

    def test_pair_operator_pair_state_actively_changes_predictions(self):
        torch.manual_seed(3)
        model = NeuroTwinPairOperator(
            input_dims={"fmri": 6},
            output_dims={"fmri": 6},
            config=NeuroTwinPairOperatorConfig(latent_dim=10, n_layers=1, pair_rank=3, use_uncertainty_head=False),
        )
        batch = {"fmri": torch.randn(2, 4, 6)}

        with_pair = model.forward_task(batch, target_modality="fmri", task="forecast")["prediction"]
        model.use_pair_state = False
        without_pair = model.forward_task(batch, target_modality="fmri", task="forecast")["prediction"]

        self.assertGreater(torch.max(torch.abs(with_pair - without_pair)).item(), 1e-8)
