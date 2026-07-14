import unittest

import torch

from neurotwin.models.baselines import TorchTCNBaseline
from neurotwin.models.pair_operator import NeuroTwinPairOperator, NeuroTwinPairOperatorConfig
from neurotwin.models.torch_models import (
    NeuralStateSpaceTranslator,
    NeuralStateSpaceTranslatorConfig,
    TinyGRUBaseline,
    TinySSMBaseline,
    TinyTransformerBaseline,
)


class CausalSequenceModelTests(unittest.TestCase):
    def test_transformer_and_tcn_do_not_read_future_samples(self):
        self._assert_causal(TinyTransformerBaseline(3, 2, latent_dim=8, n_heads=2, n_layers=1))
        self._assert_causal(TorchTCNBaseline(3, 2, hidden_dim=8, kernel_size=3))

    def test_tiny_ssm_is_a_state_space_recurrence_not_a_gru_alias(self):
        model = TinySSMBaseline(3, 2, latent_dim=8, n_layers=2)

        self.assertFalse(any(isinstance(module, torch.nn.GRU) for module in model.modules()))
        self._assert_causal(model)

    def test_legacy_ssm_fallback_is_explicitly_a_gru(self):
        model = TinyGRUBaseline(3, 2, latent_dim=8, n_layers=2)

        self.assertTrue(any(isinstance(module, torch.nn.GRU) for module in model.modules()))
        self._assert_causal(model)

    def test_pair_operator_hrf_path_is_causal_and_reports_real_mixing_not_fake_utilization(self):
        torch.manual_seed(8)
        model = NeuroTwinPairOperator(
            input_dims={"eeg": 3},
            output_dims={"fmri": 4},
            config=NeuroTwinPairOperatorConfig(
                latent_dim=8,
                n_layers=1,
                backbone="gru",
                pair_rank=2,
                hrf_delay_steps=1,
            ),
        )
        x = torch.randn(2, 6, 3)
        baseline = model.forward_task({"eeg": x}, target_modality="fmri", task="forecast")
        changed = x.clone()
        changed[:, 4:, :] += 100.0
        perturbed = model.forward_task({"eeg": changed}, target_modality="fmri", task="forecast")

        self.assertTrue(torch.allclose(baseline["prediction"][:, :4], perturbed["prediction"][:, :4], atol=1e-6))
        self.assertNotIn("expert_utilization", baseline)
        self.assertIn("pair_mixing_matrix", baseline)

    def test_translator_temporal_conv_and_transformer_are_causal(self):
        model = NeuralStateSpaceTranslator(
            input_dims={"eeg": 3},
            output_dims={"eeg": 2},
            config=NeuralStateSpaceTranslatorConfig(
                latent_dim=8,
                n_layers=1,
                backbone="transformer",
                encoder="temporal_conv",
                n_heads=2,
            ),
        )

        self._assert_causal(lambda values: model(values, target_modality="eeg", task="forecast"), mapping_input=True)

    def test_ambiguous_ssm_backbone_does_not_silently_build_a_gru(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            NeuralStateSpaceTranslator(
                input_dims={"eeg": 3},
                output_dims={"eeg": 2},
                config=NeuralStateSpaceTranslatorConfig(latent_dim=8, backbone="ssm"),
            )
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            NeuroTwinPairOperator(
                input_dims={"eeg": 3},
                output_dims={"eeg": 2},
                config=NeuroTwinPairOperatorConfig(latent_dim=8, backbone="ssm"),
            )

    def _assert_causal(self, model: torch.nn.Module, *, mapping_input: bool = False) -> None:
        torch.manual_seed(4)
        if isinstance(model, torch.nn.Module):
            model.eval()
        x = torch.randn(2, 6, 3)
        baseline = model({"eeg": x}) if mapping_input else model(x)
        changed = x.clone()
        changed[:, 4:, :] += 100.0
        perturbed = model({"eeg": changed}) if mapping_input else model(changed)

        self.assertTrue(torch.allclose(baseline[:, :4], perturbed[:, :4], atol=1e-6))
        self.assertFalse(torch.allclose(baseline[:, 4:], perturbed[:, 4:]))


if __name__ == "__main__":
    unittest.main()
