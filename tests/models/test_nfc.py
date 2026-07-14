import unittest

import torch

import neurotwin.models as models
from neurotwin.models.nfc import (
    BehaviorObservationOperator,
    EEGObservationOperator,
    FMRIObservationOperator,
    FieldUpdateOperator,
    LatentNeuralField,
    LowRankPairKernel,
    NeuralFieldCompiler,
    NeuralFieldCompilerConfig,
    StimulusConditioningOperator,
    UncertaintyMapHead,
)
from neurotwin.training.prepared_metrics import gaussian_nll_loss


class NeuralFieldCompilerShapeTests(unittest.TestCase):
    def test_models_package_exports_nfc_experimental_architecture(self):
        self.assertIs(models.NeuralFieldCompiler, NeuralFieldCompiler)
        self.assertEqual(NeuralFieldCompiler.model_id, "neurotwin_nfc")
        self.assertEqual(NeuralFieldCompiler.model_status, "experimental_architecture")

    def test_latent_field_shape_contract(self):
        field = LatentNeuralField(input_dim=3, latent_dim=8, subject_dim=4, stimulus_dim=5)
        observations = torch.randn(2, 6, 7, 3)
        subject_state = torch.randn(2, 4)
        stimulus_state = torch.randn(2, 6, 5)

        latent = field(observations, subject_state=subject_state, stimulus_state=stimulus_state)

        self.assertEqual(latent.shape, (2, 6, 7, 8))
        self.assertTrue(torch.isfinite(latent).all())

    def test_field_update_is_shape_stable_and_causal(self):
        updater = FieldUpdateOperator(latent_dim=8, stimulus_dim=3, subject_dim=2)
        field = torch.randn(2, 5, 4, 8)
        stimulus = torch.randn(2, 5, 3)
        subject = torch.randn(2, 2)

        baseline = updater(field, stimulus_state=stimulus, subject_state=subject)
        changed_future = stimulus.clone()
        changed_future[:, -1, :] += 100.0
        perturbed = updater(field, stimulus_state=changed_future, subject_state=subject)

        self.assertEqual(baseline.shape, field.shape)
        self.assertTrue(torch.isfinite(baseline).all())
        self.assertTrue(torch.allclose(baseline[:, :-1], perturbed[:, :-1], atol=1e-5))
        self.assertFalse(torch.allclose(baseline[:, -1], perturbed[:, -1]))

    def test_field_update_backends_and_depth_change_the_computation(self):
        torch.manual_seed(17)
        gru = FieldUpdateOperator(latent_dim=8, backend="gru", n_layers=2, n_heads=2)
        torch.manual_seed(17)
        transformer = FieldUpdateOperator(latent_dim=8, backend="transformer", n_layers=2, n_heads=2)
        field = torch.randn(2, 5, 4, 8)

        gru_output = gru(field)
        transformer_output = transformer(field)

        self.assertNotEqual(tuple(gru.state_dict()), tuple(transformer.state_dict()))
        self.assertFalse(torch.allclose(gru_output, transformer_output))
        self.assertEqual(gru.temporal.num_layers, 2)
        self.assertEqual(transformer.temporal.num_layers, 2)

    def test_field_update_rejects_ambiguous_structural_prior(self):
        updater = FieldUpdateOperator(latent_dim=8)
        field = torch.randn(2, 5, 4, 8)

        with self.assertRaisesRegex(ValueError, "structural_prior must have shape"):
            updater(field, anatomy=torch.zeros(2, 4, 4))

    def test_low_rank_pair_kernel_changes_output_when_enabled(self):
        torch.manual_seed(3)
        z = torch.randn(2, 5, 8)
        full = LowRankPairKernel(latent_dim=8, rank=3, use_pair_state=True)
        disabled = LowRankPairKernel(latent_dim=8, rank=3, use_pair_state=False)

        full_out = full(z)
        disabled_out = disabled(z)
        weights = full.pair_weights(z)

        self.assertEqual(full_out.shape, z.shape)
        self.assertEqual(disabled_out.shape, z.shape)
        self.assertEqual(weights.shape, (2, 5, 5))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-5))
        self.assertFalse(torch.allclose(full_out, disabled_out))

    def test_observation_operators_return_modality_shapes(self):
        latent = torch.randn(2, 6, 7, 8)

        fmri = FMRIObservationOperator(latent_dim=8, output_dim=7)
        eeg = EEGObservationOperator(latent_dim=8, output_dim=3)
        behavior = BehaviorObservationOperator(latent_dim=8, output_dim=2)

        self.assertEqual(fmri(latent).shape, (2, 6, 7))
        self.assertEqual(eeg(latent).shape, (2, 6, 3))
        self.assertEqual(behavior(latent).shape, (2, 6, 2))

    def test_fmri_observation_operator_cannot_read_future_latent_samples(self):
        torch.manual_seed(9)
        operator = FMRIObservationOperator(latent_dim=8, output_dim=7, hrf_delay_steps=1)
        latent = torch.randn(2, 6, 7, 8)
        baseline = operator(latent)
        changed = latent.clone()
        changed[:, 4:, :, :] += 100.0
        perturbed = operator(changed)

        self.assertTrue(torch.allclose(baseline[:, :4], perturbed[:, :4], atol=1e-6))
        self.assertFalse(torch.allclose(baseline[:, 4:], perturbed[:, 4:]))

    def test_stimulus_conditioning_is_causal(self):
        conditioner = StimulusConditioningOperator(stimulus_dim=4, latent_dim=8, lag_steps=2)
        stimulus = torch.randn(2, 5, 4)

        baseline = conditioner(stimulus)
        changed_future = stimulus.clone()
        changed_future[:, -1, :] += 100.0
        perturbed = conditioner(changed_future)

        self.assertEqual(baseline.shape, (2, 5, 8))
        self.assertTrue(torch.allclose(baseline[:, :-1], perturbed[:, :-1], atol=1e-5))
        self.assertFalse(torch.allclose(baseline[:, -1], perturbed[:, -1]))

    def test_uncertainty_head_outputs_region_time_and_pair_maps(self):
        latent = torch.randn(2, 6, 7, 8)
        head = UncertaintyMapHead(latent_dim=8, pair_uncertainty=True)

        maps = head(latent)

        self.assertEqual(maps["region_uncertainty"].shape, (2, 6, 7))
        self.assertEqual(maps["time_uncertainty"].shape, (2, 6))
        self.assertEqual(maps["pair_uncertainty"].shape, (2, 7, 7))
        self.assertTrue(torch.isfinite(maps["region_uncertainty"]).all())

    def test_gaussian_objective_trains_nfc_uncertainty_head(self):
        model = NeuralFieldCompiler(
            input_dims={"eeg": 3},
            output_dims={"eeg": 3},
            config=NeuralFieldCompilerConfig(latent_dim=8, use_uncertainty=True),
        )
        output = model.forward_task({"eeg": torch.randn(2, 6, 3)}, target_modality="eeg", task="forecast")
        loss = gaussian_nll_loss(output["prediction"], torch.randn(2, 6, 3), output["uncertainty"])

        loss.backward()

        gradient = model.uncertainty_head.region_head.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_neural_field_compiler_forward_contract(self):
        model = NeuralFieldCompiler(
            input_dims={"stimulus": 4},
            output_dims={"fmri": 7, "eeg": 3},
            config=NeuralFieldCompilerConfig(
                latent_dim=8,
                pair_rank=3,
                use_pair_kernel=True,
                use_observation_operator=True,
                use_uncertainty=True,
            ),
        )

        output = model.forward_task(
            {"stimulus": torch.randn(2, 6, 4)},
            target_modality="fmri",
            task="stimulus_to_fmri_response",
        )

        self.assertEqual(output["prediction"].shape, (2, 6, 7))
        self.assertEqual(output["latent_field"].shape, (2, 6, 7, 8))
        self.assertEqual(output["projection"].shape[0], 2)
        self.assertIn("uncertainty", output)
        self.assertIn("modality_weights", output)
        self.assertNotIn("expert_utilization", output)
        self.assertTrue(torch.isfinite(output["prediction"]).all())

    def test_neural_field_compiler_fuses_every_available_modality(self):
        torch.manual_seed(11)
        model = NeuralFieldCompiler(
            input_dims={"eeg": 3, "fmri": 2},
            output_dims={"eeg": 3},
            config=NeuralFieldCompilerConfig(latent_dim=8, pair_rank=3),
        )
        eeg = torch.randn(2, 6, 3)
        fmri = torch.randn(2, 6, 2)
        baseline = model.forward_task({"eeg": eeg, "fmri": fmri}, target_modality="eeg")
        perturbed = model.forward_task({"eeg": eeg, "fmri": fmri + 20.0}, target_modality="eeg")

        self.assertEqual(baseline["modality_weights"].shape, (2,))
        self.assertTrue(torch.allclose(baseline["modality_weights"].sum(), torch.tensor(1.0), atol=1e-6))
        self.assertFalse(torch.allclose(baseline["prediction"], perturbed["prediction"]))

    def test_neural_field_compiler_uses_coordinates_and_rejects_subject_ids(self):
        torch.manual_seed(21)
        model = NeuralFieldCompiler(
            input_dims={"eeg": 3},
            output_dims={"eeg": 3},
            config=NeuralFieldCompilerConfig(latent_dim=8, geometry_dim=2),
        )
        eeg = torch.randn(2, 6, 3)
        coordinates = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        baseline = model.forward_task(
            {"eeg": eeg},
            target_modality="eeg",
            geometry={"coordinates": coordinates},
        )
        perturbed = model.forward_task(
            {"eeg": eeg},
            target_modality="eeg",
            geometry={"coordinates": coordinates.flip(0)},
        )

        self.assertFalse(torch.allclose(baseline["prediction"], perturbed["prediction"]))
        with self.assertRaisesRegex(ValueError, "subject_ids are forbidden"):
            model.forward_task(
                {"eeg": eeg},
                target_modality="eeg",
                geometry={"coordinates": coordinates},
                subject_ids=torch.tensor([0, 1]),
            )

    def test_forward_uses_structural_prior_geometry_without_tensor_truthiness(self):
        model = self._geometry_model()
        spy = _SpyFieldUpdate()
        model.field_update = spy
        structural = torch.zeros(7, 7)
        anatomy = torch.ones(7, 7)

        model.forward_task(
            {"stimulus": torch.randn(2, 6, 4)},
            target_modality="fmri",
            geometry={"structural_prior": structural, "anatomy": anatomy},
        )

        self.assertIs(spy.anatomy, structural)

    def test_forward_uses_anatomy_geometry_fallback(self):
        model = self._geometry_model()
        spy = _SpyFieldUpdate()
        model.field_update = spy
        anatomy = torch.ones(7, 7)

        model.forward_task(
            {"stimulus": torch.randn(2, 6, 4)},
            target_modality="fmri",
            geometry={"anatomy": anatomy},
        )

        self.assertIs(spy.anatomy, anatomy)

    def test_forward_allows_missing_geometry(self):
        model = self._geometry_model()
        spy = _SpyFieldUpdate()
        model.field_update = spy

        output = model.forward_task({"stimulus": torch.randn(2, 6, 4)}, target_modality="fmri")

        self.assertIsNone(spy.anatomy)
        self.assertEqual(output["prediction"].shape, (2, 6, 7))

    @staticmethod
    def _geometry_model() -> NeuralFieldCompiler:
        return NeuralFieldCompiler(
            input_dims={"stimulus": 4},
            output_dims={"fmri": 7},
            config=NeuralFieldCompilerConfig(latent_dim=8, pair_rank=3, use_observation_operator=True),
        )


class _SpyFieldUpdate(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anatomy: torch.Tensor | None = None

    def forward(
        self,
        field: torch.Tensor,
        *,
        stimulus_state: torch.Tensor | None = None,
        anatomy: torch.Tensor | None = None,
        subject_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del stimulus_state, subject_state
        self.anatomy = anatomy
        return field


if __name__ == "__main__":
    unittest.main()
