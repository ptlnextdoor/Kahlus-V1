import unittest

import torch

from neurotwin.models.torch_models import NeuralStateSpaceTranslator, NeuralStateSpaceTranslatorConfig


class ModelMetadataGeometryTests(unittest.TestCase):
    def test_translator_accepts_metadata_and_geometry_features(self):
        model = NeuralStateSpaceTranslator(
            input_dims={"eeg": 4},
            output_dims={"eeg": 4},
            config=NeuralStateSpaceTranslatorConfig(latent_dim=16, metadata_dim=3, geometry_dim=2),
        )
        batch = {"eeg": torch.randn(2, 5, 4)}
        metadata = torch.randn(2, 5, 3)
        geometry = {"eeg": torch.randn(2, 5, 2)}

        output = model.forward_task(
            batch,
            target_modality="eeg",
            task="reconstruction",
            metadata=metadata,
            geometry=geometry,
        )

        self.assertEqual(output["prediction"].shape, (2, 5, 4))
        self.assertEqual(output["latent"].shape, (2, 5, 16))
