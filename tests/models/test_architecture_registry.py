import unittest

from neurotwin.models.architecture_registry import (
    architecture_registry,
    architecture_status,
    build_architecture_model,
    estimate_architecture_extra_parameters,
    normalize_architecture_type,
)
from neurotwin.models.nfc import NeuralFieldCompiler
from neurotwin.models.pair_operator import NeuroTwinPairOperator
from neurotwin.models.torch_models import NeuralStateSpaceTranslator
from neurotwin.config_types import resolve_prepared_config


class ArchitectureRegistryTests(unittest.TestCase):
    def test_registry_normalizes_supported_aliases(self):
        self.assertEqual(normalize_architecture_type("current_neurotwin"), "NeuralStateSpaceTranslator")
        self.assertEqual(normalize_architecture_type("pair_operator"), "NeuroTwinPairOperator")
        self.assertEqual(normalize_architecture_type("neurotwin_nfc"), "NeuralFieldCompiler")
        self.assertEqual(normalize_architecture_type("neural-field-compiler"), "NeuralFieldCompiler")

    def test_registry_factories_build_expected_models(self):
        base = {"input_dims": {"stimulus": 2}, "output_dims": {"fmri": 5}, "latent_dim": 8, "n_layers": 1}

        current = build_architecture_model({"type": "current_neurotwin", **base})
        pair = build_architecture_model({"type": "pair_operator", **base})
        nfc = build_architecture_model({"type": "nfc", **base})

        self.assertIsInstance(current, NeuralStateSpaceTranslator)
        self.assertIsInstance(pair, NeuroTwinPairOperator)
        self.assertIsInstance(nfc, NeuralFieldCompiler)

    def test_registry_status_and_estimate_hook_are_canonical(self):
        pair_config = resolve_prepared_config({"model": {"type": "pair_operator", "latent_dim": 8, "output_dim": 5, "pair_rank": 3}}).model
        nfc_config = resolve_prepared_config({"model": {"type": "neurotwin_nfc", "latent_dim": 8, "output_dim": 5, "pair_rank": 3}}).model

        self.assertEqual(architecture_status("pair_operator"), "local_baseline")
        self.assertEqual(architecture_status("nfc"), "experimental_architecture")
        self.assertGreater(estimate_architecture_extra_parameters(pair_config), 0)
        self.assertGreater(estimate_architecture_extra_parameters(nfc_config), estimate_architecture_extra_parameters(pair_config))

    def test_registry_entries_declare_supported_tasks(self):
        entries = architecture_registry()

        self.assertTrue(all("future_state_forecasting" in entry.supported_tasks for entry in entries))
        self.assertTrue(all(callable(entry.config_parser) for entry in entries))


if __name__ == "__main__":
    unittest.main()
