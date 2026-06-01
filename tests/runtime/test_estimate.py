import unittest

from neurotwin.runtime.estimate import estimate_config


class EstimateConfigTests(unittest.TestCase):
    def test_default_config_estimate_is_directly_covered(self):
        estimate = estimate_config({})

        self.assertEqual(estimate["estimated_parameters"], 221184)
        self.assertEqual(estimate["estimated_activation_mb"], 1.0)
        self.assertEqual(estimate["estimated_optimizer_mb"], 1.688)
        self.assertEqual(estimate["estimated_checkpoint_mb"], 0.844)
        self.assertEqual(estimate["effective_batch_size"], 8)
        self.assertEqual(estimate["backbone"], "ssm_fallback")
        self.assertEqual(estimate["precision"], "fp32")

    def test_transformer_backbone_estimates_more_parameters_than_fallback(self):
        base = {
            "model": {
                "latent_dim": 32,
                "n_layers": 3,
                "input_dim": 4,
                "output_dim": 5,
                "modalities": ["eeg", "fmri"],
            }
        }

        fallback = estimate_config({**base, "model": {**base["model"], "backbone": "ssm_fallback"}})
        transformer = estimate_config({**base, "model": {**base["model"], "backbone": "transformer"}})

        self.assertGreater(transformer["estimated_parameters"], fallback["estimated_parameters"])
        self.assertEqual(transformer["backbone"], "transformer")
        self.assertEqual(fallback["backbone"], "ssm_fallback")

    def test_bf16_precision_halves_activation_and_checkpoint_estimates(self):
        base = {"model": {"latent_dim": 32, "n_layers": 2, "input_dim": 4}, "batch_size": 2, "window_size": 16}

        fp32 = estimate_config({**base, "precision": "fp32"})
        bf16 = estimate_config({**base, "precision": "bf16"})

        self.assertEqual(bf16["precision"], "bf16")
        self.assertAlmostEqual(float(bf16["estimated_activation_mb"]) * 2, float(fp32["estimated_activation_mb"]))
        self.assertAlmostEqual(float(bf16["estimated_checkpoint_mb"]) * 2, float(fp32["estimated_checkpoint_mb"]), places=2)
        self.assertEqual(bf16["estimated_optimizer_mb"], fp32["estimated_optimizer_mb"])

    def test_nested_training_fallbacks_and_effective_batch_size(self):
        estimate = estimate_config(
            {
                "training": {
                    "batch_size": 3,
                    "gradient_accumulation_steps": 4,
                    "precision": "bfloat16",
                    "compile": True,
                }
            }
        )

        self.assertEqual(estimate["effective_batch_size"], 12)
        self.assertEqual(estimate["gradient_accumulation_steps"], 4)
        self.assertEqual(estimate["precision"], "bfloat16")
        self.assertEqual(estimate["compile"], "True")

    def test_invalid_modalities_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "unsupported model modality"):
            estimate_config({"model": {"modalities": ["eeg", "bogus"]}})

        with self.assertRaisesRegex(ValueError, "non-empty list"):
            estimate_config({"model": {"modalities": "eeg"}})

        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            estimate_config({"model": {"modalities": ["eeg", ""]}})


if __name__ == "__main__":
    unittest.main()
