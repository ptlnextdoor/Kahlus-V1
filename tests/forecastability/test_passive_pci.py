from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurotwin.forecastability.complexity_features import (
    complexity_block,
    lempel_ziv_complexity,
    multiscale_entropy,
    permutation_entropy,
    spectral_slope,
    spectral_slope_block,
)
from neurotwin.forecastability.passive_pci import (
    make_passive_pci_fixture,
    run_passive_pci_gate,
)


class PassivePciTests(unittest.TestCase):
    def test_complexity_features_are_deterministic(self) -> None:
        signal = np.sin(np.linspace(0.0, 8.0 * np.pi, 64))
        first = (
            lempel_ziv_complexity(signal),
            permutation_entropy(signal),
            multiscale_entropy(signal),
            spectral_slope(signal),
        )
        second = (
            lempel_ziv_complexity(signal),
            permutation_entropy(signal),
            multiscale_entropy(signal),
            spectral_slope(signal),
        )
        self.assertEqual(first, second)
        self.assertGreaterEqual(permutation_entropy(signal), 0.0)
        self.assertLessEqual(permutation_entropy(signal), 1.0)

    def test_complexity_block_shape(self) -> None:
        windows = np.random.default_rng(0).normal(size=(5, 64, 2)).astype(np.float32)
        block = complexity_block(windows)
        self.assertEqual(block.shape, (5, 6))
        slopes = spectral_slope_block(windows)
        self.assertEqual(slopes.shape, (5, 1))

    def test_synthetic_known_and_null_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = run_passive_pci_gate(
                Path(tmp),
                seed=3,
                sleep_edf_root=None,
                bootstrap_mode="smoke",
            )
            known = gate["synthetic_known"]["states"][0]["residual_model"]
            null = gate["synthetic_null"]["states"][0]["residual_model"]
            self.assertGreater(known["rfs_bits"], 0.02)
            self.assertLess(abs(null["rfs_bits"]), 0.03)
            self.assertLessEqual(null["rfs_ci_high"], 0.05)
            shuffled = gate["synthetic_known"]["states"][0]["controls"]["label_shuffle"]
            self.assertLess(shuffled["rfs_bits"], known["rfs_bits"] * 0.4)

    def test_gate_schema_and_claim_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = run_passive_pci_gate(Path(tmp), seed=1, sleep_edf_root=None, bootstrap_mode="smoke")
            self.assertEqual(gate["schema"], "kahlus.forecastability.passive_pci.v1")
            self.assertIn("claim_scope", gate)
            self.assertIn("stop_reason", gate)
            self.assertIn("macrostates", gate)

    def test_fixture_has_multiple_subjects(self) -> None:
        fixture = make_passive_pci_fixture(seed=1, residual_complexity=True)
        self.assertGreaterEqual(len(set(fixture.subject.tolist())), 4)
        self.assertEqual(set(fixture.macrostate.tolist()), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
