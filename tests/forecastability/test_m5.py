from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurotwin.forecastability.m5 import _run_fixture, estimate_pic_bits, make_pic_fixture, run_m5_gate


class ForecastabilityM5Tests(unittest.TestCase):
    def test_integrated_world_has_positive_pic(self) -> None:
        fixture = make_pic_fixture(seed=3, world="integrated_predictive")
        payload = estimate_pic_bits(fixture.windows, fixture.future, fixture.patient, nuisance=fixture.nuisance)

        self.assertGreater(payload["pic_bits"], 0.05)
        self.assertGreater(payload["pic_ci_low"], 0.0)

    def test_independent_and_white_noise_worlds_do_not_have_pic(self) -> None:
        independent = make_pic_fixture(seed=4, world="independent_predictable")
        white = make_pic_fixture(seed=5, world="white_noise")
        nuisance = make_pic_fixture(seed=6, world="nuisance_only")
        independent_payload = estimate_pic_bits(independent.windows, independent.future, independent.patient, nuisance=independent.nuisance)
        white_payload = estimate_pic_bits(white.windows, white.future, white.patient, nuisance=white.nuisance)
        nuisance_payload = estimate_pic_bits(nuisance.windows, nuisance.future, nuisance.patient, nuisance=nuisance.nuisance)

        self.assertLess(abs(independent_payload["pic_bits"]), 0.05)
        self.assertLess(abs(white_payload["pic_bits"]), 0.05)
        self.assertLess(abs(nuisance_payload["pic_bits"]), 0.05)
        self.assertLess(independent_payload["pic_ci_high"], 0.05)
        self.assertLess(white_payload["pic_ci_high"], 0.05)
        self.assertLess(nuisance_payload["pic_ci_high"], 0.05)

    def test_m5_synthetic_gate_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = run_m5_gate(Path(tmp), seed=6)
            out = Path(tmp)
            self.assertTrue((out / "m5_gate_report.json").exists())
            self.assertTrue((out / "M5_EVIDENCE_REPORT.md").exists())

            self.assertTrue(gate["synthetic_gate_passed"])
            self.assertTrue(gate["gate_passed"])
            self.assertEqual(gate["gate_failures"], [])
            self.assertEqual(gate["validation_scope"], "synthetic_instrument_validity_only")
            self.assertFalse(gate["external_generalization"])
            self.assertFalse(gate["public_data_used"])
            self.assertTrue(gate["nuisance_conditioned"])
            self.assertEqual(gate["claim_scope"], "passive_predictive_complexity_synthetic_method_only")
            self.assertIn("no_consciousness_claim", gate["blocked_claims"])

    def test_nuisance_only_world_fails_residual_rfs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = run_m5_gate(Path(tmp), seed=7)

        nuisance = gate["worlds"]["nuisance_only"]["pic_residual"]
        self.assertLess(nuisance["rfs_bits"], 0.02)
        self.assertLessEqual(nuisance["rfs_ci_low"], 0.0)

    def test_residual_rfs_does_not_use_future_derived_pic_rows(self) -> None:
        fixture = make_pic_fixture(seed=8, world="integrated_predictive")
        zero_future = replace(fixture, future=np.zeros_like(fixture.future))
        original = _run_fixture(fixture, seed=8)
        changed_future = _run_fixture(zero_future, seed=8)

        self.assertNotAlmostEqual(original["pic"]["pic_bits"], changed_future["pic"]["pic_bits"], places=2)
        self.assertAlmostEqual(original["pic_residual"]["rfs_bits"], changed_future["pic_residual"]["rfs_bits"], places=12)

    def test_m5_attribution_contract_is_finite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = run_m5_gate(Path(tmp), seed=6)

        for payload in gate["worlds"].values():
            attribution = payload["attribution"]
            self.assertIn("time_summary", attribution)
            self.assertIn("spectral_power", attribution)
            for section in attribution.values():
                self.assertTrue(np.isfinite(section["pic_bits"]))
                self.assertTrue(np.isfinite(section["pic_ci_low"]))
                self.assertTrue(np.isfinite(section["pic_ci_high"]))


if __name__ == "__main__":
    unittest.main()
