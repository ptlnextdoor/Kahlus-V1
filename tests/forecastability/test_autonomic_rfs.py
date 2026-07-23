from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurotwin.forecastability.amrith_acceptance import run_amrith_acceptance
from neurotwin.forecastability.autonomic_rfs import (
    make_autonomic_fixture,
    run_autonomic_rfs_gate,
)


class AmrithAcceptanceTests(unittest.TestCase):
    def test_overlap_trap_acceptance_passes(self) -> None:
        payload = run_amrith_acceptance(seed=0)
        self.assertTrue(payload["passed"])
        self.assertTrue(payload["checks"]["overlap_copy_fake_good"])


class AutonomicRfsTests(unittest.TestCase):
    def test_synthetic_known_and_null_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = run_autonomic_rfs_gate(
                Path(tmp),
                seed=4,
                mesa_root=None,
                shhs_root=None,
                bootstrap_mode="smoke",
            )
            known = gate["synthetic_known"]["horizons"][0]["residual_model"]
            null = gate["synthetic_null"]["horizons"][0]["residual_model"]
            self.assertGreater(known["rfs_bits"], 0.02)
            self.assertLess(abs(null["rfs_bits"]), 0.03)
            self.assertLessEqual(null["rfs_ci_high"], 0.05)

    def test_fixture_has_multiple_subjects(self) -> None:
        fixture = make_autonomic_fixture(seed=2, residual_signal=True)
        self.assertGreaterEqual(len(set(fixture.subject.tolist())), 8)


if __name__ == "__main__":
    unittest.main()
