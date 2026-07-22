from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurotwin.forecastability.interoception_scout import (
    make_interoception_fixture,
    run_interoception_rfs_gate,
)


class InteroceptionScoutTests(unittest.TestCase):
    def test_synthetic_known_and_null_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = run_interoception_rfs_gate(
                Path(tmp),
                seed=3,
                sleep_edf_root=None,
                bootstrap_mode="smoke",
            )
            known = gate["synthetic_known"]["horizons"][0]["residual_model"]
            null = gate["synthetic_null"]["horizons"][0]["residual_model"]
            self.assertGreater(known["rfs_bits"], 0.02)
            self.assertLess(abs(null["rfs_bits"]), 0.03)
            self.assertLessEqual(null["rfs_ci_high"], 0.05)
            shuffled = gate["synthetic_known"]["horizons"][0]["controls"]["label_shuffle"]
            self.assertLess(shuffled["rfs_bits"], known["rfs_bits"] * 0.4)

    def test_fixture_has_multiple_subjects(self) -> None:
        fixture = make_interoception_fixture(seed=1, residual_signal=True)
        self.assertGreaterEqual(len(set(fixture.subject.tolist())), 4)


if __name__ == "__main__":
    unittest.main()
