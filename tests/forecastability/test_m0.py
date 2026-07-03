from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurotwin.forecastability import run_m0_gate


class ForecastabilityM0Tests(unittest.TestCase):
    def test_m0_baseline_table_is_bit_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = run_m0_gate(
                Path(tmp),
                seed=3,
                train_steps=1,
                enforce_clean_worktree=False,
                n_subjects=6,
                n_time=32,
                n_channels=3,
            )
            self.assertTrue(gate["bit_stable_baseline_table"])
            self.assertFalse(gate["missing_rows"])
            self.assertTrue((Path(tmp) / "M0_EVIDENCE_REPORT.md").exists())


if __name__ == "__main__":
    unittest.main()
