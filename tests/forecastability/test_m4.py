from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurotwin.forecastability.m4 import patient_horizon_labels, run_m4_gate


class ForecastabilityM4Tests(unittest.TestCase):
    def test_patient_horizon_labels_do_not_cross_patients(self) -> None:
        labels = patient_horizon_labels([0, 1, 0, 1], [0, 0, 1, 1], horizons=(1, 2))
        self.assertEqual(labels[1].tolist(), [0, 1, 0, 1])
        self.assertEqual(labels[2].tolist(), [1, 0, 1, 0])

    def test_m4_synthetic_curve_passes_without_sleep_edf_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = run_m4_gate(Path(tmp), seed=5)

        self.assertTrue(gate["synthetic_gate_passed"])
        self.assertFalse(gate["gate_passed"])
        self.assertEqual(gate["sleep_edf_smoke"]["status"], "not_run_no_local_sleep_edf_root")
        self.assertGreater(gate["synthetic_known_signal"]["curve"][0]["rfs_ci_low"], 0.0)


if __name__ == "__main__":
    unittest.main()
