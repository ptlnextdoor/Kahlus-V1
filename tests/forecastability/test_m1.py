from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurotwin.forecastability.m1 import discrete_survival_labels, make_transition_fixture, run_m1_gate


class ForecastabilityM1Tests(unittest.TestCase):
    def test_discrete_survival_labels_shift_future_events(self) -> None:
        labels = discrete_survival_labels([0, 1, 0, 1], bins=2)
        self.assertEqual(labels.tolist(), [[1, 0], [0, 1], [1, 0], [0, 0]])

    def test_known_and_null_synthetic_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = run_m1_gate(Path(tmp), seed=5)
            self.assertTrue(gate["gate_passed"])
            self.assertGreater(gate["known_signal"]["logistic_full"]["rfs_ci_low"], 0.0)
            self.assertLessEqual(gate["synthetic_null"]["logistic_full"]["rfs_ci_low"], 0.0)

    def test_fixture_has_event_patients(self) -> None:
        fixture = make_transition_fixture(seed=7, residual_signal=True)
        self.assertGreaterEqual(len(set(fixture.patient[fixture.y == 1])), 8)


if __name__ == "__main__":
    unittest.main()
