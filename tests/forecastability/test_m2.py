from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurotwin.forecastability.m2 import fetch_sleep_edf_records_index, make_synthetic_sleep_fixture, run_m2_gate


class ForecastabilityM2Tests(unittest.TestCase):
    def test_synthetic_sleep_fixture_has_transition_events(self) -> None:
        fixture = make_synthetic_sleep_fixture(seed=2)
        self.assertGreater(int(fixture.transition.sum()), 80)
        self.assertGreaterEqual(len(set(fixture.subject[fixture.transition == 1])), 10)

    def test_m2_refuses_real_gate_without_sleep_edf_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = run_m2_gate(Path(tmp), seed=2)
            self.assertTrue(gate["synthetic_sleep_machinery_passed"])
            self.assertFalse(gate["gate_passed"])
            self.assertEqual(gate["real_sleep_edf"]["status"], "not_run_no_local_sleep_edf_root")

    def test_sleep_edf_records_index_source(self) -> None:
        records = fetch_sleep_edf_records_index()
        self.assertIn("sleep-cassette/SC4001E0-PSG.edf", records)


if __name__ == "__main__":
    unittest.main()
