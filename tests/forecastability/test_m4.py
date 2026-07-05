from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurotwin.forecastability.m4 import (
    NUISANCE_PROBE_KEYS,
    _known_curve_passes,
    _nuisance_probe_failures,
    _sleep_curve_failures,
    patient_horizon_labels,
    run_m4_gate,
)


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
        first = gate["synthetic_known_signal"]["curve"][0]
        self.assertGreater(first["rfs_ci_low"], 0.0)
        self.assertEqual(set(first["nuisance_probes"]), set(NUISANCE_PROBE_KEYS))
        self.assertEqual(first["nuisance_probe_failures"], [])

    def test_m4_nuisance_probe_failures_are_claim_blockers(self) -> None:
        probes = {
            "patient": {"accuracy": 0.81, "chance": 0.50},
            "site": {"accuracy": 0.55, "chance": 0.50},
            "time_bucket": {"accuracy": 0.30, "chance": 0.25},
        }
        failures = _nuisance_probe_failures(probes, prefix="horizon_2")

        self.assertIn("horizon_2_nuisance_probe_patient_above_threshold", failures)
        self.assertIn("horizon_2_nuisance_probe_session_missing", failures)
        self.assertNotIn("horizon_2_nuisance_probe_site_above_threshold", failures)

    def test_m4_nuisance_probe_failures_fail_closed_for_malformed_payloads(self) -> None:
        missing_failures = _nuisance_probe_failures(None, prefix="horizon_3")
        self.assertEqual(
            set(missing_failures),
            {f"horizon_3_nuisance_probe_{key}_missing" for key in NUISANCE_PROBE_KEYS},
        )

        malformed_failures = _nuisance_probe_failures(
            {
                "patient": {"accuracy": "not-a-number", "chance": 0.50},
                "site": {"accuracy": float("nan"), "chance": 0.50},
                "time_bucket": {"accuracy": 0.30, "chance": 0.25},
                "session": {"accuracy": 0.40, "chance": 0.50},
            },
            prefix="horizon_3",
        )

        self.assertIn("horizon_3_nuisance_probe_patient_invalid", malformed_failures)
        self.assertIn("horizon_3_nuisance_probe_site_nonfinite", malformed_failures)

    def test_m4_known_curve_blocks_any_horizon_probe_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = run_m4_gate(Path(tmp), seed=5)

        known = gate["synthetic_known_signal"]
        self.assertTrue(_known_curve_passes(known))
        known["curve"][1]["nuisance_probe_failures"] = [
            "horizon_2_nuisance_probe_patient_above_threshold"
        ]
        self.assertFalse(_known_curve_passes(known))

    def test_sleep_curve_failures_include_any_horizon_probe_failure(self) -> None:
        payload = {
            "status": "completed_sleep_edf_smoke",
            "curve": [
                {
                    "horizon": 1,
                    "event_patients": 8,
                    "rfs_ci_low": 0.01,
                    "rfs_bits": 0.10,
                    "shuffled_rfs_bits": 0.01,
                    "time_shift_rfs_bits": 0.01,
                    "nuisance_probe_failures": [],
                },
                {
                    "horizon": 2,
                    "event_patients": 8,
                    "rfs_ci_low": 0.01,
                    "rfs_bits": 0.10,
                    "shuffled_rfs_bits": 0.01,
                    "time_shift_rfs_bits": 0.01,
                    "nuisance_probe_failures": [
                        "horizon_2_nuisance_probe_patient_above_threshold"
                    ],
                },
            ],
        }

        self.assertIn(
            "sleep_edf_horizon_2_nuisance_probe_patient_above_threshold",
            _sleep_curve_failures(payload),
        )

    def test_m4_report_mentions_nuisance_probe_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_m4_gate(Path(tmp), seed=5)
            report = Path(tmp, "M4_EVIDENCE_REPORT.md").read_text(encoding="utf-8")

        self.assertIn("Nuisance probes are reported for every M4 horizon", report)
        self.assertIn("| horizon | RFS bits | CI low | CI high |", report)
        self.assertIn("nuisance probes", report)


if __name__ == "__main__":
    unittest.main()
