from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurotwin.forecastability.m1 import TransitionFixture, make_transition_fixture
from neurotwin.forecastability.m4 import (
    NUISANCE_PROBE_KEYS,
    _cluster_permutation_rfs,
    _curve_payload,
    _filter_fixture,
    _known_curve_passes,
    _nuisance_probe_failures,
    _null_curve_passes,
    _sleep_curve_failures,
    patient_horizon_labels,
    patient_horizon_valid_masks,
    run_m4_gate,
)


class ForecastabilityM4Tests(unittest.TestCase):
    def test_patient_horizon_labels_do_not_cross_patients(self) -> None:
        labels = patient_horizon_labels([0, 1, 0, 1], [0, 0, 1, 1], horizons=(1, 2))
        masks = patient_horizon_valid_masks([0, 1, 0, 1], [0, 0, 1, 1], horizons=(1, 2))
        self.assertEqual(labels[1].tolist(), [0, 1, 0, 1])
        self.assertEqual(labels[2].tolist(), [1, 0, 1, 0])
        self.assertEqual(masks[1].tolist(), [True, True, True, True])
        self.assertEqual(masks[2].tolist(), [True, False, True, False])

    def test_filter_fixture_preserves_row_alignment(self) -> None:
        fixture = TransitionFixture(
            windows=np.arange(24, dtype=np.float32).reshape(4, 3, 2),
            nuisance=np.arange(12, dtype=np.float32).reshape(4, 3),
            y=np.asarray([0, 1, 0, 1], dtype=np.int64),
            patient=np.asarray([10, 10, 20, 20], dtype=np.int64),
            site=np.asarray([1, 1, 2, 2], dtype=np.int64),
            time_bucket=np.asarray([0, 1, 2, 3], dtype=np.int64),
            session=np.asarray([0, 0, 1, 1], dtype=np.int64),
        )

        filtered = _filter_fixture(fixture, np.asarray([True, False, True, False]))

        self.assertEqual(filtered.y.tolist(), [0, 0])
        self.assertEqual(filtered.patient.tolist(), [10, 20])
        self.assertEqual(filtered.site.tolist(), [1, 2])
        self.assertEqual(filtered.time_bucket.tolist(), [0, 2])
        self.assertEqual(filtered.session.tolist(), [0, 1])
        self.assertEqual(filtered.windows.shape, (2, 3, 2))
        self.assertEqual(filtered.nuisance.shape, (2, 3))

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
        rows = gate["synthetic_known_signal"]["curve"]
        self.assertEqual(rows[0]["invalid_terminal_rows"], 0)
        self.assertEqual(rows[1]["invalid_terminal_rows"], 12)
        self.assertEqual(rows[2]["invalid_terminal_rows"], 24)
        for row in rows:
            self.assertEqual(row["valid_rows"] + row["invalid_terminal_rows"], row["total_rows"])
            self.assertEqual(row["evaluated_rows"], row["valid_rows"])
            self.assertIn("cluster_permutation", row)
            self.assertEqual(row["cluster_permutation_failures"], [])
        self.assertEqual(gate["primary_horizon"], 1)

    def test_cluster_permutation_rfs_detects_consistent_cluster_gain(self) -> None:
        y = np.asarray([0, 1] * 8, dtype=np.int64)
        patient = np.repeat(np.arange(8), 2)
        baseline = np.full(len(y), 0.50, dtype=np.float64)
        pred = np.where(y == 1, 0.90, 0.10)

        out = _cluster_permutation_rfs(y, baseline, pred, patient, seed=11)

        self.assertEqual(out["permutation_unit"], "patient_cluster_sign_flip")
        self.assertEqual(out["n_clusters"], 8)
        self.assertEqual(out["mode"], "exact")
        self.assertEqual(out["n_permutations"], 256)
        self.assertGreater(out["observed_rfs_bits"], 0.0)
        self.assertLessEqual(out["p_value"], 0.05)

    def test_cluster_permutation_rfs_is_deterministic(self) -> None:
        y = np.asarray([0, 1] * 16, dtype=np.int64)
        patient = np.repeat(np.arange(16), 2)
        baseline = np.full(len(y), 0.50, dtype=np.float64)
        pred = np.where(y == 1, 0.75, 0.25)

        first = _cluster_permutation_rfs(y, baseline, pred, patient, seed=13, n_permutations=101)
        second = _cluster_permutation_rfs(y, baseline, pred, patient, seed=13, n_permutations=101)

        self.assertEqual(first, second)
        self.assertEqual(first["mode"], "sampled")
        self.assertEqual(first["n_permutations"], 101)

    def test_cluster_permutation_rfs_returns_non_significant_when_predictions_match_baseline(self) -> None:
        y = np.asarray([0, 1] * 8, dtype=np.int64)
        patient = np.repeat(np.arange(8), 2)
        baseline = np.full(len(y), 0.50, dtype=np.float64)

        out = _cluster_permutation_rfs(y, baseline, baseline, patient, seed=17)

        self.assertEqual(out["observed_rfs_bits"], 0.0)
        self.assertEqual(out["p_value"], 1.0)

    def test_m4_known_curve_blocks_primary_permutation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = run_m4_gate(Path(tmp), seed=5)

        known = gate["synthetic_known_signal"]
        self.assertTrue(_known_curve_passes(known, primary_horizon=gate["primary_horizon"]))
        known["curve"][0]["cluster_permutation"]["p_value"] = 1.0
        known["curve"][0]["cluster_permutation_failures"] = [
            "horizon_1_cluster_permutation_not_significant"
        ]
        self.assertFalse(_known_curve_passes(known, primary_horizon=gate["primary_horizon"]))

    def test_m4_null_curve_blocks_significant_permutation_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = run_m4_gate(Path(tmp), seed=5)

        null = gate["synthetic_null"]
        self.assertTrue(_null_curve_passes(null, primary_horizon=gate["primary_horizon"]))
        null["curve"][1]["cluster_permutation"]["p_value"] = 0.01
        null["curve"][1]["cluster_permutation"]["observed_rfs_bits"] = 0.10
        null["curve"][1]["cluster_permutation_failures"] = []
        self.assertTrue(_null_curve_passes(null, primary_horizon=gate["primary_horizon"]))
        null["curve"][0]["cluster_permutation"]["p_value"] = 0.01
        null["curve"][0]["cluster_permutation"]["observed_rfs_bits"] = 0.10
        self.assertFalse(_null_curve_passes(null, primary_horizon=gate["primary_horizon"]))

    def test_m4_null_curve_fails_closed_when_primary_permutation_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = run_m4_gate(Path(tmp), seed=5)

        null = gate["synthetic_null"]
        del null["curve"][0]["cluster_permutation"]
        self.assertFalse(_null_curve_passes(null, primary_horizon=gate["primary_horizon"]))

    def test_m4_rejects_horizon_without_valid_future_labels(self) -> None:
        fixture = make_transition_fixture(
            seed=5,
            residual_signal=True,
            n_patients=4,
            steps_per_patient=2,
        )

        with self.assertRaisesRegex(ValueError, "no valid within-patient future labels"):
            _curve_payload(fixture, horizons=(3,), seed=5)

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
                    "cluster_permutation": {"p_value": 0.01},
                    "cluster_permutation_failures": [],
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
                    "cluster_permutation": {"p_value": 0.01},
                    "cluster_permutation_failures": [],
                },
            ],
        }

        self.assertIn(
            "sleep_edf_horizon_2_nuisance_probe_patient_above_threshold",
            _sleep_curve_failures(payload),
        )

    def test_sleep_curve_failures_include_primary_permutation_failure(self) -> None:
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
                    "cluster_permutation": {"p_value": 0.50},
                    "cluster_permutation_failures": [
                        "horizon_1_cluster_permutation_not_significant"
                    ],
                }
            ],
        }

        self.assertIn(
            "sleep_edf_primary_cluster_permutation_not_significant",
            _sleep_curve_failures(payload),
        )

    def test_sleep_curve_failures_use_explicit_primary_horizon(self) -> None:
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
                    "cluster_permutation": {"p_value": 0.01, "observed_rfs_bits": 0.10},
                    "cluster_permutation_failures": [],
                },
                {
                    "horizon": 2,
                    "event_patients": 8,
                    "rfs_ci_low": 0.01,
                    "rfs_bits": 0.10,
                    "shuffled_rfs_bits": 0.01,
                    "time_shift_rfs_bits": 0.01,
                    "nuisance_probe_failures": [],
                    "cluster_permutation": {"p_value": 0.50, "observed_rfs_bits": 0.01},
                    "cluster_permutation_failures": [
                        "horizon_2_cluster_permutation_not_significant"
                    ],
                },
            ],
        }

        self.assertNotIn(
            "sleep_edf_primary_cluster_permutation_not_significant",
            _sleep_curve_failures(payload, primary_horizon=1),
        )
        self.assertIn(
            "sleep_edf_primary_cluster_permutation_not_significant",
            _sleep_curve_failures(payload, primary_horizon=2),
        )

    def test_m4_report_mentions_nuisance_probe_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_m4_gate(Path(tmp), seed=5)
            report = Path(tmp, "M4_EVIDENCE_REPORT.md").read_text(encoding="utf-8")

        self.assertIn("Nuisance probes are reported for every M4 horizon", report)
        self.assertIn("| horizon | total rows | valid rows | evaluated rows | invalid terminal |", report)
        self.assertIn("patient-cluster sign-flip permutation p-values", report)
        self.assertIn("cluster p", report)
        self.assertIn("nuisance probes", report)
        self.assertIn("terminal rows without a within-patient future label are excluded", report)

    def test_committed_m4_artifact_reports_current_row_accounting(self) -> None:
        root = Path(__file__).parents[2] / "artifacts" / "forecastability_trial0_m4"
        gate = json.loads((root / "m4_gate_report.json").read_text(encoding="utf-8"))
        report = (root / "M4_EVIDENCE_REPORT.md").read_text(encoding="utf-8")

        self.assertEqual(gate["sleep_edf_smoke"]["status"], "not_run_no_local_sleep_edf_root")
        self.assertIn("| horizon | total rows | valid rows | evaluated rows | invalid terminal |", report)
        self.assertIn("nuisance_probes", json.dumps(gate, sort_keys=True))
        self.assertIn("cluster_permutation", json.dumps(gate, sort_keys=True))
        self.assertEqual(gate["primary_horizon"], 1)
        for section in ("synthetic_known_signal", "synthetic_null"):
            for row in gate[section]["curve"]:
                self.assertEqual(row["valid_rows"] + row["invalid_terminal_rows"], row["total_rows"])
                self.assertEqual(row["evaluated_rows"], row["valid_rows"])
                self.assertEqual(
                    row["horizon_label_policy"],
                    "drop_terminal_rows_without_within_patient_future_label",
                )
                self.assertIn("p_value", row["cluster_permutation"])


if __name__ == "__main__":
    unittest.main()
