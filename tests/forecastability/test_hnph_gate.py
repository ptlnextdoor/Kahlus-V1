from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import yaml

from neurotwin.forecastability import (
    HnphFeasibilityEvidence,
    build_hnph_baseline_feasibility,
    run_hnph_baseline_feasibility,
)


_ROOT = Path(__file__).resolve().parents[2]
_PROTOCOL = _ROOT / "configs" / "protocol" / "hnph_phase0_v0.2.yaml"


def _evidence(**changes: object) -> HnphFeasibilityEvidence:
    protocol = yaml.safe_load(_PROTOCOL.read_text(encoding="utf-8"))
    values: dict[str, object] = {
        "protocol_path": _PROTOCOL,
        "artifact_hashes": {
            "physical_registry": "a" * 64,
            "source_manifest": "b" * 64,
            "split_audit": "c" * 64,
            "anchor_targets": "d" * 64,
            "transform_lineage": "e" * 64,
            "baseline_results": "f" * 64,
        },
        "seed": 7,
        "bootstrap_replicates": protocol["primary_endpoint"]["inference"]["bootstrap_replicates"],
        "subject_cluster_max_t_passed": True,
        "independent_subject_clusters": protocol["feasibility_gate"]["minimum_independent_subject_clusters"],
        "event_subjects": protocol["feasibility_gate"]["minimum_event_subjects"],
        "positive_primary_band_anchors": protocol["feasibility_gate"]["minimum_positive_primary_band_anchors"],
        "simulated_power": protocol["feasibility_gate"]["endpoint_power_minimum"],
        "finite": True,
        "firebreak_passed": True,
        "complete": True,
        "baseline_ids": tuple(protocol["baseline_ladder"]["required"]),
        "selected_best_baseline": "semi_markov_competing_risk",
        "controls_passed": {name: True for name in protocol["control_suite"]["required"]},
        "classical_eeg_incremental_log_skill_lcb": 0.01,
    }
    values.update(changes)
    return HnphFeasibilityEvidence(**values)  # type: ignore[arg-type]


class HnphBaselineGateTests(unittest.TestCase):
    def test_passing_feasibility_writes_hashed_json_and_markdown_without_local_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = run_hnph_baseline_feasibility(tmp, _evidence())
            report = Path(tmp) / "HNPH_BASELINE_FEASIBILITY.md"
            manifest = Path(tmp) / "hnph_baseline_feasibility_manifest.json"

            self.assertTrue(payload["model_work_authorized"])
            self.assertEqual(payload["decision"]["outcome_class"], "dynamics_only_pass")
            self.assertEqual(payload["exit_code"], 0)
            self.assertTrue(report.exists())
            self.assertTrue(manifest.exists())
            self.assertNotIn(str(_ROOT), report.read_text(encoding="utf-8"))
            self.assertIn("hnph_baseline_feasibility_json", json.loads(manifest.read_text(encoding="utf-8"))["artifact_hashes"])

    def test_nonpositive_classical_residual_skill_is_a_complete_bounded_stop(self) -> None:
        payload = build_hnph_baseline_feasibility(_evidence(classical_eeg_incremental_log_skill_lcb=0.0))

        self.assertFalse(payload["model_work_authorized"])
        self.assertEqual(payload["decision"]["outcome_class"], "calibrated_null")
        self.assertEqual(payload["exit_code"], 0)

    def test_missing_control_or_bootstrap_standard_fails_closed(self) -> None:
        controls = dict(_evidence().controls_passed)
        controls["label_shuffle"] = False
        control_failure = build_hnph_baseline_feasibility(_evidence(controls_passed=controls))
        bootstrap_failure = build_hnph_baseline_feasibility(_evidence(bootstrap_replicates=1999))

        self.assertEqual(control_failure["decision"]["outcome_class"], "invalid_experiment")
        self.assertEqual(control_failure["exit_code"], 1)
        self.assertEqual(bootstrap_failure["decision"]["outcome_class"], "invalid_experiment")
        self.assertEqual(bootstrap_failure["exit_code"], 1)

    def test_missing_max_t_inference_fails_closed(self) -> None:
        payload = build_hnph_baseline_feasibility(_evidence(subject_cluster_max_t_passed=False))

        self.assertEqual(payload["decision"]["outcome_class"], "invalid_experiment")


if __name__ == "__main__":
    unittest.main()
