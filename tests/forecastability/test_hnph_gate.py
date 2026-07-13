from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import math
from pathlib import Path
from statistics import NormalDist
import tempfile
import unittest

import yaml

from neurotwin.forecastability import (
    HnphFeasibilityEvidence,
    LabelReproducibilityFamilyResult,
    LabelReproducibilityReference,
    build_hnph_baseline_feasibility,
    run_hnph_baseline_feasibility,
)
from neurotwin.forecastability.hnph_baselines import HNPH_CLASSICAL_BASELINE_SCHEMA
from neurotwin.repro import hash_file


_ROOT = Path(__file__).resolve().parents[2]
_PROTOCOL = _ROOT / "configs" / "protocol" / "hnph_phase0_v0.3.yaml"
_PREREGISTRATION = _ROOT / "docs" / "research" / "hnph_b2_preregistration_addendum.md"
_ARTIFACT_NAMES = (
    "physical_registry",
    "source_manifest",
    "split_audit",
    "anchor_targets",
    "transform_lineage",
    "baseline_results",
    "preregistration_addendum",
    "power_analysis",
    "max_t_inference",
    "comparator_acceptance",
    "label_reliability_reference",
    "future_canary",
)


def _frozen_required_subjects(protocol: dict[str, object], sigma_bits: float) -> int:
    effect = protocol["effect_threshold"]
    assert isinstance(effect, dict)
    critical = NormalDist().inv_cdf(1 - effect["familywise_alpha"] / effect["familywise_hypothesis_count"])
    critical += NormalDist().inv_cdf(effect["target_power"])
    return math.ceil((critical * sigma_bits / effect["epsilon_bits_per_anchor"]) ** 2)


def _family(protocol: dict[str, object], *, lcb: float = 0.02, raters: int = 3) -> LabelReproducibilityFamilyResult:
    band_ids = tuple(row["id"] for row in protocol["lead_bands_minutes"])
    references = {
        band_id: LabelReproducibilityReference(
            subject_balanced_log_skill_bits=0.05,
            subject_bootstrap_lcb_95_bits=lcb,
            subject_count=12,
            independent_rater_count=raters,
            bootstrap_replicates=2000,
            interval_method="subject_cluster_bootstrap_max_t_one_sided",
        )
        for band_id in band_ids
    }
    return LabelReproducibilityFamilyResult(
        references_by_family_cell=references,
        family_cell_ids=band_ids,
        subject_count=12,
        independent_rater_count=raters,
        bootstrap_replicates=2000,
        max_t_critical_value=1.0,
        max_t_passed=True,
    )


@contextmanager
def _evidence(**changes: object):
    protocol = yaml.safe_load(_PROTOCOL.read_text(encoding="utf-8"))
    primary_band = protocol["primary_band_id"]
    controls = protocol["control_suite"]
    negative_controls = tuple(
        name
        for name in controls["required"]
        if name not in {controls["positive_control"], controls["nuisance_probe_control"]}
    )
    sigma_bits = 0.03
    required_subjects = _frozen_required_subjects(protocol, sigma_bits)
    values: dict[str, object] = {
        "protocol_path": _PROTOCOL,
        "preregistration_path": _PREREGISTRATION,
        "seed": 7,
        "bootstrap_replicates": protocol["primary_endpoint"]["inference"]["bootstrap_replicates"],
        "subject_cluster_max_t_passed": True,
        "independent_subject_clusters": protocol["feasibility_gate"]["minimum_independent_subject_clusters"],
        "event_subjects": protocol["feasibility_gate"]["minimum_event_subjects"],
        "positive_primary_band_anchors": protocol["feasibility_gate"]["minimum_positive_primary_band_anchors"],
        "simulated_power_by_band": {primary_band: protocol["effect_threshold"]["target_power"]},
        "power_sigma_bits_by_band": {primary_band: sigma_bits},
        "required_subjects_by_band": {primary_band: required_subjects},
        "available_event_subjects_by_band": {primary_band: required_subjects},
        "power_source": protocol["effect_threshold"]["sigma_source"],
        "finite": True,
        "firebreak_passed": True,
        "future_canary_passed": True,
        "complete": True,
        "baseline_ids": tuple(protocol["baseline_ladder"]["required"]),
        "selected_best_baseline": "semi_markov_competing_risk",
        "baseline_claim_mode_comparator_eligible": True,
        "baseline_claim_mode_max_t_inference_eligible": True,
        "baseline_claim_mode_primary_target_eligible": True,
        "negative_control_upper_95_log_skill_bits_by_control": {name: 0.0 for name in negative_controls},
        "real_minus_control_lower_95_log_skill_bits_by_control": {name: 0.001 for name in negative_controls},
        "synthetic_known_signal_lcb_bits_by_band": {primary_band: 0.02},
        "nuisance_probe_accuracy_above_chance": controls["nuisance_probe_maximum_accuracy_above_chance"],
        "comparator_acceptance_checks": {
            name: True for name in protocol["semi_markov_comparator_acceptance"]["required"]
        },
        "comparator_nuisance_challenger_lcb_bits_by_band": {primary_band: 0.019},
        "primary_ambiguity_handling": protocol["target"]["b2_primary_outcome"]["ambiguous_handling"],
        "complete_follow_up_excluded_count_by_band": {primary_band: 0},
        "label_reproducibility_family": _family(protocol),
        "label_rater_target_provenance_sha256": "b" * 64,
        "label_chief_comparator_prediction_sha256": "c" * 64,
        "classical_eeg_incremental_log_skill_lcb_by_band": {primary_band: 0.02},
    }
    values.update(changes)
    with tempfile.TemporaryDirectory() as tmp:
        artifact_paths = _write_bound_artifacts(Path(tmp), protocol, values)
        values["artifact_paths"] = artifact_paths
        values["artifact_hashes"] = {name: hash_file(path) for name, path in artifact_paths.items()}
        yield HnphFeasibilityEvidence(**values)  # type: ignore[arg-type]


def _write_bound_artifacts(
    directory: Path,
    protocol: dict[str, object],
    values: dict[str, object],
) -> dict[str, Path]:
    primary_band = protocol["primary_band_id"]
    family = values["label_reproducibility_family"]
    paths: dict[str, Path] = {}

    def write(name: str, payload: dict[str, object]) -> None:
        path = directory / f"{name}.json"
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        paths[name] = path

    for name in ("physical_registry", "source_manifest", "transform_lineage"):
        write(name, {"schema": f"kahlus.hnph.{name}.v1"})
    write(
        "split_audit",
        {
            "schema": "kahlus.hnph.split_audit.v1",
            "finite": values["finite"],
            "firebreak_passed": values["firebreak_passed"],
            "complete": values["complete"],
        },
    )
    write(
        "anchor_targets",
        {
            "schema": "kahlus.hnph.anchor_targets.v1",
            "outcome_alphabet": ["no_event", "Wake", "NREM", "REM", "Ambiguous"],
            "primary_target_kind": "leave_one_rater_out_soft_label",
            "primary_target_provenance_sha256": values["label_rater_target_provenance_sha256"],
            "primary_ambiguity_handling": values["primary_ambiguity_handling"],
            "complete_follow_up_excluded_count_by_band": values["complete_follow_up_excluded_count_by_band"],
            "positive_primary_band_anchors": values["positive_primary_band_anchors"],
        },
    )
    write(
        "power_analysis",
        {
            "schema": "kahlus.hnph.power_analysis.v1",
            "seed": values["seed"],
            "independent_subject_clusters": values["independent_subject_clusters"],
            "event_subjects": values["event_subjects"],
            "simulated_power_by_band": values["simulated_power_by_band"],
            "sigma_bits_by_band": values["power_sigma_bits_by_band"],
            "required_subjects_by_band": values["required_subjects_by_band"],
            "available_event_subjects_by_band": values["available_event_subjects_by_band"],
            "source": values["power_source"],
        },
    )
    write(
        "baseline_results",
        {
            "schema": HNPH_CLASSICAL_BASELINE_SCHEMA,
            "validation_nll_by_model": {name: 1.0 for name in values["baseline_ids"]},
            "selected_best_baseline": values["selected_best_baseline"],
            "claim_mode_comparator_eligible": values["baseline_claim_mode_comparator_eligible"],
            "claim_mode_max_t_inference_eligible": values["baseline_claim_mode_max_t_inference_eligible"],
            "claim_mode_primary_target_eligible": values["baseline_claim_mode_primary_target_eligible"],
            "primary_target_provenance_sha256": values["label_rater_target_provenance_sha256"],
            "chief_comparator_prediction_sha256": values["label_chief_comparator_prediction_sha256"],
        },
    )
    write(
        "max_t_inference",
        {
            "schema": "kahlus.hnph.subject_cluster_max_t.v1",
            "bootstrap_replicates": values["bootstrap_replicates"],
            "passed": values["subject_cluster_max_t_passed"],
            "negative_control_upper_95_log_skill_bits_by_control": values[
                "negative_control_upper_95_log_skill_bits_by_control"
            ],
            "real_minus_control_lower_95_log_skill_bits_by_control": values[
                "real_minus_control_lower_95_log_skill_bits_by_control"
            ],
            "synthetic_known_signal_lcb_bits_by_band": values["synthetic_known_signal_lcb_bits_by_band"],
            "nuisance_probe_accuracy_above_chance": values["nuisance_probe_accuracy_above_chance"],
            "classical_eeg_incremental_log_skill_lcb_by_band": values[
                "classical_eeg_incremental_log_skill_lcb_by_band"
            ],
        },
    )
    write(
        "comparator_acceptance",
        {
            "schema": "kahlus.hnph.comparator_acceptance.v1",
            "checks": values["comparator_acceptance_checks"],
            "nuisance_challenger_lcb_bits_by_band": values[
                "comparator_nuisance_challenger_lcb_bits_by_band"
            ],
        },
    )
    label_payload: dict[str, object] = {"schema": "kahlus.hnph.label_reproducibility_unavailable.v1"}
    if family is not None:
        assert isinstance(family, LabelReproducibilityFamilyResult)
        label_payload = {
            **family.to_dict(),
            "rater_target_provenance_sha256": values["label_rater_target_provenance_sha256"],
            "chief_comparator_prediction_sha256": values["label_chief_comparator_prediction_sha256"],
        }
    write("label_reliability_reference", label_payload)
    write(
        "future_canary",
        {"schema": "kahlus.hnph.future_canary.v1", "passed": values["future_canary_passed"]},
    )
    paths["preregistration_addendum"] = _PREREGISTRATION
    self_names = set(paths)
    assert self_names == set(_ARTIFACT_NAMES)
    assert primary_band in values["simulated_power_by_band"]
    return paths


class HnphBaselineGateTests(unittest.TestCase):
    def test_exact_epsilon_pass_writes_redacted_hashed_artifacts(self) -> None:
        with _evidence() as evidence, tempfile.TemporaryDirectory() as tmp:
            payload = run_hnph_baseline_feasibility(tmp, evidence)
            report = Path(tmp) / "HNPH_BASELINE_FEASIBILITY.md"
            json_artifact = Path(tmp) / "hnph_baseline_feasibility.json"
            manifest = Path(tmp) / "hnph_baseline_feasibility_manifest.json"

            self.assertTrue(payload["model_work_authorized"])
            self.assertEqual(payload["decision"]["outcome_class"], "dynamics_only_pass")
            self.assertEqual(payload["stop_reason"], "pass_authorize_h3")
            self.assertEqual(payload["effect_size"]["bits_per_anchor"], 0.02)
            self.assertEqual(payload["preregistration_hash"], hash_file(_PREREGISTRATION))
            self.assertIn("hnph_baseline_feasibility_json", json.loads(manifest.read_text())["artifact_hashes"])
            self.assertTrue(report.exists())
            self.assertTrue(json_artifact.exists())
            self.assertNotIn(str(_ROOT), report.read_text())
            self.assertNotIn(str(_ROOT), json_artifact.read_text())

    def test_below_epsilon_is_a_complete_bounded_stop(self) -> None:
        with _evidence(classical_eeg_incremental_log_skill_lcb_by_band={"B2": 0.019999}) as evidence:
            payload = build_hnph_baseline_feasibility(evidence)

        self.assertEqual(payload["stop_reason"], "residual_below_epsilon")
        self.assertEqual(payload["exit_code"], 0)

    def test_nuisance_challenger_at_epsilon_rejects_comparator(self) -> None:
        with _evidence(comparator_nuisance_challenger_lcb_bits_by_band={"B2": 0.02}) as evidence:
            payload = build_hnph_baseline_feasibility(evidence)

        self.assertEqual(payload["stop_reason"], "comparator_challenger_exceeds_epsilon")
        self.assertEqual(payload["exit_code"], 0)

    def test_placeholder_baseline_cannot_authorize_h3(self) -> None:
        with _evidence(baseline_claim_mode_comparator_eligible=False) as evidence:
            payload = build_hnph_baseline_feasibility(evidence)

        self.assertEqual(payload["stop_reason"], "baseline_fail")
        self.assertEqual(payload["exit_code"], 0)

    def test_missing_label_family_is_a_complete_bounded_stop(self) -> None:
        with _evidence(label_reproducibility_family=None) as evidence:
            payload = build_hnph_baseline_feasibility(evidence)

        self.assertEqual(payload["stop_reason"], "label_reliability_unavailable")
        self.assertEqual(payload["exit_code"], 0)

    def test_fewer_than_three_independent_raters_is_not_h3_eligible(self) -> None:
        protocol = yaml.safe_load(_PROTOCOL.read_text())
        with _evidence(label_reproducibility_family=_family(protocol, raters=2)) as evidence:
            payload = build_hnph_baseline_feasibility(evidence)

        self.assertEqual(payload["stop_reason"], "label_reliability_unavailable")

    def test_control_failure_is_a_complete_bounded_stop(self) -> None:
        protocol = yaml.safe_load(_PROTOCOL.read_text())
        controls = protocol["control_suite"]
        negative = next(name for name in controls["required"] if name not in {controls["positive_control"], controls["nuisance_probe_control"]})
        with _evidence(negative_control_upper_95_log_skill_bits_by_control={negative: 0.000001}) as evidence:
            payload = build_hnph_baseline_feasibility(evidence)

        self.assertEqual(payload["stop_reason"], "control_fail")
        self.assertEqual(payload["exit_code"], 0)

    def test_hash_mismatch_invalidates_instead_of_trusting_hash_shaped_input(self) -> None:
        with _evidence() as evidence:
            hashes = dict(evidence.artifact_hashes)
            hashes["baseline_results"] = "0" * 64
            payload = build_hnph_baseline_feasibility(replace(evidence, artifact_hashes=hashes))

        self.assertEqual(payload["decision"]["outcome_class"], "invalid_experiment")
        self.assertEqual(payload["stop_reason"], "integrity_fail")
        self.assertEqual(payload["exit_code"], 1)

    def test_caller_boolean_cannot_override_a_bound_baseline_artifact(self) -> None:
        with _evidence() as evidence:
            payload = build_hnph_baseline_feasibility(
                replace(evidence, baseline_claim_mode_comparator_eligible=False)
            )

        self.assertEqual(payload["decision"]["outcome_class"], "invalid_experiment")
        self.assertEqual(payload["stop_reason"], "integrity_fail")

    def test_noncanonical_same_id_protocol_cannot_control_gate(self) -> None:
        with _evidence() as evidence, tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "hnph_phase0_v0.3.yaml"
            copied.write_text(_PROTOCOL.read_text(), encoding="utf-8")
            payload = build_hnph_baseline_feasibility(replace(evidence, protocol_path=copied))

        self.assertEqual(payload["decision"]["outcome_class"], "invalid_experiment")
        self.assertEqual(payload["stop_reason"], "integrity_fail")


if __name__ == "__main__":
    unittest.main()
