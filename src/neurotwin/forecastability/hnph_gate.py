"""Fail-closed HNPH v0.3 B2 preregistration and baseline-feasibility gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from statistics import NormalDist
from typing import Any, Mapping

import yaml

from neurotwin.forecastability.contracts import EvidenceDecision
from neurotwin.forecastability.hnph_baselines import HNPH_CLASSICAL_BASELINE_SCHEMA
from neurotwin.forecastability.label_reliability import (
    LABEL_REPRODUCIBILITY_FAMILY_SCHEMA,
    LabelReproducibilityFamilyResult,
)
from neurotwin.repro import hash_file, write_json


HNPH_BASELINE_FEASIBILITY_SCHEMA = "kahlus.hnph.baseline_feasibility.v3"
_HNPH_PROTOCOL_ID = "kahlus.hnph.phase0.v0.3"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL_PROTOCOL_PATH = _REPOSITORY_ROOT / "configs" / "protocol" / "hnph_phase0_v0.3.yaml"
_CANONICAL_PREREGISTRATION_PATH = _REPOSITORY_ROOT / "docs" / "research" / "hnph_b2_preregistration_addendum.md"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FROZEN_PROTOCOL_SHA256 = re.compile(r"\*\*Frozen v0\.3 protocol SHA-256:\*\* `([0-9a-f]{64})`")
_REQUIRED_ARTIFACT_HASHES = frozenset(
    {
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
    }
)


@dataclass(frozen=True)
class HnphFeasibilityEvidence:
    """Hash-bound B2 inputs; raw data, rater labels, and local paths stay outside the packet."""

    protocol_path: str | Path
    preregistration_path: str | Path
    artifact_hashes: Mapping[str, str]
    artifact_paths: Mapping[str, str | Path]
    seed: int
    bootstrap_replicates: int
    subject_cluster_max_t_passed: bool
    independent_subject_clusters: int
    event_subjects: int
    positive_primary_band_anchors: int
    simulated_power_by_band: Mapping[str, float]
    power_sigma_bits_by_band: Mapping[str, float]
    required_subjects_by_band: Mapping[str, int]
    available_event_subjects_by_band: Mapping[str, int]
    power_source: str
    finite: bool
    firebreak_passed: bool
    future_canary_passed: bool
    complete: bool
    baseline_ids: tuple[str, ...]
    selected_best_baseline: str
    baseline_claim_mode_comparator_eligible: bool
    baseline_claim_mode_max_t_inference_eligible: bool
    baseline_claim_mode_primary_target_eligible: bool
    negative_control_upper_95_log_skill_bits_by_control: Mapping[str, float]
    real_minus_control_lower_95_log_skill_bits_by_control: Mapping[str, float]
    synthetic_known_signal_lcb_bits_by_band: Mapping[str, float]
    nuisance_probe_accuracy_above_chance: float
    comparator_acceptance_checks: Mapping[str, bool]
    comparator_nuisance_challenger_lcb_bits_by_band: Mapping[str, float]
    primary_ambiguity_handling: str
    complete_follow_up_excluded_count_by_band: Mapping[str, int]
    label_reproducibility_family: LabelReproducibilityFamilyResult | None
    label_rater_target_provenance_sha256: str | None
    label_chief_comparator_prediction_sha256: str | None
    classical_eeg_incremental_log_skill_lcb_by_band: Mapping[str, float]


def build_hnph_baseline_feasibility(evidence: HnphFeasibilityEvidence) -> dict[str, Any]:
    """Evaluate the frozen B2 protocol and return a publication-safe evidence packet."""

    protocol, protocol_error = _load_yaml_object(evidence.protocol_path, "HNPH protocol")
    preregistration, preregistration_error = _load_text_hash(evidence.preregistration_path, "preregistration addendum")
    protocol_sha256 = _hash_or_none(evidence.protocol_path)
    frozen_protocol_sha256, frozen_protocol_error = _load_frozen_protocol_hash(evidence.preregistration_path)
    thresholds = _thresholds(protocol)
    structural_failures = _structural_failures(
        evidence,
        protocol,
        protocol_error,
        preregistration,
        preregistration_error,
        protocol_sha256,
        frozen_protocol_sha256,
        frozen_protocol_error,
        thresholds,
    )
    claim_scope = "internal_b2_feasibility_only_no_external_or_clinical_claim"
    if structural_failures:
        stop_reason = "integrity_fail"
        decision = EvidenceDecision(
            protocol_version=str(protocol.get("protocol_version", "unknown")),
            gate_passed=False,
            outcome_class="invalid_experiment",
            failed_requirements=tuple(structural_failures),
            allowed_claims=(),
            blocked_claims=_blocked_claims(),
            claim_scope=claim_scope,
            stop_reason=stop_reason,
        )
    else:
        stop_reason, gate_failures = _gate_failures(evidence, thresholds)
        if gate_failures:
            decision = EvidenceDecision(
                protocol_version=str(protocol["protocol_version"]),
                gate_passed=False,
                outcome_class="calibrated_null",
                failed_requirements=tuple(gate_failures),
                allowed_claims=("bounded HNPH B2 preregistration result",),
                blocked_claims=_blocked_claims(),
                claim_scope=claim_scope,
                stop_reason=stop_reason,
            )
        else:
            stop_reason = "pass_authorize_h3"
            decision = EvidenceDecision(
                protocol_version=str(protocol["protocol_version"]),
                gate_passed=True,
                outcome_class="dynamics_only_pass",
                failed_requirements=(),
                allowed_claims=("internal B2 evidence authorizes a frozen small H3 implementation",),
                blocked_claims=_blocked_claims(),
                claim_scope=claim_scope,
                stop_reason=stop_reason,
            )
    primary_band = thresholds["primary_band_id"]
    payload = {
        "schema": HNPH_BASELINE_FEASIBILITY_SCHEMA,
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": protocol_sha256,
        "frozen_protocol_sha256": frozen_protocol_sha256,
        "preregistration_hash": preregistration,
        "artifact_hashes": _safe_hashes(evidence.artifact_hashes),
        "seed": _safe_int(evidence.seed),
        "bootstrap_replicates": _safe_int(evidence.bootstrap_replicates),
        "epsilon_bits_per_anchor": thresholds["epsilon_bits_per_anchor"],
        "effect_size": {
            "bits_per_anchor": thresholds["epsilon_bits_per_anchor"],
            "role": "preregistered_design_sensitivity_not_clinical_or_biological_effect",
        },
        "thresholds": thresholds,
        "frozen_power_inputs": {
            "source": evidence.power_source,
            "sigma_bits_by_band": _safe_float_mapping(evidence.power_sigma_bits_by_band),
            "required_subjects_by_band": _safe_int_mapping(evidence.required_subjects_by_band),
            "available_event_subjects_by_band": _safe_int_mapping(evidence.available_event_subjects_by_band),
        },
        "observed": {
            "independent_subject_clusters": _safe_int(evidence.independent_subject_clusters),
            "event_subjects": _safe_int(evidence.event_subjects),
            "positive_primary_band_anchors": _safe_int(evidence.positive_primary_band_anchors),
            "subject_cluster_max_t_passed": bool(evidence.subject_cluster_max_t_passed),
            "simulated_power_by_band": _safe_float_mapping(evidence.simulated_power_by_band),
            "classical_eeg_incremental_log_skill_lcb_by_band": _safe_float_mapping(evidence.classical_eeg_incremental_log_skill_lcb_by_band),
            "selected_best_baseline": evidence.selected_best_baseline,
            "baseline_ids": sorted(set(evidence.baseline_ids)),
            "baseline_claim_mode_comparator_eligible": bool(evidence.baseline_claim_mode_comparator_eligible),
            "baseline_claim_mode_max_t_inference_eligible": bool(evidence.baseline_claim_mode_max_t_inference_eligible),
            "baseline_claim_mode_primary_target_eligible": bool(evidence.baseline_claim_mode_primary_target_eligible),
            "negative_control_upper_95_log_skill_bits_by_control": _safe_float_mapping(
                evidence.negative_control_upper_95_log_skill_bits_by_control
            ),
            "real_minus_control_lower_95_log_skill_bits_by_control": _safe_float_mapping(
                evidence.real_minus_control_lower_95_log_skill_bits_by_control
            ),
            "synthetic_known_signal_lcb_bits_by_band": _safe_float_mapping(
                evidence.synthetic_known_signal_lcb_bits_by_band
            ),
            "nuisance_probe_accuracy_above_chance": (
                float(evidence.nuisance_probe_accuracy_above_chance)
                if _finite_number(evidence.nuisance_probe_accuracy_above_chance)
                else None
            ),
            "comparator_acceptance_checks": _safe_bool_mapping(evidence.comparator_acceptance_checks),
            "comparator_nuisance_challenger_lcb_bits_by_band": _safe_float_mapping(evidence.comparator_nuisance_challenger_lcb_bits_by_band),
            "primary_ambiguity_handling": evidence.primary_ambiguity_handling,
            "complete_follow_up_excluded_count_by_band": _safe_int_mapping(evidence.complete_follow_up_excluded_count_by_band),
            "label_rater_target_provenance_sha256": evidence.label_rater_target_provenance_sha256,
            "label_chief_comparator_prediction_sha256": evidence.label_chief_comparator_prediction_sha256,
            "label_reproducibility_family": (
                evidence.label_reproducibility_family.to_dict()
                if evidence.label_reproducibility_family is not None
                else None
            ),
            "primary_band_id": primary_band,
        },
        "decision": asdict(decision),
        "model_work_authorized": decision.gate_passed,
        "claim_scope": claim_scope,
        "stop_reason": stop_reason,
        "exit_code": 1 if decision.outcome_class == "invalid_experiment" else 0,
    }
    return payload


def run_hnph_baseline_feasibility(
    out_dir: str | Path,
    evidence: HnphFeasibilityEvidence,
) -> dict[str, Any]:
    """Write the required JSON, Markdown, and hash manifest for a B2 decision."""

    payload = build_hnph_baseline_feasibility(evidence)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = write_json(out / "hnph_baseline_feasibility.json", payload)
    markdown_path = out / "HNPH_BASELINE_FEASIBILITY.md"
    markdown_path.write_text(format_hnph_baseline_feasibility(payload), encoding="utf-8")
    manifest = {
        "schema": "kahlus.hnph.baseline_feasibility_manifest.v3",
        "protocol_sha256": payload["protocol_sha256"],
        "frozen_protocol_sha256": payload["frozen_protocol_sha256"],
        "preregistration_hash": payload["preregistration_hash"],
        "artifact_hashes": {
            "hnph_baseline_feasibility_json": hash_file(json_path),
            "hnph_baseline_feasibility_markdown": hash_file(markdown_path),
            **payload["artifact_hashes"],
        },
    }
    write_json(out / "hnph_baseline_feasibility_manifest.json", manifest)
    return payload


def format_hnph_baseline_feasibility(payload: Mapping[str, Any]) -> str:
    """Render the paired human-readable evidence artifact without source paths."""

    decision = payload["decision"]
    lines = [
        "# HNPH B2 Baseline Feasibility",
        "",
        f"- protocol: `{payload.get('protocol_id', 'unknown')}`",
        f"- protocol_sha256: `{payload.get('protocol_sha256', 'missing')}`",
        f"- frozen_protocol_sha256: `{payload.get('frozen_protocol_sha256', 'missing')}`",
        f"- preregistration_hash: `{payload.get('preregistration_hash', 'missing')}`",
        f"- epsilon_bits_per_anchor: `{payload.get('epsilon_bits_per_anchor', 'missing')}`",
        f"- outcome_class: `{decision['outcome_class']}`",
        f"- model_work_authorized: `{payload['model_work_authorized']}`",
        f"- claim_scope: `{payload['claim_scope']}`",
        f"- stop_reason: `{payload['stop_reason']}`",
        f"- exit_code: `{payload['exit_code']}`",
        "",
        "## Failed Requirements",
        "",
    ]
    failures = decision["failed_requirements"]
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- none")
    lines.extend(["", "## Claim Boundary", ""])
    if decision["allowed_claims"]:
        lines.extend(f"- allowed: {claim}" for claim in decision["allowed_claims"])
    else:
        lines.append("- allowed: none")
    lines.extend(f"- blocked: {claim}" for claim in decision["blocked_claims"])
    return "\n".join(lines) + "\n"


def _structural_failures(
    evidence: HnphFeasibilityEvidence,
    protocol: Mapping[str, Any],
    protocol_error: str | None,
    preregistration_hash: str | None,
    preregistration_error: str | None,
    protocol_sha256: str | None,
    frozen_protocol_sha256: str | None,
    frozen_protocol_error: str | None,
    thresholds: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    if protocol_error:
        failures.append(protocol_error)
    if protocol.get("protocol_id") != _HNPH_PROTOCOL_ID:
        failures.append("frozen HNPH v0.3 protocol is missing or mismatched")
    if not _is_canonical_protocol_path(evidence.protocol_path):
        failures.append("HNPH gate must use the repository's canonical v0.3 protocol path")
    if not _is_canonical_preregistration_path(evidence.preregistration_path):
        failures.append("HNPH gate must use the repository's canonical v0.3 preregistration addendum")
    if preregistration_error or preregistration_hash is None:
        failures.append(preregistration_error or "preregistration addendum hash is missing")
    if frozen_protocol_error or frozen_protocol_sha256 is None:
        failures.append(frozen_protocol_error or "preregistration addendum lacks the frozen protocol hash")
    elif protocol_sha256 != frozen_protocol_sha256:
        failures.append("canonical protocol hash differs from the preregistered frozen protocol hash")
    if not evidence.complete:
        failures.append("baseline-feasibility evidence is incomplete")
    if not evidence.finite:
        failures.append("baseline-feasibility payload contains non-finite values")
    if not evidence.firebreak_passed:
        failures.append("forecast firebreak audit did not pass")
    if not evidence.future_canary_passed:
        failures.append("future-canary audit did not pass")
    if not _is_int(evidence.seed):
        failures.append("seed must be an integer")
    if not _is_int(evidence.bootstrap_replicates) or evidence.bootstrap_replicates < thresholds["bootstrap_replicates"]:
        failures.append("subject-cluster bootstrap replicate count is below the frozen requirement")
    if not evidence.subject_cluster_max_t_passed:
        failures.append("subject-cluster max-t inference did not pass")
    if evidence.primary_ambiguity_handling != thresholds["primary_ambiguity_handling"]:
        failures.append("primary ambiguity handling differs from the frozen protocol")
    if (
        thresholds["positive_control_id"] not in thresholds["required_controls"]
        or thresholds["nuisance_probe_control_id"] not in thresholds["required_controls"]
    ):
        failures.append("frozen control-suite roles are incomplete")
    artifact_keys = set(evidence.artifact_hashes)
    missing_artifacts = sorted(_REQUIRED_ARTIFACT_HASHES - artifact_keys)
    if missing_artifacts:
        failures.append("missing required artifact hashes: " + ",".join(missing_artifacts))
    malformed = sorted(
        str(key)
        for key, value in evidence.artifact_hashes.items()
        if not _SHA256.fullmatch(str(value).lower()) or "path" in str(key).lower() or "uri" in str(key).lower()
    )
    if malformed:
        failures.append("artifact hashes must be SHA-256 values with non-path names: " + ",".join(malformed))
    if preregistration_hash is not None and evidence.artifact_hashes.get("preregistration_addendum") != preregistration_hash:
        failures.append("preregistration addendum hash does not bind the emitted packet")
    failures.extend(_artifact_hash_binding_failures(evidence))
    failures.extend(_artifact_content_failures(evidence))
    if evidence.power_source != thresholds["power_sigma_source"]:
        failures.append("power inputs were not frozen from the permitted source")
    for field, value in (
        ("label rater-target provenance hash", evidence.label_rater_target_provenance_sha256),
        ("label chief-comparator provenance hash", evidence.label_chief_comparator_prediction_sha256),
    ):
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            failures.append(f"{field} is missing or malformed")
    primary_band = thresholds["primary_band_id"]
    if not _finite_nonnegative(evidence.power_sigma_bits_by_band.get(primary_band)):
        failures.append("primary-band frozen power sigma is missing or invalid")
    if not _is_int(evidence.required_subjects_by_band.get(primary_band)) or not _is_int(evidence.available_event_subjects_by_band.get(primary_band)):
        failures.append("primary-band required and available subject counts are missing or invalid")
    else:
        required = evidence.required_subjects_by_band[primary_band]
        sigma = evidence.power_sigma_bits_by_band.get(primary_band)
        if _finite_nonnegative(sigma):
            minimum_required = _minimum_required_subjects(float(sigma), thresholds)
            if required < minimum_required:
                failures.append("primary-band required subject count is below the frozen power calculation")
    return failures


def _gate_failures(evidence: HnphFeasibilityEvidence, thresholds: Mapping[str, Any]) -> tuple[str, list[str]]:
    primary_band = thresholds["primary_band_id"]
    if evidence.independent_subject_clusters < thresholds["minimum_independent_subject_clusters"] or evidence.event_subjects < thresholds["minimum_event_subjects"] or evidence.positive_primary_band_anchors < thresholds["minimum_positive_primary_band_anchors"]:
        return "underpowered", ["subject, event-subject, or primary-anchor feasibility count is below threshold"]
    required_subjects = evidence.required_subjects_by_band[primary_band]
    available_subjects = evidence.available_event_subjects_by_band[primary_band]
    simulated_power = evidence.simulated_power_by_band.get(primary_band)
    if required_subjects > available_subjects or not _finite_number(simulated_power) or float(simulated_power) < thresholds["target_power"]:
        return "underpowered", ["primary-band power gate is not met before claim-mode evaluation"]
    baseline_ids = set(evidence.baseline_ids)
    missing_baselines = sorted(set(thresholds["required_baselines"]) - baseline_ids)
    if missing_baselines or evidence.selected_best_baseline not in baseline_ids:
        return "baseline_fail", ["frozen nuisance baseline ladder is incomplete or unselected"]
    if evidence.selected_best_baseline != "semi_markov_competing_risk":
        return "baseline_fail", ["chief semi-Markov comparator is not the selected nuisance baseline"]
    if (
        not evidence.baseline_claim_mode_comparator_eligible
        or not evidence.baseline_claim_mode_max_t_inference_eligible
        or not evidence.baseline_claim_mode_primary_target_eligible
    ):
        return "baseline_fail", ["baseline artifact is marked diagnostic-only rather than claim-mode eligible"]
    failed_comparator = sorted(name for name in thresholds["required_comparator_checks"] if not evidence.comparator_acceptance_checks.get(name, False))
    if failed_comparator:
        return "baseline_fail", ["chief comparator acceptance checks failed: " + ",".join(failed_comparator)]
    challenger_lcb = evidence.comparator_nuisance_challenger_lcb_bits_by_band.get(primary_band)
    if not _finite_number(challenger_lcb):
        return "baseline_fail", ["primary-band nuisance-challenger adequacy audit is missing"]
    if float(challenger_lcb) >= thresholds["epsilon_bits_per_anchor"]:
        return "comparator_challenger_exceeds_epsilon", ["a nuisance-only challenger improves the chief comparator by epsilon or more"]
    failed_controls = sorted(
        name
        for name in thresholds["negative_controls"]
        if not _finite_number(evidence.negative_control_upper_95_log_skill_bits_by_control.get(name))
        or float(evidence.negative_control_upper_95_log_skill_bits_by_control[name]) > 0
        or not _finite_number(evidence.real_minus_control_lower_95_log_skill_bits_by_control.get(name))
        or float(evidence.real_minus_control_lower_95_log_skill_bits_by_control[name]) <= 0
    )
    if failed_controls:
        return "control_fail", ["required control confidence bands did not pass: " + ",".join(failed_controls)]
    synthetic_lcb = evidence.synthetic_known_signal_lcb_bits_by_band.get(primary_band)
    if not _finite_number(synthetic_lcb) or float(synthetic_lcb) < thresholds["epsilon_bits_per_anchor"]:
        return "control_fail", ["synthetic known-signal lower bound is below epsilon"]
    if (
        not _finite_number(evidence.nuisance_probe_accuracy_above_chance)
        or float(evidence.nuisance_probe_accuracy_above_chance) > thresholds["nuisance_probe_maximum_accuracy_above_chance"]
    ):
        return "control_fail", ["nuisance probe exceeds the frozen chance margin"]
    family = evidence.label_reproducibility_family
    if family is None:
        return "label_reliability_unavailable", ["label-reproducibility family is unavailable"]
    if (
        family.schema != LABEL_REPRODUCIBILITY_FAMILY_SCHEMA
        or family.method != thresholds["label_family_method"]
        or tuple(family.outcome_alphabet) != thresholds["label_outcome_alphabet"]
        or family.probability_floor != thresholds["probability_floor"]
        or family.bootstrap_replicates < thresholds["bootstrap_replicates"]
        or not family.max_t_passed
    ):
        return "label_reliability_unavailable", ["label-reproducibility family lacks frozen five-way leave-one-rater-out max-t provenance"]
    if tuple(family.family_cell_ids) != thresholds["lead_band_ids"]:
        return "label_reliability_unavailable", ["label-reproducibility family does not cover every preregistered lead band"]
    if any(
        reference.bootstrap_replicates < thresholds["bootstrap_replicates"]
        or reference.independent_rater_count < thresholds["minimum_independent_raters"]
        or reference.subject_count < thresholds["minimum_independent_subject_clusters"]
        or reference.probability_floor != thresholds["probability_floor"]
        or reference.interval_method != "subject_cluster_bootstrap_max_t_one_sided"
        or not _finite_number(reference.subject_balanced_log_skill_bits)
        or not _finite_number(reference.subject_bootstrap_lcb_95_bits)
        for reference in family.references_by_family_cell.values()
    ):
        return "label_reliability_unavailable", ["one or more label-reproducibility band references are incomplete"]
    reference = family.references_by_family_cell.get(primary_band)
    if reference is None or not _finite_number(reference.subject_bootstrap_lcb_95_bits):
        return "label_reliability_unavailable", ["primary-band label-reproducibility reference is incomplete"]
    if (
        family.independent_rater_count < thresholds["minimum_independent_raters"]
        or family.subject_count < thresholds["minimum_independent_subject_clusters"]
    ):
        return "label_reliability_unavailable", ["label-reproducibility family lacks frozen rater or subject support"]
    if reference.subject_bootstrap_lcb_95_bits < thresholds["epsilon_bits_per_anchor"]:
        return "label_reliability_below_epsilon", ["label-reproducibility reference lower bound is below epsilon"]
    residual_lcb = evidence.classical_eeg_incremental_log_skill_lcb_by_band.get(primary_band)
    if not _finite_number(residual_lcb) or float(residual_lcb) < thresholds["epsilon_bits_per_anchor"]:
        return "residual_below_epsilon", ["classical EEG-plus-nuisance residual-skill lower bound is below epsilon"]
    return "pass_authorize_h3", []


def _thresholds(protocol: Mapping[str, Any]) -> dict[str, Any]:
    feasibility = _mapping(protocol.get("feasibility_gate"))
    inference = _mapping(_mapping(protocol.get("primary_endpoint")).get("inference"))
    ladder = _mapping(protocol.get("baseline_ladder"))
    controls = _mapping(protocol.get("control_suite"))
    effect = _mapping(protocol.get("effect_threshold"))
    comparator = _mapping(protocol.get("semi_markov_comparator_acceptance"))
    target = _mapping(_mapping(protocol.get("target")).get("b2_primary_outcome"))
    label_reliability = _mapping(protocol.get("label_construct_validity"))
    lead_bands = protocol.get("lead_bands_minutes")
    required_controls = tuple(str(value) for value in controls.get("required", ()))
    positive_control_id = str(controls.get("positive_control", ""))
    nuisance_probe_control_id = str(controls.get("nuisance_probe_control", ""))
    return {
        "minimum_independent_subject_clusters": int(feasibility.get("minimum_independent_subject_clusters", 10**9)),
        "minimum_event_subjects": int(feasibility.get("minimum_event_subjects", 10**9)),
        "minimum_positive_primary_band_anchors": int(feasibility.get("minimum_positive_primary_band_anchors", 10**9)),
        "endpoint_power_minimum": float(feasibility.get("endpoint_power_minimum", math.inf)),
        "bootstrap_replicates": int(inference.get("bootstrap_replicates", 10**9)),
        "required_baselines": tuple(str(value) for value in ladder.get("required", ())),
        "required_controls": tuple(str(value) for value in controls.get("required", ())),
    }


def _blocked_claims() -> tuple[str, ...]:
    return (
        "external HNPH result",
        "Passive-PCI evidence",
        "clinical, diagnostic, treatment, or seizure-warning claim",
        "neural architecture expansion without a passing B2 gate",
    )
