"""Fail-closed HNPH Phase-0 baseline-feasibility evidence gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from neurotwin.forecastability.contracts import EvidenceDecision
from neurotwin.repro import hash_file, write_json


HNPH_BASELINE_FEASIBILITY_SCHEMA = "kahlus.hnph.baseline_feasibility.v1"
_HNPH_PROTOCOL_ID = "kahlus.hnph.phase0.v0.2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_ARTIFACT_HASHES = frozenset(
    {
        "physical_registry",
        "source_manifest",
        "split_audit",
        "anchor_targets",
        "transform_lineage",
        "baseline_results",
    }
)


@dataclass(frozen=True)
class HnphFeasibilityEvidence:
    """Hash-bound B2 inputs; raw data and local paths are intentionally absent."""

    protocol_path: str | Path
    artifact_hashes: Mapping[str, str]
    seed: int
    bootstrap_replicates: int
    subject_cluster_max_t_passed: bool
    independent_subject_clusters: int
    event_subjects: int
    positive_primary_band_anchors: int
    simulated_power: float
    finite: bool
    firebreak_passed: bool
    complete: bool
    baseline_ids: tuple[str, ...]
    selected_best_baseline: str
    controls_passed: Mapping[str, bool]
    classical_eeg_incremental_log_skill_lcb: float


def build_hnph_baseline_feasibility(evidence: HnphFeasibilityEvidence) -> dict[str, Any]:
    """Evaluate the frozen B2 gate and return a publication-safe evidence payload."""

    protocol, protocol_error = _load_protocol(evidence.protocol_path)
    protocol_sha256 = _hash_or_none(evidence.protocol_path)
    invalid = _invalid_requirements(evidence, protocol, protocol_error)
    thresholds = _thresholds(protocol)
    if invalid:
        decision = EvidenceDecision(
            protocol_version=str(protocol.get("protocol_version", "unknown")),
            gate_passed=False,
            outcome_class="invalid_experiment",
            failed_requirements=tuple(invalid),
            allowed_claims=(),
            blocked_claims=_blocked_claims(),
        )
        stop_reason = "invalid_or_incomplete_hnph_baseline_evidence"
    else:
        null_failures = _null_requirements(evidence, thresholds)
        if null_failures:
            decision = EvidenceDecision(
                protocol_version=str(protocol["protocol_version"]),
                gate_passed=False,
                outcome_class="calibrated_null",
                failed_requirements=tuple(null_failures),
                allowed_claims=("bounded HNPH Phase-0 baseline-feasibility result",),
                blocked_claims=_blocked_claims(),
            )
            stop_reason = "bounded_or_calibrated_null_stops_h3"
        else:
            decision = EvidenceDecision(
                protocol_version=str(protocol["protocol_version"]),
                gate_passed=True,
                outcome_class="dynamics_only_pass",
                failed_requirements=(),
                allowed_claims=("internal HNPH classical-baseline feasibility supports a small H3 model test",),
                blocked_claims=_blocked_claims(),
            )
            stop_reason = "h3_authorized_after_classical_eeg_residual_skill"
    payload = {
        "schema": HNPH_BASELINE_FEASIBILITY_SCHEMA,
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": protocol_sha256,
        "artifact_hashes": dict(sorted((str(key), str(value)) for key, value in evidence.artifact_hashes.items())),
        "seed": int(evidence.seed),
        "bootstrap_replicates": int(evidence.bootstrap_replicates),
        "thresholds": thresholds,
        "observed": {
            "independent_subject_clusters": int(evidence.independent_subject_clusters),
            "subject_cluster_max_t_passed": bool(evidence.subject_cluster_max_t_passed),
            "event_subjects": int(evidence.event_subjects),
            "positive_primary_band_anchors": int(evidence.positive_primary_band_anchors),
            "simulated_power": float(evidence.simulated_power),
            "classical_eeg_incremental_log_skill_lcb": float(evidence.classical_eeg_incremental_log_skill_lcb),
            "selected_best_baseline": evidence.selected_best_baseline,
            "baseline_ids": sorted(set(evidence.baseline_ids)),
            "controls_passed": dict(sorted((str(key), bool(value)) for key, value in evidence.controls_passed.items())),
        },
        "decision": asdict(decision),
        "model_work_authorized": decision.gate_passed,
        "claim_scope": "internal_baseline_feasibility_only_not_external_or_clinical_evidence",
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
        "schema": "kahlus.hnph.baseline_feasibility_manifest.v1",
        "protocol_sha256": payload["protocol_sha256"],
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
        "# HNPH Phase-0 Baseline Feasibility",
        "",
        f"- protocol: `{payload.get('protocol_id', 'unknown')}`",
        f"- protocol_sha256: `{payload.get('protocol_sha256', 'missing')}`",
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
    lines.extend(f"- {failure}" for failure in failures) if failures else lines.append("- none")
    lines.extend(["", "## Claim Boundary", ""])
    lines.extend(f"- allowed: {claim}" for claim in decision["allowed_claims"]) if decision["allowed_claims"] else lines.append("- allowed: none")
    lines.extend(f"- blocked: {claim}" for claim in decision["blocked_claims"])
    return "\n".join(lines) + "\n"


def _load_protocol(path: str | Path) -> tuple[dict[str, Any], str | None]:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return {}, f"HNPH protocol could not be read: {exc}"
    if not isinstance(payload, dict):
        return {}, "HNPH protocol must be a YAML object"
    return payload, None


def _hash_or_none(path: str | Path) -> str | None:
    try:
        return hash_file(path)
    except OSError:
        return None


def _invalid_requirements(
    evidence: HnphFeasibilityEvidence,
    protocol: Mapping[str, Any],
    protocol_error: str | None,
) -> list[str]:
    failures: list[str] = []
    if protocol_error:
        failures.append(protocol_error)
    if protocol.get("protocol_id") != _HNPH_PROTOCOL_ID:
        failures.append("frozen HNPH v0.2 protocol is missing or mismatched")
    if not evidence.complete:
        failures.append("baseline-feasibility evidence is incomplete")
    if not evidence.finite:
        failures.append("baseline-feasibility payload contains non-finite values")
    if not evidence.firebreak_passed:
        failures.append("forecast firebreak audit did not pass")
    if isinstance(evidence.seed, bool) or not isinstance(evidence.seed, int):
        failures.append("seed must be an integer")
    thresholds = _thresholds(protocol)
    if evidence.bootstrap_replicates < thresholds["bootstrap_replicates"]:
        failures.append("subject-cluster bootstrap replicate count is below the frozen requirement")
    if not evidence.subject_cluster_max_t_passed:
        failures.append("subject-cluster max-t inference did not pass")
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
    required_baselines = set(thresholds["required_baselines"])
    baseline_ids = set(evidence.baseline_ids)
    missing_baselines = sorted(required_baselines - baseline_ids)
    if missing_baselines:
        failures.append("frozen baseline ladder is incomplete: " + ",".join(missing_baselines))
    if evidence.selected_best_baseline not in baseline_ids:
        failures.append("selected best baseline is not present in the evaluated baseline set")
    if evidence.selected_best_baseline == "fixed_standard_eeg_features_plus_nuisance":
        failures.append("EEG-plus-nuisance model cannot define the best nuisance baseline")
    required_controls = set(thresholds["required_controls"])
    failed_controls = sorted(name for name in required_controls if not evidence.controls_passed.get(name, False))
    if failed_controls:
        failures.append("required controls did not pass: " + ",".join(failed_controls))
    return failures


def _null_requirements(evidence: HnphFeasibilityEvidence, thresholds: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for observed_name, threshold_name in (
        ("independent_subject_clusters", "minimum_independent_subject_clusters"),
        ("event_subjects", "minimum_event_subjects"),
        ("positive_primary_band_anchors", "minimum_positive_primary_band_anchors"),
    ):
        if int(getattr(evidence, observed_name)) < int(thresholds[threshold_name]):
            failures.append(f"{observed_name} is below the frozen feasibility threshold")
    if not math.isfinite(float(evidence.simulated_power)) or evidence.simulated_power < float(thresholds["endpoint_power_minimum"]):
        failures.append("simulated primary-endpoint power is below the frozen feasibility threshold")
    skill = float(evidence.classical_eeg_incremental_log_skill_lcb)
    if not math.isfinite(skill) or skill <= 0:
        failures.append("classical EEG-plus-nuisance residual-skill lower bound is not positive")
    return failures


def _thresholds(protocol: Mapping[str, Any]) -> dict[str, Any]:
    feasibility = protocol.get("feasibility_gate") if isinstance(protocol.get("feasibility_gate"), Mapping) else {}
    inference = protocol.get("primary_endpoint", {}).get("inference", {}) if isinstance(protocol.get("primary_endpoint"), Mapping) else {}
    ladder = protocol.get("baseline_ladder") if isinstance(protocol.get("baseline_ladder"), Mapping) else {}
    controls = protocol.get("control_suite") if isinstance(protocol.get("control_suite"), Mapping) else {}
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
