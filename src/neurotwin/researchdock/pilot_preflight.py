"""RD-4 pre-collection pilot-readiness artifacts for ResearchDock.

RD-4 is not data collection. It records the local safety and evidence checks
that must pass before any validation-scale ResearchDock pilot is considered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from neurotwin.researchdock.protocol import ResearchDockSessionProtocol
from neurotwin.researchdock.schemas import ResearchDockSession


REQUIRED_PRIOR_EVIDENCE = (
    "rd0_synthetic_fixture_gate",
    "rd1_session_export_contract",
    "rd2_observation_model_baselines",
    "rd3_public_dataset_review",
)

SAFE_MODALITIES = (
    "webcam_pupil_gaze_design_only",
    "reaction_time",
    "task_accuracy",
    "self_report_sliders",
    "optional_ppg_hrv_schema_only",
)

SAFETY_BOUNDARIES = (
    "no diagnosis claims",
    "no treatment claims",
    "no stimulation",
    "no trauma exposure tasks",
    "no unsupervised photic testing",
    "no raw identifiers",
    "no raw participant data",
)


def build_researchdock_pilot_manifest(
    *,
    sessions: Sequence[ResearchDockSession],
    protocol: ResearchDockSessionProtocol,
) -> dict[str, Any]:
    task_battery = tuple(block.task_name for block in protocol.blocks)
    return {
        "sprint": "RD-4",
        "collection_status": "pre_collection_preflight_only",
        "claimed_use": "response-profile measurement pilot-readiness scaffold",
        "required_prior_evidence": list(REQUIRED_PRIOR_EVIDENCE),
        "safe_modalities": list(SAFE_MODALITIES),
        "task_battery": list(task_battery),
        "planned_export_format": "researchdock_csv_v1",
        "synthetic_fixture_sessions": len(sessions),
        "synthetic_fixture_trials": sum(len(session.trials) for session in sessions),
        "contains_pii": False,
        "contains_real_participant_data": False,
        "contains_clinical_labels": False,
        "hardware_access_enabled": False,
        "safety_boundaries": list(SAFETY_BOUNDARIES),
        "operator_checklist": [
            "review protocol with appropriate institutional oversight before any real participant session",
            "verify local storage path is outside source control before collection",
            "verify participant identifiers are hashed or replaced before export",
            "run RD-4 preflight and archive its gate before collection",
        ],
    }


def run_researchdock_pilot_preflight(manifest: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if manifest.get("collection_status") != "pre_collection_preflight_only":
        failures.append("RD-4 must remain pre-collection only")
    if manifest.get("hardware_access_enabled"):
        failures.append("hardware access must be disabled for RD-4")
    if manifest.get("contains_real_participant_data"):
        failures.append("real participant data is not allowed in RD-4")
    if manifest.get("contains_pii"):
        failures.append("PII is not allowed in RD-4 artifacts")
    if manifest.get("contains_clinical_labels"):
        failures.append("clinical labels are not allowed in RD-4 artifacts")
    claim = str(manifest.get("claimed_use", "")).lower()
    if any(term in claim for term in ("diagnos", "treat", "ptsd", "depression", "anhedonia", "epilepsy")):
        failures.append("clinical or diagnostic claims are blocked")
    prior = set(str(item) for item in manifest.get("required_prior_evidence", ()))
    missing_prior = [item for item in REQUIRED_PRIOR_EVIDENCE if item not in prior]
    for item in missing_prior:
        failures.append(f"missing required prior evidence: {item}")
    boundaries = set(str(item) for item in manifest.get("safety_boundaries", ()))
    for item in SAFETY_BOUNDARIES:
        if item not in boundaries:
            failures.append(f"missing safety boundary: {item}")
    task_battery = set(str(item) for item in manifest.get("task_battery", ()))
    for task_name in ("reward_anticipation", "effort_for_reward", "mild_frustration", "recovery_rest"):
        if task_name not in task_battery:
            failures.append(f"missing required pilot task: {task_name}")
    return {
        "sprint": "RD-4",
        "passed": not failures,
        "failure_reasons": failures,
        "claim_boundary": "pilot_readiness_pre_collection_only_no_clinical_claims",
    }


def write_researchdock_pilot_preflight_artifacts(out_dir: str | Path, manifest: dict[str, Any]) -> dict[str, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    gate = run_researchdock_pilot_preflight(manifest)
    manifest_path = out_path / "researchdock_pilot_manifest.json"
    gate_path = out_path / "researchdock_pilot_preflight_gate.json"
    report_path = out_path / "researchdock_pilot_preflight_report.md"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_format_pilot_preflight_report(manifest, gate), encoding="utf-8")
    return {"manifest": str(manifest_path), "gate": str(gate_path), "report": str(report_path)}


def _format_pilot_preflight_report(manifest: dict[str, Any], gate: dict[str, Any]) -> str:
    lines = [
        "# ResearchDock RD-4 Pilot Preflight",
        "",
        "RD-4 is pre-collection only. It does not open hardware, collect real participant data,",
        "write raw participant data, or support clinical claims.",
        "",
        f"- collection_status: {manifest['collection_status']}",
        f"- preflight_passed: {gate['passed']}",
        f"- synthetic_fixture_sessions: {manifest['synthetic_fixture_sessions']}",
        f"- synthetic_fixture_trials: {manifest['synthetic_fixture_trials']}",
        "",
        "## Required Prior Evidence",
        "",
    ]
    lines.extend(f"- {item}" for item in manifest["required_prior_evidence"])
    lines.extend(["", "## Safety Boundaries", ""])
    lines.extend(f"- {item}" for item in manifest["safety_boundaries"])
    lines.extend(["", "## Operator Checklist", ""])
    lines.extend(f"- {item}" for item in manifest["operator_checklist"])
    if gate["failure_reasons"]:
        lines.extend(["", "## Gate Failures", ""])
        lines.extend(f"- {item}" for item in gate["failure_reasons"])
    return "\n".join(lines) + "\n"
