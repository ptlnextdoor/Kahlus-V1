#!/usr/bin/env python3
"""Audit NV-1 prompt requirement coverage against generated local evidence.

This is a local metadata/intake coverage check. It does not download datasets,
execute adapters, run baselines or models, check raw EEG files, or launch
A100/cluster jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from _bootstrap import ensure_src_import_path

REPO_ROOT = ensure_src_import_path(__file__)

from neurotwin.neurovisual import REQUIRED_ONTOLOGY_FIELDS  # noqa: E402
from neurotwin.repro import write_json  # noqa: E402


EXPECTED_FILES = (
    "src/neurotwin/neurovisual/__init__.py",
    "src/neurotwin/neurovisual/ontology.py",
    "src/neurotwin/neurovisual/intake.py",
    "src/neurotwin/neurovisual/condition_map.py",
    "src/neurotwin/neurovisual/dataset_registry.py",
    "src/neurotwin/neurovisual/gates.py",
    "scripts/build_neurovisual_dataset_registry.py",
    "scripts/run_neurovisual_intake_smoke.py",
    "tests/neurovisual/test_neurovisual_ontology.py",
    "tests/neurovisual/test_neurovisual_intake.py",
    "tests/neurovisual/test_neurovisual_dataset_registry.py",
    "tests/neurovisual/test_neurovisual_claim_gates.py",
    "docs/research/kahlus_neurovisual_symptom_ontology.md",
    "docs/research/kahlus_neurovisual_dataset_registry.md",
)

REQUIRED_ONTOLOGY_PROMPT_FIELDS = (
    "onset_speed",
    "duration_seconds",
    "episode_frequency",
    "course_change_recent",
    "awareness_retained",
    "memory_retained",
    "impaired_awareness_flag",
    "visual_field_location",
    "color_distortion",
    "shape_or_object_distortion",
    "pattern_or_outline_effect",
    "motion_or_flicker",
    "expansion_or_spreading",
    "light_or_glare_sensitivity",
    "screen_or_sun_trigger",
    "moving_object_tracking_trigger",
    "derealization",
    "depersonalization",
    "body_detachment",
    "alarm_or_impending_doom",
    "neck_or_head_sensation",
    "headache",
    "photophobia",
    "nausea",
    "confusion_after",
    "fatigue_after",
    "motor_symptoms",
    "speech_symptoms",
    "prior_seizure_history",
    "migraine_history",
    "concussion_history",
    "medication_context",
    "caffeine_or_stimulant_context",
    "sleep_context",
    "hydration_context",
    "stress_context",
    "no_new_objects_seen",
    "no_minutes_long_progression",
    "no_loss_of_consciousness",
    "no_postictal_confusion_reported",
    "no_motor_event_reported",
    "urgent_red_flags",
    "clinician_questions",
    "should_seek_medical_evaluation",
    "not_diagnosis_notice",
)

REQUIRED_CONDITIONS = (
    "occipital_focal_aware_seizure_visual_aura",
    "reflex_photosensitive_seizure_activity",
    "migraine_aura",
    "visual_snow_syndrome",
    "concussion_related_visual_symptoms",
    "panic_derealization_episodes",
    "functional_psychogenic_episodes",
    "medication_substance_metabolic_contributors",
)

REQUIRED_BLOCKED_CLAIMS = (
    "predicts_seizure",
    "detects_epilepsy",
    "diagnoses_epilepsy",
    "diagnoses_migraine_aura",
    "diagnoses_visual_snow",
    "diagnoses_psychiatric_condition",
    "detects_hallucinations",
    "decodes_visual_experience",
    "safe_for_unsupervised_photic_testing",
    "triggers_or_recommends_photic_stimulation",
    "safe_for_users_with_epilepsy_history",
    "replaces_neurologist",
    "clinical_diagnostic_report",
    "provides_medical_advice",
    "treatment_recommendation",
    "medication_guidance",
)

EXPECTED_TOP_PRIORITIES = (
    "HBN-EEG / EEG Foundation Challenge",
    "CHB-MIT Scalp EEG Database",
    "TUSZ / TUH EEG Seizure Corpus",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    intake_dir = out_dir / "intake_smoke"
    registry_dir = out_dir / "registry_package"
    audit_path = out_dir / "neurovisual_requirement_coverage_audit.json"
    report_path = out_dir / "neurovisual_requirement_coverage_audit.md"

    intake_result = _run_local_script("scripts/run_neurovisual_intake_smoke.py", "--out-dir", str(intake_dir))
    registry_result = _run_local_script("scripts/build_neurovisual_dataset_registry.py", "--out", str(registry_dir))

    audit = _build_audit(
        out_dir=out_dir,
        intake_dir=intake_dir,
        registry_dir=registry_dir,
        intake_returncode=intake_result.returncode,
        registry_returncode=registry_result.returncode,
    )
    write_json(audit_path, audit)
    report_path.write_text(_format_report(audit), encoding="utf-8")

    print(f"branch=nv1 out_dir={out_dir.resolve()}")
    print(f"intake_smoke_dir={intake_dir}")
    print(f"registry_package_dir={registry_dir}")
    print(f"audit={audit_path}")
    print(f"report={report_path}")
    print(f"passed={audit['passed']}")
    print("bulk_dataset_download=false")
    print("a100_jobs_launched=false")
    return 0 if audit["passed"] else 1


def _run_local_script(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def _build_audit(
    *,
    out_dir: Path,
    intake_dir: Path,
    registry_dir: Path,
    intake_returncode: int,
    registry_returncode: int,
) -> dict[str, Any]:
    failures: list[str] = []
    intake = _read_json(intake_dir / "neurovisual_intake_smoke.json", failures)
    registry = _read_json(registry_dir / "neurovisual_dataset_registry.json", failures)
    summary = _read_json(registry_dir / "neurovisual_registry_verification_summary.json", failures)
    adapter_plan = _read_json(registry_dir / "neurovisual_adapter_plan.json", failures)
    query_plan = _read_json(registry_dir / "neurovisual_metadata_query_plan.json", failures)
    claim_gate = _read_json(registry_dir / "neurovisual_registry_claim_gate.json", failures)
    symptom_doc = _read_text(REPO_ROOT / "docs/research/kahlus_neurovisual_symptom_ontology.md", failures)
    registry_doc = _read_text(REPO_ROOT / "docs/research/kahlus_neurovisual_dataset_registry.md", failures)

    if intake_returncode != 0:
        failures.append(f"intake_smoke_returncode:{intake_returncode}")
    if registry_returncode != 0:
        failures.append(f"registry_build_returncode:{registry_returncode}")

    requirements = {
        "expected_files": _check_expected_files(failures),
        "ontology": _check_ontology(intake, failures),
        "intake_smoke": _check_intake_smoke(intake, failures),
        "condition_matrix": _check_condition_matrix(intake, failures),
        "dataset_registry": _check_dataset_registry(registry, summary, failures),
        "claim_gate": _check_claim_gate(intake, claim_gate, failures),
        "documentation": _check_documentation(symptom_doc, registry_doc, failures),
        "deferred_model_experiment_documented_only": _check_deferred_model_doc(
            symptom_doc,
            registry_doc,
            adapter_plan,
            failures,
        ),
        "metadata_query_plan": _check_metadata_query_plan(query_plan, failures),
    }

    execution = _aggregate_execution(intake, registry, summary, adapter_plan, query_plan)
    for flag in (
        "bulk_dataset_download",
        "a100_jobs_launched",
        "cluster_jobs_launched",
        "adapters_implemented",
        "baselines_run",
        "models_run",
    ):
        if execution.get(flag) is not False:
            failures.append(f"execution_flag_not_false:{flag}")

    openneuro_results = summary.get("openneuro_verification_results", []) if isinstance(summary, dict) else []
    if not isinstance(openneuro_results, list):
        openneuro_results = []
        failures.append("openneuro_verification_results_missing")

    score_3_assignments = summary.get("score_3_assignments", []) if isinstance(summary, dict) else []
    if not isinstance(score_3_assignments, list):
        score_3_assignments = []
        failures.append("score_3_assignments_not_list")

    top_priorities = [
        row.get("dataset_name")
        for row in summary.get("top_adapter_priorities", [])
        if isinstance(row, dict)
    ] if isinstance(summary, dict) else []

    return {
        "schema": "kahlus.nv1.requirement_coverage_audit.v1",
        "scope": "NV-1 prompt requirement coverage against generated local evidence",
        "passed": not failures,
        "failures": failures,
        "out_dir": str(out_dir),
        "generated_artifacts": {
            "intake_smoke": str(intake_dir / "neurovisual_intake_smoke.json"),
            "registry": str(registry_dir / "neurovisual_dataset_registry.json"),
            "registry_claim_gate": str(registry_dir / "neurovisual_registry_claim_gate.json"),
        },
        "requirements": requirements,
        "execution": execution,
        "top_adapter_priorities": top_priorities,
        "openneuro_verification_results": openneuro_results,
        "score_3_assignments": score_3_assignments,
        "datasets_rejected": summary.get("rejected_datasets", []) if isinstance(summary, dict) else [],
    }


def _check_expected_files(failures: list[str]) -> dict[str, Any]:
    missing = [path for path in EXPECTED_FILES if not (REPO_ROOT / path).exists()]
    for path in missing:
        failures.append(f"missing_expected_file:{path}")
    return {"status": "covered" if not missing else "missing", "missing": missing, "files": list(EXPECTED_FILES)}


def _check_ontology(intake: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    ontology_fields = set(REQUIRED_ONTOLOGY_FIELDS)
    missing_fields = sorted(set(REQUIRED_ONTOLOGY_PROMPT_FIELDS) - ontology_fields)
    profile = intake.get("episode_phenotype_profile", {}) if isinstance(intake, dict) else {}
    profile_missing = sorted(set(REQUIRED_ONTOLOGY_PROMPT_FIELDS) - set(profile)) if isinstance(profile, dict) else []
    structured_history = profile.get("structured_history_h_t", {}) if isinstance(profile, dict) else {}
    if missing_fields:
        failures.append(f"ontology_missing_fields:{','.join(missing_fields)}")
    if profile_missing:
        failures.append(f"profile_missing_fields:{','.join(profile_missing)}")
    if not isinstance(structured_history, dict) or structured_history.get("schema") != "kahlus.nv1.structured_history.v1":
        failures.append("structured_history_h_t_missing")
    return {
        "status": "covered" if not missing_fields and not profile_missing else "missing",
        "field_count": len(REQUIRED_ONTOLOGY_FIELDS),
        "missing_fields": missing_fields,
        "profile_missing_fields": profile_missing,
    }


def _check_intake_smoke(intake: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    required = (
        "episode_phenotype_profile",
        "missing_clinician_questions",
        "red_flag_checklist",
        "claim_gate",
        "execution",
    )
    missing = [key for key in required if key not in intake]
    questions = intake.get("missing_clinician_questions", [])
    red_flags = intake.get("red_flag_checklist", [])
    profile = intake.get("episode_phenotype_profile", {})
    notice = profile.get("not_diagnosis_notice", "") if isinstance(profile, dict) else ""
    if missing:
        failures.append(f"intake_missing_keys:{','.join(missing)}")
    if not questions:
        failures.append("intake_missing_clinician_questions_empty")
    if not isinstance(red_flags, list):
        failures.append("red_flag_checklist_not_list")
    if "not a diagnosis" not in str(notice).lower():
        failures.append("intake_not_diagnosis_notice_missing")
    return {"status": "covered" if not missing else "missing", "missing": missing}


def _check_condition_matrix(intake: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    matrix = intake.get("condition_matrix", []) if isinstance(intake, dict) else []
    condition_names = {row.get("condition") for row in matrix if isinstance(row, dict)}
    missing = sorted(set(REQUIRED_CONDITIONS) - condition_names)
    if missing:
        failures.append(f"condition_matrix_missing:{','.join(missing)}")
    for row in matrix if isinstance(matrix, list) else []:
        if isinstance(row, dict) and "not diagnostic" not in str(row.get("claim_boundary", "")).lower():
            failures.append(f"condition_claim_boundary_missing:{row.get('condition')}")
    return {"status": "covered" if not missing else "missing", "missing": missing, "condition_count": len(condition_names)}


def _check_dataset_registry(
    registry: dict[str, Any],
    summary: dict[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    entries = registry.get("entries", []) if isinstance(registry, dict) else []
    names = {row.get("dataset_name") for row in entries if isinstance(row, dict)}
    missing = sorted(set(EXPECTED_TOP_PRIORITIES) - names)
    if missing:
        failures.append(f"dataset_registry_missing_priority_entries:{','.join(missing)}")
    for row in entries if isinstance(entries, list) else []:
        if not isinstance(row, dict):
            failures.append("dataset_registry_non_object_entry")
            continue
        score = row.get("neurovisual_relevance_score")
        if score not in (0, 1, 2, 3):
            failures.append(f"bad_neurovisual_relevance_score:{row.get('dataset_name')}")
        if score == 3 and not (
            row.get("subjective_symptom_annotations_available")
            or "clinician-coded neurovisual phenotype" in str(row.get("notes", "")).lower()
        ):
            failures.append(f"score3_without_subjective_evidence:{row.get('dataset_name')}")
    top_priorities = [
        row.get("dataset_name")
        for row in summary.get("top_adapter_priorities", [])
        if isinstance(row, dict)
    ] if isinstance(summary, dict) else []
    if top_priorities[:3] != list(EXPECTED_TOP_PRIORITIES):
        failures.append("top_adapter_priorities_mismatch")
    return {
        "status": "covered" if not missing else "missing",
        "missing": missing,
        "entry_count": len(entries) if isinstance(entries, list) else 0,
    }


def _check_claim_gate(
    intake: dict[str, Any],
    registry_claim_gate: dict[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    intake_gate = intake.get("claim_gate", {}) if isinstance(intake, dict) else {}
    blocked_terms = set(registry_claim_gate.get("blocked_claim_terms", [])) if isinstance(registry_claim_gate, dict) else set()
    missing_blocked = sorted(set(REQUIRED_BLOCKED_CLAIMS) - blocked_terms)
    if missing_blocked:
        failures.append(f"claim_gate_missing_blocked_terms:{','.join(missing_blocked)}")
    if not registry_claim_gate.get("passed"):
        failures.append("registry_claim_gate_not_passed")
    if registry_claim_gate.get("blocked_claims_found"):
        failures.append("registry_claim_gate_blocked_claims_found")
    if not isinstance(intake_gate, dict) or not intake_gate.get("passed"):
        failures.append("intake_claim_gate_not_passed")
    return {
        "status": "covered" if not missing_blocked else "missing",
        "missing_blocked_terms": missing_blocked,
    }


def _check_documentation(symptom_doc: str, registry_doc: str, failures: list[str]) -> dict[str, Any]:
    checks = {
        "symptom_side_branch": "side branch" in symptom_doc.lower(),
        "symptom_not_diagnosis": "not a diagnosis" in symptom_doc.lower(),
        "registry_side_branch": "side branch" in registry_doc.lower(),
        "registry_no_download": "no bulk dataset download" in registry_doc.lower(),
        "registry_unverified_openneuro": "openneuro" in registry_doc.lower() and "unverified" in registry_doc.lower(),
    }
    for key, passed in checks.items():
        if not passed:
            failures.append(f"documentation_check_failed:{key}")
    return {"status": "covered" if all(checks.values()) else "missing", "checks": checks}


def _check_deferred_model_doc(
    symptom_doc: str,
    registry_doc: str,
    adapter_plan: dict[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    combined = f"{symptom_doc}\n{registry_doc}".lower()
    checks = {
        "event_window_forecasting_documented": "event-aligned eeg window forecasting" in combined,
        "prospective_prediction_blocked": "prospective seizure prediction" in combined,
        "adapters_not_implemented": not adapter_plan.get("execution", {}).get("adapters_implemented", True),
    }
    for key, passed in checks.items():
        if not passed:
            failures.append(f"deferred_model_doc_check_failed:{key}")
    return {"status": "covered" if all(checks.values()) else "missing", "checks": checks}


def _check_metadata_query_plan(query_plan: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    targets = query_plan.get("query_targets", []) if isinstance(query_plan, dict) else []
    target_ids = {row.get("target_id") for row in targets if isinstance(row, dict)}
    required = {
        "openneuro_nemar_multimodal_working_memory",
        "openneuro_nemar_bdi_reward_selection",
        "moabb_neurovisual_relevant_eeg",
        "eegdash_public_multimodal_eeg",
    }
    missing = sorted(required - target_ids)
    if missing:
        failures.append(f"metadata_query_plan_missing_targets:{','.join(missing)}")
    if query_plan.get("execution", {}).get("metadata_queries_executed") is not False:
        failures.append("metadata_queries_executed_not_false")
    return {"status": "covered" if not missing else "missing", "missing": missing}


def _aggregate_execution(*payloads: dict[str, Any]) -> dict[str, bool]:
    flags = {
        "bulk_dataset_download": False,
        "a100_jobs_launched": False,
        "cluster_jobs_launched": False,
        "metadata_queries_executed": False,
        "adapters_implemented": False,
        "baselines_run": False,
        "models_run": False,
        "private_patient_data_used": False,
    }
    for payload in payloads:
        execution = payload.get("execution", {}) if isinstance(payload, dict) else {}
        if not isinstance(execution, dict):
            continue
        for key in flags:
            flags[key] = bool(flags[key] or execution.get(key, False))
    return flags


def _read_json(path: Path, failures: list[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        failures.append(f"missing_json:{path.name}")
        return {}
    except json.JSONDecodeError:
        failures.append(f"invalid_json:{path.name}")
        return {}
    if not isinstance(payload, dict):
        failures.append(f"json_not_object:{path.name}")
        return {}
    return payload


def _read_text(path: Path, failures: list[str]) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        failures.append(f"missing_text:{path}")
        return ""


def _format_report(audit: dict[str, Any]) -> str:
    lines = [
        "# NV-1 Requirement Coverage Audit",
        "",
        f"- passed: {str(audit['passed']).lower()}",
        f"- failures: {len(audit['failures'])}",
        "",
        "## Requirements",
        "",
    ]
    requirements = audit["requirements"]
    assert isinstance(requirements, dict)
    for key, value in requirements.items():
        status = value.get("status") if isinstance(value, dict) else "unknown"
        lines.append(f"- {key}: {status}")
    execution = audit["execution"]
    assert isinstance(execution, dict)
    lines.extend(
        [
            "",
            "## Execution Boundary",
            "",
            f"- bulk_dataset_download: {str(execution['bulk_dataset_download']).lower()}",
            f"- a100_jobs_launched: {str(execution['a100_jobs_launched']).lower()}",
            f"- cluster_jobs_launched: {str(execution['cluster_jobs_launched']).lower()}",
            f"- adapters_implemented: {str(execution['adapters_implemented']).lower()}",
            f"- baselines_run: {str(execution['baselines_run']).lower()}",
            f"- models_run: {str(execution['models_run']).lower()}",
            "",
            "The deferred model experiment remains documentation-only in this audit.",
        ]
    )
    if audit["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in audit["failures"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
