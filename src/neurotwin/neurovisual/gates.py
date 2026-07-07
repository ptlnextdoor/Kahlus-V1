from __future__ import annotations

import json
from typing import Any, Iterable


NEUROVISUAL_ALLOWED_CLAIM_SCOPES: tuple[str, ...] = (
    "neurovisual_symptom_mapping_research_ready",
    "dataset_registry_ready",
    "structured_intake_schema_ready",
)

NEUROVISUAL_BLOCKED_CLAIMS: tuple[str, ...] = (
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


def evaluate_neurovisual_claim_gate(
    *,
    claim_scope: str,
    payloads: Iterable[Any] = (),
) -> dict[str, Any]:
    failure_reasons: list[str] = []
    if claim_scope not in NEUROVISUAL_ALLOWED_CLAIM_SCOPES:
        failure_reasons.append(f"unsupported_claim_scope:{claim_scope}")
    text = "\n".join(_payload_text(payload).lower() for payload in payloads)
    blocked_found = [claim for claim in NEUROVISUAL_BLOCKED_CLAIMS if claim.lower() in text]
    failure_reasons.extend(f"blocked_claim:{claim}" for claim in blocked_found)
    return {
        "schema": "kahlus.nv1.claim_gate.v1",
        "claim_scope": claim_scope,
        "passed": not failure_reasons,
        "scientific_claim_allowed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "blocked_claims_found": blocked_found,
        "blocked_claim_terms": list(NEUROVISUAL_BLOCKED_CLAIMS),
        "allowed_claim_scopes": list(NEUROVISUAL_ALLOWED_CLAIM_SCOPES),
    }


def _payload_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, sort_keys=True)
    except TypeError:
        return str(payload)
