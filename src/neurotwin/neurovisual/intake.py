from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neurotwin.neurovisual.gates import evaluate_neurovisual_claim_gate
from neurotwin.neurovisual.ontology import NOT_DIAGNOSIS_NOTICE, NeurovisualEpisodeProfile, default_episode_profile


@dataclass(frozen=True)
class NeurovisualIntakeProfile:
    episode_phenotype_profile: NeurovisualEpisodeProfile
    missing_clinician_questions: tuple[str, ...]
    red_flag_checklist: tuple[str, ...]
    claim_gate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kahlus.nv1.intake_profile.v1",
            "scope": "NV-1 side branch: structured symptom mapping, not Kahlus v1/v2/v3 replacement",
            "episode_phenotype_profile": self.episode_phenotype_profile.to_dict(),
            "missing_clinician_questions": list(self.missing_clinician_questions),
            "red_flag_checklist": list(self.red_flag_checklist),
            "claim_gate": self.claim_gate,
            "not_diagnosis_notice": NOT_DIAGNOSIS_NOTICE,
        }


def build_episode_intake_profile(raw: dict[str, Any] | None = None) -> NeurovisualIntakeProfile:
    raw = dict(raw or {})
    allowed_fields = set(NeurovisualEpisodeProfile.__dataclass_fields__)
    profile_values = {key: value for key, value in raw.items() if key in allowed_fields}
    red_flags = tuple(_red_flags(raw))
    questions = tuple(_missing_questions(raw))
    profile = default_episode_profile(
        **profile_values,
        clinician_questions=questions,
        urgent_red_flags=red_flags,
        impaired_awareness_flag=bool(raw.get("awareness_retained") is False or raw.get("memory_retained") is False),
        should_seek_medical_evaluation=True,
    )
    payload = {
        "profile": profile.to_dict(),
        "missing_clinician_questions": questions,
        "red_flag_checklist": red_flags,
        "not_diagnosis_notice": NOT_DIAGNOSIS_NOTICE,
    }
    gate = evaluate_neurovisual_claim_gate(
        claim_scope="structured_intake_schema_ready",
        payloads=[payload],
    )
    return NeurovisualIntakeProfile(profile, questions, red_flags, gate)


def _missing_questions(raw: dict[str, Any]) -> list[str]:
    checks = (
        ("duration_seconds", "What was the exact episode duration in seconds?"),
        ("awareness_retained", "Was awareness fully retained throughout the episode?"),
        ("memory_retained", "Was memory retained for the full episode?"),
        ("headache", "Was headache present before, during, or after the episode?"),
        ("photophobia", "Was photophobia present?"),
        ("nausea", "Was nausea present?"),
        ("medication_context", "Were there recent medication or substance changes?"),
        ("sleep_context", "What was the sleep context before the episode?"),
    )
    return [question for key, question in checks if key not in raw or raw.get(key) in (None, "", "unknown")]


def _red_flags(raw: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if raw.get("prior_seizure_history") is True:
        flags.append("prior seizure history requires clinician-supervised interpretation")
    if raw.get("awareness_retained") is False or raw.get("no_loss_of_consciousness") is False:
        flags.append("possible impaired awareness or loss of consciousness")
    if raw.get("motor_symptoms") is True:
        flags.append("motor symptoms reported")
    if raw.get("speech_symptoms") is True:
        flags.append("speech symptoms reported")
    if raw.get("course_change_recent") is True:
        flags.append("recent change in episode course")
    if not flags:
        flags.append("no urgent red flag asserted in synthetic intake; clinician review boundary still applies")
    return flags
