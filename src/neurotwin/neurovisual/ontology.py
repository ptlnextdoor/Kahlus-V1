from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


NOT_DIAGNOSIS_NOTICE = (
    "NV-1 provides structured symptom mapping for clinician or researcher review; "
    "it is not a diagnosis, medical advice, seizure prediction, or a replacement for a neurologist."
)

REQUIRED_ONTOLOGY_FIELDS: tuple[str, ...] = (
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


@dataclass(frozen=True)
class NeurovisualEpisodeProfile:
    onset_speed: str = "unknown"
    duration_seconds: int | None = None
    episode_frequency: str = "unknown"
    course_change_recent: bool | None = None
    awareness_retained: bool | None = None
    memory_retained: bool | None = None
    impaired_awareness_flag: bool = False
    visual_field_location: str = "unknown"
    color_distortion: bool = False
    shape_or_object_distortion: bool = False
    pattern_or_outline_effect: bool = False
    motion_or_flicker: bool = False
    expansion_or_spreading: bool = False
    light_or_glare_sensitivity: bool = False
    screen_or_sun_trigger: bool = False
    moving_object_tracking_trigger: bool = False
    derealization: bool = False
    depersonalization: bool = False
    body_detachment: bool = False
    alarm_or_impending_doom: bool = False
    neck_or_head_sensation: bool = False
    headache: bool = False
    photophobia: bool = False
    nausea: bool = False
    confusion_after: bool = False
    fatigue_after: bool = False
    motor_symptoms: bool = False
    speech_symptoms: bool = False
    prior_seizure_history: bool = False
    migraine_history: bool = False
    concussion_history: bool = False
    medication_context: str = "unknown"
    caffeine_or_stimulant_context: str = "unknown"
    sleep_context: str = "unknown"
    hydration_context: str = "unknown"
    stress_context: str = "unknown"
    no_new_objects_seen: bool | None = None
    no_minutes_long_progression: bool | None = None
    no_loss_of_consciousness: bool | None = None
    no_postictal_confusion_reported: bool | None = None
    no_motor_event_reported: bool | None = None
    urgent_red_flags: tuple[str, ...] = field(default_factory=tuple)
    clinician_questions: tuple[str, ...] = field(default_factory=tuple)
    should_seek_medical_evaluation: bool = True
    not_diagnosis_notice: str = NOT_DIAGNOSIS_NOTICE
    source_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["structured_history_h_t"] = {
            "schema": "kahlus.nv1.structured_history.v1",
            "field_count": len(REQUIRED_ONTOLOGY_FIELDS),
            "tokenizable_fields": [
                field_name
                for field_name in REQUIRED_ONTOLOGY_FIELDS
                if field_name not in {"clinician_questions", "urgent_red_flags", "not_diagnosis_notice"}
            ],
        }
        return payload


def default_episode_profile(**overrides: Any) -> NeurovisualEpisodeProfile:
    values = {
        "clinician_questions": (
            "What was the exact episode duration?",
            "Was awareness fully retained throughout the episode?",
            "Were there headache, photophobia, nausea, motor, or speech symptoms?",
            "Were there medication, stimulant, sleep, hydration, or stress context changes?",
        ),
        "urgent_red_flags": (),
    }
    values.update(overrides)
    if values.get("duration_seconds") is not None and int(values["duration_seconds"]) < 0:
        raise ValueError("duration_seconds must be non-negative when provided")
    return NeurovisualEpisodeProfile(**values)


def episode_profile_to_dict(profile: NeurovisualEpisodeProfile) -> dict[str, Any]:
    return profile.to_dict()
