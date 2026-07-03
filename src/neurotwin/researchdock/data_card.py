from __future__ import annotations

from typing import Any, Sequence

from neurotwin.researchdock.schemas import ResearchDockSession


def build_researchdock_data_card(sessions: Sequence[ResearchDockSession]) -> dict[str, Any]:
    profiles = sorted({str(session.metadata.get("profile", "unknown")) for session in sessions})
    quality_flags = sorted({flag for session in sessions for flag in session.quality_flags})
    return {
        "dataset_id": "researchdock_synthetic_v0",
        "branch": "researchdock",
        "n_sessions": len(sessions),
        "n_trials": sum(len(session.trials) for session in sessions),
        "profiles": profiles,
        "modalities": ["behavior", "pupil_gaze", "ppg_hrv_proxy", "self_report"],
        "contains_pii": False,
        "contains_real_participant_data": False,
        "contains_clinical_labels": False,
        "contains_stimulation": False,
        "quality_flags": quality_flags,
        "claim_boundary": "synthetic response-profile measurement scaffold only; not diagnosis or treatment",
    }
