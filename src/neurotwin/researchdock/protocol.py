from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from neurotwin.researchdock.tasks import researchdock_task_templates


@dataclass(frozen=True)
class ResearchDockProtocolBlock:
    task_name: str
    n_trials: int
    target_duration_s: float
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...] = ()
    quality_checks: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResearchDockSessionProtocol:
    protocol_id: str
    seed: int
    blocks: tuple[ResearchDockProtocolBlock, ...]
    hardware_required: bool
    clinical_claim_allowed: bool
    export_format: str
    notes: tuple[str, ...] = ()


def build_rd1_session_protocol(*, seed: int = 0) -> ResearchDockSessionProtocol:
    """Build the deterministic RD-1 local task/session protocol."""

    templates = researchdock_task_templates()
    trial_counts = {
        "reward_anticipation": 12,
        "probabilistic_reward_learning": 16,
        "effort_for_reward": 12,
        "mild_frustration": 8,
        "recovery_rest": 6,
        "visual_attention": 12,
    }
    blocks = tuple(
        ResearchDockProtocolBlock(
            task_name=template.task_name,
            n_trials=trial_counts[template.task_name],
            target_duration_s=float(trial_counts[template.task_name] * 4),
            required_inputs=("reaction_time_ms", "accuracy", "task_event"),
            optional_inputs=tuple(signal for signal in template.primary_signals if signal not in {"reaction_time_ms", "accuracy"}),
            quality_checks=("missing_pupil", "implausible_reaction_time", "invalid_accuracy"),
        )
        for template in templates
    )
    return ResearchDockSessionProtocol(
        protocol_id="researchdock_rd1_local_task_protocol",
        seed=int(seed),
        blocks=blocks,
        hardware_required=False,
        clinical_claim_allowed=False,
        export_format="researchdock_csv_v1",
        notes=(
            "Local app/session prototype only; no camera or PPG device is opened.",
            "Measures response profiles and quality flags, not diagnosis or treatment.",
        ),
    )


def protocol_to_dict(protocol: ResearchDockSessionProtocol) -> dict[str, Any]:
    return asdict(protocol)


def researchdock_interface_contract() -> dict[str, Any]:
    """Describe the RD-1 interface without depending on local hardware."""

    return {
        "mode": "design_contract_only",
        "opens_camera": False,
        "requires_ppg_device": False,
        "input_channels": {
            "webcam_pupil_gaze": {
                "status": "optional_design_target",
                "fields": ["pupil_diameter", "gaze_x", "gaze_y"],
                "quality_flags": ["missing_pupil", "gaze_out_of_bounds"],
            },
            "optional_ppg_hrv": {
                "status": "optional_design_target",
                "fields": ["ppg_value", "hrv_proxy"],
                "quality_flags": ["missing_ppg_value", "low_ppg_quality"],
            },
            "behavior": {
                "status": "required_for_local_prototype",
                "fields": ["reaction_time_ms", "accuracy", "task_name", "event_type"],
                "quality_flags": ["implausible_reaction_time", "invalid_accuracy"],
            },
            "self_report": {
                "status": "optional_for_local_prototype",
                "fields": ["valence", "arousal", "motivation"],
                "quality_flags": ["missing_self_report"],
            },
        },
        "outputs": ["csv_session_export", "quality_flags", "synthetic_metrics", "evidence_gate"],
        "claim_boundary": "response-profile measurement scaffold only; no diagnosis, treatment, or stimulation",
    }
