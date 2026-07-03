from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchDockTaskTemplate:
    task_name: str
    description: str
    primary_signals: tuple[str, ...]
    allowed_event_types: tuple[str, ...]


def researchdock_task_templates() -> tuple[ResearchDockTaskTemplate, ...]:
    """Return the safe RD-0 behavioral task battery."""

    return (
        ResearchDockTaskTemplate(
            task_name="reward_anticipation",
            description="Neutral and reward cue blocks for non-diagnostic reward response-profile measurement.",
            primary_signals=("pupil_diameter", "reaction_time_ms", "self_report_motivation"),
            allowed_event_types=("cue", "response", "feedback"),
        ),
        ResearchDockTaskTemplate(
            task_name="probabilistic_reward_learning",
            description="Choice-feedback task for reward-learning response-profile measurement.",
            primary_signals=("accuracy", "reaction_time_ms", "pupil_diameter"),
            allowed_event_types=("choice", "feedback"),
        ),
        ResearchDockTaskTemplate(
            task_name="effort_for_reward",
            description="Voluntary effort choices with low and high effort levels.",
            primary_signals=("effort_level", "reaction_time_ms", "accuracy"),
            allowed_event_types=("offer", "response", "feedback"),
        ),
        ResearchDockTaskTemplate(
            task_name="mild_frustration",
            description="Mild timing difficulty block for stress-recovery response-profile measurement.",
            primary_signals=("reaction_time_ms", "pupil_diameter", "ppg_value"),
            allowed_event_types=("prompt", "response", "feedback"),
        ),
        ResearchDockTaskTemplate(
            task_name="recovery_rest",
            description="Quiet recovery block after task demand.",
            primary_signals=("pupil_diameter", "hrv_proxy", "self_report_arousal"),
            allowed_event_types=("rest_start", "rest_sample", "rest_end"),
        ),
        ResearchDockTaskTemplate(
            task_name="visual_attention",
            description="Brief visual attention task without unsupervised photic stimulation.",
            primary_signals=("gaze_x", "gaze_y", "reaction_time_ms", "accuracy"),
            allowed_event_types=("target", "response"),
        ),
    )
