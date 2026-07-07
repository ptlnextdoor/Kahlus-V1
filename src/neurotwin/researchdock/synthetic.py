from __future__ import annotations

from typing import Sequence

import numpy as np

from neurotwin.researchdock.schemas import (
    ResearchDockSensorPacket,
    ResearchDockSelfReport,
    ResearchDockSession,
    ResearchDockTaskEvent,
    ResearchDockTrial,
)


def make_synthetic_researchdock_sessions(*, seed: int = 0) -> tuple[ResearchDockSession, ...]:
    rng = np.random.default_rng(seed)
    profiles = (
        ("reward_responsive", 0.42, 0.0),
        ("blunted_reward_response", 0.08, 0.0),
        ("high_noise", 0.25, 0.35),
        ("missing_pupil", 0.0, 0.0),
    )
    return tuple(
        _session(
            rng,
            participant_id_hash=f"sha256:synthetic-{idx:02d}",
            session_id=f"researchdock_synth_{profile}",
            profile=profile,
            reward_gain=reward_gain,
            noise_scale=noise_scale,
            missing_pupil=profile == "missing_pupil",
        )
        for idx, (profile, reward_gain, noise_scale) in enumerate(profiles)
    )


def _session(
    rng: np.random.Generator,
    *,
    participant_id_hash: str,
    session_id: str,
    profile: str,
    reward_gain: float,
    noise_scale: float,
    missing_pupil: bool,
) -> ResearchDockSession:
    trials: list[ResearchDockTrial] = []
    timestamp = 0.0
    for idx in range(8):
        reward_condition = "reward" if idx % 2 == 0 else "neutral"
        task_name = "reward_anticipation" if idx < 4 else "effort_for_reward"
        effort = 0.25 if idx < 4 else 0.75
        base_pupil = 3.0 + (reward_gain if reward_condition == "reward" else 0.0)
        pupil = None if missing_pupil else base_pupil + float(rng.normal(0.0, 0.04 + noise_scale))
        reaction_time = 430.0 - (35.0 if reward_condition == "reward" else 0.0) + float(rng.normal(0.0, 8.0 + 20.0 * noise_scale))
        accuracy = 1.0 if idx % 5 else 0.0
        flags = ("synthetic_high_noise",) if noise_scale > 0.0 else ()
        trials.append(
            _trial(
                participant_id_hash,
                session_id,
                timestamp,
                task_name,
                "response",
                reward_condition,
                effort,
                reaction_time,
                accuracy,
                pupil,
                hrv_proxy=62.0 + float(rng.normal(0.0, 2.0 + 8.0 * noise_scale)),
                quality_flags=flags,
            )
        )
        timestamp += 1.0
    for idx in range(4):
        pupil = None if missing_pupil else 3.25 - 0.08 * idx + float(rng.normal(0.0, 0.03 + noise_scale))
        trials.append(
            _trial(
                participant_id_hash,
                session_id,
                timestamp,
                "recovery_rest",
                "rest_sample",
                None,
                None,
                None,
                None,
                pupil,
                hrv_proxy=65.0 + idx + float(rng.normal(0.0, 1.5 + 4.0 * noise_scale)),
                quality_flags=("missing_pupil",) if missing_pupil else (),
            )
        )
        timestamp += 1.0
    session_flags = ("missing_pupil",) if missing_pupil else ()
    return ResearchDockSession(
        participant_id_hash=participant_id_hash,
        session_id=session_id,
        trials=tuple(trials),
        sensor_packets=tuple(trial.sensor_packet for trial in trials if trial.sensor_packet is not None),
        task_events=tuple(trial.task_event for trial in trials if trial.task_event is not None),
        self_reports=tuple(trial.self_report for trial in trials if trial.self_report is not None),
        quality_flags=session_flags,
        metadata={"dataset_id": "researchdock_synthetic_v0", "profile": profile, "synthetic": True},
    )


def _trial(
    participant_id_hash: str,
    session_id: str,
    timestamp: float,
    task_name: str,
    event_type: str,
    reward_condition: str | None,
    effort_level: float | None,
    reaction_time_ms: float | None,
    accuracy: float | None,
    pupil_diameter: float | None,
    *,
    hrv_proxy: float,
    quality_flags: Sequence[str] = (),
) -> ResearchDockTrial:
    sensor = ResearchDockSensorPacket(
        timestamp=timestamp,
        pupil_diameter=pupil_diameter,
        gaze_x=0.0 if pupil_diameter is not None else None,
        gaze_y=0.0 if pupil_diameter is not None else None,
        ppg_value=0.8 + 0.01 * hrv_proxy,
        hrv_proxy=hrv_proxy,
        quality_flags=tuple(quality_flags),
    )
    event = ResearchDockTaskEvent(
        timestamp=timestamp,
        task_name=task_name,
        event_type=event_type,
        reward_condition=reward_condition,
        effort_level=effort_level,
        reaction_time_ms=reaction_time_ms,
        accuracy=accuracy,
        quality_flags=tuple(quality_flags),
    )
    report = ResearchDockSelfReport(
        timestamp=timestamp,
        valence=0.2 if reward_condition == "reward" else 0.0,
        arousal=0.3 if task_name != "recovery_rest" else 0.1,
        motivation=0.6 if reward_condition == "reward" else 0.4,
        quality_flags=tuple(quality_flags),
    )
    return ResearchDockTrial(
        participant_id_hash=participant_id_hash,
        session_id=session_id,
        timestamp=timestamp,
        task_name=task_name,
        event_type=event_type,
        reward_condition=reward_condition,
        effort_level=effort_level,
        reaction_time_ms=reaction_time_ms,
        accuracy=accuracy,
        sensor_packet=sensor,
        task_event=event,
        self_report=report,
        quality_flags=tuple(quality_flags),
    )
