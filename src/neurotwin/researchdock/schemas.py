from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


QualityFlags = tuple[str, ...]


@dataclass(frozen=True)
class ResearchDockSensorPacket:
    timestamp: float
    pupil_diameter: float | None = None
    gaze_x: float | None = None
    gaze_y: float | None = None
    ppg_value: float | None = None
    hrv_proxy: float | None = None
    quality_flags: QualityFlags = ()


@dataclass(frozen=True)
class ResearchDockTaskEvent:
    timestamp: float
    task_name: str
    event_type: str
    reward_condition: str | None = None
    effort_level: float | None = None
    reaction_time_ms: float | None = None
    accuracy: float | None = None
    quality_flags: QualityFlags = ()


@dataclass(frozen=True)
class ResearchDockSelfReport:
    timestamp: float
    valence: float | None = None
    arousal: float | None = None
    motivation: float | None = None
    quality_flags: QualityFlags = ()


@dataclass(frozen=True)
class ResearchDockTrial:
    participant_id_hash: str
    session_id: str
    timestamp: float
    task_name: str
    event_type: str
    reward_condition: str | None = None
    effort_level: float | None = None
    reaction_time_ms: float | None = None
    accuracy: float | None = None
    sensor_packet: ResearchDockSensorPacket | None = None
    task_event: ResearchDockTaskEvent | None = None
    self_report: ResearchDockSelfReport | None = None
    quality_flags: QualityFlags = ()


@dataclass(frozen=True)
class ResearchDockSession:
    participant_id_hash: str
    session_id: str
    trials: tuple[ResearchDockTrial, ...]
    sensor_packets: tuple[ResearchDockSensorPacket, ...] = ()
    task_events: tuple[ResearchDockTaskEvent, ...] = ()
    self_reports: tuple[ResearchDockSelfReport, ...] = ()
    quality_flags: QualityFlags = ()
    metadata: dict[str, Any] = field(default_factory=dict)
