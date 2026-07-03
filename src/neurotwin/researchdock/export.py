from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from neurotwin.repro import write_json
from neurotwin.researchdock.quality import assess_trial_quality, summarize_quality_flags
from neurotwin.researchdock.schemas import ResearchDockSession, ResearchDockTrial


def export_researchdock_sessions(sessions: Sequence[ResearchDockSession], out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    session_rows = [
        {
            "participant_id_hash": session.participant_id_hash,
            "session_id": session.session_id,
            "n_trials": len(session.trials),
            "quality_flags": _flags(session.quality_flags),
            "metadata_json": json.dumps(session.metadata, sort_keys=True),
        }
        for session in sessions
    ]
    trial_rows = [_trial_row(session, trial, idx) for session in sessions for idx, trial in enumerate(session.trials)]
    sensor_rows = [_sensor_row(session, trial, idx) for session in sessions for idx, trial in enumerate(session.trials) if trial.sensor_packet is not None]
    event_rows = [_event_row(session, trial, idx) for session in sessions for idx, trial in enumerate(session.trials) if trial.task_event is not None]
    report_rows = [_report_row(session, trial, idx) for session in sessions for idx, trial in enumerate(session.trials) if trial.self_report is not None]

    paths = {
        "sessions_csv": _write_rows(out / "researchdock_sessions.csv", session_rows, _SESSION_FIELDS),
        "trials_csv": _write_rows(out / "researchdock_trials.csv", trial_rows, _TRIAL_FIELDS),
        "sensor_packets_csv": _write_rows(out / "researchdock_sensor_packets.csv", sensor_rows, _SENSOR_FIELDS),
        "task_events_csv": _write_rows(out / "researchdock_task_events.csv", event_rows, _EVENT_FIELDS),
        "self_reports_csv": _write_rows(out / "researchdock_self_reports.csv", report_rows, _REPORT_FIELDS),
    }
    quality = summarize_quality_flags(tuple(trial for session in sessions for trial in session.trials))
    manifest = {
        "export_format": "researchdock_csv_v1",
        "n_sessions": len(sessions),
        "n_trials": len(trial_rows),
        "contains_pii": False,
        "files": {key: Path(path).name for key, path in paths.items()},
        "quality_summary": quality,
    }
    write_json(out / "researchdock_session_manifest.json", manifest)
    return manifest


_SESSION_FIELDS = ("participant_id_hash", "session_id", "n_trials", "quality_flags", "metadata_json")
_TRIAL_FIELDS = (
    "participant_id_hash",
    "session_id",
    "trial_index",
    "timestamp",
    "task_name",
    "event_type",
    "reward_condition",
    "effort_level",
    "reaction_time_ms",
    "accuracy",
    "quality_flags",
)
_SENSOR_FIELDS = ("participant_id_hash", "session_id", "trial_index", "timestamp", "pupil_diameter", "gaze_x", "gaze_y", "ppg_value", "hrv_proxy", "quality_flags")
_EVENT_FIELDS = ("participant_id_hash", "session_id", "trial_index", "timestamp", "task_name", "event_type", "reward_condition", "effort_level", "reaction_time_ms", "accuracy", "quality_flags")
_REPORT_FIELDS = ("participant_id_hash", "session_id", "trial_index", "timestamp", "valence", "arousal", "motivation", "quality_flags")


def _write_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Iterable[str]) -> Path:
    names = tuple(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in names})
    return path


def _trial_row(session: ResearchDockSession, trial: ResearchDockTrial, trial_index: int) -> dict[str, Any]:
    return {
        "participant_id_hash": session.participant_id_hash,
        "session_id": session.session_id,
        "trial_index": int(trial_index),
        "timestamp": trial.timestamp,
        "task_name": trial.task_name,
        "event_type": trial.event_type,
        "reward_condition": trial.reward_condition,
        "effort_level": trial.effort_level,
        "reaction_time_ms": trial.reaction_time_ms,
        "accuracy": trial.accuracy,
        "quality_flags": _flags(assess_trial_quality(trial)),
    }


def _sensor_row(session: ResearchDockSession, trial: ResearchDockTrial, trial_index: int) -> dict[str, Any]:
    sensor = trial.sensor_packet
    assert sensor is not None
    return {
        "participant_id_hash": session.participant_id_hash,
        "session_id": session.session_id,
        "trial_index": int(trial_index),
        "timestamp": sensor.timestamp,
        "pupil_diameter": sensor.pupil_diameter,
        "gaze_x": sensor.gaze_x,
        "gaze_y": sensor.gaze_y,
        "ppg_value": sensor.ppg_value,
        "hrv_proxy": sensor.hrv_proxy,
        "quality_flags": _flags(sensor.quality_flags),
    }


def _event_row(session: ResearchDockSession, trial: ResearchDockTrial, trial_index: int) -> dict[str, Any]:
    event = trial.task_event
    assert event is not None
    return {
        "participant_id_hash": session.participant_id_hash,
        "session_id": session.session_id,
        "trial_index": int(trial_index),
        "timestamp": event.timestamp,
        "task_name": event.task_name,
        "event_type": event.event_type,
        "reward_condition": event.reward_condition,
        "effort_level": event.effort_level,
        "reaction_time_ms": event.reaction_time_ms,
        "accuracy": event.accuracy,
        "quality_flags": _flags(event.quality_flags),
    }


def _report_row(session: ResearchDockSession, trial: ResearchDockTrial, trial_index: int) -> dict[str, Any]:
    report = trial.self_report
    assert report is not None
    return {
        "participant_id_hash": session.participant_id_hash,
        "session_id": session.session_id,
        "trial_index": int(trial_index),
        "timestamp": report.timestamp,
        "valence": report.valence,
        "arousal": report.arousal,
        "motivation": report.motivation,
        "quality_flags": _flags(report.quality_flags),
    }


def _flags(flags: Sequence[str]) -> str:
    return ";".join(sorted(str(flag) for flag in flags if str(flag)))
