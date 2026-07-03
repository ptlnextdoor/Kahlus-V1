from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from neurotwin.researchdock.schemas import ResearchDockTrial


def assess_trial_quality(trial: ResearchDockTrial) -> tuple[str, ...]:
    flags = set(trial.quality_flags)
    if trial.sensor_packet is not None:
        flags.update(trial.sensor_packet.quality_flags)
    pupil = trial.sensor_packet.pupil_diameter if trial.sensor_packet is not None else None
    if pupil is None:
        flags.add("missing_pupil")
    elif pupil <= 0.0 or pupil > 10.0:
        flags.add("invalid_pupil_diameter")

    if trial.reaction_time_ms is not None and (trial.reaction_time_ms < 100.0 or trial.reaction_time_ms > 3000.0):
        flags.add("implausible_reaction_time")
    if trial.accuracy is not None and (trial.accuracy < 0.0 or trial.accuracy > 1.0):
        flags.add("invalid_accuracy")
    return tuple(sorted(flags))


def summarize_quality_flags(trials: Sequence[ResearchDockTrial]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for trial in trials:
        counts.update(assess_trial_quality(trial))
    return {
        "n_trials": len(trials),
        "flag_counts": dict(sorted(counts.items())),
        "n_flagged_trials": sum(1 for trial in trials if assess_trial_quality(trial)),
    }
