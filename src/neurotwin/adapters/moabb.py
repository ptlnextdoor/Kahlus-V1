from __future__ import annotations

import importlib.util
import os
from typing import Any, Iterable

import numpy as np

from neurotwin.data.schemas import NeuralEventBatch
from neurotwin.data.split_manifest import RecordingRecord


class MissingOptionalDependency(RuntimeError):
    pass


def moabb_optional_status() -> dict[str, bool]:
    if os.environ.get("NEUROTWIN_FORCE_MISSING_MOABB") == "1":
        return {"moabb": False, "mne": False}
    return {
        "moabb": importlib.util.find_spec("moabb") is not None,
        "mne": importlib.util.find_spec("mne") is not None,
    }


def require_moabb() -> None:
    status = moabb_optional_status()
    missing = [name for name, present in status.items() if not present]
    if missing:
        raise MissingOptionalDependency(
            "MOABB adapter requires optional dependencies. Install with: pip install -e '.[moabb]'. "
            f"Missing: {', '.join(missing)}"
        )


def load_moabb_trials(
    dataset_name: str,
    subjects: tuple[int, ...] | None = None,
    paradigm: str = "LeftRightImagery",
    max_trials: int | None = None,
    sampling_rate: float | None = None,
) -> list[dict[str, Any]]:
    """Load a small MOABB dataset through its paradigm API when optional deps exist."""

    require_moabb()
    dataset_cls = _resolve_moabb_dataset(dataset_name)
    dataset = dataset_cls()
    paradigm_obj = _build_moabb_paradigm(paradigm)
    x, labels, metadata = paradigm_obj.get_data(dataset=dataset, subjects=list(subjects) if subjects else None)
    trials = []
    for idx, signal in enumerate(x):
        if max_trials is not None and len(trials) >= max_trials:
            break
        meta = _metadata_row(metadata, idx)
        trials.append(
            {
                "signal": np.asarray(signal, dtype=np.float32),
                "subject": meta.get("subject", meta.get("subject_id", "unknown")),
                "session": meta.get("session", meta.get("session_id", "0")),
                "run": meta.get("run", meta.get("run_id", str(idx))),
                "label": labels[idx] if idx < len(labels) else None,
                "sampling_rate": float(sampling_rate or getattr(dataset, "sfreq", 1.0) or 1.0),
                "channel_names": _listlike(getattr(dataset, "channels", None)) or _listlike(meta.get("channel_names")) or [],
                "metadata": meta,
            }
        )
    return trials


def balanced_trial_subset(
    trials: Iterable[dict[str, Any]],
    split_policy: str,
    max_trials: int | None,
) -> list[dict[str, Any]]:
    """Cap MOABB smoke trials without collapsing a group-held-out split."""

    trials = list(trials)
    if max_trials is None:
        return trials
    if max_trials <= 0:
        raise ValueError("max_trials must be positive when provided")
    if split_policy == "time":
        return trials[:max_trials]
    key = {
        "subject": "subject",
        "session": "session",
        "site": "site",
        "dataset": "dataset",
    }.get(split_policy)
    if key is None:
        return trials[:max_trials]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        grouped.setdefault(str(trial.get(key, "unknown")), []).append(trial)
    if len(grouped) < 3:
        raise ValueError(
            f"MOABB smoke split {split_policy!r} requires at least 3 {key} groups "
            "so train/val/test are all represented."
        )
    if max_trials < len(grouped):
        raise ValueError(
            f"max_trials={max_trials} is too small for {len(grouped)} {key} groups; "
            f"use at least {len(grouped)}."
        )

    selected: list[dict[str, Any]] = []
    offsets = {group: 0 for group in grouped}
    group_order = sorted(grouped)
    while len(selected) < max_trials:
        progressed = False
        for group in group_order:
            offset = offsets[group]
            group_trials = grouped[group]
            if offset >= len(group_trials):
                continue
            selected.append(group_trials[offset])
            offsets[group] = offset + 1
            progressed = True
            if len(selected) >= max_trials:
                break
        if not progressed:
            break
    return selected


def trials_to_recordings(trials: Iterable[dict[str, Any]] | None, dataset_id: str, site_id: str = "moabb") -> list[RecordingRecord]:
    if trials is None:
        require_moabb()
        raise ValueError("trials cannot be None when MOABB dependencies are installed")
    records = []
    for idx, trial in enumerate(trials):
        record_id = _build_record_id(dataset_id, idx, trial)
        signal = _time_by_channel_signal(trial)
        sampling_rate = float(trial.get("sampling_rate", 1.0))
        subject_id = _prefixed("sub", trial.get("subject", "unknown"))
        session_id = _prefixed("ses", trial.get("session", "0"))
        run_id = _prefixed("run", trial.get("run", str(idx)))
        start_time = float(trial.get("start_time", 0.0))
        end_time = float(trial.get("end_time", start_time + signal.shape[0] / sampling_rate))
        record_id = f"{dataset_id}_{subject_id}_{session_id}_{run_id}_trial-{idx:05d}"
        records.append(
            RecordingRecord(
                record_id=record_id,
                modality="eeg",
                dataset=dataset_id,
                subject_id=subject_id,
                session_id=session_id,
                site_id=site_id,
                start_time=start_time,
                end_time=end_time,
                stimulus_id=str(trial.get("label")) if trial.get("label") is not None else None,
                path=trial.get("path"),
                metadata={
                    "record_id": record_id,
                    "source_record_id": record_id,
                    "adapter": "moabb",
                    "run_id": run_id,
                    "sampling_rate": sampling_rate,
                    "channel_names": list(trial.get("channel_names", [])),
                    "montage": trial.get("montage"),
                },
            )
        )
    return records


def trials_to_event_batches(trials: Iterable[dict[str, Any]], dataset_id: str, site_id: str = "moabb") -> list[NeuralEventBatch]:
    batches = []
    for idx, trial in enumerate(trials):
        record_id = _build_record_id(dataset_id, idx, trial)
        signal = _time_by_channel_signal(trial)
        sampling_rate = float(trial.get("sampling_rate", 1.0))
        n_time, n_space = signal.shape[:2]
        subject_id = _prefixed("sub", trial.get("subject", "unknown"))
        session_id = _prefixed("ses", trial.get("session", "0"))
        run_id = _prefixed("run", trial.get("run", str(idx)))
        batches.append(
            NeuralEventBatch(
                modality="eeg",
                dataset=dataset_id,
                subject_id=subject_id,
                session_id=session_id,
                site_id=site_id,
                time=np.arange(n_time, dtype=np.float32) / sampling_rate,
                signal=signal,
                mask=np.ones_like(signal, dtype=bool),
                stimulus_embedding=None,
                behavior={},
                space_index=np.arange(n_space),
                provenance={"adapter": "moabb", "record_index": idx},
                metadata={
                    "record_id": record_id,
                    "source_record_id": record_id,
                    "run_id": run_id,
                    "sampling_rate": sampling_rate,
                    "channel_names": list(trial.get("channel_names", [])),
                    "montage": trial.get("montage"),
                },
            )
        )
    return batches


def _prefixed(prefix: str, value: Any) -> str:
    value_str = str(value)
    return value_str if value_str.startswith(prefix + "-") else f"{prefix}-{value_str}"


def _build_record_id(dataset_id: str, index: int, trial: dict[str, Any]) -> str:
    subject_id = _prefixed("sub", trial.get("subject", "unknown"))
    session_id = _prefixed("ses", trial.get("session", "0"))
    run_id = _prefixed("run", trial.get("run", str(index)))
    return str(trial.get("record_id") or f"{dataset_id}_{subject_id}_{session_id}_{run_id}_trial-{index:05d}")


def _time_by_channel_signal(trial: dict[str, Any]) -> np.ndarray:
    """Normalize MOABB/MNE trial arrays to NeuroTwin [time, channel] layout."""

    signal = np.asarray(trial["signal"], dtype=np.float32)
    if signal.ndim == 1:
        signal = signal[:, None]
    elif signal.ndim > 2:
        signal = signal.reshape(signal.shape[0], -1)
    channel_names = _listlike(trial.get("channel_names"))
    if signal.ndim != 2:
        raise ValueError("MOABB trial signal must be coercible to a 2D array")
    if channel_names:
        if len(channel_names) == signal.shape[0] and len(channel_names) != signal.shape[1]:
            signal = signal.T
    elif signal.shape[0] < signal.shape[1]:
        # MOABB/MNE epochs are typically [channels, time]; NeuroTwin stores [time, channels].
        signal = signal.T
    return np.ascontiguousarray(signal, dtype=np.float32)


def _resolve_moabb_dataset(dataset_name: str) -> Any:
    import moabb.datasets as datasets

    if not hasattr(datasets, dataset_name):
        raise ValueError(f"Unknown MOABB dataset {dataset_name!r}")
    return getattr(datasets, dataset_name)


def _build_moabb_paradigm(name: str) -> Any:
    import moabb.paradigms as paradigms

    if not hasattr(paradigms, name):
        raise ValueError(f"Unknown MOABB paradigm {name!r}")
    return getattr(paradigms, name)()


def _metadata_row(metadata: Any, idx: int) -> dict[str, Any]:
    if isinstance(metadata, list):
        return dict(metadata[idx])
    if hasattr(metadata, "iloc"):
        return dict(metadata.iloc[idx].to_dict())
    if isinstance(metadata, dict):
        return {key: value[idx] if hasattr(value, "__len__") and not isinstance(value, str) else value for key, value in metadata.items()}
    return {}


def _listlike(value: Any) -> list[Any]:
    if value is None or isinstance(value, str):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return []
