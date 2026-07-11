from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError, version
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
    paradigm_obj = _build_moabb_paradigm(paradigm, resample=sampling_rate)
    x, labels, metadata = paradigm_obj.get_data(
        dataset=dataset,
        subjects=list(subjects) if subjects else None,
        return_epochs=True,
    )
    signal, resolved_sampling_rate, channel_names, signal_unit, unit_factor_provenance = _epoch_array_metadata(
        x,
        dataset=dataset,
        sampling_rate=sampling_rate,
    )
    source_provenance = _moabb_source_provenance(
        dataset_name=dataset_name,
        paradigm_name=paradigm,
        paradigm_obj=paradigm_obj,
        unit_factor_provenance=unit_factor_provenance,
    )
    trials = []
    for idx, trial_signal in enumerate(signal):
        if max_trials is not None and len(trials) >= max_trials:
            break
        meta = _metadata_row(metadata, idx)
        trials.append(
            {
                "signal": np.asarray(trial_signal, dtype=np.float32),
                "subject": meta.get("subject", meta.get("subject_id", "unknown")),
                "session": meta.get("session", meta.get("session_id", "0")),
                "run": meta.get("run", meta.get("run_id", str(idx))),
                "label": labels[idx] if idx < len(labels) else None,
                "sampling_rate": resolved_sampling_rate,
                "channel_names": channel_names,
                "signal_unit": signal_unit,
                **source_provenance,
                "metadata": meta,
            }
        )
    return trials


def load_balanced_moabb_subject_trials(
    dataset_name: str,
    subjects: tuple[int, ...],
    *,
    paradigm: str = "LeftRightImagery",
    max_trials: int | None,
    sampling_rate: float | None = None,
) -> list[dict[str, Any]]:
    """Load subjects incrementally and preserve subject-round-robin selection.

    At most one subject's MOABB epochs plus the bounded selected trial copies are
    retained at once. The returned ordering matches ``balanced_trial_subset`` on
    the same per-subject trial streams.
    """

    if not subjects:
        raise ValueError("subjects must contain at least one subject")
    if max_trials is None:
        trials: list[dict[str, Any]] = []
        for subject in subjects:
            trials.extend(
                load_moabb_trials(
                    dataset_name,
                    subjects=(subject,),
                    paradigm=paradigm,
                    sampling_rate=sampling_rate,
                )
            )
        return trials
    if max_trials <= 0:
        raise ValueError("max_trials must be positive when provided")
    subject_order = sorted(dict.fromkeys(subjects), key=lambda value: str(value))
    quota = (max_trials + len(subject_order) - 1) // len(subject_order)
    retained: dict[int, list[dict[str, Any]]] = {}
    available: dict[int, int] = {}
    for subject in subject_order:
        loaded = load_moabb_trials(
            dataset_name,
            subjects=(subject,),
            paradigm=paradigm,
            sampling_rate=sampling_rate,
        )
        available[subject] = len(loaded)
        retained[subject] = [_copy_trial(trial) for trial in loaded[:quota]]

    nonempty_groups = sum(count > 0 for count in available.values())
    if nonempty_groups < 3:
        raise ValueError(
            "MOABB smoke split 'subject' requires at least 3 subject groups "
            "so train/val/test are all represented."
        )
    if max_trials < nonempty_groups:
        raise ValueError(
            f"max_trials={max_trials} is too small for {nonempty_groups} subject groups; "
            f"use at least {nonempty_groups}."
        )

    target_counts = _balanced_group_counts(available, max_trials=max_trials)
    for subject in subject_order:
        target = target_counts[subject]
        if target <= len(retained[subject]):
            continue
        loaded = load_moabb_trials(
            dataset_name,
            subjects=(subject,),
            paradigm=paradigm,
            sampling_rate=sampling_rate,
        )
        retained[subject] = [_copy_trial(trial) for trial in loaded[:target]]

    selected: list[dict[str, Any]] = []
    for offset in range(max(target_counts.values(), default=0)):
        for subject in subject_order:
            if offset < target_counts[subject]:
                selected.append(retained[subject][offset])
    return selected


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
                    "signal_unit": str(trial.get("signal_unit", "unknown")),
                    "channel_names": list(trial.get("channel_names", [])),
                    "montage": trial.get("montage"),
                    **_trial_source_metadata(trial),
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
                provenance={"adapter": "moabb", "record_index": idx, **_trial_source_metadata(trial)},
                metadata={
                    "record_id": record_id,
                    "source_record_id": record_id,
                    "run_id": run_id,
                    "sampling_rate": sampling_rate,
                    "signal_unit": str(trial.get("signal_unit", "unknown")),
                    "channel_names": list(trial.get("channel_names", [])),
                    "montage": trial.get("montage"),
                    **_trial_source_metadata(trial),
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


def _build_moabb_paradigm(name: str, *, resample: float | None = None) -> Any:
    import moabb.paradigms as paradigms

    if not hasattr(paradigms, name):
        raise ValueError(f"Unknown MOABB paradigm {name!r}")
    kwargs = {"resample": float(resample)} if resample is not None else {}
    return getattr(paradigms, name)(**kwargs)


def _epoch_array_metadata(
    data: Any,
    *,
    dataset: Any,
    sampling_rate: float | None,
) -> tuple[np.ndarray, float, list[str], str, dict[str, Any]]:
    """Extract MOABB epoch arrays with their true sampling rate and signal unit.

    MOABB's regular array path scales MNE epoch data by ``dataset.unit_factor``
    (normally 1e6) into microvolts. This adapter requests epochs so it can
    retain the sampling rate and channel names, then performs that same
    documented scaling explicitly.
    """

    if hasattr(data, "get_data") and hasattr(data, "info"):
        array = np.asarray(data.get_data(), dtype=np.float32)
        info = getattr(data, "info")
        info_rate = float(info["sfreq"])
        if sampling_rate is not None and not np.isclose(info_rate, float(sampling_rate), rtol=0.0, atol=1e-8):
            raise RuntimeError(
                f"MOABB resampling requested {sampling_rate} Hz but epochs report {info_rate} Hz"
            )
        unit_factor = float(getattr(dataset, "unit_factor", 1e6))
        return (
            array * unit_factor,
            info_rate,
            _listlike(getattr(data, "ch_names", None)),
            "uV",
            {
                "factor": unit_factor,
                "factor_source": f"{type(dataset).__module__}.{type(dataset).__name__}.unit_factor",
                "input_source": "MNE Epochs.get_data() returned by MOABB",
                "operation": "multiply",
                "output_unit": "uV",
            },
        )

    # This fallback exists for simple test doubles and nonstandard MOABB-like
    # objects. It cannot establish physical units, so downstream figure code
    # must not label it as real microvolt data.
    return (
        np.asarray(data, dtype=np.float32),
        float(sampling_rate or getattr(dataset, "sfreq", 1.0) or 1.0),
        _listlike(getattr(dataset, "channels", None)),
        "unknown",
        {
            "factor": None,
            "factor_source": "unavailable for non-MNE MOABB-like array",
            "input_source": "array returned by MOABB-like paradigm",
            "operation": "none",
            "output_unit": "unknown",
        },
    )


def _moabb_source_provenance(
    *,
    dataset_name: str,
    paradigm_name: str,
    paradigm_obj: Any,
    unit_factor_provenance: dict[str, Any],
) -> dict[str, Any]:
    try:
        moabb_version = version("moabb")
    except PackageNotFoundError:
        moabb_version = "unknown"
    get_params = getattr(paradigm_obj, "get_params", None)
    parameters = get_params(deep=False) if callable(get_params) else {}
    filters = getattr(paradigm_obj, "filters", None)
    return {
        "signal_source": "MOABB-preprocessed epochs",
        "moabb_dataset": dataset_name,
        "moabb_paradigm": paradigm_name,
        "moabb_version": moabb_version,
        "moabb_filters": {"source": "paradigm.filters", "value": _json_safe(filters)},
        "moabb_preprocessing": {
            "api": "paradigm.get_data(return_epochs=True)",
            "paradigm_class": f"{type(paradigm_obj).__module__}.{type(paradigm_obj).__name__}",
            "parameters": _json_safe(parameters),
        },
        "unit_factor_provenance": unit_factor_provenance,
    }


def _trial_source_metadata(trial: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "signal_source",
        "moabb_dataset",
        "moabb_paradigm",
        "moabb_version",
        "moabb_filters",
        "moabb_preprocessing",
        "unit_factor_provenance",
    )
    return {key: trial[key] for key in keys if key in trial}


def _copy_trial(trial: dict[str, Any]) -> dict[str, Any]:
    copied = dict(trial)
    copied["signal"] = np.array(trial["signal"], dtype=np.float32, copy=True)
    copied["metadata"] = dict(trial.get("metadata", {}))
    return copied


def _balanced_group_counts(available: dict[int, int], *, max_trials: int) -> dict[int, int]:
    counts = {subject: 0 for subject in available}
    while sum(counts.values()) < max_trials:
        progressed = False
        for subject in available:
            if counts[subject] >= available[subject]:
                continue
            counts[subject] += 1
            progressed = True
            if sum(counts.values()) >= max_trials:
                break
        if not progressed:
            break
    return counts


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return repr(value)


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
