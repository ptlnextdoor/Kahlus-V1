from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neurotwin.data.schemas import NeuralEventBatch
from neurotwin.data.split_manifest import SplitManifest
from neurotwin.data.windows import WindowSpec, batch_to_windows


@dataclass(frozen=True)
class PreparedSuiteConfig:
    event_manifest: str | Path
    split_manifest: str | Path
    window_length: int = 8
    stride: int = 8
    seed: int = 0
    train_steps: int = 5
    require_ci: bool = True


@dataclass(frozen=True)
class SupervisedWindowTask:
    task_id: str
    source_modality: str
    target_modality: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    metric_mask: np.ndarray | None = None
    x_val: np.ndarray | None = None
    y_val: np.ndarray | None = None
    val_metric_mask: np.ndarray | None = None
    notes: tuple[str, ...] = ()


def build_prepared_window_tasks(
    batches: list[NeuralEventBatch],
    split: SplitManifest,
    window_length: int,
    stride: int,
    seed: int = 0,
) -> tuple[tuple[SupervisedWindowTask, ...], list[dict[str, str]]]:
    skipped: list[dict[str, str]] = []
    windows_by_split = prepared_windows_by_split(batches, split, window_length=window_length, stride=stride)
    split_keys = _split_record_keys(split)
    for batch in batches:
        if _record_id(batch) not in split_keys:
            skipped.append({"task_id": "all", "reason": f"event not present in split manifest: {_record_id(batch)}"})

    tasks: list[SupervisedWindowTask] = []
    future = _future_task_from_windows(windows_by_split)
    if future is not None:
        tasks.append(future)
    else:
        skipped.append({"task_id": "future_state_forecasting", "reason": "need train and test windows for one modality"})

    masked = _masked_task_from_windows(windows_by_split, seed=seed)
    if masked is not None:
        tasks.append(masked)
    else:
        skipped.append({"task_id": "masked_neural_reconstruction", "reason": "need train and test windows for one modality"})

    cross_modal = _cross_modal_task_from_windows(windows_by_split)
    if cross_modal is not None:
        tasks.append(cross_modal)
    else:
        skipped.append({"task_id": "cross_modal_translation", "reason": "need paired train/test windows for two modalities"})

    stimulus_fmri = _stimulus_fmri_task_from_windows(windows_by_split)
    if stimulus_fmri is not None:
        tasks.append(stimulus_fmri)
    else:
        skipped.append({"task_id": "stimulus_to_fmri_response", "reason": "need fMRI train/test windows with aligned stimulus embeddings"})
    return tuple(tasks), skipped


def prepared_windows_by_split(
    batches: list[NeuralEventBatch],
    split: SplitManifest,
    window_length: int,
    stride: int,
) -> dict[str, list[NeuralEventBatch]]:
    spec = WindowSpec(length=window_length, stride=stride)
    split_keys = _split_record_keys(split)
    windows_by_split: dict[str, list[NeuralEventBatch]] = {"train": [], "val": [], "test": []}
    for batch in batches:
        split_name = split_keys.get(_record_id(batch))
        if split_name is None:
            continue
        windows_by_split[split_name].extend(batch_to_windows(batch, spec))
    return windows_by_split


def first_prepared_modality_with_splits(windows_by_split: dict[str, list[NeuralEventBatch]]) -> str | None:
    train_modalities = {window.modality for window in windows_by_split["train"]}
    test_modalities = {window.modality for window in windows_by_split["test"]}
    for preferred in ("eeg", "fmri", "meg", "spikes"):
        if preferred in train_modalities and preferred in test_modalities:
            return preferred
    shared = sorted(train_modalities & test_modalities)
    return shared[0] if shared else None


def _future_task_from_windows(windows_by_split: dict[str, list[NeuralEventBatch]]) -> SupervisedWindowTask | None:
    modality = first_prepared_modality_with_splits(windows_by_split)
    if modality is None:
        return None
    train = [window.signal for window in windows_by_split["train"] if window.modality == modality]
    val = [window.signal for window in windows_by_split["val"] if window.modality == modality]
    test = [window.signal for window in windows_by_split["test"] if window.modality == modality]
    x_train, y_train = _future_xy(train)
    x_val, y_val = _future_xy(val)
    x_test, y_test = _future_xy(test)
    if x_train is None or x_test is None or y_train is None or y_test is None:
        return None
    return SupervisedWindowTask(
        task_id="future_state_forecasting",
        source_modality=modality,
        target_modality=modality,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_val=x_val,
        y_val=y_val,
        notes=(f"prepared {modality} next-state windows",),
    )


def _masked_task_from_windows(
    windows_by_split: dict[str, list[NeuralEventBatch]],
    seed: int,
) -> SupervisedWindowTask | None:
    modality = first_prepared_modality_with_splits(windows_by_split)
    if modality is None:
        return None
    train = np.asarray([window.signal for window in windows_by_split["train"] if window.modality == modality], dtype=np.float32)
    val = np.asarray([window.signal for window in windows_by_split["val"] if window.modality == modality], dtype=np.float32)
    test = np.asarray([window.signal for window in windows_by_split["test"] if window.modality == modality], dtype=np.float32)
    if train.size == 0 or test.size == 0:
        return None
    rng = np.random.default_rng(seed + 31)
    train_mask = rng.random(train.shape) < 0.2
    val_mask = rng.random(val.shape) < 0.2 if val.size else None
    test_mask = rng.random(test.shape) < 0.2
    x_train = train.copy()
    x_val = val.copy() if val.size else None
    x_test = test.copy()
    x_train[train_mask] = 0.0
    if x_val is not None and val_mask is not None:
        x_val[val_mask] = 0.0
    x_test[test_mask] = 0.0
    return SupervisedWindowTask(
        task_id="masked_neural_reconstruction",
        source_modality=modality,
        target_modality=modality,
        x_train=x_train,
        y_train=train,
        x_test=x_test,
        y_test=test,
        metric_mask=test_mask,
        x_val=x_val,
        y_val=val if val.size else None,
        val_metric_mask=val_mask,
        notes=(f"prepared {modality} masked reconstruction",),
    )


def _cross_modal_task_from_windows(windows_by_split: dict[str, list[NeuralEventBatch]]) -> SupervisedWindowTask | None:
    modalities = sorted({window.modality for split_windows in windows_by_split.values() for window in split_windows})
    if len(modalities) < 2:
        return None
    source = "eeg" if "eeg" in modalities else modalities[0]
    target_candidates = [modality for modality in modalities if modality != source]
    target = "fmri" if "fmri" in target_candidates else target_candidates[0]
    train_pairs = _paired_windows(windows_by_split["train"], source, target)
    val_pairs = _paired_windows(windows_by_split["val"], source, target)
    test_pairs = _paired_windows(windows_by_split["test"], source, target)
    if not train_pairs or not test_pairs:
        return None
    x_train, y_train = _stack_pairs(train_pairs)
    x_val, y_val = _stack_pairs(val_pairs) if val_pairs else (None, None)
    x_test, y_test = _stack_pairs(test_pairs)
    return SupervisedWindowTask(
        task_id="cross_modal_translation",
        source_modality=source,
        target_modality=target,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_val=x_val,
        y_val=y_val,
        notes=(f"prepared paired {source}->{target} windows",),
    )


def _stimulus_fmri_task_from_windows(windows_by_split: dict[str, list[NeuralEventBatch]]) -> SupervisedWindowTask | None:
    train = _stimulus_fmri_xy(windows_by_split["train"])
    val = _stimulus_fmri_xy(windows_by_split["val"])
    test = _stimulus_fmri_xy(windows_by_split["test"])
    if train is None or test is None:
        return None
    x_train, y_train = train
    x_test, y_test = test
    x_val, y_val = val if val is not None else (None, None)
    return SupervisedWindowTask(
        task_id="stimulus_to_fmri_response",
        source_modality="stimulus",
        target_modality="fmri",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_val=x_val,
        y_val=y_val,
        notes=(
            "prepared stimulus->fMRI response windows",
            "tribe_style is a clean_room_approximation, not an exact TRIBE v2 reproduction",
        ),
    )


def _future_xy(signals: list[np.ndarray]) -> tuple[np.ndarray | None, np.ndarray | None]:
    usable = [np.asarray(signal, dtype=np.float32) for signal in signals if signal.shape[0] >= 2]
    if not usable:
        return None, None
    return np.asarray([signal[:-1] for signal in usable], dtype=np.float32), np.asarray([signal[1:] for signal in usable], dtype=np.float32)


def _paired_windows(
    windows: list[NeuralEventBatch],
    source: str,
    target: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str, str, str, int], dict[str, NeuralEventBatch]] = {}
    for window in windows:
        key = (
            window.dataset,
            window.subject_id,
            window.session_id,
            window.site_id,
            int(window.metadata.get("window_start_index", 0)),
        )
        grouped.setdefault(key, {})[window.modality] = window
    pairs = []
    for group in grouped.values():
        if source in group and target in group:
            source_signal = group[source].signal
            target_signal = group[target].signal
            n_time = min(source_signal.shape[0], target_signal.shape[0])
            pairs.append((source_signal[:n_time], target_signal[:n_time]))
    return pairs


def _stimulus_fmri_xy(windows: list[NeuralEventBatch]) -> tuple[np.ndarray, np.ndarray] | None:
    pairs = [
        (np.asarray(window.stimulus_embedding, dtype=np.float32), np.asarray(window.signal, dtype=np.float32))
        for window in windows
        if window.modality == "fmri"
        and window.stimulus_embedding is not None
        and window.stimulus_embedding.shape[0] == window.signal.shape[0]
    ]
    if not pairs:
        return None
    return (
        np.asarray([stimulus for stimulus, _ in pairs], dtype=np.float32),
        np.asarray([signal for _, signal in pairs], dtype=np.float32),
    )


def _stack_pairs(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([source for source, _ in pairs], dtype=np.float32),
        np.asarray([target for _, target in pairs], dtype=np.float32),
    )


def _record_id(batch: NeuralEventBatch) -> str:
    return str(batch.metadata.get("record_id") or batch.metadata.get("source_record_id"))


def _split_record_keys(split: SplitManifest) -> dict[str, str]:
    keys = {}
    for split_name in ("train", "val", "test"):
        for record in getattr(split, split_name):
            keys[record.record_id] = split_name
    return keys
