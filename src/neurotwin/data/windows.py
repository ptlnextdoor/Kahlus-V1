from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurotwin.data.schemas import NeuralEventBatch


@dataclass(frozen=True)
class WindowSpec:
    length: int
    stride: int
    drop_last: bool = True


def batch_to_windows(batch: NeuralEventBatch, spec: WindowSpec) -> list[NeuralEventBatch]:
    if spec.length <= 0 or spec.stride <= 0:
        raise ValueError("WindowSpec length and stride must be positive")
    windows: list[NeuralEventBatch] = []
    starts = range(0, batch.n_time, spec.stride)
    for window_idx, start in enumerate(starts):
        end = start + spec.length
        if end > batch.n_time:
            if spec.drop_last:
                break
            start = max(0, batch.n_time - spec.length)
            end = batch.n_time
        signal = batch.signal[start:end]
        if signal.shape[0] == 0:
            continue
        stimulus = batch.stimulus_embedding[start:end] if batch.stimulus_embedding is not None else None
        behavior = {
            key: _slice_behavior(value, start, end)
            for key, value in batch.behavior.items()
        }
        windows.append(
            NeuralEventBatch(
                modality=batch.modality,
                dataset=batch.dataset,
                subject_id=batch.subject_id,
                session_id=batch.session_id,
                site_id=batch.site_id,
                time=batch.time[start:end],
                signal=signal,
                mask=batch.mask[start:end],
                stimulus_embedding=stimulus,
                behavior=behavior,
                space_index=batch.space_index,
                uncertainty=batch.uncertainty[start:end] if batch.uncertainty is not None else None,
                provenance={
                    **batch.provenance,
                    "source_subject_id": batch.subject_id,
                    "source_session_id": batch.session_id,
                    "window_index": window_idx,
                },
                metadata={
                    **batch.metadata,
                    "window_start_index": start,
                    "window_end_index": end,
                    "source_record_id": batch.metadata.get("source_record_id", batch.metadata.get("record_id")),
                },
            )
        )
        if end >= batch.n_time and not spec.drop_last:
            break
    return windows


def _slice_behavior(value: object, start: int, end: int) -> object:
    if hasattr(value, "shape"):
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] >= end:
            return arr[start:end]
    return value
