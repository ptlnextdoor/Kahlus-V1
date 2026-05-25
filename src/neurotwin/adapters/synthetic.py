from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256

import numpy as np

from neurotwin.data.schemas import NeuralEventBatch
from neurotwin.data.split_manifest import RecordingRecord


def make_synthetic_recordings(
    n_subjects: int = 8,
    sessions_per_subject: int = 2,
    modalities: Sequence[str] = ("fmri", "eeg", "meg"),
    sites: Sequence[str] = ("site-a", "site-b"),
    datasets: Sequence[str] = ("synthetic_a", "synthetic_b"),
) -> list[RecordingRecord]:
    records: list[RecordingRecord] = []
    for subject_idx in range(n_subjects):
        subject_id = f"sub-{subject_idx:03d}"
        site_id = sites[subject_idx % len(sites)]
        dataset = datasets[subject_idx % len(datasets)]
        for session_idx in range(sessions_per_subject):
            session_id = f"ses-{session_idx:02d}"
            start_time = float(session_idx * 100)
            end_time = start_time + 64.0
            for modality in modalities:
                record_id = f"{dataset}_{site_id}_{subject_id}_{session_id}_{modality}"
                records.append(
                    RecordingRecord(
                        record_id=record_id,
                        modality=modality,
                        dataset=dataset,
                        subject_id=subject_id,
                        session_id=session_id,
                        site_id=site_id,
                        start_time=start_time,
                        end_time=end_time,
                        stimulus_id=f"stim-{session_idx:02d}",
                        path=f"synthetic://{record_id}",
                        metadata={"synthetic": True, "split_stage": "recording_manifest"},
                    )
                )
    return records


def make_synthetic_event_batch(
    modality: str = "fmri",
    n_time: int = 16,
    n_space: int = 8,
    stimulus_dim: int = 12,
    seed: int = 0,
) -> NeuralEventBatch:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(n_time, n_space)).astype(np.float32)
    return NeuralEventBatch(
        modality=modality,
        dataset=f"synthetic_{modality}",
        subject_id="sub-000",
        session_id="ses-00",
        site_id="site-a",
        time=np.arange(n_time, dtype=np.float32),
        signal=signal,
        mask=np.ones_like(signal, dtype=bool),
        stimulus_embedding=rng.normal(size=(n_time, stimulus_dim)).astype(np.float32),
        behavior={"response": rng.normal(size=n_time).astype(np.float32)},
        space_index=np.arange(n_space),
        uncertainty=np.full_like(signal, 0.05, dtype=np.float32),
        provenance={"source": "synthetic", "split_stage": "recording_manifest"},
    )


def make_synthetic_event_batches(
    n_subjects: int = 8,
    sessions_per_subject: int = 2,
    modalities: Sequence[str] = ("fmri", "eeg", "meg"),
    sites: Sequence[str] = ("site-a", "site-b"),
    datasets: Sequence[str] = ("synthetic_a", "synthetic_b"),
    n_time: int = 64,
) -> list[NeuralEventBatch]:
    """Create paired synthetic recordings that mirror the recording manifest."""

    batches: list[NeuralEventBatch] = []
    projections = {
        "fmri": _projection("fmri", 4, 5),
        "eeg": _projection("eeg", 4, 6),
        "meg": _projection("meg", 4, 7),
        "spikes": _projection("spikes", 4, 8),
    }
    for subject_idx in range(n_subjects):
        subject_id = f"sub-{subject_idx:03d}"
        site_id = sites[subject_idx % len(sites)]
        dataset = datasets[subject_idx % len(datasets)]
        for session_idx in range(sessions_per_subject):
            session_id = f"ses-{session_idx:02d}"
            latent = _latent(subject_idx=subject_idx, session_idx=session_idx, n_time=n_time)
            time = np.arange(n_time, dtype=np.float32) + float(session_idx * 100)
            stimulus = np.stack(
                [
                    np.sin(time / 7.0),
                    np.cos(time / 11.0),
                    np.full_like(time, subject_idx / max(1, n_subjects - 1), dtype=np.float32),
                ],
                axis=1,
            ).astype(np.float32)
            for modality in modalities:
                projection = projections.get(modality, _projection(modality, 4, 5))
                signal = (latent @ projection).astype(np.float32)
                signal += _noise(modality, subject_idx, session_idx, signal.shape)
                record_id = f"{dataset}_{site_id}_{subject_id}_{session_id}_{modality}"
                batches.append(
                    NeuralEventBatch(
                        modality=modality,
                        dataset=dataset,
                        subject_id=subject_id,
                        session_id=session_id,
                        site_id=site_id,
                        time=time,
                        signal=signal,
                        mask=np.ones_like(signal, dtype=bool),
                        stimulus_embedding=stimulus,
                        behavior={"response": latent[:, 0].astype(np.float32)},
                        space_index=np.arange(signal.shape[1]),
                        uncertainty=np.full_like(signal, 0.05, dtype=np.float32),
                        provenance={"source": "synthetic", "split_stage": "recording_manifest"},
                        metadata={
                            "record_id": record_id,
                            "synthetic": True,
                            "source_record_id": record_id,
                        },
                    )
                )
    return batches


def _latent(subject_idx: int, session_idx: int, n_time: int) -> np.ndarray:
    rng = np.random.default_rng(subject_idx * 101 + session_idx * 17)
    latent = rng.normal(size=(n_time, 4)).astype(np.float32)
    for idx in range(1, n_time):
        latent[idx] += 0.45 * latent[idx - 1]
    return latent


def _projection(name: str, in_dim: int, out_dim: int) -> np.ndarray:
    seed = int.from_bytes(sha256(name.encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    return rng.normal(size=(in_dim, out_dim)).astype(np.float32)


def _noise(modality: str, subject_idx: int, session_idx: int, shape: tuple[int, ...]) -> np.ndarray:
    seed = int.from_bytes(sha256(f"{modality}-{subject_idx}-{session_idx}".encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    return (0.03 * rng.normal(size=shape)).astype(np.float32)
