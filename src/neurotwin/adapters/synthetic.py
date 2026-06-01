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


def make_synthetic_multimodal_recordings(
    n_subjects: int = 8,
    sessions_per_subject: int = 2,
    include_unpaired: bool = True,
) -> list[RecordingRecord]:
    """Create multimodal synthetic recording records for smoke tests only."""

    records: list[RecordingRecord] = []
    modalities = ("eeg", "fmri", "behavior", "stimulus")
    rates = _multimodal_sampling_rates()
    for subject_idx in range(n_subjects):
        subject_id = f"sub-{subject_idx:03d}"
        site_id = "site-a" if subject_idx % 2 == 0 else "site-b"
        dataset = "synthetic_multimodal"
        missing = {"fmri"} if include_unpaired and subject_idx == n_subjects - 1 else set()
        for session_idx in range(sessions_per_subject):
            session_id = f"ses-{session_idx:02d}"
            for modality in modalities:
                if modality in missing:
                    continue
                record_id = f"{dataset}_{site_id}_{subject_id}_{session_id}_{modality}"
                records.append(
                    RecordingRecord(
                        record_id=record_id,
                        modality=modality,
                        dataset=dataset,
                        subject_id=subject_id,
                        session_id=session_id,
                        site_id=site_id,
                        start_time=float(session_idx * 100),
                        end_time=float(session_idx * 100 + 64),
                        stimulus_id=f"stim-{session_idx:02d}",
                        path=f"synthetic://{record_id}",
                        metadata={
                            "synthetic": True,
                            "split_stage": "recording_manifest",
                            "sampling_rate": rates[modality],
                            "source_hash": _stable_id(f"source:{record_id}"),
                            "preprocessing_hash": _stable_id(f"prep:{record_id}"),
                            "stimulus_segment_id": f"{subject_id}:{session_id}:stim-{session_idx:02d}:0:64",
                            "missing_modality_mask": {name: name not in missing for name in modalities},
                        },
                    )
                )
    return records


def make_synthetic_multimodal_event_batches(
    n_subjects: int = 8,
    sessions_per_subject: int = 2,
    n_time: int = 64,
    include_unpaired: bool = True,
) -> list[NeuralEventBatch]:
    """Create EEG+fMRI+behavior+stimulus batches with aligned latent structure."""

    batches: list[NeuralEventBatch] = []
    modalities = ("eeg", "fmri", "behavior", "stimulus")
    rates = _multimodal_sampling_rates()
    dims = {"eeg": 6, "fmri": 5, "behavior": 2, "stimulus": 3}
    projections = {modality: _projection(f"multimodal-{modality}", 4, dims[modality]) for modality in modalities}
    for subject_idx in range(n_subjects):
        subject_id = f"sub-{subject_idx:03d}"
        site_id = "site-a" if subject_idx % 2 == 0 else "site-b"
        dataset = "synthetic_multimodal"
        missing = {"fmri"} if include_unpaired and subject_idx == n_subjects - 1 else set()
        for session_idx in range(sessions_per_subject):
            session_id = f"ses-{session_idx:02d}"
            latent = _latent(subject_idx=subject_idx, session_idx=session_idx, n_time=n_time)
            for modality in modalities:
                if modality in missing:
                    continue
                modality_time = max(8, n_time // 4) if modality == "fmri" else n_time
                indices = np.linspace(0, n_time - 1, modality_time).round().astype(int)
                time = (indices.astype(np.float32) / float(rates[modality])) + float(session_idx * 100)
                signal = (latent[indices] @ projections[modality]).astype(np.float32)
                signal += _noise(modality, subject_idx, session_idx, signal.shape)
                mask = np.ones_like(signal, dtype=bool)
                if modality in {"eeg", "fmri"} and signal.size:
                    mask[::7, 0] = False
                stimulus = np.stack(
                    [
                        np.sin(time / 7.0),
                        np.cos(time / 11.0),
                        np.full_like(time, subject_idx / max(1, n_subjects - 1), dtype=np.float32),
                    ],
                    axis=1,
                ).astype(np.float32)
                record_id = f"{dataset}_{site_id}_{subject_id}_{session_id}_{modality}"
                batches.append(
                    NeuralEventBatch(
                        modality=modality,
                        dataset=dataset,
                        subject_id=subject_id,
                        session_id=session_id,
                        site_id=site_id,
                        time=time.astype(np.float32),
                        signal=signal,
                        mask=mask,
                        stimulus_embedding=stimulus,
                        behavior={
                            "response": latent[indices, 0].astype(np.float32),
                            "reaction_time": np.abs(latent[indices, 1]).astype(np.float32),
                        },
                        space_index=np.arange(signal.shape[1]),
                        uncertainty=np.full_like(signal, 0.05, dtype=np.float32),
                        provenance={"source": "synthetic_multimodal", "split_stage": "recording_manifest"},
                        metadata={
                            "record_id": record_id,
                            "recording_id": record_id,
                            "dataset_id": dataset,
                            "task_id": "synthetic_multimodal_translation",
                            "synthetic": True,
                            "source_record_id": record_id,
                            "sampling_rate": rates[modality],
                            "source_hash": _stable_id(f"source:{record_id}"),
                            "preprocessing_hash": _stable_id(f"prep:{record_id}"),
                            "stimulus_id": f"stim-{session_idx:02d}",
                            "stimulus_segment_id": f"{subject_id}:{session_id}:stim-{session_idx:02d}:0:64",
                            "stimulus_alignment": {
                                "stimulus_id": f"stim-{session_idx:02d}",
                                "segment_id": f"{subject_id}:{session_id}:stim-{session_idx:02d}:0:64",
                                "timebase": "seconds",
                            },
                            "behavior_metadata": {"features": ["response", "reaction_time"]},
                            "missing_modality_mask": {name: name not in missing for name in modalities},
                            "available_modalities": [name for name in modalities if name not in missing],
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


def _multimodal_sampling_rates() -> dict[str, float]:
    return {"eeg": 256.0, "fmri": 0.5, "behavior": 64.0, "stimulus": 30.0}


def _stable_id(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()
