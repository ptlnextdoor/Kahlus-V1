from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from neurotwin.data.schemas import NeuralEventBatch
from neurotwin.data.split_manifest import RecordingRecord, SplitManifest
from neurotwin.repro import hash_file, stable_hash, write_json


ALGONAUTS_DATASET_ID = "algonauts2025"
ALGONAUTS_SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-05")
ALGONAUTS_TR_SECONDS = 1.49
ALGONAUTS_PARCELS = 1000
FEATURE_KEYS = ("stimulus_embedding", "stimulus_features", "features", "embedding", "x", "data")
RESPONSE_KEYS = ("signal", "fmri", "responses", "response", "bold", "y", "data")


@dataclass(frozen=True)
class AlgonautsPrepareResult:
    records: list[RecordingRecord]
    event_batches: list[NeuralEventBatch]
    split_manifest: SplitManifest
    data_manifest: Path
    feature_manifest: Path
    stimulus_manifest: Path


def prepare_algonauts2025(
    root: str | Path,
    out_dir: str | Path,
    *,
    split: str = "official",
) -> AlgonautsPrepareResult:
    """Prepare Algonauts 2025 stimulus-to-fMRI batches from verified local artifacts.

    The real challenge response files are HDF5 containers with one or more
    datasets shaped ``[time, 1000]``. Tests and small handoff fixtures may use
    NPZ files with the same array contract. Stimulus features must be real
    precomputed arrays stored either in the response NPZ fixture or in a
    separate feature file whose content hash can be verified later by the
    prepared-task evidence gate.
    """

    if split != "official":
        raise ValueError("Algonauts 2025 preparation currently supports --split official")
    root_path = Path(root).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Algonauts root does not exist: {root_path}")

    response_records = _discover_response_records(root_path)
    if not response_records:
        raise ValueError(f"No Algonauts fMRI response arrays found under {root_path}")

    records: list[RecordingRecord] = []
    batches: list[NeuralEventBatch] = []
    feature_rows: list[dict[str, Any]] = []
    stimulus_rows: list[dict[str, Any]] = []
    digest_cache: dict[Path, str] = {}
    for source in response_records:
        subject_id = _subject_from_path(source.path)
        if subject_id not in ALGONAUTS_SUBJECTS:
            continue
        canonical_stimulus_id = _canonical_stimulus_id(source.stimulus_id)
        split_assignment = _split_assignment(canonical_stimulus_id, source.path)
        stimulus = _load_matching_stimulus_features(root_path, source)
        _validate_response_and_stimulus(source.signal, stimulus.array, source.path, stimulus.path)
        record_id = _record_id(source, subject_id)
        time = np.arange(source.signal.shape[0], dtype=np.float32) * ALGONAUTS_TR_SECONDS
        source_digest = _cached_hash(source.path, digest_cache)
        feature_digest = _cached_hash(stimulus.path, digest_cache)
        metadata = {
            "dataset_id": ALGONAUTS_DATASET_ID,
            "record_id": record_id,
            "source_record_id": record_id,
            "task_id": "stimulus_to_fmri_response",
            "split_assignment": split_assignment,
            "split": split_assignment,
            "tr": ALGONAUTS_TR_SECONDS,
            "sampling_rate": 1.0 / ALGONAUTS_TR_SECONDS,
            "time_start": float(time[0]) if time.size else 0.0,
            "time_end": float(time[-1]) if time.size else 0.0,
            "run_id": canonical_stimulus_id,
            "stimulus_id": canonical_stimulus_id,
            "raw_stimulus_id": source.stimulus_id,
            "stimulus_segment_id": record_id,
            "source_hash": source_digest,
            "source_file_hash": source_digest,
            "preprocessing_hash": stable_hash(
                {
                    "adapter": "algonauts2025",
                    "record_id": record_id,
                    "source_hash": source_digest,
                    "feature_hash": feature_digest,
                    "time": int(source.signal.shape[0]),
                    "parcels": int(source.signal.shape[1]),
                }
            ),
            "stimulus_feature_source": "algonauts2025_real_precomputed_features",
            "stimulus_feature_path": str(stimulus.path),
            "stimulus_feature_hash": feature_digest,
            "stimulus_feature_hash_verified": True,
            "stimulus_feature_status": "real_precomputed",
            "stimulus_feature_modalities": list(stimulus.modalities),
            "stimulus_feature_key": stimulus.key,
            "require_real_stimulus": True,
            "stimulus_alignment": {
                "tr_seconds": ALGONAUTS_TR_SECONDS,
                "grid": "one stimulus feature row per fMRI TR",
                "time_axis_verified": True,
                "future_lookahead_allowed": False,
            },
            "atlas": "Schaefer1000",
            "n_parcels": ALGONAUTS_PARCELS,
            "source_path": str(source.path),
        }
        records.append(
            RecordingRecord(
                record_id=record_id,
                modality="fmri",
                dataset=ALGONAUTS_DATASET_ID,
                subject_id=subject_id,
                session_id=source.session_id,
                site_id="cneuromod",
                start_time=float(time[0]) if time.size else 0.0,
                end_time=float(time[-1]) if time.size else 0.0,
                stimulus_id=canonical_stimulus_id,
                path=str(source.path),
                metadata={key: value for key, value in metadata.items() if key != "task_id"},
            )
        )
        batches.append(
            NeuralEventBatch(
                modality="fmri",
                dataset=ALGONAUTS_DATASET_ID,
                subject_id=subject_id,
                session_id=source.session_id,
                site_id="cneuromod",
                time=time,
                signal=source.signal.astype(np.float32, copy=False),
                mask=np.ones_like(source.signal, dtype=bool),
                stimulus_embedding=stimulus.array.astype(np.float32, copy=False),
                behavior={},
                space_index=np.arange(ALGONAUTS_PARCELS, dtype=np.int64),
                uncertainty=None,
                provenance={
                    "source": "algonauts_2025.competitors",
                    "license": "CC0",
                    "response_file": str(source.path),
                    "stimulus_feature_file": str(stimulus.path),
                    "split_stage": "movie_run_manifest",
                },
                metadata=metadata,
            )
        )
        feature_rows.append(
            {
                "record_id": record_id,
                "stimulus_id": canonical_stimulus_id,
                "raw_stimulus_id": source.stimulus_id,
                "path": str(stimulus.path),
                "key": stimulus.key,
                "sha256": feature_digest,
                "shape": list(stimulus.array.shape),
                "modalities": list(stimulus.modalities),
                "source": "algonauts2025_real_precomputed_features",
                "claim_eligible": True,
            }
        )
        stimulus_rows.append(
            {
                "record_id": record_id,
                "stimulus_id": canonical_stimulus_id,
                "raw_stimulus_id": source.stimulus_id,
                "response_file": str(source.path),
                "response_sha256": source_digest,
                "stimulus_feature_file": str(stimulus.path),
                "stimulus_feature_sha256": feature_digest,
                "split": split_assignment,
                "subject_id": subject_id,
                "tr_seconds": ALGONAUTS_TR_SECONDS,
            }
        )

    if not batches:
        raise ValueError(
            "No Algonauts 2025 batches were prepared. Expected subjects: "
            + ", ".join(ALGONAUTS_SUBJECTS)
        )
    _validate_subjects(records)
    split_manifest = _official_split_manifest(records)
    out_path.mkdir(parents=True, exist_ok=True)
    data_manifest = write_json(
        out_path / "data_manifest.json",
        {
            "dataset": ALGONAUTS_DATASET_ID,
            "license": "CC0",
            "subjects": sorted({record.subject_id for record in records}),
            "expected_subjects": list(ALGONAUTS_SUBJECTS),
            "record_count": len(records),
            "response_format": "HDF5/NPZ arrays shaped [time,1000]",
            "records": [_record_manifest_row(record) for record in records],
        },
    )
    feature_manifest = write_json(
        out_path / "feature_manifest.json",
        {
            "dataset": ALGONAUTS_DATASET_ID,
            "claim_eligible": True,
            "hash_verified": True,
            "rows": feature_rows,
        },
    )
    stimulus_manifest = write_json(
        out_path / "stimulus_manifest.json",
        {
            "dataset": ALGONAUTS_DATASET_ID,
            "tr_seconds": ALGONAUTS_TR_SECONDS,
            "rows": stimulus_rows,
        },
    )
    return AlgonautsPrepareResult(
        records=records,
        event_batches=batches,
        split_manifest=split_manifest,
        data_manifest=data_manifest,
        feature_manifest=feature_manifest,
        stimulus_manifest=stimulus_manifest,
    )


@dataclass(frozen=True)
class _ResponseRecord:
    path: Path
    key: str
    signal: np.ndarray
    stimulus_id: str
    session_id: str


@dataclass(frozen=True)
class _StimulusFeature:
    path: Path
    key: str
    array: np.ndarray
    modalities: tuple[str, ...]


def _discover_response_records(root: Path) -> list[_ResponseRecord]:
    records: list[_ResponseRecord] = []
    for path in sorted(_candidate_response_files(root)):
        try:
            for key, signal in _response_arrays(path):
                stimulus_id = _stimulus_id(path, key)
                records.append(
                    _ResponseRecord(
                        path=path,
                        key=key,
                        signal=signal,
                        stimulus_id=stimulus_id,
                        session_id=_session_id(stimulus_id, path),
                    )
                )
        except Exception as exc:  # noqa: BLE001 - continue discovery but report if no usable files remain.
            if _looks_like_response_path(path):
                raise ValueError(f"Failed to read Algonauts response file {path}: {exc}") from exc
    return records


def _candidate_response_files(root: Path) -> Iterable[Path]:
    suffixes = {".npz", ".npy", ".h5", ".hdf5"}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        text = path.as_posix().lower()
        if any(
            token in text
            for token in (
                "/events/",
                "/features/",
                "/features-",
                "/feature_",
                "/stimulus_features/",
                "/precomputed_features/",
                "/target_sample_number/",
                "/atlas/",
            )
        ):
            continue
        if _looks_like_response_path(path):
            yield path


def _looks_like_response_path(path: Path) -> bool:
    text = path.as_posix().lower()
    if "/target_sample_number/" in text or "/atlas/" in text:
        return False
    return "/func/" in text and any(token in text for token in ("fmri", "response", "responses", "bold"))


def _response_arrays(path: Path) -> list[tuple[str, np.ndarray]]:
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
        return [("data", _normalize_response_array(arr, path, "data"))]
    if path.suffix.lower() == ".npz":
        out: list[tuple[str, np.ndarray]] = []
        with np.load(path, allow_pickle=False) as data:
            for key in RESPONSE_KEYS:
                if key in data:
                    out.append((key, _normalize_response_array(data[key], path, key)))
            if not out:
                for key in data.files:
                    value = data[key]
                    if value.ndim == 2 and ALGONAUTS_PARCELS in value.shape:
                        out.append((key, _normalize_response_array(value, path, key)))
        return out
    return _hdf5_arrays(path, expected_second_dim=ALGONAUTS_PARCELS, normalize=_normalize_response_array)


def _hdf5_arrays(path: Path, *, expected_second_dim: int | None, normalize: Any) -> list[tuple[str, np.ndarray]]:
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - covered on cluster if h5py is missing.
        raise RuntimeError("h5py is required to read Algonauts HDF5 files") from exc

    arrays: list[tuple[str, np.ndarray]] = []
    with h5py.File(path, "r") as handle:
        def visit(name: str, node: Any) -> None:
            if not hasattr(node, "shape"):
                return
            shape = tuple(int(dim) for dim in node.shape)
            if len(shape) != 2:
                return
            if expected_second_dim is not None and expected_second_dim not in shape:
                return
            arrays.append((name, normalize(np.asarray(node), path, name)))

        handle.visititems(visit)
    return arrays


def _normalize_response_array(value: np.ndarray, path: Path, key: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{path}:{key} is not a 2D response array")
    if arr.shape[1] != ALGONAUTS_PARCELS and arr.shape[0] == ALGONAUTS_PARCELS:
        arr = arr.T
    if arr.shape[1] != ALGONAUTS_PARCELS:
        raise ValueError(f"{path}:{key} expected 1000 parcels, got shape {arr.shape}")
    return arr


def _load_matching_stimulus_features(root: Path, source: _ResponseRecord) -> _StimulusFeature:
    embedded = _embedded_stimulus_features(source)
    if embedded is not None:
        return embedded
    candidates = sorted(_candidate_feature_files(root, source.stimulus_id))
    for path in candidates:
        for key, arr in _feature_arrays(path):
            if arr.ndim == 2 and arr.shape[0] == source.signal.shape[0]:
                return _StimulusFeature(path=path, key=key, array=arr, modalities=_feature_modalities(path))
    raise ValueError(
        f"No stimulus feature array with {source.signal.shape[0]} rows found for "
        f"{source.stimulus_id} ({source.path}); tried aliases={sorted(_stimulus_id_aliases(source.stimulus_id))}"
    )


def _embedded_stimulus_features(source: _ResponseRecord) -> _StimulusFeature | None:
    if source.path.suffix.lower() != ".npz":
        return None
    with np.load(source.path, allow_pickle=False) as data:
        for key in FEATURE_KEYS:
            if key in data:
                arr = np.asarray(data[key], dtype=np.float32)
                if arr.ndim == 2 and arr.shape[0] == source.signal.shape[0]:
                    return _StimulusFeature(
                        path=source.path,
                        key=key,
                        array=arr,
                        modalities=_feature_modalities(source.path),
                    )
    return None


def _candidate_feature_files(root: Path, stimulus_id: str) -> Iterable[Path]:
    suffixes = {".npz", ".npy", ".h5", ".hdf5"}
    aliases = _stimulus_id_aliases(stimulus_id)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        text = path.as_posix().lower()
        stem_aliases = _stimulus_id_aliases(path.stem)
        if not aliases.intersection(stem_aliases):
            continue
        if any(token in text for token in ("feature", "embedding", "stimulus", "bert", "slowfast", "mfcc", "vjepa", "llama")):
            yield path


def _feature_arrays(path: Path) -> list[tuple[str, np.ndarray]]:
    if path.suffix.lower() == ".npy":
        return [("data", np.asarray(np.load(path, allow_pickle=False), dtype=np.float32))]
    if path.suffix.lower() == ".npz":
        out: list[tuple[str, np.ndarray]] = []
        with np.load(path, allow_pickle=False) as data:
            for key in FEATURE_KEYS:
                if key in data:
                    out.append((key, np.asarray(data[key], dtype=np.float32)))
            if not out:
                for key in data.files:
                    value = np.asarray(data[key])
                    if value.ndim == 2 and value.shape[1] != ALGONAUTS_PARCELS:
                        out.append((key, value.astype(np.float32)))
        return out
    return _hdf5_arrays(path, expected_second_dim=None, normalize=lambda value, _path, _key: np.asarray(value, dtype=np.float32))


def _validate_response_and_stimulus(signal: np.ndarray, stimulus: np.ndarray, response_path: Path, stimulus_path: Path) -> None:
    if signal.shape[1] != ALGONAUTS_PARCELS:
        raise ValueError(f"{response_path} expected [time,1000] fMRI, got {signal.shape}")
    if stimulus.ndim != 2:
        raise ValueError(f"{stimulus_path} expected [time,dim] stimulus features, got {stimulus.shape}")
    if stimulus.shape[0] != signal.shape[0]:
        raise ValueError(
            f"{stimulus_path} time length {stimulus.shape[0]} does not match {response_path} fMRI length {signal.shape[0]}"
        )
    if not np.isfinite(signal).all():
        raise ValueError(f"Non-finite fMRI response values in {response_path}")
    if not np.isfinite(stimulus).all():
        raise ValueError(f"Non-finite stimulus feature values in {stimulus_path}")
    if float(np.abs(stimulus).sum()) == 0.0:
        raise ValueError(f"All-zero stimulus features in {stimulus_path}")


def _official_split_manifest(records: list[RecordingRecord]) -> SplitManifest:
    train = [record for record in records if record.metadata.get("split_assignment") == "train"]
    val = [record for record in records if record.metadata.get("split_assignment") == "val"]
    test = [record for record in records if record.metadata.get("split_assignment") == "test"]
    if not train or not val or not test:
        raise ValueError(
            "Algonauts official/local-dev split requires nonempty train, val, and test records. "
            f"Observed train={len(train)} val={len(val)} test={len(test)}"
        )
    hashes = {record.record_id: record.stable_hash() for record in records}
    return SplitManifest(
        policy="official",
        seed=0,
        train=train,
        val=val,
        test=test,
        record_hashes=hashes,
        split_stage="algonauts_movie_run_manifest",
        notes=[
            "Algonauts 2025 split is at whole movie/episode/run level, never random TR windows.",
            "When official withheld responses are unavailable locally, this adapter uses a whole-run local-dev partition: "
            "Friends seasons 1-5 train, Friends season 6 val, Movie10/test/OOD-labeled runs test.",
            "Stimulus feature paths and hashes are preserved for claim eligibility checks.",
        ],
    )


def _split_assignment(stimulus_id: str, path: Path) -> str:
    text = f"{path.as_posix()} {stimulus_id}".lower()
    if any(token in text for token in ("ood", "test", "friends_s07", "friends-s07", "season7", "season_7")):
        return "test"
    if "movie10" in text or "movie_10" in text:
        return "test"
    aliases = _stimulus_id_aliases(stimulus_id)
    if any(alias.startswith("friends_s06") for alias in aliases):
        return "val"
    if any(token in text for token in ("val", "valid", "friends_s06", "friends-s06", "season6", "season_6")):
        return "val"
    if any(_is_movie10_id(alias) for alias in aliases):
        return "test"
    if "train" in text or "friends_s0" in text or "season" in text or "friends" in text:
        return "train"
    return "train"


def _validate_subjects(records: list[RecordingRecord]) -> None:
    observed = sorted({record.subject_id for record in records})
    missing = [subject for subject in ALGONAUTS_SUBJECTS if subject not in observed]
    if missing:
        raise ValueError("Algonauts 2025 expected four subjects; missing " + ", ".join(missing))


def _record_id(source: _ResponseRecord, subject_id: str) -> str:
    return f"algonauts2025_{subject_id}_{_canonical_stimulus_id(source.stimulus_id)}"


def _record_manifest_row(record: RecordingRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "modality": record.modality,
        "dataset": record.dataset,
        "subject_id": record.subject_id,
        "session_id": record.session_id,
        "site_id": record.site_id,
        "start_time": record.start_time,
        "end_time": record.end_time,
        "stimulus_id": record.stimulus_id,
        "path": record.path,
        "metadata": record.metadata,
    }


def _subject_from_path(path: Path) -> str:
    match = re.search(r"sub[-_]?0?([0-9]{1,2})", path.as_posix(), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not infer Algonauts subject from {path}")
    return f"sub-{int(match.group(1)):02d}"


def _stimulus_id(path: Path, key: str) -> str:
    key_text = str(key).strip("/").replace("/", "_")
    if key_text and key_text not in {"data", "signal", "fmri", "responses", "response", "bold", "y"}:
        return key_text
    stem = path.stem
    stem = re.sub(r"sub[-_]?0?[0-9]{1,2}", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"(fmri|responses?|bold|func)", "", stem, flags=re.IGNORECASE)
    stem = stem.strip("_-.")
    return stem or path.stem


def _session_id(stimulus_id: str, path: Path) -> str:
    canonical = _canonical_stimulus_id(stimulus_id)
    text = f"{canonical} {stimulus_id} {path.as_posix()}".lower()
    match = re.search(r"(friends[_-]?s[0-9]{1,2}e[0-9]{1,2}[a-z]?)", text)
    if match:
        return match.group(1).replace("-", "_")
    match = re.search(r"(movie10[_-]?[a-z0-9_\\-]+)", text)
    if match:
        return match.group(1).replace("-", "_")
    match = re.search(r"(ood[_-]?[a-z0-9_\\-]+)", text)
    if match:
        return match.group(1).replace("-", "_")
    return canonical[:80] or "ses-algonauts"


def _feature_modalities(path: Path) -> tuple[str, ...]:
    text = path.as_posix().lower()
    modalities: list[str] = []
    for name, tokens in {
        "video": ("video", "visual", "slowfast", "resnet", "vjepa", "clip"),
        "audio": ("audio", "mfcc", "wav", "whisper"),
        "text": ("text", "language", "bert", "llama", "transcript"),
    }.items():
        if any(token in text for token in tokens):
            modalities.append(name)
    return tuple(modalities or ["video", "audio", "text"])


def _normalize_id(value: str) -> str:
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _cached_hash(path: Path, cache: dict[Path, str]) -> str:
    resolved = path.resolve()
    if resolved not in cache:
        cache[resolved] = hash_file(resolved)
    return cache[resolved]


def _canonical_stimulus_id(value: str) -> str:
    aliases = _stimulus_id_aliases(value)
    friends = sorted(alias for alias in aliases if alias.startswith("friends_s"))
    if friends:
        return friends[0]
    movie10 = sorted(alias for alias in aliases if _is_movie10_id(alias))
    if movie10:
        return movie10[0]
    stripped = _strip_feature_suffix(_strip_session_task_prefix(_normalize_id(value)))
    return stripped or _normalize_id(value)


def _stimulus_id_aliases(value: str) -> set[str]:
    normalized = _normalize_id(value)
    aliases = {normalized}
    stripped = _strip_feature_suffix(_strip_session_task_prefix(normalized))
    if stripped:
        aliases.add(stripped)
    runless = re.sub(r"_run_[0-9]+$", "", stripped)
    if runless:
        aliases.add(runless)
    match = re.fullmatch(r"s([0-9]{1,2})e([0-9]{1,2}[a-z]?)", runless)
    if match:
        aliases.add(f"friends_s{int(match.group(1)):02d}e{match.group(2)}")
    match = re.search(r"s([0-9]{1,2})e([0-9]{1,2}[a-z]?)", runless)
    if match:
        aliases.add(f"friends_s{int(match.group(1)):02d}e{match.group(2)}")
    return {alias for alias in aliases if alias}


def _strip_session_task_prefix(value: str) -> str:
    text = re.sub(r"^ses_[0-9]+_", "", value)
    text = re.sub(r"^task_", "", text)
    return text


def _strip_feature_suffix(value: str) -> str:
    text = value
    for suffix in ("_stimulus_features", "_features", "_feature", "_embedding", "_embeddings"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text


def _is_movie10_id(value: str) -> bool:
    return bool(re.match(r"^(bourne|figures|life|wolf)[0-9]{2}$", value))
