from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from neurotwin.data.audit import audit_split_manifest
from neurotwin.data.event_io import save_event_batches
from neurotwin.data.manifest_io import save_data_manifest, save_split_manifest
from neurotwin.data.prepared_tasks import prepared_windows_by_split
from neurotwin.data.schemas import NeuralEventBatch
from neurotwin.data.split_manifest import RecordingRecord, SplitManifest, build_split_manifest
from neurotwin.repro import manifest_hash, write_json


REQUIRED_DATASETS = (
    "sleep_edf_expanded",
    "chb_mit_physionet",
    "eegmmi_physionet",
    "siena_scalp_eeg",
)


class MissingOptionalDependency(RuntimeError):
    """Raised when EDF support is unavailable."""


@dataclass(frozen=True)
class MultiDatasetRegistryEntry:
    dataset_id: str
    name: str
    official_url: str
    citation: str
    access_class: str
    license_or_terms: str
    raw_path_policy: str
    processed_path_policy: str
    subject_key: str
    session_key: str
    checksum_status: str
    kaggle_mirror_note: str | None = None


@dataclass(frozen=True)
class DatasetLoadResult:
    dataset_id: str
    root: str
    records: list[RecordingRecord]
    batches: list[NeuralEventBatch]
    report: dict[str, Any] = field(default_factory=dict)


def multidataset_registry() -> dict[str, MultiDatasetRegistryEntry]:
    return {
        "sleep_edf_expanded": MultiDatasetRegistryEntry(
            dataset_id="sleep_edf_expanded",
            name="Sleep-EDF Expanded",
            official_url="https://physionet.org/content/sleep-edfx/1.0.0/",
            citation="Kemp et al., IEEE Transactions on Biomedical Engineering, 2000; PhysioNet Sleep-EDF Expanded.",
            access_class="public_physionet",
            license_or_terms="PhysioNet credential/terms apply; verify before cluster download.",
            raw_path_policy="external_root_only_no_raw_edf_committed",
            processed_path_policy="prepared_npz_and_manifests_only",
            subject_key="sleep_edf_subject_from_psg_filename",
            session_key="psg_night_file_stem",
            checksum_status="computed_only_when_files_are_available",
            kaggle_mirror_note="Kaggle mirrors may exist, but official PhysioNet is the provenance source.",
        ),
        "chb_mit_physionet": MultiDatasetRegistryEntry(
            dataset_id="chb_mit_physionet",
            name="CHB-MIT Scalp EEG Database",
            official_url="https://physionet.org/content/chbmit/1.0.0/",
            citation="Goldberger et al., Circulation, 2000; CHB-MIT Scalp EEG Database on PhysioNet.",
            access_class="public_physionet",
            license_or_terms="PhysioNet credential/terms apply; verify before cluster download.",
            raw_path_policy="external_root_only_no_raw_edf_committed",
            processed_path_policy="prepared_npz_and_manifests_only",
            subject_key="patient_folder_with_chb21_canonicalized_to_chb01",
            session_key="edf_file_stem",
            checksum_status="computed_only_when_files_are_available",
            kaggle_mirror_note="Kaggle mirrors may exist, but official PhysioNet is the provenance source.",
        ),
        "eegmmi_physionet": MultiDatasetRegistryEntry(
            dataset_id="eegmmi_physionet",
            name="EEG Motor Movement/Imagery Database",
            official_url="https://physionet.org/content/eegmmidb/1.0.0/",
            citation="Schalk et al., IEEE TBME, 2004; EEGMMI Database on PhysioNet.",
            access_class="public_physionet",
            license_or_terms="PhysioNet credential/terms apply; verify before cluster download.",
            raw_path_policy="external_root_only_no_raw_edf_committed",
            processed_path_policy="prepared_npz_and_manifests_only",
            subject_key="S### subject folder",
            session_key="S###R## run file stem",
            checksum_status="computed_only_when_files_are_available",
            kaggle_mirror_note="Kaggle mirrors may exist, but official PhysioNet is the provenance source.",
        ),
        "siena_scalp_eeg": MultiDatasetRegistryEntry(
            dataset_id="siena_scalp_eeg",
            name="Siena Scalp EEG Database",
            official_url="https://physionet.org/content/siena-scalp-eeg/1.0.0/",
            citation="Detti et al.; Siena Scalp EEG Database on PhysioNet.",
            access_class="public_physionet",
            license_or_terms="PhysioNet credential/terms apply; verify before cluster download.",
            raw_path_policy="external_root_only_no_raw_edf_committed",
            processed_path_policy="prepared_npz_and_manifests_only",
            subject_key="patient_folder",
            session_key="edf_file_stem",
            checksum_status="computed_only_when_files_are_available",
            kaggle_mirror_note="Kaggle mirrors may exist, but official PhysioNet is the provenance source.",
        ),
    }


def prepare_multidataset_a100_evidence(
    root: str | Path | None,
    out_dir: str | Path,
    *,
    seed: int = 0,
    max_records_per_dataset: int | None = None,
    max_samples_per_record: int = 16384,
    max_channels: int = 32,
    window_length: int = 128,
    stride: int = 128,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    registry = multidataset_registry()
    discovered = discover_multidataset_roots(root)
    load_results: list[DatasetLoadResult] = []
    missing: list[str] = []
    errors: dict[str, str] = {}

    for dataset_id in REQUIRED_DATASETS:
        dataset_root = discovered.get(dataset_id)
        if dataset_root is None:
            missing.append(dataset_id)
            continue
        try:
            load_results.append(
                _load_dataset(
                    dataset_id,
                    dataset_root,
                    max_records=max_records_per_dataset,
                    max_samples_per_record=max_samples_per_record,
                    max_channels=max_channels,
                )
            )
        except Exception as exc:  # noqa: BLE001 - evidence package should report per-dataset adapter failures.
            errors[dataset_id] = str(exc)

    records = [record for result in load_results for record in result.records]
    batches = _harmonize_space([batch for result in load_results for batch in result.batches])
    if not records or not batches:
        message = {
            "error": "no supported public EEG dataset roots produced records",
            "required_datasets": list(REQUIRED_DATASETS),
            "missing_dataset_roots": missing,
            "adapter_errors": errors,
        }
        write_json(out / "multidataset_evidence_gate.json", message)
        raise ValueError(json.dumps(message, sort_keys=True))

    split_manifest = build_split_manifest(records, policy="subject", seed=seed)
    split_manifest.notes.extend(
        [
            "Multi-dataset evidence split is subject/patient-held-out before windowing.",
            "Sleep-EDF night-held-out is secondary only; subject-held-out is the primary gate.",
            "CHB-MIT chb21 is canonicalized to chb01 for subject leakage control.",
            "Siena supports CHB-MIT to Siena transfer reporting when both datasets are present.",
        ]
    )
    split_path = save_split_manifest(split_manifest, out / "split_manifest.json")
    data_path = save_data_manifest(records, out / "data_manifest.json")
    split_audit = audit_split_manifest(split_manifest, "subject")
    source_violations = _source_record_split_violations(split_manifest)
    manifest_digest = manifest_hash([record.__dict__ for record in split_manifest.all_records])
    event_path = save_event_batches(
        batches,
        out,
        manifest_metadata={
            "dataset": "multidataset_a100",
            "split_policy": "subject",
            "manifest_hash": manifest_digest,
            "split_manifest": str(split_path),
            "data_manifest": str(data_path),
            "window_length": window_length,
            "stride": stride,
            "required_datasets": list(REQUIRED_DATASETS),
            "supported_datasets": [result.dataset_id for result in load_results],
            "claim_scope": "multidataset_eeg_forecasting_completion_benchmark_ready",
        },
    )
    window_counts = _window_counts(batches, split_manifest, window_length=window_length, stride=stride)
    normalization = _normalization_report(batches)
    transfer_manifest_path = _write_chb_to_siena_transfer_manifest(records, out, seed=seed)
    supported = [result.dataset_id for result in load_results]
    missing_required = [dataset_id for dataset_id in REQUIRED_DATASETS if dataset_id not in supported]
    gate = {
        "claim_scope": "multidataset_eeg_forecasting_completion_benchmark_ready",
        "passed": not missing_required
        and split_audit.passed
        and not source_violations
        and all(window_counts.get(split, 0) > 0 for split in ("train", "val", "test")),
        "required_datasets": list(REQUIRED_DATASETS),
        "supported_datasets": supported,
        "missing_required_datasets": missing_required,
        "missing_dataset_roots": missing,
        "adapter_errors": errors,
        "blocked_claims": [
            "clinical seizure prediction",
            "sleep diagnosis",
            "treatment prediction",
            "brain foundation model",
            "recovery or diagnosis claim",
        ],
        "split_audit_passed": split_audit.passed,
        "split_audit_violations": list(split_audit.violations),
        "source_record_split_violations": source_violations,
        "window_counts_by_split": window_counts,
    }
    provenance = {
        dataset_id: asdict(entry) for dataset_id, entry in registry.items()
    } | {
        "kaggle_policy": "Kaggle is a convenience mirror only; official PhysioNet/other dataset hosts remain ground truth."
    }
    summary = {
        "registry": provenance,
        "load_reports": [result.report for result in load_results],
        "event_manifest": str(event_path),
        "split_manifest": str(split_path),
        "data_manifest": str(data_path),
        "normalization_report": normalization,
        "transfer_manifest": str(transfer_manifest_path) if transfer_manifest_path else None,
        "gate": gate,
    }
    write_json(out / "dataset_registry.json", provenance)
    write_json(out / "channel_sampling_normalization_report.json", normalization)
    write_json(out / "multidataset_evidence_gate.json", gate)
    write_json(out / "multidataset_evidence_bundle_manifest.json", summary)
    _write_markdown_report(out / "README_MULTIDATASET_EVIDENCE.md", summary)
    return summary


def discover_multidataset_roots(root: str | Path | None) -> dict[str, Path]:
    env = {
        "sleep_edf_expanded": os.environ.get("SLEEP_EDF_ROOT"),
        "chb_mit_physionet": os.environ.get("CHB_MIT_ROOT"),
        "eegmmi_physionet": os.environ.get("EEGMMI_ROOT"),
        "siena_scalp_eeg": os.environ.get("SIENA_ROOT"),
    }
    discovered: dict[str, Path] = {}
    for dataset_id, value in env.items():
        if value and Path(value).expanduser().exists():
            discovered[dataset_id] = Path(value).expanduser().resolve()

    if root is None:
        return discovered
    base = Path(root).expanduser().resolve()
    if not base.exists():
        return discovered
    candidates = {
        "sleep_edf_expanded": ("sleep_edf_expanded", "sleep-edf-expanded", "sleep-edfx", "sleep-edf", "sleep"),
        "chb_mit_physionet": ("chb_mit", "chb-mit", "chbmit", "chb_mit_physionet"),
        "eegmmi_physionet": ("eegmmi", "eegmmidb", "eeg-motor-movement-imagery", "eegmmi_physionet"),
        "siena_scalp_eeg": ("siena", "siena-scalp-eeg", "siena_scalp_eeg"),
    }
    for dataset_id, names in candidates.items():
        if dataset_id in discovered:
            continue
        for name in names:
            candidate = base / name
            if candidate.exists():
                discovered[dataset_id] = candidate.resolve()
                break
    return discovered


def load_sleep_edf(
    root: str | Path,
    *,
    max_records: int | None = None,
    max_samples_per_record: int = 16384,
    max_channels: int = 32,
) -> DatasetLoadResult:
    dataset_id = "sleep_edf_expanded"
    root_path = Path(root)
    psg_files = sorted(root_path.rglob("*PSG.edf"))
    if max_records is not None:
        psg_files = psg_files[:max_records]
    records: list[RecordingRecord] = []
    batches: list[NeuralEventBatch] = []
    for psg in psg_files:
        session = psg.stem.replace("-PSG", "")
        subject = f"sub-{session[:6]}"
        hypnogram = _find_sleep_hypnogram(psg)
        annotations = _read_edf_annotations(hypnogram) if hypnogram else []
        new_records, new_batches = _edf_to_records_and_batches(
            psg,
            dataset_id=dataset_id,
            subject_id=subject,
            session_id=f"ses-{session}",
            site_id="physionet_sleep_edf",
            task_id="sleep_edf_recording",
            channel_policy="sleep",
            annotations=annotations,
            max_samples_per_record=max_samples_per_record,
            max_channels=max_channels,
            extra_metadata={"hypnogram_path": str(hypnogram) if hypnogram else None},
        )
        records.extend(new_records)
        batches.extend(new_batches)
    return DatasetLoadResult(
        dataset_id=dataset_id,
        root=str(root_path),
        records=records,
        batches=batches,
        report=_load_report(dataset_id, root_path, records, batches),
    )


def load_chb_mit(
    root: str | Path,
    *,
    max_records: int | None = None,
    max_samples_per_record: int = 16384,
    max_channels: int = 32,
) -> DatasetLoadResult:
    from neurotwin.stf.chb_mit import parse_chb_mit_summary_dir

    dataset_id = "chb_mit_physionet"
    root_path = Path(root)
    seizure_intervals = parse_chb_mit_summary_dir(root_path)
    edfs = _chb_edf_files(root_path)
    if max_records is not None:
        edfs = edfs[:max_records]
    records: list[RecordingRecord] = []
    batches: list[NeuralEventBatch] = []
    for edf in edfs:
        patient = _canonical_chb_patient(edf.parent.name)
        session = edf.stem
        intervals = seizure_intervals.get(edf.name, ())
        new_records, new_batches = _edf_to_records_and_batches(
            edf,
            dataset_id=dataset_id,
            subject_id=f"sub-{patient}",
            session_id=f"ses-{session}",
            site_id="physionet_chb_mit",
            task_id="chb_mit_recording",
            channel_policy="eeg_only",
            annotations=_seizure_annotations(intervals),
            seizure_intervals=intervals,
            max_samples_per_record=max_samples_per_record,
            max_channels=max_channels,
            extra_metadata={
                "patient_id": patient,
                "original_patient_folder": edf.parent.name,
                "chb21_canonicalized_to_chb01": edf.parent.name == "chb21",
            },
        )
        records.extend(new_records)
        batches.extend(new_batches)
    return DatasetLoadResult(
        dataset_id=dataset_id,
        root=str(root_path),
        records=records,
        batches=batches,
        report=_load_report(dataset_id, root_path, records, batches),
    )


def load_eegmmi(
    root: str | Path,
    *,
    max_records: int | None = None,
    max_samples_per_record: int = 16384,
    max_channels: int = 32,
) -> DatasetLoadResult:
    dataset_id = "eegmmi_physionet"
    root_path = Path(root)
    edfs = sorted(root_path.rglob("S*/S*R*.edf"))
    if not edfs:
        edfs = sorted(root_path.rglob("*.edf"))
    if max_records is not None:
        edfs = edfs[:max_records]
    records: list[RecordingRecord] = []
    batches: list[NeuralEventBatch] = []
    for edf in edfs:
        subject = edf.parent.name if edf.parent.name.startswith("S") else edf.stem.split("R", 1)[0]
        new_records, new_batches = _edf_to_records_and_batches(
            edf,
            dataset_id=dataset_id,
            subject_id=f"sub-{subject}",
            session_id=f"run-{edf.stem}",
            site_id="physionet_eegmmi",
            task_id="eegmmi_motor_imagery",
            channel_policy="eeg_only",
            annotations=_read_edf_annotations(edf),
            max_samples_per_record=max_samples_per_record,
            max_channels=max_channels,
            extra_metadata={"edfplus_task_annotations": True},
        )
        records.extend(new_records)
        batches.extend(new_batches)
    return DatasetLoadResult(
        dataset_id=dataset_id,
        root=str(root_path),
        records=records,
        batches=batches,
        report=_load_report(dataset_id, root_path, records, batches),
    )


def load_siena(
    root: str | Path,
    *,
    max_records: int | None = None,
    max_samples_per_record: int = 16384,
    max_channels: int = 32,
) -> DatasetLoadResult:
    dataset_id = "siena_scalp_eeg"
    root_path = Path(root)
    intervals = _parse_generic_seizure_texts(root_path)
    edfs = sorted(root_path.rglob("*.edf"))
    if max_records is not None:
        edfs = edfs[:max_records]
    records: list[RecordingRecord] = []
    batches: list[NeuralEventBatch] = []
    for edf in edfs:
        patient = edf.parent.name
        session = edf.stem
        seizure_intervals = intervals.get(edf.name, ())
        annotations = [*_read_edf_annotations(edf), *_seizure_annotations(seizure_intervals)]
        new_records, new_batches = _edf_to_records_and_batches(
            edf,
            dataset_id=dataset_id,
            subject_id=f"sub-{patient}",
            session_id=f"ses-{session}",
            site_id="physionet_siena",
            task_id="siena_recording",
            channel_policy="siena",
            annotations=annotations,
            seizure_intervals=seizure_intervals,
            max_samples_per_record=max_samples_per_record,
            max_channels=max_channels,
            extra_metadata={"patient_id": patient, "ecg_ekg_supported_if_present": True},
        )
        records.extend(new_records)
        batches.extend(new_batches)
    return DatasetLoadResult(
        dataset_id=dataset_id,
        root=str(root_path),
        records=records,
        batches=batches,
        report=_load_report(dataset_id, root_path, records, batches),
    )


def _load_dataset(
    dataset_id: str,
    root: Path,
    *,
    max_records: int | None,
    max_samples_per_record: int,
    max_channels: int,
) -> DatasetLoadResult:
    loaders = {
        "sleep_edf_expanded": load_sleep_edf,
        "chb_mit_physionet": load_chb_mit,
        "eegmmi_physionet": load_eegmmi,
        "siena_scalp_eeg": load_siena,
    }
    return loaders[dataset_id](
        root,
        max_records=max_records,
        max_samples_per_record=max_samples_per_record,
        max_channels=max_channels,
    )


def _edf_to_records_and_batches(
    edf_path: Path,
    *,
    dataset_id: str,
    subject_id: str,
    session_id: str,
    site_id: str,
    task_id: str,
    channel_policy: str,
    annotations: list[dict[str, Any]],
    max_samples_per_record: int,
    max_channels: int,
    seizure_intervals: Iterable[tuple[float, float]] = (),
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[list[RecordingRecord], list[NeuralEventBatch]]:
    edfio = _require_edfio()
    edf = edfio.read_edf(edf_path, lazy_load_data=True)
    grouped: dict[tuple[str, float], list[Any]] = {}
    for signal in edf.signals:
        modality = _modality_for_label(signal.label, channel_policy)
        if modality is None:
            continue
        grouped.setdefault((modality, float(signal.sampling_frequency)), []).append(signal)

    records: list[RecordingRecord] = []
    batches: list[NeuralEventBatch] = []
    for (modality, sampling_rate), signals in sorted(grouped.items(), key=lambda item: item[0]):
        kept = signals[:max_channels]
        if not kept:
            continue
        n_samples = min(max_samples_per_record, min(int(signal.data.shape[0]) for signal in kept))
        if n_samples < 2:
            continue
        signal_matrix = np.stack([np.asarray(signal.data[:n_samples], dtype=np.float32) for signal in kept], axis=1)
        time = (np.arange(n_samples, dtype=np.float32) / np.float32(sampling_rate)).astype(np.float32)
        mask = np.isfinite(signal_matrix).astype(np.float32)
        signal_matrix = np.nan_to_num(signal_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        record_id = f"{dataset_id}:{edf_path.stem}:{modality}:{int(round(sampling_rate))}"
        metadata = {
            "record_id": record_id,
            "source_record_id": f"{dataset_id}:{edf_path.stem}",
            "source_path": str(edf_path),
            "channel_names": [str(signal.label) for signal in kept],
            "omitted_channel_count": max(0, len(signals) - len(kept)),
            "sampling_rate": sampling_rate,
            "checksum_status": "not_computed_raw_external",
            "annotations": annotations[:128],
            "annotation_count": len(annotations),
            "seizure_interval_count": len(tuple(seizure_intervals)),
            "clinical_claim_allowed": False,
            **(extra_metadata or {}),
        }
        batch = NeuralEventBatch(
            modality=modality,
            dataset=dataset_id,
            subject_id=subject_id,
            session_id=session_id,
            site_id=site_id,
            time=time,
            signal=signal_matrix,
            mask=mask,
            stimulus_embedding=None,
            behavior={},
            space_index=np.arange(signal_matrix.shape[1], dtype=np.int64),
            provenance={"source": "external_public_edf", "path": str(edf_path), "adapter": "multidataset"},
            metadata=metadata,
        )
        records.append(
            RecordingRecord(
                record_id=record_id,
                modality=modality,
                dataset=dataset_id,
                subject_id=subject_id,
                session_id=session_id,
                site_id=site_id,
                start_time=0.0,
                end_time=float(time[-1]),
                path=str(edf_path),
                metadata={
                    "source_record_id": metadata["source_record_id"],
                    "task_id": task_id,
                    "modality": modality,
                    "sampling_rate": sampling_rate,
                    "checksum_status": metadata["checksum_status"],
                },
            )
        )
        batches.append(batch)
    return records, batches


def _require_edfio() -> Any:
    try:
        import edfio  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional runtime.
        raise MissingOptionalDependency("Install edfio to prepare public EDF datasets") from exc
    return edfio


def _read_edf_annotations(edf_path: Path | None) -> list[dict[str, Any]]:
    if edf_path is None or not edf_path.exists():
        return []
    edfio = _require_edfio()
    try:
        edf = edfio.read_edf(edf_path, lazy_load_data=True)
    except Exception:
        return []
    rows = []
    for annotation in getattr(edf, "annotations", []):
        rows.append(
            {
                "onset": float(annotation.onset),
                "duration": None if annotation.duration is None else float(annotation.duration),
                "text": str(annotation.text),
            }
        )
    return rows


def _find_sleep_hypnogram(psg_path: Path) -> Path | None:
    exact = psg_path.with_name(psg_path.name.replace("-PSG.edf", "-Hypnogram.edf"))
    if exact.exists():
        return exact
    prefix = psg_path.name[:6]
    matches = sorted(psg_path.parent.glob(f"{prefix}*Hypnogram.edf"))
    return matches[0] if matches else None


def _chb_edf_files(root: Path) -> list[Path]:
    records_file = root / "RECORDS"
    if records_file.exists():
        rows = []
        for line in records_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.endswith(".edf"):
                rows.append(root / line)
        existing = [path for path in rows if path.exists()]
        if existing:
            return sorted(existing)
    return sorted(root.rglob("chb*.edf"))


def _canonical_chb_patient(patient: str) -> str:
    return "chb01" if patient == "chb21" else patient


def _seizure_annotations(intervals: Iterable[tuple[float, float]]) -> list[dict[str, Any]]:
    return [
        {"onset": float(start), "duration": float(end) - float(start), "text": "seizure_interval"}
        for start, end in intervals
    ]


def _parse_generic_seizure_texts(root: Path) -> dict[str, tuple[tuple[float, float], ...]]:
    intervals: dict[str, list[tuple[float, float]]] = {}
    current_file: str | None = None
    start: float | None = None
    file_re = re.compile(r"(?:file name|filename)\s*[:=]\s*(\S+\.edf)", re.IGNORECASE)
    start_re = re.compile(r"(?:start time|seizure start(?: time)?)\s*[:=]\s*([0-9.]+)", re.IGNORECASE)
    end_re = re.compile(r"(?:end time|seizure end(?: time)?)\s*[:=]\s*([0-9.]+)", re.IGNORECASE)
    for text_file in sorted(root.rglob("*.txt")):
        for line in text_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            file_match = file_re.search(line)
            if file_match:
                current_file = Path(file_match.group(1)).name
            start_match = start_re.search(line)
            if start_match:
                start = float(start_match.group(1))
            end_match = end_re.search(line)
            if current_file and start is not None and end_match:
                end = float(end_match.group(1))
                if end > start:
                    intervals.setdefault(current_file, []).append((start, end))
                start = None
    return {key: tuple(value) for key, value in intervals.items()}


def _modality_for_label(label: str, channel_policy: str) -> str | None:
    lowered = label.lower()
    if channel_policy == "sleep":
        if "eeg" in lowered or lowered.startswith(("fp", "c3", "c4", "o1", "o2")):
            return "eeg"
        if any(token in lowered for token in ("eog", "emg", "ecg", "ekg", "resp", "airflow", "oro", "nasal", "temp")):
            return "clinical"
        return None
    if channel_policy == "siena" and any(token in lowered for token in ("ecg", "ekg")):
        return "clinical"
    return "eeg"


def _harmonize_space(batches: list[NeuralEventBatch]) -> list[NeuralEventBatch]:
    mins: dict[str, int] = {}
    for batch in batches:
        mins[batch.modality] = min(mins.get(batch.modality, batch.n_space), batch.n_space)
    harmonized: list[NeuralEventBatch] = []
    for batch in batches:
        n_space = mins[batch.modality]
        if batch.n_space == n_space:
            harmonized.append(batch)
            continue
        metadata = dict(batch.metadata)
        metadata["space_harmonized_to_channels"] = n_space
        metadata["original_n_space"] = batch.n_space
        if "channel_names" in metadata:
            metadata["channel_names"] = list(metadata["channel_names"])[:n_space]
        harmonized.append(
            NeuralEventBatch(
                modality=batch.modality,
                dataset=batch.dataset,
                subject_id=batch.subject_id,
                session_id=batch.session_id,
                site_id=batch.site_id,
                time=batch.time,
                signal=batch.signal[:, :n_space],
                mask=batch.mask[:, :n_space],
                space_index=batch.space_index[:n_space],
                stimulus_embedding=batch.stimulus_embedding,
                behavior=batch.behavior,
                uncertainty=batch.uncertainty[:, :n_space] if batch.uncertainty is not None else None,
                provenance=batch.provenance,
                metadata=metadata,
            )
        )
    return harmonized


def _window_counts(batches: list[NeuralEventBatch], split_manifest: SplitManifest, *, window_length: int, stride: int) -> dict[str, int]:
    windows = prepared_windows_by_split(batches, split_manifest, window_length=window_length, stride=stride)
    return {split: len(items) for split, items in windows.items()}


def _normalization_report(batches: list[NeuralEventBatch]) -> dict[str, Any]:
    rows = []
    for batch in batches:
        rows.append(
            {
                "dataset": batch.dataset,
                "modality": batch.modality,
                "recording_id": batch.recording_id,
                "n_time": batch.n_time,
                "n_space": batch.n_space,
                "sampling_rate": batch.sampling_rate,
                "channels": list(batch.metadata.get("channel_names", [])),
            }
        )
    return {
        "policy": "truncate_to_common_channel_count_per_modality; no raw EDF copied",
        "rows": rows,
        "sampling_rates_by_dataset_modality": _sampling_rates_by_dataset_modality(batches),
    }


def _sampling_rates_by_dataset_modality(batches: list[NeuralEventBatch]) -> dict[str, list[float]]:
    values: dict[str, set[float]] = {}
    for batch in batches:
        values.setdefault(f"{batch.dataset}:{batch.modality}", set()).add(float(batch.sampling_rate or 0.0))
    return {key: sorted(value) for key, value in values.items()}


def _source_record_split_violations(manifest: SplitManifest) -> list[str]:
    by_source: dict[str, set[str]] = {}
    for split, records in (("train", manifest.train), ("val", manifest.val), ("test", manifest.test)):
        for record in records:
            source = str(record.metadata.get("source_record_id", record.record_id))
            by_source.setdefault(source, set()).add(split)
    return [
        f"source_record_id {source!r} appears in splits {','.join(sorted(splits))}"
        for source, splits in sorted(by_source.items())
        if len(splits) > 1
    ]


def _write_chb_to_siena_transfer_manifest(records: list[RecordingRecord], out: Path, *, seed: int) -> Path | None:
    train = [record for record in records if record.dataset == "chb_mit_physionet"]
    test = [record for record in records if record.dataset == "siena_scalp_eeg"]
    if not train or not test:
        return None
    manifest = SplitManifest(
        policy="dataset",
        seed=seed,
        train=train,
        val=[],
        test=test,
        record_hashes={record.record_id: record.stable_hash() for record in [*train, *test]},
        notes=[
            "Transfer manifest only: CHB-MIT train to Siena test.",
            "No clinical seizure prediction claim is permitted from this manifest alone.",
        ],
    )
    return save_split_manifest(manifest, out / "chbmit_to_siena_transfer_split_manifest.json")


def _load_report(
    dataset_id: str,
    root: Path,
    records: list[RecordingRecord],
    batches: list[NeuralEventBatch],
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "root": str(root),
        "record_count": len(records),
        "event_batch_count": len(batches),
        "subjects": sorted({record.subject_id for record in records}),
        "modalities": sorted({batch.modality for batch in batches}),
        "sessions": sorted({record.session_id for record in records})[:32],
    }


def _write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    gate = summary["gate"]
    lines = [
        "# Kahlus Multi-Dataset A100 Evidence Package",
        "",
        "This is a public-dataset benchmark package. It is not a clinical seizure predictor, sleep diagnostic system, treatment predictor, or brain foundation model.",
        "",
        f"- gate_passed: {gate['passed']}",
        f"- claim_scope: {gate['claim_scope']}",
        f"- supported_datasets: {', '.join(gate['supported_datasets'])}",
        f"- missing_required_datasets: {', '.join(gate['missing_required_datasets']) or 'none'}",
        f"- window_counts_by_split: {gate['window_counts_by_split']}",
        "",
        "## Expected Baselines",
        "",
        "Run the prepared baseline suite with persistence, ridge, AR/VAR-style autoregressive ridge, TinySSM, TCN/Transformer when available, shuffled-target, time-shift, and patient/session nuisance controls before interpreting any model result.",
        "",
        "## Official Data Sources",
        "",
    ]
    registry = summary["registry"]
    for dataset_id in REQUIRED_DATASETS:
        entry = registry[dataset_id]
        lines.append(f"- {dataset_id}: {entry['official_url']} ({entry['access_class']})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
