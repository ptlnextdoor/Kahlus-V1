from __future__ import annotations

from datetime import date
from typing import Any


REGISTRY_REQUIRED_FIELDS: tuple[str, ...] = (
    "dataset_name",
    "source_url_or_identifier",
    "verification_status",
    "verification_source",
    "metadata_source_url",
    "date_checked",
    "modality",
    "task",
    "population",
    "labels_available",
    "event_annotations_available",
    "subjective_symptom_annotations_available",
    "eeg_montage_and_channels",
    "sampling_rate",
    "file_format",
    "license_and_access_requirements",
    "download_size",
    "split_strategy_feasibility",
    "leakage_risks",
    "neurovisual_relevance_score",
    "adapter_priority",
    "notes",
)

SCORE_DEFINITIONS: dict[int, str] = {
    0: "no visual, seizure, perceptual, or relevant EEG content",
    1: "incidental visual task or generic EEG during visual stimulation",
    2: "EEG with seizure/event annotations, visual ERP, photic stimulation, or perceptual-task structure, but no subjective symptom labels",
    3: "EEG/event data paired with subjective visual symptom labels, aura descriptions, clinician-coded neurovisual episode phenotype, or explicit symptom-event alignment",
}


def build_seed_dataset_registry(*, date_checked: str | None = None) -> dict[str, Any]:
    checked = date_checked or date.today().isoformat()
    entries = [
        _entry(
            date_checked=checked,
            dataset_name="HBN-EEG / EEG Foundation Challenge",
            source_url_or_identifier="https://arxiv.org/abs/2506.19141",
            verification_status="confirmed",
            verification_source="prompt-provided source URL; metadata recorded without bulk download",
            metadata_source_url="https://arxiv.org/abs/2506.19141",
            modality="EEG",
            task="multiple active and passive EEG tasks; cross-task and cross-subject challenge framing",
            population="3,000+ child-to-young-adult participants",
            labels_available="psychopathology factor prediction challenge labels described by source metadata",
            event_annotations_available=False,
            subjective_symptom_annotations_available=False,
            eeg_montage_and_channels="128-channel high-density EEG",
            sampling_rate="metadata source required for exact sampling rate before adapter",
            file_format="challenge metadata; raw files not downloaded",
            license_and_access_requirements="verify challenge data terms before adapter",
            download_size="not downloaded in NV-1",
            split_strategy_feasibility="strong cross-subject generalization benchmark anchor",
            leakage_risks="subject/task leakage must be audited before model use",
            neurovisual_relevance_score=2,
            adapter_priority=1,
            notes="Best cross-subject EEG benchmark anchor, not a neurovisual symptom dataset.",
        ),
        _entry(
            date_checked=checked,
            dataset_name="CHB-MIT Scalp EEG Database",
            source_url_or_identifier="https://physionet.org/content/chbmit/1.0.0/",
            verification_status="confirmed",
            verification_source="prompt-provided PhysioNet source URL; metadata recorded without bulk download",
            metadata_source_url="https://physionet.org/content/chbmit/1.0.0/",
            modality="scalp EEG",
            task="pediatric seizure EEG recordings with annotated seizure onsets and ends",
            population="pediatric subjects; 22 subjects and 23 cases per prompt metadata",
            labels_available="seizure timing annotations",
            event_annotations_available=True,
            subjective_symptom_annotations_available=False,
            eeg_montage_and_channels="scalp EEG; exact montage must be adapter-verified",
            sampling_rate="dataset metadata must be read before adapter",
            file_format="EDF",
            license_and_access_requirements="PhysioNet access terms; no bulk download in NV-1",
            download_size="42.6 GB uncompressed per prompt metadata",
            split_strategy_feasibility="subject-held-out and patient-held-out event-window research splits after adapter review",
            leakage_risks="seizure event adjacency, patient identity, file/session leakage",
            neurovisual_relevance_score=2,
            adapter_priority=2,
            notes="Annotated seizure EEG, but not subjective visual aura or symptom-labelled data.",
        ),
        _entry(
            date_checked=checked,
            dataset_name="TUSZ / TUH EEG Seizure Corpus",
            source_url_or_identifier="https://isip.piconepress.com/projects/nedc/html/tuh_eeg/",
            verification_status="confirmed",
            verification_source="prompt-provided TUH/TUSZ source description; metadata recorded without bulk download",
            metadata_source_url="https://isip.piconepress.com/projects/nedc/html/tuh_eeg/",
            modality="clinical EEG",
            task="seizure/event annotation corpus",
            population="clinical EEG population; exact cohort metadata requires access-term review",
            labels_available="seizure/event annotations",
            event_annotations_available=True,
            subjective_symptom_annotations_available=False,
            eeg_montage_and_channels="clinical EEG; montage varies and must be adapter-audited",
            sampling_rate="varies; must be adapter-verified",
            file_format="EDF-style TUH EEG corpus files after approved access",
            license_and_access_requirements="Temple/TUH access may require registration or agreement",
            download_size="not downloaded in NV-1",
            split_strategy_feasibility="patient-held-out event-window research splits after adapter review",
            leakage_risks="patient/session leakage, annotation adjacency, site/equipment artifacts",
            neurovisual_relevance_score=2,
            adapter_priority=3,
            notes="Large seizure EEG corpus, but symptom-phenotype labels must be verified before scoring as 3.",
        ),
        _entry(
            date_checked=checked,
            dataset_name="OpenNeuro lead: EEG plus pupillometry plus PPG working-memory task",
            source_url_or_identifier="OpenNeuro/NEMAR catalog query required; accession not confirmed",
            verification_status="unverified",
            verification_source="unverified lead retained without hardcoded accession",
            metadata_source_url="UNVERIFIED",
            modality="EEG plus possible pupil/PPG",
            task="working memory task candidate",
            population="candidate college-age or healthy adult population; unverified",
            labels_available="unverified",
            event_annotations_available=False,
            subjective_symptom_annotations_available=False,
            eeg_montage_and_channels="unverified",
            sampling_rate="unverified",
            file_format="BIDS candidate; unverified",
            license_and_access_requirements="must verify accession, license, subject count, modalities, and task",
            download_size="not downloaded in NV-1",
            split_strategy_feasibility="unknown until accession is verified",
            leakage_risks="unknown until accession is verified",
            neurovisual_relevance_score=1,
            adapter_priority=20,
            notes="OpenNeuro IDs from prior notes remain unverified and are not hardcoded as confirmed.",
        ),
        _entry(
            date_checked=checked,
            dataset_name="OpenNeuro lead: EEG plus BDI reward/probabilistic selection task",
            source_url_or_identifier="OpenNeuro/NEMAR catalog query required; accession not confirmed",
            verification_status="unverified",
            verification_source="unverified lead retained without hardcoded accession",
            metadata_source_url="UNVERIFIED",
            modality="EEG",
            task="reward or probabilistic selection task candidate",
            population="healthy adult or college-age candidate; unverified",
            labels_available="BDI or task labels unverified",
            event_annotations_available=False,
            subjective_symptom_annotations_available=False,
            eeg_montage_and_channels="unverified",
            sampling_rate="unverified",
            file_format="BIDS candidate; unverified",
            license_and_access_requirements="must verify accession, license, subject count, modalities, and task",
            download_size="not downloaded in NV-1",
            split_strategy_feasibility="unknown until accession is verified",
            leakage_risks="unknown until accession is verified",
            neurovisual_relevance_score=0,
            adapter_priority=21,
            notes="Relevant to ResearchDock/anhedonia if verified later, not current NV-1 adapter work.",
        ),
        _entry(
            date_checked=checked,
            dataset_name="Private or unprovided personal medical notes",
            source_url_or_identifier="not available to NV-1",
            verification_status="rejected",
            verification_source="personal-data boundary",
            metadata_source_url="REJECTED",
            modality="not applicable",
            task="not applicable",
            population="not applicable",
            labels_available="not used",
            event_annotations_available=False,
            subjective_symptom_annotations_available=False,
            eeg_montage_and_channels="not applicable",
            sampling_rate="not applicable",
            file_format="not applicable",
            license_and_access_requirements="rejected: private personal data boundary",
            download_size="none",
            split_strategy_feasibility="not eligible",
            leakage_risks="PII/private medical context",
            neurovisual_relevance_score=0,
            adapter_priority=999,
            notes="Rejected for NV-1: no private patient data, notes, exact dates, locations, or clinician identifiers are used.",
        ),
    ]
    for entry in entries:
        validate_registry_entry(entry)
    return {
        "schema": "kahlus.nv1.dataset_registry.v1",
        "scope": "NV-1 side branch dataset discovery registry",
        "score_definitions": SCORE_DEFINITIONS,
        "execution": {
            "bulk_dataset_download": False,
            "a100_jobs_launched": False,
            "metadata_only": True,
        },
        "entries": entries,
    }


def validate_registry_entry(entry: dict[str, Any]) -> None:
    missing = [field for field in REGISTRY_REQUIRED_FIELDS if field not in entry]
    if missing:
        raise ValueError(f"missing registry fields: {', '.join(missing)}")
    if entry["verification_status"] not in {"confirmed", "unverified", "rejected"}:
        raise ValueError("verification_status must be confirmed, unverified, or rejected")
    score = entry["neurovisual_relevance_score"]
    if score not in SCORE_DEFINITIONS:
        raise ValueError("neurovisual_relevance_score must be one of 0, 1, 2, 3")
    if int(score) == 3 and not entry["subjective_symptom_annotations_available"]:
        notes = str(entry.get("notes", "")).lower()
        if "clinician-coded neurovisual phenotype" not in notes and "aura descriptions" not in notes:
            raise ValueError("score 3 requires subjective symptom or clinician-coded neurovisual phenotype evidence")
    if not isinstance(entry["adapter_priority"], int) or entry["adapter_priority"] < 1:
        raise ValueError("adapter_priority must be a positive integer")


def build_registry_verification_summary(registry: dict[str, Any]) -> dict[str, Any]:
    entries = list(registry.get("entries", ()))
    counts = {
        "confirmed": sum(1 for entry in entries if entry.get("verification_status") == "confirmed"),
        "unverified": sum(1 for entry in entries if entry.get("verification_status") == "unverified"),
        "rejected": sum(1 for entry in entries if entry.get("verification_status") == "rejected"),
    }
    dates = sorted({str(entry.get("date_checked")) for entry in entries if entry.get("date_checked")})
    top_priorities = [
        _summary_row(entry)
        for entry in sorted(entries, key=lambda item: int(item.get("adapter_priority", 999999)))
        if entry.get("verification_status") == "confirmed"
    ][:3]
    metadata_source_urls = sorted(
        {
            str(entry.get("metadata_source_url"))
            for entry in entries
            if str(entry.get("metadata_source_url", "")).startswith("http")
        }
    )
    openneuro_results = [
        {
            "dataset_name": entry["dataset_name"],
            "verification_status": entry["verification_status"],
            "confirmed_accession": None,
            "metadata_source": entry["source_url_or_identifier"],
            "metadata_source_url": entry["metadata_source_url"],
            "date_checked": entry["date_checked"],
            "notes": entry["notes"],
        }
        for entry in entries
        if "OpenNeuro" in str(entry.get("dataset_name", ""))
    ]
    return {
        "schema": "kahlus.nv1.registry_verification_summary.v1",
        "scope": "NV-1 side branch metadata verification summary",
        "metadata_retrieval_date": dates[-1] if dates else None,
        "execution": {
            "bulk_dataset_download": False,
            "a100_jobs_launched": False,
            "cluster_jobs_launched": False,
            "metadata_only": True,
        },
        "counts": counts,
        "dataset_sources_searched": [
            "prompt-provided HBN-EEG / EEG Foundation Challenge metadata source",
            "prompt-provided PhysioNet CHB-MIT metadata source",
            "prompt-provided TUH/TUSZ metadata source",
            "OpenNeuro/NEMAR lead placeholders retained as unverified; no accession confirmed",
        ],
        "metadata_source_urls": metadata_source_urls,
        "top_adapter_priorities": top_priorities,
        "openneuro_verification_results": openneuro_results,
        "score3_assignments": [
            _summary_row(entry) for entry in entries if entry.get("neurovisual_relevance_score") == 3
        ],
        "rejected_datasets": [
            _summary_row(entry) for entry in entries if entry.get("verification_status") == "rejected"
        ],
    }


def build_metadata_only_adapter_plan(registry: dict[str, Any]) -> dict[str, Any]:
    entries = [
        entry
        for entry in sorted(registry.get("entries", ()), key=lambda item: int(item.get("adapter_priority", 999999)))
        if entry.get("verification_status") == "confirmed" and int(entry.get("adapter_priority", 999999)) <= 3
    ]
    return {
        "schema": "kahlus.nv1.adapter_plan.v1",
        "scope": "metadata-only adapter planning; no adapter implementation in NV-1D",
        "execution": {
            "bulk_dataset_download": False,
            "a100_jobs_launched": False,
            "cluster_jobs_launched": False,
            "adapters_implemented": False,
            "metadata_only": True,
        },
        "local_manifest_required_fields": [
            "record_id",
            "subject_id",
            "session_id",
            "dataset_name",
            "signal_path",
            "sampling_rate",
            "channel_names",
            "event_annotations_path",
            "task_label",
            "license_or_access_confirmation",
        ],
        "planned_adapters": [_planned_adapter_row(entry) for entry in entries],
    }


def build_split_audit_plan(registry: dict[str, Any]) -> dict[str, Any]:
    entries = [
        entry
        for entry in sorted(registry.get("entries", ()), key=lambda item: int(item.get("adapter_priority", 999999)))
        if entry.get("verification_status") == "confirmed" and int(entry.get("adapter_priority", 999999)) <= 3
    ]
    return {
        "schema": "kahlus.nv1.split_audit_plan.v1",
        "scope": "metadata-only split audit planning; no split audit execution without local manifest records",
        "execution": {
            "split_audit_executed": False,
            "baselines_run": False,
            "models_run": False,
            "adapters_implemented": False,
            "bulk_dataset_download": False,
            "a100_jobs_launched": False,
            "cluster_jobs_launched": False,
            "metadata_only": True,
        },
        "global_required_split_keys": [
            "dataset_name",
            "subject_id",
            "session_id",
            "record_id",
            "task_label",
        ],
        "global_leakage_checks": [
            "subject_overlap",
            "session_overlap",
            "record_overlap",
            "event_adjacency_overlap",
            "duplicate_signal_path",
            "task_label_leakage",
        ],
        "global_gates_before_model": [
            "local_manifest_contract_passed",
            "split_audit_executed",
            "baseline_ladder_before_model",
            "claim_gate_passed",
            "evidence_manifest_verified",
        ],
        "dataset_split_plans": [_split_audit_row(entry) for entry in entries],
    }


def build_metadata_query_plan(registry: dict[str, Any]) -> dict[str, Any]:
    unverified_leads = [
        _summary_row(entry) for entry in registry.get("entries", ()) if entry.get("verification_status") == "unverified"
    ]
    required_fields = [
        "accession_number",
        "subject_count",
        "task_description",
        "modalities",
        "population",
        "license_or_access_terms",
        "metadata_source_url",
        "date_checked",
    ]
    return {
        "schema": "kahlus.nv1.metadata_query_plan.v1",
        "scope": "planned metadata discovery only; no live query, no adapter, no download",
        "execution": {
            "bulk_dataset_download": False,
            "a100_jobs_launched": False,
            "cluster_jobs_launched": False,
            "metadata_queries_executed": False,
            "metadata_only": True,
        },
        "required_verification_fields": required_fields,
        "unverified_registry_leads": unverified_leads,
        "query_targets": [
            _query_target(
                target_id="openneuro_nemar_multimodal_working_memory",
                catalog_names=["OpenNeuro", "NEMAR"],
                search_terms=[
                    "EEG pupillometry PPG working memory",
                    "EEG pupil PPG BIDS working memory",
                    "college age EEG pupillometry PPG",
                ],
                required_matches=required_fields,
                notes="Do not confirm an accession until catalog metadata verifies modalities, task, subject count, and license.",
            ),
            _query_target(
                target_id="openneuro_nemar_bdi_reward_selection",
                catalog_names=["OpenNeuro", "NEMAR"],
                search_terms=[
                    "EEG BDI reward probabilistic selection",
                    "EEG depression inventory reward task BIDS",
                    "EEG probabilistic selection task college age",
                ],
                required_matches=required_fields,
                notes="Potential ResearchDock/anhedonia lead only; keep out of NV-1 adapters until independently verified.",
            ),
            _query_target(
                target_id="openneuro_nemar_visual_event_eeg",
                catalog_names=["OpenNeuro", "NEMAR"],
                search_terms=[
                    "EEG photic stimulation visual ERP",
                    "EEG perceptual episode visual task",
                    "EEG migraine aura visual symptoms",
                ],
                required_matches=required_fields
                + ["subjective_symptom_annotations", "event_symptom_alignment"],
                notes="Score 3 is allowed only if symptom descriptions, aura annotations, or clinician-coded phenotype fields are verified.",
            ),
            _query_target(
                target_id="moabb_neurovisual_relevant_eeg",
                catalog_names=["MOABB"],
                search_terms=[
                    "visual ERP EEG",
                    "visual stimulation EEG",
                    "event annotated EEG visual task",
                ],
                required_matches=required_fields,
                notes="Catalog additional MOABB datasets without downloading raw data in this sprint.",
            ),
            _query_target(
                target_id="eegdash_public_multimodal_eeg",
                catalog_names=["EEGDash", "public multimodal EEG catalogs"],
                search_terms=[
                    "EEG fNIRS ECG behavior visual task",
                    "multimodal EEG PPG behavior visual perception",
                    "public EEG ECG behavior perceptual task",
                ],
                required_matches=required_fields,
                notes="Catalog public multimodal leads only after source metadata and access terms are confirmed.",
            ),
        ],
    }


def _query_target(
    *,
    target_id: str,
    catalog_names: list[str],
    search_terms: list[str],
    required_matches: list[str],
    notes: str,
) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "catalog_names": catalog_names,
        "search_terms": search_terms,
        "required_matches": required_matches,
        "verification_status": "planned_not_executed",
        "confirmed_accession": None,
        "adapter_allowed_after_verification": False,
        "notes": notes,
    }


def _planned_adapter_row(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_name": entry["dataset_name"],
        "adapter_priority": entry["adapter_priority"],
        "metadata_source_url": entry["metadata_source_url"],
        "implementation_status": "planned_not_implemented",
        "local_manifest_required_fields": [
            "record_id",
            "subject_id",
            "session_id",
            "signal_path",
            "sampling_rate",
            "channel_names",
            "event_annotations_path",
        ],
        "split_audit_strategy": entry["split_strategy_feasibility"],
        "leakage_risks": entry["leakage_risks"],
        "claim_boundary": "research-only planning; no diagnosis, seizure prediction, epilepsy detection, or clinical triage",
        "next_adapter_gate": (
            "Verify access terms, build local manifest from user-provided paths, run split/leakage audit, "
            "then run baselines before any model work."
        ),
    }


def _split_audit_row(entry: dict[str, Any]) -> dict[str, Any]:
    event_keys = ["event_annotations_path"] if entry.get("event_annotations_available") else []
    return {
        "dataset_name": entry["dataset_name"],
        "adapter_priority": entry["adapter_priority"],
        "metadata_source_url": entry["metadata_source_url"],
        "implementation_status": "planned_not_executed",
        "required_split_keys": [
            "dataset_name",
            "subject_id",
            "session_id",
            "record_id",
            "task_label",
            *event_keys,
        ],
        "split_strategy": entry["split_strategy_feasibility"],
        "leakage_checks": [
            "subject_overlap",
            "session_overlap",
            "record_overlap",
            "event_adjacency_overlap",
            "duplicate_signal_path",
            "task_label_leakage",
        ],
        "gates_before_model": [
            "local_manifest_contract_passed",
            "split_audit_executed",
            "baseline_ladder_before_model",
            "claim_gate_passed",
            "evidence_manifest_verified",
        ],
        "baseline_requirement": "persistence, ridge, and other baseline ladder results must exist before model work",
        "claim_boundary": "research-only split planning; no diagnosis, seizure prediction, epilepsy detection, or clinical triage",
    }


def _summary_row(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_name": entry["dataset_name"],
        "verification_status": entry["verification_status"],
        "adapter_priority": entry["adapter_priority"],
        "neurovisual_relevance_score": entry["neurovisual_relevance_score"],
        "source_url_or_identifier": entry["source_url_or_identifier"],
        "metadata_source_url": entry["metadata_source_url"],
        "date_checked": entry["date_checked"],
        "notes": entry["notes"],
    }


def _entry(**values: Any) -> dict[str, Any]:
    return dict(values)
