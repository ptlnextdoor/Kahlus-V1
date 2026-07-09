"""RD-3 public dataset review for ResearchDock.

This module is a source-backed review artifact, not a loader. It records what
can be safely mapped from public affect datasets into the ResearchDock roadmap
without downloading data, committing raw participant records, or claiming
diagnostic utility.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResearchDockPublicDatasetReview:
    dataset_id: str
    display_name: str
    official_urls: tuple[str, ...]
    citation: str
    citation_status: str
    access_status: str
    license_status: str
    modalities: tuple[str, ...]
    task_labels: tuple[str, ...]
    researchdock_fields: tuple[str, ...]
    researchdock_fit: tuple[str, ...]
    source_notes: tuple[str, ...]
    boundary_notes: tuple[str, ...]
    loader_status: str = "review_only_no_loader"


def researchdock_public_dataset_reviews() -> tuple[ResearchDockPublicDatasetReview, ...]:
    """Return RD-3 public dataset mappings in source-review order."""
    return (
        ResearchDockPublicDatasetReview(
            dataset_id="wesad",
            display_name="WESAD",
            official_urls=(
                "https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection",
                "https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html",
            ),
            citation=(
                "Schmidt, Reiss, Duerichen, Marberger, and Van Laerhoven, "
                "Introducing WESAD, a multimodal dataset for wearable stress and affect detection, ICMI 2018."
            ),
            citation_status="verified_from_uci_and_siegen_pages",
            access_status="public_download_for_scientific_noncommercial_use",
            license_status="scientific_noncommercial_with_credit_per_siegen_dataset_page",
            modalities=(
                "bvp",
                "ecg",
                "eda",
                "emg",
                "respiration",
                "temperature",
                "acceleration",
                "self_report",
            ),
            task_labels=("neutral", "stress", "amusement"),
            researchdock_fields=(
                "ppg_hrv_proxy",
                "stress_recovery_proxy",
                "self_report",
                "motion_quality",
            ),
            researchdock_fit=(
                "Best local review candidate for physiology-only stress and recovery baselines.",
                "Does not provide ResearchDock reward-task behavior or synchronized pupil observations.",
            ),
            source_notes=(
                "UCI lists 15 subjects, wrist and chest physiological sensors, and neutral/stress/amusement states.",
                "Siegen page provides the original download link and noncommercial scientific-use disclaimer.",
            ),
            boundary_notes=(
                "no raw participant data",
                "review artifact only; no raw participant data is downloaded or committed",
                "stress labels are affect-task labels, not diagnosis labels",
            ),
        ),
        ResearchDockPublicDatasetReview(
            dataset_id="deap",
            display_name="DEAP",
            official_urls=(
                "https://www.eecs.qmul.ac.uk/mmv/datasets/deap/",
                "https://doi.org/10.1109/T-AFFC.2011.15",
                "https://research.utwente.nl/en/publications/deap-a-database-for-emotion-analysis-using-physiological-signals/",
            ),
            citation=(
                "Koelstra, Muhl, Soleymani, Lee, Yazdani, Ebrahimi, Pun, Nijholt, and Patras, "
                "DEAP: A Database for Emotion Analysis Using Physiological Signals, IEEE TAC 2012."
            ),
            citation_status="verified_from_publication_record; dataset_page_unavailable_at_review",
            access_status="historical_public_dataset_page_unavailable_at_review",
            license_status="terms_unresolved_until_official_dataset_page_or_access_flow_is_available",
            modalities=("eeg", "peripheral_physiology", "face_video", "self_report", "stimulus_video"),
            task_labels=("music_video_affect", "valence", "arousal", "dominance", "liking", "familiarity"),
            researchdock_fields=(
                "eeg_observation",
                "ppg_hrv_proxy",
                "valence_arousal_self_report",
                "stimulus_context",
            ),
            researchdock_fit=(
                "Useful for affective EEG plus physiology pretraining once access and terms are rechecked.",
                "Does not contain ResearchDock reward, effort, or frustration task structure.",
            ),
            source_notes=(
                "Publication record reports 32 participants watching 40 one-minute music video excerpts.",
                "Official historical QMUL dataset URL was checked but was unavailable during RD-3 review.",
            ),
            boundary_notes=(
                "no raw participant data",
                "review artifact only; no raw participant data is downloaded or committed",
                "affect ratings are not diagnosis labels",
            ),
        ),
        ResearchDockPublicDatasetReview(
            dataset_id="seed",
            display_name="SEED",
            official_urls=(
                "https://bcmi.sjtu.edu.cn/home/seed/",
                "https://bcmi.sjtu.edu.cn/home/seed/seed.html",
                "https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/",
            ),
            citation="SEED dataset, BCMI Lab, Shanghai Jiao Tong University.",
            citation_status="verified_from_bcmi_dataset_pages",
            access_status="application_required_with_institution_email_and_license_upload",
            license_status="academic_research_license_required_before_download",
            modalities=("eeg", "eye_tracking", "self_report", "stimulus_video"),
            task_labels=("positive", "negative", "neutral"),
            researchdock_fields=(
                "eeg_observation",
                "pupil_gaze_proxy",
                "valence_arousal_self_report",
                "stimulus_context",
            ),
            researchdock_fit=(
                "Useful for EEG plus eye-movement missing-modality experiments after approved access.",
                "Emotion video protocol is adjacent to, but not a substitute for, ResearchDock reward/effort/stress tasks.",
            ),
            source_notes=(
                "BCMI states SEED has EEG and eye movement data for 12 subjects plus EEG-only data for 3 subjects.",
                "BCMI application form requires institution email, dataset checklist, and uploaded license file.",
            ),
            boundary_notes=(
                "no raw participant data",
                "review artifact only; no raw participant data is downloaded or committed",
                "emotion labels are stimulus-response labels, not diagnosis labels",
            ),
        ),
    )


def write_researchdock_public_dataset_review(out_dir: str | Path) -> dict[str, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    reviews = researchdock_public_dataset_reviews()
    payload = {
        "sprint": "RD-3",
        "claim_boundary": "public_dataset_mapping_review_only_no_clinical_claims",
        "datasets": [asdict(review) for review in reviews],
    }
    json_path = out_path / "researchdock_public_dataset_review.json"
    report_path = out_path / "researchdock_public_dataset_review.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_format_public_dataset_review_report(reviews), encoding="utf-8")
    return {"json": str(json_path), "report": str(report_path)}


def _format_public_dataset_review_report(reviews: tuple[ResearchDockPublicDatasetReview, ...]) -> str:
    lines = [
        "# ResearchDock RD-3 Public Dataset Review",
        "",
        "This sprint maps public affect datasets to ResearchDock fields. It does not download data, add loaders,",
        "commit raw participant data, or support diagnosis claims.",
        "",
        "## No Loaders",
        "",
        "All datasets are marked `review_only_no_loader` until access terms and local source manifests are reviewed.",
        "",
        "## Dataset Mapping",
        "",
        "| dataset | access | modalities | ResearchDock fields | loader status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for review in reviews:
        lines.append(
            "| {name} | {access} | {modalities} | {fields} | {loader} |".format(
                name=review.display_name,
                access=review.access_status,
                modalities=", ".join(review.modalities),
                fields=", ".join(review.researchdock_fields),
                loader=review.loader_status,
            )
        )
    lines.extend(["", "## Source Notes", ""])
    for review in reviews:
        lines.append(f"### {review.display_name}")
        lines.append("")
        lines.append(f"- citation_status: {review.citation_status}")
        lines.append(f"- license_status: {review.license_status}")
        for note in review.source_notes:
            lines.append(f"- {note}")
        for note in review.boundary_notes:
            lines.append(f"- {note}")
        lines.append("")
    lines.extend(
        [
            "## Claim Boundary",
            "",
            "RD-3 supports dataset triage and feature mapping only. It is not diagnosis, treatment,",
            "clinical decision support, or evidence of ResearchDock/Kahlus clinical utility.",
        ]
    )
    return "\n".join(lines) + "\n"
