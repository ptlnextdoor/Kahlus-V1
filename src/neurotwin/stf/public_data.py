from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CHB_MIT_DATASET_ID = "chb_mit_physionet"
CHB_MIT_PHYSIONET_URL = "https://physionet.org/content/chbmit/1.0.0/"
CHB_MIT_FILE_BASE_URL = "https://physionet.org/files/chbmit/1.0.0/"
CHB_MIT_RECORDS_WITH_SEIZURES_URL = (
    "https://physionet.org/content/chbmit/1.0.0/RECORDS-WITH-SEIZURES"
)


@dataclass(frozen=True)
class CHBMITRootAudit:
    dataset_id: str
    data_root: str
    passed: bool
    failure_reasons: tuple[str, ...]
    record_count: int
    seizure_record_count: int
    patients: tuple[str, ...]
    missing_seizure_annotations: tuple[str, ...]
    raw_data_in_repo: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "kahlus.stf.chb_mit_root_audit.v1",
            "dataset_id": self.dataset_id,
            "data_root": self.data_root,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "record_count": self.record_count,
            "seizure_record_count": self.seizure_record_count,
            "patients": list(self.patients),
            "missing_seizure_annotations": list(self.missing_seizure_annotations),
            "raw_data_in_repo": self.raw_data_in_repo,
            "source_url": CHB_MIT_PHYSIONET_URL,
            "records_with_seizures_url": CHB_MIT_RECORDS_WITH_SEIZURES_URL,
            "claim_boundary": (
                "local public-data readiness audit only; no diagnosis, treatment, "
                "seizure prevention, medication, stimulation, or A100 claim"
            ),
        }


def stf_public_dataset_registry() -> dict[str, dict[str, Any]]:
    return {
        CHB_MIT_DATASET_ID: {
            "dataset_id": CHB_MIT_DATASET_ID,
            "name": "CHB-MIT Scalp EEG Database",
            "source_url": CHB_MIT_PHYSIONET_URL,
            "records_with_seizures_url": CHB_MIT_RECORDS_WITH_SEIZURES_URL,
            "first_use": "local public-data smoke after synthetic STF gate",
            "required_local_files": ["RECORDS", "RECORDS-WITH-SEIZURES"],
            "blocked": ["raw EDF commit", "clinical diagnosis", "seizure prevention claim"],
        }
    }


def audit_chb_mit_root(data_root: str | Path, *, repo_root: str | Path | None = None) -> CHBMITRootAudit:
    root = Path(data_root).expanduser().resolve()
    repo = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    failures: list[str] = []
    records_path = root / "RECORDS"
    seizure_records_path = root / "RECORDS-WITH-SEIZURES"

    raw_data_in_repo = _is_relative_to(root, repo)
    if raw_data_in_repo:
        failures.append("public raw EEG root is inside the repository; keep raw data outside git")
    if not root.exists():
        failures.append(f"data root does not exist: {root}")
    if not records_path.exists():
        failures.append("missing RECORDS")
    if not seizure_records_path.exists():
        failures.append("missing RECORDS-WITH-SEIZURES")

    records = _read_record_list(records_path) if records_path.exists() else []
    seizure_records = _read_record_list(seizure_records_path) if seizure_records_path.exists() else []
    if records_path.exists() and not records:
        failures.append("RECORDS contains no EDF records")
    if seizure_records_path.exists() and not seizure_records:
        failures.append("RECORDS-WITH-SEIZURES contains no seizure EDF records")

    record_set = set(records)
    missing_from_records = [record for record in seizure_records if record not in record_set]
    if missing_from_records:
        failures.append("seizure records are not a subset of RECORDS")
    missing_annotations = [
        record for record in seizure_records if not (root / f"{record}.seizures").exists()
    ]
    if missing_annotations:
        failures.append("missing .edf.seizures annotation files for seizure records")

    patients = sorted({record.split("/", 1)[0] for record in records if "/" in record})
    return CHBMITRootAudit(
        dataset_id=CHB_MIT_DATASET_ID,
        data_root=str(root),
        passed=not failures,
        failure_reasons=tuple(failures),
        record_count=len(records),
        seizure_record_count=len(seizure_records),
        patients=tuple(patients),
        missing_seizure_annotations=tuple(missing_annotations),
        raw_data_in_repo=raw_data_in_repo,
    )


def write_chb_mit_root_audit(out_dir: str | Path, audit: CHBMITRootAudit) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "audit": out / "chb_mit_root_audit.json",
        "registry": out / "stf_public_dataset_registry.json",
        "report": out / "stf_public_data_report.md",
    }
    paths["audit"].write_text(json.dumps(audit.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    paths["registry"].write_text(
        json.dumps(stf_public_dataset_registry(), indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["report"].write_text(_audit_report(audit), encoding="utf-8")
    return paths


def fetch_chb_mit_smoke_subset(
    out_root: str | Path,
    *,
    patients: int = 2,
    records_per_patient: int = 2,
    base_url: str = CHB_MIT_FILE_BASE_URL,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(out_root).expanduser().resolve()
    repo = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    if _is_relative_to(root, repo):
        raise ValueError("refusing to fetch raw CHB-MIT data inside the repository")
    root.mkdir(parents=True, exist_ok=True)

    records_text = _read_url_text(_url(base_url, "RECORDS"))
    seizure_text = _read_url_text(_url(base_url, "RECORDS-WITH-SEIZURES"))
    records = _record_rows(records_text)
    seizure_records = _record_rows(seizure_text)
    selected = _select_smoke_records(records, seizure_records, patients, records_per_patient)
    selected_seizure_records = [record for record in selected if record in set(seizure_records)]
    selected_patients = sorted({record.split("/", 1)[0] for record in selected})

    for record in selected:
        _download(_url(base_url, record), root / record)
    for patient in selected_patients:
        _download(_url(base_url, f"{patient}/{patient}-summary.txt"), root / patient / f"{patient}-summary.txt")
    for record in selected_seizure_records:
        _download(_url(base_url, f"{record}.seizures"), root / f"{record}.seizures")

    (root / "RECORDS").write_text("\n".join(selected) + "\n", encoding="utf-8")
    (root / "RECORDS-WITH-SEIZURES").write_text(
        "\n".join(selected_seizure_records) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "kahlus.stf.chb_mit_smoke_subset.v1",
        "dataset_id": CHB_MIT_DATASET_ID,
        "source_base_url": base_url,
        "data_root": str(root),
        "selected_records": selected,
        "selected_seizure_records": selected_seizure_records,
        "patients": selected_patients,
        "a100_jobs_launched": False,
        "claim_boundary": "public-data smoke subset only; no clinical, device, or A100 claim",
    }
    (root / "kahlus_stf_smoke_subset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _read_record_list(path: Path) -> list[str]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        row = raw.strip()
        if row and not row.startswith("#") and row.endswith(".edf"):
            rows.append(row)
    return rows


def _record_rows(text: str) -> list[str]:
    return [
        row.strip()
        for row in text.splitlines()
        if row.strip() and not row.startswith("#") and row.strip().endswith(".edf")
    ]


def _select_smoke_records(
    records: list[str],
    seizure_records: list[str],
    patients: int,
    records_per_patient: int,
) -> list[str]:
    if patients < 2:
        raise ValueError("patients must be at least 2 for patient-held-out smoke")
    if records_per_patient < 1:
        raise ValueError("records_per_patient must be positive")
    selected: list[str] = []
    seizure_patient_ids = sorted({record.split("/", 1)[0] for record in seizure_records})
    for patient in seizure_patient_ids[:patients]:
        patient_records = [record for record in records if record.startswith(f"{patient}/")]
        patient_seizures = [record for record in seizure_records if record.startswith(f"{patient}/")]
        for record in patient_seizures + patient_records:
            if record not in selected:
                selected.append(record)
            if sum(row.startswith(f"{patient}/") for row in selected) >= records_per_patient:
                break
    if len({record.split("/", 1)[0] for record in selected}) < 2:
        raise ValueError("could not select records from at least two seizure patients")
    return selected


def _read_url_text(url: str) -> str:
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(url, path)


def _url(base_url: str, rel: str) -> str:
    return base_url.rstrip("/") + "/" + rel


def _audit_report(audit: CHBMITRootAudit) -> str:
    failures = audit.failure_reasons or ("none",)
    lines = [
        "# Kahlus-STF public-data audit",
        "",
        f"- dataset_id: {audit.dataset_id}",
        f"- source_url: {CHB_MIT_PHYSIONET_URL}",
        f"- passed: {audit.passed}",
        f"- record_count: {audit.record_count}",
        f"- seizure_record_count: {audit.seizure_record_count}",
        f"- raw_data_in_repo: {audit.raw_data_in_repo}",
        "- a100_jobs_launched: false",
        "",
        "## Failure Reasons",
    ]
    lines.extend(f"- {reason}" for reason in failures)
    return "\n".join(lines) + "\n"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
