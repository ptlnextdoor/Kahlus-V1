#!/usr/bin/env python3
"""Fetch Kahlus public EDF datasets from official PhysioNet file endpoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urljoin, urlparse
from urllib.request import urlopen


CLAIM_BOUNDARY = "public_edf_fetch_only_no_clinical_or_model_superiority_claims"
EXPECTED_FOLDERS = ("sleep-edf", "chb-mit", "eegmmi", "siena")
REPO_ROOT = Path(__file__).resolve().parents[1]
HREF_RE = re.compile(r"""href=["']([^"']+)["']""", re.IGNORECASE)


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    folder: str
    official_url: str
    files_base_url: str


DATASETS: dict[str, DatasetSpec] = {
    "sleep_edf_expanded": DatasetSpec(
        dataset_id="sleep_edf_expanded",
        folder="sleep-edf",
        official_url="https://physionet.org/content/sleep-edfx/1.0.0/",
        files_base_url="https://physionet.org/files/sleep-edfx/1.0.0/",
    ),
    "chb_mit_physionet": DatasetSpec(
        dataset_id="chb_mit_physionet",
        folder="chb-mit",
        official_url="https://physionet.org/content/chbmit/1.0.0/",
        files_base_url="https://physionet.org/files/chbmit/1.0.0/",
    ),
    "eegmmi_physionet": DatasetSpec(
        dataset_id="eegmmi_physionet",
        folder="eegmmi",
        official_url="https://physionet.org/content/eegmmidb/1.0.0/",
        files_base_url="https://physionet.org/files/eegmmidb/1.0.0/",
    ),
    "siena_scalp_eeg": DatasetSpec(
        dataset_id="siena_scalp_eeg",
        folder="siena",
        official_url="https://physionet.org/content/siena-scalp-eeg/1.0.0/",
        files_base_url="https://physionet.org/files/siena-scalp-eeg/1.0.0/",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", required=True, help="persistent raw root, e.g. /data/kahlus/raw")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="download all primary EDF records listed by PhysioNet")
    group.add_argument(
        "--max-records-per-dataset",
        type=int,
        default=16,
        help="smoke subset size per dataset; default: 16",
    )
    parser.add_argument("--allow-repo-root", action="store_true", help="allow writing raw EDFs under this git repo")
    args = parser.parse_args()

    if args.max_records_per_dataset is not None and args.max_records_per_dataset <= 0:
        parser.error("--max-records-per-dataset must be positive")

    out_root = Path(args.out_root).expanduser().resolve()
    if is_path_within(out_root, REPO_ROOT) and not args.allow_repo_root:
        print(
            f"refusing to write public raw EDF data inside repo: {out_root}\n"
            "Use a persistent external path such as /data/kahlus/raw, or pass --allow-repo-root explicitly.",
            file=sys.stderr,
        )
        return 2

    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "kahlus.public_edf_download_manifest.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "out_root": str(out_root),
        "claim_boundary": CLAIM_BOUNDARY,
        "kaggle_policy": "Kaggle may be used only as an exact verified mirror; official PhysioNet is ground truth.",
        "expected_raw_folders": list(EXPECTED_FOLDERS),
        "full_download": bool(args.full),
        "max_records_per_dataset": None if args.full else int(args.max_records_per_dataset),
        "datasets": [],
    }
    required_failures: list[str] = []
    for spec in DATASETS.values():
        dataset_manifest, failures = fetch_dataset(
            spec,
            out_root / spec.folder,
            full=bool(args.full),
            max_records=None if args.full else int(args.max_records_per_dataset),
        )
        manifest["datasets"].append(dataset_manifest)
        required_failures.extend(failures)

    manifest_path = out_root / "kahlus_public_edf_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest={manifest_path}")
    if required_failures:
        print("required_download_failures=" + str(len(required_failures)), file=sys.stderr)
        return 3
    print("public_edf_fetch_complete=true")
    return 0


def fetch_dataset(spec: DatasetSpec, dataset_root: Path, *, full: bool, max_records: int | None) -> tuple[dict, list[str]]:
    print(f"[{spec.dataset_id}] root={dataset_root}")
    dataset_root.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    skipped: list[str] = []
    failed: list[dict[str, object]] = []
    required_failures: list[str] = []

    records_result = download_file(url_for(spec, "RECORDS"), dataset_root / "RECORDS", required=True)
    _track_download(records_result, downloaded, skipped, failed, required_failures)
    records = read_records(dataset_root / "RECORDS")
    selected = select_primary_records(spec.dataset_id, records, max_records=max_records, full=full)
    files: dict[str, bool] = {record: True for record in selected}

    if spec.dataset_id == "sleep_edf_expanded":
        files.update({record: False for record in sleep_hypnogram_records(spec, selected)})
    elif spec.dataset_id == "chb_mit_physionet":
        seizure_result = download_file(url_for(spec, "RECORDS-WITH-SEIZURES"), dataset_root / "RECORDS-WITH-SEIZURES", required=True)
        _track_download(seizure_result, downloaded, skipped, failed, required_failures)
        seizure_records = set(read_records(dataset_root / "RECORDS-WITH-SEIZURES"))
        for record in selected:
            if record in seizure_records:
                files[f"{record}.seizures"] = False
        for patient in sorted({Path(record).parent.as_posix() for record in selected}):
            files[f"{patient}/{Path(patient).name}-summary.txt"] = False
    elif spec.dataset_id == "siena_scalp_eeg":
        for record in selected:
            parent = Path(record).parent.as_posix()
            for href in directory_hrefs(url_for(spec, parent + "/")):
                if href.lower().endswith((".txt", ".tsv", ".csv")):
                    files[f"{parent}/{href}"] = False

    for rel_path, required in sorted(files.items()):
        result = download_file(url_for(spec, rel_path), dataset_root / rel_path, required=required)
        _track_download(result, downloaded, skipped, failed, required_failures)

    return {
        "dataset_id": spec.dataset_id,
        "folder": spec.folder,
        "official_url": spec.official_url,
        "source_file_base_url": spec.files_base_url,
        "records_url": url_for(spec, "RECORDS"),
        "selected_records": selected,
        "downloaded_files": downloaded,
        "skipped_existing_files": skipped,
        "failed_files": failed,
        "claim_boundary": CLAIM_BOUNDARY,
    }, required_failures


def select_primary_records(dataset_id: str, records: list[str], *, max_records: int | None, full: bool) -> list[str]:
    if dataset_id == "sleep_edf_expanded":
        selected = [record for record in records if record.endswith("-PSG.edf")]
    else:
        selected = [record for record in records if record.endswith(".edf")]
    if full:
        return selected
    return selected[: max_records or 0]


def sleep_hypnogram_records(spec: DatasetSpec, psg_records: list[str]) -> list[str]:
    found: list[str] = []
    href_cache: dict[str, list[str]] = {}
    for psg in psg_records:
        parent = Path(psg).parent.as_posix()
        hrefs = href_cache.setdefault(parent, directory_hrefs(url_for(spec, parent + "/")))
        match = match_sleep_hypnogram(psg, hrefs)
        if match:
            found.append(f"{parent}/{match}")
    return found


def match_sleep_hypnogram(psg_record: str, hrefs: list[str]) -> str | None:
    prefix = Path(psg_record).name[:6]
    matches = sorted(
        href
        for href in hrefs
        if Path(href).name.startswith(prefix) and Path(href).name.endswith("-Hypnogram.edf")
    )
    return matches[0] if matches else None


def directory_hrefs(url: str, fetch_text: Callable[[str], str] | None = None) -> list[str]:
    try:
        html = (fetch_text or read_url_text)(url)
    except (HTTPError, URLError, TimeoutError, OSError):
        return []
    hrefs = []
    for raw in HREF_RE.findall(html):
        href = unquote(urlparse(raw).path.split("/")[-1])
        if href and href not in {".", ".."} and not href.startswith("?"):
            hrefs.append(href)
    return sorted(set(hrefs))


def read_records(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def download_file(url: str, destination: Path, *, required: bool) -> dict[str, object]:
    rel = destination.as_posix()
    if destination.exists() and destination.stat().st_size > 0:
        print(f"skip existing {destination}")
        return {"status": "skipped_existing", "url": url, "path": rel, "required": required}
    print(f"download {url} -> {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".part")
    try:
        with urlopen(url, timeout=60) as response, temp.open("wb") as handle:  # nosec B310 - fixed public PhysioNet URLs.
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        temp.replace(destination)
        return {
            "status": "downloaded",
            "url": url,
            "path": rel,
            "required": required,
            "bytes": destination.stat().st_size,
        }
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        temp.unlink(missing_ok=True)
        print(f"failed {'required' if required else 'optional'} {url}: {exc}", file=sys.stderr)
        return {"status": "failed", "url": url, "path": rel, "required": required, "error": str(exc)}


def _track_download(result: dict[str, object], downloaded: list[str], skipped: list[str], failed: list[dict[str, object]], required_failures: list[str]) -> None:
    status = result["status"]
    path = str(result["path"])
    if status == "downloaded":
        downloaded.append(path)
    elif status == "skipped_existing":
        skipped.append(path)
    elif status == "failed":
        failed.append(result)
        if result.get("required"):
            required_failures.append(path)


def url_for(spec: DatasetSpec, rel_path: str) -> str:
    return urljoin(spec.files_base_url, quote(rel_path, safe="/._-+"))


def read_url_text(url: str) -> str:
    with urlopen(url, timeout=60) as response:  # nosec B310 - fixed public PhysioNet URLs.
        return response.read().decode("utf-8", errors="replace")


def is_path_within(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


if __name__ == "__main__":
    raise SystemExit(main())
