#!/usr/bin/env python3
"""Fetch NSRR MESA / SHHS polysomnography (credentialed) — fingerprint only, no raw in git."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_MESA = Path("/Users/aayu/datasets/kahlus_multidataset_public/nsrr/mesa")
DEFAULT_SHHS = Path("/Users/aayu/datasets/kahlus_multidataset_public/nsrr/shhs")

NSRR_ACCESS_URL = "https://sleepdata.org/datasets"
MESA_DOCS = "https://sleepdata.org/datasets/mesa"
SHHS_DOCS = "https://sleepdata.org/datasets/shhs"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_recordings(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(
        {
            str(p.relative_to(root))
            for p in root.rglob("*.edf")
            if not p.name.startswith(".")
        }
    )


def write_fingerprint(root: Path, *, dataset: str) -> dict[str, object]:
    edf_files = discover_recordings(root)
    xml_files = sorted(str(p.relative_to(root)) for p in root.rglob("*-nsrr.xml"))
    payload = {
        "dataset": dataset,
        "nsrr_portal": NSRR_ACCESS_URL,
        "mesa_docs": MESA_DOCS,
        "shhs_docs": SHHS_DOCS,
        "n_edf_files": len(edf_files),
        "n_xml_files": len(xml_files),
        "edf_files_sample": edf_files[:30],
        "edf_truncated": len(edf_files) > 30,
        "access_note": (
            "NSRR requires credentialed account + DUA. Download EDF + *-nsrr.xml "
            "via the NSRR portal; this script does not auto-download protected data."
        ),
    }
    if (root / "mesa-sleep-dataset-0.6.0.csv").is_file():
        payload["cohort_csv_sha256"] = _sha256(root / "mesa-sleep-dataset-0.6.0.csv")
    out = root / "dataset_fingerprint.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def print_access_instructions() -> None:
    print(
        "\n".join(
            [
                "NSRR credentialed access required:",
                f"  1. Create account at {NSRR_ACCESS_URL}",
                "  2. Complete data use agreement for MESA and/or SHHS",
                "  3. Download polysomnography EDF + companion *-nsrr.xml annotation files",
                "  4. Place under:",
                f"     MESA: {DEFAULT_MESA}",
                f"     SHHS: {DEFAULT_SHHS}",
                "  5. Re-run with --skip-download to write dataset_fingerprint.json",
                "",
                "Expected pairing: mesa-sleep-XXXX.edf + mesa-sleep-XXXX-nsrr.xml",
            ]
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="NSRR MESA/SHHS fetch helper (fingerprint + access guide).")
    parser.add_argument("--mesa-target", type=Path, default=DEFAULT_MESA)
    parser.add_argument("--shhs-target", type=Path, default=DEFAULT_SHHS)
    parser.add_argument("--dataset", choices=("mesa", "shhs", "both"), default="both")
    parser.add_argument("--skip-download", action="store_true", help="Only verify layout + write fingerprint.")
    args = parser.parse_args()

    if not args.skip_download:
        print_access_instructions()
        print("\nNo automatic download — NSRR data is credentialed. Use --skip-download after manual fetch.")

    results: dict[str, object] = {}
    if args.dataset in {"mesa", "both"}:
        args.mesa_target.mkdir(parents=True, exist_ok=True)
        results["mesa"] = write_fingerprint(args.mesa_target, dataset="mesa")
        print(json.dumps(results["mesa"], indent=2))
    if args.dataset in {"shhs", "both"}:
        args.shhs_target.mkdir(parents=True, exist_ok=True)
        results["shhs"] = write_fingerprint(args.shhs_target, dataset="shhs")
        print(json.dumps(results["shhs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
