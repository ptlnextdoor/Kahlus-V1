#!/usr/bin/env python3
"""Package a local Kahlus handoff for an A100 cluster without launching jobs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tarfile

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.a100_handoff import A100HandoffError, package_kahlus_a100_handoff  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="clean git checkout to package")
    parser.add_argument("--out-dir", required=True, help="directory for the handoff zip")
    parser.add_argument("--gpu-count", type=int, default=7, help="honest A100 GPU count for this package")
    parser.add_argument(
        "--include-raw-data-root",
        help="explicitly create a separate large raw EDF tarball from this path; never added to the code zip",
    )
    args = parser.parse_args()

    try:
        package = package_kahlus_a100_handoff(Path(args.repo_root), Path(args.out_dir), gpu_count=args.gpu_count)
    except (A100HandoffError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"handoff_zip={package.zip_path}")
    print(f"commit={package.commit_hash}")
    print(f"gpu_label={package.manifest['gpu_label']}")
    print(f"expected_gpu_count={package.manifest['expected_gpu_count']}")
    print(f"runner_tarball={package.runner_tarball}")
    if args.include_raw_data_root:
        raw_archive = _write_raw_data_archive(Path(args.include_raw_data_root), Path(args.out_dir), package.commit_hash)
        print(f"raw_data_archive={raw_archive}")
    return 0


def _write_raw_data_archive(raw_root: Path, out_dir: Path, commit: str) -> Path:
    raw = raw_root.expanduser().resolve()
    if not raw.exists() or not raw.is_dir():
        raise A100HandoffError(f"--include-raw-data-root must point to an existing directory: {raw}")
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / f"kahlus-public-edf-raw-data-{commit[:7]}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(raw.rglob("*")):
            if path.is_symlink():
                raise A100HandoffError(f"refusing symlink in raw data archive: {path}")
            if path.is_file():
                archive.add(path, arcname=f"kahlus-public-edf-raw-data/{path.relative_to(raw).as_posix()}")
    return archive_path


if __name__ == "__main__":
    raise SystemExit(main())
