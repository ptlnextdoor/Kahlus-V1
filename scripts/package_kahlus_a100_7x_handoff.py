#!/usr/bin/env python3
"""Package a local Kahlus handoff for a 7xA100 cluster without launching jobs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.a100_handoff import A100HandoffError, package_kahlus_a100_7x_handoff  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="clean git checkout to package")
    parser.add_argument("--out-dir", required=True, help="directory for the handoff zip")
    args = parser.parse_args()

    try:
        package = package_kahlus_a100_7x_handoff(Path(args.repo_root), Path(args.out_dir))
    except (A100HandoffError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"handoff_zip={package.zip_path}")
    print(f"commit={package.commit_hash}")
    print(f"gpu_label={package.manifest['gpu_label']}")
    print(f"expected_gpu_count={package.manifest['expected_gpu_count']}")
    print(f"runner_tarball={package.runner_tarball}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
