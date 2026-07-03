#!/usr/bin/env python3
"""Audit a local Kahlus v1 EEG baseline evidence checksum manifest.

This is a CPU/local verification lane only. It does not launch A100, torchrun, or cluster jobs.
"""

from __future__ import annotations

import argparse
import json

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.eeg_v1 import audit_eeg_v1_checksum_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    args = parser.parse_args()

    payload = audit_eeg_v1_checksum_manifest(args.artifact_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
