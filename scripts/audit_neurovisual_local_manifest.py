#!/usr/bin/env python3
"""Audit an explicit NV-1 local manifest against a dataset registry JSON.

This validates manifest structure and registry status only. It does not check raw
file existence, download datasets, execute adapters, or launch A100/cluster jobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.neurovisual import validate_local_manifest_records  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to a JSON list of local manifest records.")
    parser.add_argument("--registry", required=True, help="Path to a neurovisual dataset registry JSON.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    registry_path = Path(args.registry)
    payload = audit_local_manifest(manifest_path=manifest_path, registry_path=registry_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


def audit_local_manifest(*, manifest_path: Path, registry_path: Path) -> dict[str, Any]:
    failures: list[str] = []
    manifest_payload = _read_json(manifest_path, "manifest", failures)
    registry_payload = _read_json(registry_path, "registry", failures)
    if not isinstance(manifest_payload, list):
        failures.append("manifest_must_be_json_list")
        manifest_records: list[dict[str, Any]] = []
    else:
        manifest_records = manifest_payload
    if not isinstance(registry_payload, dict):
        failures.append("registry_must_be_json_object")
        registry_payload = {}

    audit = validate_local_manifest_records(manifest_records, registry=registry_payload)
    merged_failures = failures + list(audit["failures"])
    audit["passed"] = not merged_failures
    audit["failures"] = merged_failures
    audit["manifest_path"] = str(manifest_path)
    audit["registry_path"] = str(registry_path)
    audit["execution"]["raw_file_existence_checked"] = False
    return audit


def _read_json(path: Path, label: str, failures: list[str]) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        failures.append(f"missing_{label}:{path}")
    except json.JSONDecodeError:
        failures.append(f"invalid_json:{label}:{path}")
    return None


if __name__ == "__main__":
    raise SystemExit(main())
