from __future__ import annotations

import json
from pathlib import Path

from neurotwin.data.forecast_contract import FORECAST_PROTOCOL_V2_NONOVERLAP
from neurotwin.eval.forecast_eligibility import (
    ForecastEligibilitySources,
    build_forecast_eligibility_from_sources,
)
from neurotwin.repro import hash_file


def build_bound_forecast_eligibility(
    root: Path,
    *,
    result_dependency_ids: tuple[str, ...] = (),
    subject_overlap_count: int = 0,
) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    raw_path = root / "source.edf"
    raw_path.write_bytes(b"test-only raw source\n")
    raw_sha = hash_file(raw_path)

    protocol = _write_json(
        root / "protocol.json",
        {"protocol_id": FORECAST_PROTOCOL_V2_NONOVERLAP, "schema_version": 2},
    )
    source_manifest = _write_json(
        root / "source_manifest.json",
        {"files": [{"path": raw_path.name, "sha256": raw_sha}]},
    )
    transform_lineage = _write_json(
        root / "transform_lineage.json",
        {
            "steps": [
                {
                    "name": "identity_test_transform",
                    "input_hashes": [raw_sha],
                    "output_hashes": ["b" * 64],
                }
            ]
        },
    )
    split_audit = _write_json(
        root / "split_audit.json",
        {
            "violations": [],
            "subject_overlap_count": subject_overlap_count,
            "recording_overlap_count": 0,
            "session_overlap_count": 0,
        },
    )
    firebreak_audit = _write_json(
        root / "firebreak_audit.json",
        {"violations": [], "target_overlaps_context": False},
    )
    invalidated_registry = _write_json(
        root / "invalidated_result_registry.json",
        {"results": [{"result_id": "legacy-overlap", "status": "invalid_experiment"}]},
    )
    return build_forecast_eligibility_from_sources(
        ForecastEligibilitySources(
            protocol=protocol,
            source_manifest=source_manifest,
            transform_lineage=transform_lineage,
            split_audit=split_audit,
            firebreak_audit=firebreak_audit,
            invalidated_result_registry=invalidated_registry,
            result_dependency_ids=result_dependency_ids,
        )
    )


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
