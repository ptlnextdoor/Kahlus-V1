from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_indexer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_labglass_index.py"
    spec = importlib.util.spec_from_file_location("build_labglass_index", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_labglass_index_finds_local_forecastability_artifacts() -> None:
    indexer = _load_indexer()
    payload = indexer.build_index(Path(__file__).resolve().parents[1] / "artifacts")

    assert payload["reports"]
    assert any(report["title"] == "M4" for report in payload["reports"])


def test_labglass_index_normalizes_gate_status() -> None:
    indexer = _load_indexer()
    report = indexer._report_from_payload(
        {"gate_passed": False, "gate_failures": ["underpowered_event_patients"], "claim_scope": "smoke"},
        path=Path("run/evidence_gate.json"),
        source=None,
        size_bytes=42,
    )

    assert report["status"] == "UNDERPOWERED"
    assert report["statusKind"] == "warning"
    assert report["failures"] == ["underpowered_event_patients"]
    assert report["gatePredicate"]["power"] == "warning"


def test_labglass_index_ignores_aggregate_json_names() -> None:
    indexer = _load_indexer()

    assert not indexer._is_report_name("aggregate.json", "run/aggregate.json")
    assert indexer._is_report_name("m5_gate_report.json", "artifacts/forecastability_trial0_m5/m5_gate_report.json")
