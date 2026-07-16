from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analysis" / "plot_hnph_preprint_figures.py"
PROTOCOL = ROOT / "configs" / "protocol" / "hnph_phase0_v0.4.yaml"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(out_dir: Path, data_root: Path | None = None) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--protocol",
        str(PROTOCOL),
        "--out-dir",
        str(out_dir),
    ]
    if data_root is not None:
        command.extend(["--data-root", str(data_root)])
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def _dataset() -> dict[str, object]:
    return {
        "version": "fixture-v1",
        "license": "fixture-only",
        "source_sha256": "a" * 64,
        "subject_identity_field": "fixture_person_id",
        "channels": [{"name": "fixture", "unit": "mV", "sampling_rate_hz": 16.0}],
        "raters": [
            {"id": f"rater-{index}", "annotation_sha256": f"{index + 1:x}" * 64}
            for index in range(5)
        ],
    }


def _qualified_fixture(data_root: Path) -> dict[str, object]:
    data_root.mkdir(parents=True, exist_ok=True)
    raw_path = data_root / "non_neural_fixture.npz"
    time = np.linspace(0, 20, 512, endpoint=False)
    signal = np.sin(2 * np.pi * 0.5 * time)
    stages = np.asarray([0, 1, 1, 2, 2, 3], dtype=np.int64)
    np.savez(raw_path, signal=signal, stages=stages)
    payload: dict[str, object] = {
        "schema": "kahlus.hnph.dod_source_qualification.v1",
        "qualified": True,
        "external_opened": False,
        "datasets": {"DOD-H": _dataset(), "DOD-O": _dataset()},
        "leave_one_rater_out": [
            {
                "dataset": "DOD-H",
                "target_rater_id": "rater-0",
                "consensus_rater_ids": ["rater-1", "rater-2", "rater-3", "rater-4"],
                "target_sha256": "b" * 64,
            }
        ],
        "raw_example": {
            "dataset": "DOD-H",
            "safe_record_id": "fixture-001",
            "npz": raw_path.name,
            "signal_sha256": _hash(raw_path),
            "annotation_sha256": "c" * 64,
            "sampling_rate_hz": 16.0,
            "unit": "mV",
            "channel_names": ["fixture"],
        },
    }
    (data_root / "source_qualification.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def test_protocol_only_generation_is_deterministic_and_marks_descriptive_transport(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_first = _run(first)
    run_second = _run(second)
    assert run_first.returncode == 0, run_first.stderr
    assert run_second.returncode == 0, run_second.stderr
    manifest = json.loads((first / "figure_manifest.json").read_text(encoding="utf-8"))
    repeated = json.loads((second / "figure_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "kahlus.hnph.figure_manifest.v1"
    assert manifest["claim_scope"] == "protocol_only_no_empirical_hnph_frontier_claim"
    assert manifest["source_qualification_status"] == "unverified"
    assert len(manifest["figures"]) == 7
    assert manifest["figures"] == repeated["figures"]
    appendix_raw = next(row for row in manifest["figures"] if row["id"] == "figA1_verified_example")
    assert appendix_raw["status"] == "descriptive_single_label_transport_not_claim_evidence"
    assert "sleep_edf_descriptive_pdf" in manifest["input_hashes"]
    assert "sleep_edf_descriptive_png" in manifest["input_hashes"]
    assert "sleep_edf_descriptive_provenance" in manifest["input_hashes"]
    assert (first / "FIGURE_PROVENANCE.md").is_file()
    assert (first / "figure_captions.tex").is_file()
    for row in manifest["figures"]:
        assert (first / f"{row['id']}.pdf").is_file()
        assert (first / f"{row['id']}.png").is_file()


def test_qualified_non_neural_fixture_generates_verified_raw_illustration(tmp_path: Path) -> None:
    data_root = tmp_path / "local_data"
    _qualified_fixture(data_root)
    out_dir = tmp_path / "figures"
    result = _run(out_dir, data_root)
    assert result.returncode == 0, result.stderr
    manifest_text = (out_dir / "figure_manifest.json").read_text(encoding="utf-8")
    provenance_text = (out_dir / "FIGURE_PROVENANCE.md").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["source_qualification_status"] == "qualified"
    appendix_raw = next(row for row in manifest["figures"] if row["id"] == "figA1_verified_example")
    assert appendix_raw["status"] == "verified_raw_illustration"
    assert str(data_root) not in manifest_text
    assert str(data_root) not in provenance_text


def test_renderer_refuses_unqualified_or_contaminated_sources(tmp_path: Path) -> None:
    mutations = {
        "missing_source_hash": lambda payload: payload["datasets"]["DOD-H"].pop("source_sha256"),
        "missing_physical_unit": lambda payload: payload["datasets"]["DOD-H"]["channels"][0].pop("unit"),
        "fewer_than_three_raters": lambda payload: payload["datasets"]["DOD-H"].update(
            {"raters": payload["datasets"]["DOD-H"]["raters"][:2]}
        ),
        "held_out_in_consensus": lambda payload: payload["leave_one_rater_out"][0]["consensus_rater_ids"].append(
            "rater-0"
        ),
        "external_opened": lambda payload: payload.update({"external_opened": True}),
    }
    for name, mutate in mutations.items():
        case_root = tmp_path / name
        payload = _qualified_fixture(case_root)
        mutate(payload)
        (case_root / "source_qualification.json").write_text(json.dumps(payload), encoding="utf-8")
        result = _run(tmp_path / f"out-{name}", case_root)
        assert result.returncode != 0, name
        assert "refused" in result.stderr.lower()


def test_renderer_refuses_empirical_result_without_claim_eligible_gate(tmp_path: Path) -> None:
    data_root = tmp_path / "local_data"
    _qualified_fixture(data_root)
    (data_root / "hnph_result.json").write_text(
        json.dumps(
            {
                "protocol_sha256": _hash(PROTOCOL),
                "claim_eligible": False,
                "gate_passed": False,
                "frontier_values": [0.1, 0.08, 0.04],
            }
        ),
        encoding="utf-8",
    )
    result = _run(tmp_path / "figures", data_root)
    assert result.returncode != 0
    assert "claim-eligible" in result.stderr
