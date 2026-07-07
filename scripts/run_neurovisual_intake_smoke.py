#!/usr/bin/env python3
"""Run a local NV-1 synthetic intake smoke.

No A100, no cluster job, no private participant data, no diagnosis, and no photic trigger testing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.neurovisual import build_condition_comparison_matrix, build_episode_intake_profile  # noqa: E402
from neurotwin.repro import write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input-json")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = _read_input(args.input_json)
    intake = build_episode_intake_profile(raw)
    payload = intake.to_dict()
    payload["condition_matrix"] = build_condition_comparison_matrix()
    payload["execution"] = {
        "bulk_dataset_download": False,
        "a100_jobs_launched": False,
        "cluster_jobs_launched": False,
        "private_patient_data_used": False,
    }
    json_path = write_json(out_dir / "neurovisual_intake_smoke.json", payload)
    md_path = out_dir / "neurovisual_intake_smoke.md"
    md_path.write_text(_format_report(payload), encoding="utf-8")

    print(f"branch=nv1 out_dir={out_dir.resolve()}")
    print(f"claim_scope={payload['claim_gate']['claim_scope']}")
    print(f"claim_gate_passed={payload['claim_gate']['passed']}")
    print("a100_jobs_launched=false")
    print("bulk_dataset_download=false")
    print(f"json={json_path}")
    print(f"report={md_path}")
    return 0


def _read_input(path: str | None) -> dict[str, object]:
    if path is None:
        return {
            "duration_seconds": 20,
            "onset_speed": "abrupt",
            "awareness_retained": True,
            "memory_retained": True,
            "visual_field_location": "peripheral",
            "motion_or_flicker": True,
            "screen_or_sun_trigger": True,
            "no_new_objects_seen": True,
            "no_loss_of_consciousness": True,
            "source_text": "synthetic generic example with no identifying details",
        }
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("input JSON must contain an object")
    return payload


def _format_report(payload: dict[str, object]) -> str:
    profile = payload["episode_phenotype_profile"]
    assert isinstance(profile, dict)
    questions = payload["missing_clinician_questions"]
    red_flags = payload["red_flag_checklist"]
    gate = payload["claim_gate"]
    assert isinstance(gate, dict)
    lines = [
        "# NV-1 Neurovisual Intake Smoke",
        "",
        "- scope: NV-1 side branch, not Kahlus v1/v2/v3 replacement",
        f"- claim_scope: {gate['claim_scope']}",
        f"- claim_gate_passed: {gate['passed']}",
        "- not_diagnosis: This is not a diagnosis, medical advice, seizure prediction, or a safety claim.",
        "- no_photic_testing: This smoke does not recommend trigger testing or photic stimulation.",
        "",
        "## Structured Episode Phenotype Profile",
        "",
        f"- onset_speed: {profile.get('onset_speed')}",
        f"- duration_seconds: {profile.get('duration_seconds')}",
        f"- visual_field_location: {profile.get('visual_field_location')}",
        f"- motion_or_flicker: {profile.get('motion_or_flicker')}",
        f"- screen_or_sun_trigger: {profile.get('screen_or_sun_trigger')}",
        "",
        "## Missing Clinician Questions",
        "",
        *[f"- {question}" for question in questions],
        "",
        "## Red-Flag Checklist",
        "",
        *[f"- {flag}" for flag in red_flags],
        "",
        "## Execution Boundary",
        "",
        "- bulk_dataset_download: false",
        "- a100_jobs_launched: false",
        "- cluster_jobs_launched: false",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
