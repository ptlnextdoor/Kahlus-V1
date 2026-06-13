#!/usr/bin/env python3
"""Kahlus-EM Stage 1 passive-logging analysis smoke runner (NO-HUMAN, OFFLINE).

Demonstrates the room/device environment logging schema and an OFFLINE geomagnetic context
load, then writes a descriptive analysis + a claim-blocking EM gate. The logger records only
operator-supplied values (no hardware access); the fetcher performs no network access.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.em import RoomEnvironmentLog, RoomEMFLogger, fetch_geomagnetic  # noqa: E402
from neurotwin.gates import evaluate_gate, write_evidence_gate  # noqa: E402
from neurotwin.repro import write_json  # noqa: E402


def _synthesize_log(logger: RoomEMFLogger, n_entries: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for i in range(n_entries):
        logger.log(
            RoomEnvironmentLog(
                timestamp=f"2026-06-13T10:{i:02d}:00Z",
                room_id="lab_a",
                device_id="eeg_amp_01",
                temperature_c=float(21.0 + rng.normal(scale=0.3)),
                humidity_pct=float(40.0 + rng.normal(scale=2.0)),
                mains_freq_hz=60.0,
                line_noise_uv=float(abs(rng.normal(loc=1.0, scale=0.2))),
                nearby_devices=["monitor", "hvac"],
                geomagnetic_index=None,
                notes="synthetic passive log entry",
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--log-file", default=None, help="Existing JSONL room log; synthesized if omitted")
    parser.add_argument("--geomagnetic-file", default=None, help="Offline JSON geomagnetic source (optional)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file else out / "room_emf_log.jsonl"
    logger = RoomEMFLogger(log_path)
    if not args.log_file:
        _synthesize_log(logger, n_entries=6, seed=args.seed)

    rows = logger.read_all()
    line_noise = [r["line_noise_uv"] for r in rows if r.get("line_noise_uv") is not None]
    geomagnetic = fetch_geomagnetic(args.geomagnetic_file)
    finite = bool(line_noise) and all(np.isfinite(line_noise))
    analysis = {
        "branch": "em",
        "stage": 1,
        "claim_status": "descriptive_no_human_passive_logging",
        "n_log_entries": len(rows),
        "line_noise_uv_mean": float(np.mean(line_noise)) if line_noise else None,
        "line_noise_uv_std": float(np.std(line_noise)) if line_noise else None,
        "geomagnetic_status": geomagnetic["status"],
        "network_access": geomagnetic["network_access"],
        "finite": finite,
    }
    gate = evaluate_gate(
        branch="em",
        dataset="em_stage1_passive_logging",
        split_audit_passed=True,
        baseline_table_present=False,
        finite_metrics=finite,
        calibration_checked=False,
        claim_scope="em_artifact_audit_no_human",
        extra_failure_reasons=["EM Stage 1 passive logging is descriptive infrastructure; no scientific claim"],
    )
    analysis_path = write_json(out / "passive_logging_analysis.json", analysis)
    gate_path = write_evidence_gate(out / "evidence_gate.json", gate)

    print(f"branch=em stage=1 out_dir={out.resolve()}")
    print(f"n_log_entries={len(rows)} geomagnetic_status={geomagnetic['status']} network_access={geomagnetic['network_access']}")
    print(f"finite={finite} scientific_claim_allowed={gate['scientific_claim_allowed']}")
    print(f"analysis={analysis_path} evidence_gate={gate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
