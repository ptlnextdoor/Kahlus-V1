#!/usr/bin/env python3
"""Kahlus-EM Stage 0 artifact-audit smoke runner (NO-HUMAN, SYNTHETIC).

Synthesizes phantom idle recordings under a baseline and a perturbed-environment condition,
extracts artifact features, and writes a descriptive audit report + a claim-blocking EM gate.
SAFETY: no stimulation, no high voltage, no human protocol, no clinical claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.config import load_config  # noqa: E402
from neurotwin.em import (  # noqa: E402
    EMContext,
    build_em_artifact_audit_gate,
    format_artifact_report_md,
    run_artifact_audit,
    synthesize_idle_recording,
)
from neurotwin.gates import write_evidence_gate  # noqa: E402
from neurotwin.repro import write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fs_hz, n_channels, n_samples, line_freq = 256.0, 8, 2048, 60.0
    baseline_strength, perturbed_strength = 0.0, 0.4
    seed = args.seed
    if args.config:
        cfg = load_config(args.config)
        rec = cfg.get("recording", {})
        fs_hz = float(rec.get("fs_hz", fs_hz))
        n_channels = int(rec.get("n_channels", n_channels))
        n_samples = int(rec.get("n_samples", n_samples))
        line_freq = float(rec.get("line_freq_hz", line_freq))
        conds = cfg.get("conditions", {})
        baseline_strength = float(conds.get("baseline", {}).get("field_strength_arb", baseline_strength))
        perturbed_strength = float(conds.get("perturbed", {}).get("field_strength_arb", perturbed_strength))
        seed = int(cfg.get("seed", seed))

    baseline_signal = synthesize_idle_recording(
        seed=seed, n_channels=n_channels, n_samples=n_samples, fs_hz=fs_hz,
        line_freq_hz=line_freq, em_field_strength_arb=baseline_strength,
    )
    condition_signal = synthesize_idle_recording(
        seed=seed + 1, n_channels=n_channels, n_samples=n_samples, fs_hz=fs_hz,
        line_freq_hz=line_freq, em_field_strength_arb=perturbed_strength,
    )
    report = run_artifact_audit(
        baseline_signal,
        condition_signal,
        fs_hz=fs_hz,
        line_freq_hz=line_freq,
        baseline_context=EMContext(condition_label="baseline", em_source="none", field_strength_arb=baseline_strength),
        condition_context=EMContext(
            condition_label="perturbed_environment", em_source="synthetic_field", field_strength_arb=perturbed_strength
        ),
    )
    gate = build_em_artifact_audit_gate(report)

    out = Path(args.out_dir)
    report_path = write_json(out / "artifact_audit_report.json", report)
    (out / "artifact_report.md").write_text(format_artifact_report_md(report), encoding="utf-8")
    gate_path = write_evidence_gate(out / "evidence_gate.json", gate)

    print(f"branch=em stage=0 out_dir={out.resolve()}")
    print(f"environment_effect_detected={report['response']['environment_effect_detected']}")
    print(f"finite={report['response']['finite']} scientific_claim_allowed={gate['scientific_claim_allowed']}")
    print(f"report={report_path} evidence_gate={gate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
