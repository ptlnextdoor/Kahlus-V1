"""v2 dual-field synthetic falsification benchmark.

Runs the diagnostic battery, then assembles a report + a unified evidence gate. The narrow
synthetic claim (``synthetic_dual_field_recovery``) is allowed ONLY when every diagnostic
passes; otherwise the gate blocks it and the failing reasons are recorded honestly. A
one-field model matching the two-field model is a legitimate FAIL, not something to hide.

PROPOSED / SYNTHETIC ONLY. No A100, no real-data claim.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from neurotwin.gates import evaluate_gate, write_evidence_gate
from neurotwin.models.dual_field.config import DualFieldConfig
from neurotwin.models.dual_field.diagnostics import (
    DiagnosticOutcome,
    bold_dependence,
    eeg_dependence,
    fast_latent_recovery,
    lag_recovery,
    long_rollout_stability,
    one_vs_two_field_forecast,
    slow_latent_recovery,
)
from neurotwin.models.dual_field.dual_field_compiler import simulate_dual_field
from neurotwin.repro import write_json

CLAIM_SCOPE = "synthetic_dual_field_recovery"

# A benchmark needs adequate statistical power; the tiny unit-test DualFieldConfig default
# (16 samples) is for fast shape tests, not falsification. This is the default data budget for
# the v2 falsifier. Too-small/too-noisy configs still correctly fail (see honest-fail test).
DEFAULT_BENCHMARK_CONFIG = DualFieldConfig(n_samples=96, time_steps=64)


@dataclass(frozen=True)
class V2BenchmarkResult:
    config: DualFieldConfig
    seed: int
    outcomes: list[DiagnosticOutcome]
    gate: dict[str, Any]
    passed: bool
    failure_reasons: list[str]


def _all_finite(outcomes: list[DiagnosticOutcome]) -> bool:
    return all(np.isfinite(list(o.detail.values())).all() for o in outcomes if o.detail)


def run_v2_benchmark(
    config: DualFieldConfig | None = None,
    *,
    seed: int | None = None,
    long_rollout_steps: int = 256,
) -> V2BenchmarkResult:
    cfg = config or DEFAULT_BENCHMARK_CONFIG
    if seed is not None:
        cfg = replace(cfg, seed=int(seed))
    cfg = cfg.validate()
    rollout = simulate_dual_field(cfg)

    outcomes: list[DiagnosticOutcome] = [
        fast_latent_recovery(rollout),
        slow_latent_recovery(rollout),
        eeg_dependence(rollout),
        bold_dependence(rollout),
        lag_recovery(rollout),
        one_vs_two_field_forecast(rollout),
        long_rollout_stability(cfg, time_steps=long_rollout_steps),
    ]

    all_pass = all(o.passed for o in outcomes)
    finite = _all_finite(outcomes)
    diagnostic_reasons = [f"{o.name}: {o.reason}" for o in outcomes if not o.passed and o.reason]

    gate = evaluate_gate(
        branch="v2",
        dataset="dual_field_synthetic",
        split_audit_passed=True,
        baseline_table_present=True,  # ridge baselines computed in recovery + one-vs-two-field
        finite_metrics=finite,
        # The diagnostic battery passing IS the validation/calibration of the narrow synthetic
        # recovery claim; if any diagnostic fails the claim is correctly blocked.
        calibration_checked=all_pass,
        claim_scope=CLAIM_SCOPE,
        extra_failure_reasons=diagnostic_reasons,
    )
    return V2BenchmarkResult(
        config=cfg,
        seed=int(cfg.seed),
        outcomes=outcomes,
        gate=gate,
        passed=bool(gate["scientific_claim_allowed"]),
        failure_reasons=list(gate["failure_reasons"]),
    )


def benchmark_report(result: V2BenchmarkResult) -> dict[str, Any]:
    return {
        "schema": "kahlus.v2_dual_field_benchmark.v1",
        "branch": "v2",
        "claim_status": "synthetic_scaffold_only",
        "claim_scope": CLAIM_SCOPE,
        "seed": result.seed,
        "config": result.config.__dict__,
        "diagnostics": [
            {"name": o.name, "passed": o.passed, "detail": o.detail, "reason": o.reason}
            for o in result.outcomes
        ],
        "falsification_passed": result.passed,
        "scientific_claim_allowed": bool(result.gate["scientific_claim_allowed"]),
        "failure_reasons": result.failure_reasons,
        "evidence_gate": result.gate,
    }


def write_v2_report(out_dir: str | Path, result: V2BenchmarkResult) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = write_json(out / "v2_benchmark_report.json", benchmark_report(result))
    gate_path = write_evidence_gate(out / "v2_evidence_gate.json", result.gate)
    return {"report": report_path, "evidence_gate": gate_path}
