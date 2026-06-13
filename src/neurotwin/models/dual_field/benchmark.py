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

from neurotwin.falsification import Outcome, assemble_gate, build_report, write_report
from neurotwin.models.dual_field.config import DualFieldConfig
from neurotwin.models.dual_field.diagnostics import (
    bold_dependence,
    eeg_dependence,
    fast_latent_recovery,
    lag_recovery,
    long_rollout_stability,
    one_vs_two_field_forecast,
    slow_latent_recovery,
)
from neurotwin.models.dual_field.dual_field_compiler import simulate_dual_field

CLAIM_SCOPE = "synthetic_dual_field_recovery"

# A benchmark needs adequate statistical power; the tiny unit-test DualFieldConfig default
# (16 samples) is for fast shape tests, not falsification. This is the default data budget for
# the v2 falsifier. Too-small/too-noisy configs still correctly fail (see honest-fail test).
DEFAULT_BENCHMARK_CONFIG = DualFieldConfig(n_samples=96, time_steps=64)


@dataclass(frozen=True)
class V2BenchmarkResult:
    config: DualFieldConfig
    seed: int
    outcomes: list[Outcome]
    gate: dict[str, Any]
    passed: bool
    failure_reasons: list[str]


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

    outcomes: list[Outcome] = [
        fast_latent_recovery(rollout),
        slow_latent_recovery(rollout),
        eeg_dependence(rollout),
        bold_dependence(rollout),
        lag_recovery(rollout),
        one_vs_two_field_forecast(rollout),
        long_rollout_stability(cfg, time_steps=long_rollout_steps),
    ]

    # Every diagnostic is gate-critical for the v2 recovery claim. Ridge baselines are computed
    # inside the recovery + one-vs-two-field probes, so the baseline table is present.
    gate = assemble_gate(
        branch="v2",
        dataset="dual_field_synthetic",
        claim_scope=CLAIM_SCOPE,
        outcomes=outcomes,
        required=[o.name for o in outcomes],
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
    return build_report(
        schema="kahlus.v2_dual_field_benchmark.v1",
        branch="v2",
        claim_scope=CLAIM_SCOPE,
        seed=result.seed,
        config=result.config.__dict__,
        outcomes=result.outcomes,
        gate=result.gate,
    )


def write_v2_report(out_dir: str | Path, result: V2BenchmarkResult) -> dict[str, Path]:
    return write_report(out_dir, report=benchmark_report(result), gate=result.gate, prefix="v2")
