"""v3 Transition Gym synthetic operator-recovery falsification benchmark.

Runs a baseline leaderboard (reusing the shared baseline runner, plus a dependency-free
retrieval-kNN and the untrained KTM scaffold), the operator-recovery diagnostic battery, and
assembles a strict v3 evidence gate. The narrow synthetic claim
``synthetic_transition_operator_recovery`` is allowed ONLY when the gate-critical diagnostics
(operator recovery, held-out composition, non-commutativity) all pass and metrics are finite.

KTM beating the baselines is reported honestly as an *informational* metric: the scaffold is
untrained and is expected to lose to ridge. Per the dossier ("ridge winning is diagnostic, not
failure") that does not, by itself, block the gym's operator-recovery claim, which concerns the
benchmark's falsifiability — not KTM superiority. PROPOSED / SYNTHETIC ONLY. No A100, no real claim.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from neurotwin.baseline_runner import regression_metrics, run_baselines, transition_gym_regression_task
from neurotwin.falsification import Outcome, assemble_gate, build_report, write_report
from neurotwin.models.ktm import KTM, KTMConfig
from neurotwin.numerics import ignore_spurious_matmul_warnings
from neurotwin.scoring.metrics import rank_models
from neurotwin.transition_gym import SyntheticWorldConfig, build_transition_gym
from neurotwin.transition_gym.operator_recovery import (
    heldout_composition_recovery,
    non_commutativity_score,
    operator_recovery,
    response_profile_distances,
)

CLAIM_SCOPE = "synthetic_transition_operator_recovery"
# retrieval_knn is a registered baseline in the shared runner, so the leaderboard is one call.
LEADERBOARD_BASELINES = ("ridge", "autoregressive_ridge", "mlp", "transformer", "ssm_fallback", "retrieval_knn")
REQUIRED_DIAGNOSTICS = ("operator_recovery", "heldout_composition_recovery", "non_commutativity")

# Adequate episode budget so operator recovery is well-posed (train split > state_dim+1).
DEFAULT_V3_BENCHMARK_CONFIG = SyntheticWorldConfig(n_episodes=96)


@dataclass(frozen=True)
class V3BenchmarkResult:
    config: SyntheticWorldConfig
    seed: int
    outcomes: list[Outcome]
    leaderboard: dict[str, dict[str, float]]
    ranking: list[dict[str, Any]]
    ktm_beats_baselines: bool
    gate: dict[str, Any]
    passed: bool
    failure_reasons: list[str]


def _score_ktm(bundle, cfg, task) -> dict[str, float]:
    """Score the untrained KTM scaffold on the same response-profile task (informational)."""

    with ignore_spurious_matmul_warnings():
        ktm = KTM(KTMConfig(seed=int(cfg.seed), history_len=cfg.history_len, eeg_channels=cfg.eeg_channels,
                            n_perturbations=cfg.n_perturbations, horizon=cfg.horizon))
        test_idx = np.asarray(bundle.splits.test_episodes, dtype=int)
        pred = ktm.predict_response_profile(np.asarray(bundle.history_eeg)[test_idx])
        return regression_metrics(task.y_test, pred.reshape(pred.shape[0], -1))


def run_v3_benchmark(
    config: SyntheticWorldConfig | None = None,
    *,
    seed: int | None = None,
    train_steps: int = 40,
) -> V3BenchmarkResult:
    cfg = config or DEFAULT_V3_BENCHMARK_CONFIG
    if seed is not None:
        cfg = replace(cfg, seed=int(seed))
    cfg = cfg.validate()
    bundle = build_transition_gym(cfg)

    # Leaderboard: shared baselines (incl. retrieval_knn) in one runner call, plus the untrained KTM.
    task = transition_gym_regression_task(cfg)
    base = run_baselines(task, models=LEADERBOARD_BASELINES, train_steps=train_steps, seed=int(cfg.seed))
    leaderboard: dict[str, dict[str, float]] = dict(base.metrics_by_model)
    leaderboard["ktm"] = _score_ktm(bundle, cfg, task)

    ranking_rows = rank_models(leaderboard, metric="mse", higher_is_better=False)
    ranking = [{"model_id": r.model_id, "metric": r.metric, "value": r.value, "rank": r.rank} for r in ranking_rows]
    best_baseline_mse = min(leaderboard[m]["mse"] for m in leaderboard if m != "ktm")
    ktm_beats_baselines = bool(leaderboard["ktm"]["mse"] < best_baseline_mse)

    outcomes: list[Outcome] = [
        operator_recovery(bundle),
        heldout_composition_recovery(bundle),
        non_commutativity_score(bundle),
        response_profile_distances(bundle),  # informational, not gate-critical
    ]
    try:
        bundle.splits.assert_no_episode_leakage()
        bundle.splits.assert_no_composition_leakage()
        split_audit_passed = True
    except ValueError:
        split_audit_passed = False

    leaderboard_finite = all(np.isfinite(list(m.values())).all() for m in leaderboard.values())
    gate = assemble_gate(
        branch="v3",
        dataset="transition_gym_synthetic",
        claim_scope=CLAIM_SCOPE,
        outcomes=outcomes,
        required=REQUIRED_DIAGNOSTICS,
        split_audit_passed=split_audit_passed,
        baseline_table_present=bool(leaderboard),
        extra_finite=leaderboard_finite,
    )
    return V3BenchmarkResult(
        config=cfg, seed=int(cfg.seed), outcomes=outcomes, leaderboard=leaderboard,
        ranking=ranking, ktm_beats_baselines=ktm_beats_baselines, gate=gate,
        passed=bool(gate["scientific_claim_allowed"]), failure_reasons=list(gate["failure_reasons"]),
    )


def benchmark_report(result: V3BenchmarkResult) -> dict[str, Any]:
    by_name = {o.name: o for o in result.outcomes}
    return build_report(
        schema="kahlus.v3_transition_gym_benchmark.v1",
        branch="v3",
        claim_scope=CLAIM_SCOPE,
        seed=result.seed,
        config=result.config.__dict__,
        outcomes=result.outcomes,
        gate=result.gate,
        extra={
            "baseline_leaderboard": result.leaderboard,
            "ranking": result.ranking,
            "operator_recovery_scores": by_name["operator_recovery"].detail,
            "heldout_composition_scores": by_name["heldout_composition_recovery"].detail,
            "non_commutativity_gap": by_name["non_commutativity"].detail,
            "ktm_beats_baselines": result.ktm_beats_baselines,
        },
    )


def write_v3_report(out_dir: str | Path, result: V3BenchmarkResult) -> dict[str, Path]:
    return write_report(out_dir, report=benchmark_report(result), gate=result.gate, prefix="v3")
