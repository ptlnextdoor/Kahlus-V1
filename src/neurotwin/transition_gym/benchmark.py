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

from neurotwin.baseline_runner import run_baselines, transition_gym_regression_task
from neurotwin.gates import evaluate_gate, write_evidence_gate
from neurotwin.models.ktm import KTM, KTMConfig
from neurotwin.numerics import ignore_spurious_matmul_warnings
from neurotwin.repro import write_json
from neurotwin.scoring.metrics import mae, mse, pearsonr, r2_score, rank_models
from neurotwin.transition_gym import SyntheticWorldConfig, build_transition_gym
from neurotwin.transition_gym.operator_recovery import (
    V3Outcome,
    heldout_composition_recovery,
    non_commutativity_score,
    operator_recovery,
    response_profile_distances,
)

CLAIM_SCOPE = "synthetic_transition_operator_recovery"
LEADERBOARD_BASELINES = ("ridge", "autoregressive_ridge", "mlp", "transformer", "ssm_fallback")

# Adequate episode budget so operator recovery is well-posed (train split > state_dim+1).
DEFAULT_V3_BENCHMARK_CONFIG = SyntheticWorldConfig(n_episodes=96)


@dataclass(frozen=True)
class V3BenchmarkResult:
    config: SyntheticWorldConfig
    seed: int
    outcomes: list[V3Outcome]
    leaderboard: dict[str, dict[str, float]]
    ranking: list[dict[str, Any]]
    ktm_beats_baselines: bool
    gate: dict[str, Any]
    passed: bool
    failure_reasons: list[str]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"mse": mse(y_true, y_pred), "mae": mae(y_true, y_pred),
            "r2": r2_score(y_true, y_pred), "pearson_r": pearsonr(y_true.ravel(), y_pred.ravel())}


def _retrieval_knn(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, k: int = 5) -> np.ndarray:
    xt = np.asarray(x_train, dtype=np.float64).reshape(x_train.shape[0], -1)
    xe = np.asarray(x_test, dtype=np.float64).reshape(x_test.shape[0], -1)
    k = max(1, min(k, xt.shape[0]))
    preds = []
    for q in xe:
        idx = np.argsort(np.linalg.norm(xt - q, axis=1))[:k]
        preds.append(np.asarray(y_train, dtype=np.float64)[idx].mean(axis=0))
    return np.asarray(preds, dtype=np.float64)


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

    # Leaderboard: shared baselines + retrieval-kNN + (untrained) KTM, scored on the same task.
    task = transition_gym_regression_task(cfg)
    base = run_baselines(task, models=LEADERBOARD_BASELINES, train_steps=train_steps, seed=int(cfg.seed))
    leaderboard: dict[str, dict[str, float]] = dict(base.metrics_by_model)

    with ignore_spurious_matmul_warnings():
        retrieval_pred = _retrieval_knn(task.x_train, task.y_train, task.x_test)
        leaderboard["retrieval_knn"] = _metrics(task.y_test, retrieval_pred)

        ktm = KTM(KTMConfig(seed=int(cfg.seed), history_len=cfg.history_len, eeg_channels=cfg.eeg_channels,
                            n_perturbations=cfg.n_perturbations, horizon=cfg.horizon))
        test_idx = np.asarray(bundle.splits.test_episodes, dtype=int)
        ktm_pred = ktm.predict_response_profile(np.asarray(bundle.history_eeg)[test_idx])
        ktm_pred = ktm_pred.reshape(ktm_pred.shape[0], -1)
        leaderboard["ktm"] = _metrics(task.y_test, ktm_pred)

    ranking_rows = rank_models(leaderboard, metric="mse", higher_is_better=False)
    ranking = [{"model_id": r.model_id, "metric": r.metric, "value": r.value, "rank": r.rank} for r in ranking_rows]
    best_baseline_mse = min(leaderboard[m]["mse"] for m in leaderboard if m != "ktm")
    ktm_beats_baselines = bool(leaderboard["ktm"]["mse"] < best_baseline_mse)

    # Diagnostics.
    outcomes = [
        operator_recovery(bundle),
        heldout_composition_recovery(bundle),
        non_commutativity_score(bundle),
        response_profile_distances(bundle),  # informational
    ]
    by_name = {o.name: o for o in outcomes}
    required = ("operator_recovery", "heldout_composition_recovery", "non_commutativity")
    required_pass = all(by_name[n].passed for n in required)
    diagnostic_reasons = [f"{by_name[n].name}: {by_name[n].reason}" for n in required
                          if not by_name[n].passed and by_name[n].reason]

    try:
        bundle.splits.assert_no_episode_leakage()
        bundle.splits.assert_no_composition_leakage()
        split_audit_passed = True
    except ValueError:
        split_audit_passed = False
        diagnostic_reasons.append("split audit failed")

    leaderboard_finite = all(np.isfinite(list(m.values())).all() for m in leaderboard.values())
    diag_finite = all(_outcome_finite(o) for o in outcomes)
    finite = bool(leaderboard_finite and diag_finite)

    gate = evaluate_gate(
        branch="v3",
        dataset="transition_gym_synthetic",
        split_audit_passed=split_audit_passed,
        baseline_table_present=bool(leaderboard),
        finite_metrics=finite,
        calibration_checked=required_pass,
        claim_scope=CLAIM_SCOPE,
        extra_failure_reasons=diagnostic_reasons,
    )
    return V3BenchmarkResult(
        config=cfg, seed=int(cfg.seed), outcomes=outcomes, leaderboard=leaderboard,
        ranking=ranking, ktm_beats_baselines=ktm_beats_baselines, gate=gate,
        passed=bool(gate["scientific_claim_allowed"]), failure_reasons=list(gate["failure_reasons"]),
    )


def _outcome_finite(outcome: V3Outcome) -> bool:
    flat: list[float] = []
    for value in outcome.detail.values():
        if isinstance(value, dict):
            flat.extend(float(v) for v in value.values())
        elif isinstance(value, (int, float)):
            flat.append(float(value))
    return bool(np.isfinite(flat).all()) if flat else True


def benchmark_report(result: V3BenchmarkResult) -> dict[str, Any]:
    by_name = {o.name: o for o in result.outcomes}
    return {
        "schema": "kahlus.v3_transition_gym_benchmark.v1",
        "branch": "v3",
        "claim_status": "synthetic_scaffold_only",
        "claim_scope": CLAIM_SCOPE,
        "seed": result.seed,
        "config": result.config.__dict__,
        "diagnostics": [{"name": o.name, "passed": o.passed, "detail": o.detail, "reason": o.reason}
                        for o in result.outcomes],
        "baseline_leaderboard": result.leaderboard,
        "ranking": result.ranking,
        "operator_recovery_scores": by_name["operator_recovery"].detail,
        "heldout_composition_scores": by_name["heldout_composition_recovery"].detail,
        "non_commutativity_gap": by_name["non_commutativity"].detail,
        "ktm_beats_baselines": result.ktm_beats_baselines,
        "falsification_passed": result.passed,
        "scientific_claim_allowed": bool(result.gate["scientific_claim_allowed"]),
        "failure_reasons": result.failure_reasons,
        "evidence_gate": result.gate,
    }


def write_v3_report(out_dir: str | Path, result: V3BenchmarkResult) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = write_json(out / "v3_benchmark_report.json", benchmark_report(result))
    gate_path = write_evidence_gate(out / "v3_evidence_gate.json", result.gate)
    return {"report": report_path, "evidence_gate": gate_path}
