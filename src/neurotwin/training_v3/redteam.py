"""Adversarial falsification of a KTM ``synthetic_ktm_recovery`` candidate (PROPOSED / SYNTHETIC ONLY).

A recovery *candidate* — a single fair run where the trained KTM beats the strongest baseline
under a matched optimizer-step budget — is **not** accepted on its own. A single-seed win on the
team's own synthetic gym, especially from a model whose inductive bias may mirror the gym's
generator, is a spark, not evidence. This module hardens the recovery scope: it stays blocked
until the candidate survives a red-team battery —

1. symmetric model selection (learned baselines get the same best-validation selection the KTM
   gets; non-iterative baselines report their fitted result),
2. multi-seed stability (the win holds across seeds, with a positive conservative lower bound),
3. generator-family generalization (the win does not vanish when the synthetic generator family
   is changed away from the one whose structure the KTM may be aligned to).

Each failing check records an explicit, auditable blocker reason. No A100, no real data, no
scientific-superiority claim — a surviving candidate is still *synthetic* recovery evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np

RECOVERY_MARGIN_DEFAULT = 0.05
MIN_SEEDS = 5
# Fallback acceptance fraction when the normal-approx lower bound is treated as advisory: at least
# this fraction of seeds must have a strictly positive relative improvement.
MIN_POSITIVE_SEED_FRACTION = 0.8

# Explicit blocker reasons (stable strings; asserted by tests and surfaced in failure_reasons).
BLOCKER_ASYMMETRIC_VALIDATION = "asymmetric_validation_selection"
BLOCKER_SINGLE_SEED = "single_seed_only"
BLOCKER_AFFINITY_NOT_TESTED = "architecture_affinity_not_tested"
BLOCKER_FAMILY_GENERALIZATION = "generator_family_generalization_failed"
BLOCKER_LOWER_BOUND = "lower_bound_not_positive"
BLOCKER_BASELINE_SELECTION = "baseline_selection_not_symmetric"
BLOCKER_MEAN_BELOW_MARGIN = "mean_relative_improvement_below_margin"
BLOCKER_BUDGET_NOT_MATCHED = "budget_not_matched"

SYMMETRIC_SELECTION_POLICY = "symmetric_best_val"


@dataclass(frozen=True)
class SeedOutcome:
    """One fair KTM-vs-baselines comparison at a single seed on the *original* generator family."""

    seed: int
    ktm_mse: float
    best_baseline: str
    best_baseline_mse: float
    relative_improvement: float
    comparison_locked: bool
    ktm_checkpoint_policy: str = "best_val"
    baseline_checkpoint_policy: str = "best_val"  # learned-baseline policy: best_val|final_step

    def selection_is_symmetric(self) -> bool:
        return self.ktm_checkpoint_policy == "best_val" and self.baseline_checkpoint_policy == "best_val"


@dataclass(frozen=True)
class GeneratorFamilyOutcome:
    """KTM-vs-baselines on one synthetic generator family (e.g. linear / nonlinear / different rank)."""

    family: str
    ktm_mse: float
    best_baseline: str
    best_baseline_mse: float
    relative_improvement: float
    margin: float = RECOVERY_MARGIN_DEFAULT

    @property
    def recovery_allowed_for_family(self) -> bool:
        return bool(self.relative_improvement >= self.margin)


def seed_summary(seeds: Sequence[SeedOutcome]) -> dict[str, Any]:
    """Mean/std/min/max + a conservative normal-approx lower bound on relative improvement.

    The lower bound is ``mean - 1.96 * std / sqrt(n)`` (a one-sample normal approximation; with
    <2 seeds it is ``-inf`` because no spread can be estimated). ``n_positive`` supports the
    advisory ``>= MIN_POSITIVE_SEED_FRACTION`` fallback when the CI is treated as advisory.
    """

    rels = [float(s.relative_improvement) for s in seeds]
    n = len(rels)
    if n == 0:
        return {
            "n_seeds": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
            "sem": float("inf"), "lower_bound_95": float("-inf"), "n_positive": 0,
            "positive_fraction": 0.0, "finite": True,
        }
    mean = float(np.mean(rels))
    std = float(np.std(rels, ddof=1)) if n >= 2 else 0.0
    sem = std / math.sqrt(n) if n >= 2 else float("inf")
    lower_bound = mean - 1.96 * sem if n >= 2 else float("-inf")
    n_positive = int(sum(1 for r in rels if r > 0.0))
    return {
        "n_seeds": n,
        "mean": mean,
        "std": std,
        "min": float(np.min(rels)),
        "max": float(np.max(rels)),
        "sem": float(sem),
        "lower_bound_95": float(lower_bound),
        "n_positive": n_positive,
        "positive_fraction": float(n_positive / n),
        "finite": bool(np.isfinite(rels).all()),
    }


def recovery_redteam_gate(
    *,
    seeds: Sequence[SeedOutcome],
    families: Sequence[GeneratorFamilyOutcome],
    margin: float = RECOVERY_MARGIN_DEFAULT,
    min_seeds: int = MIN_SEEDS,
) -> dict[str, Any]:
    """Decide whether a recovery candidate survives the red-team battery.

    Returns a JSON-able dossier: ``recovery_allowed`` (True only if **every** check passes) and a
    sorted, de-duplicated ``blocker_reasons`` list. The function is the single source of truth for
    the ``synthetic_ktm_recovery`` allowance under red-team discipline; callers must not relax it.
    """

    margin = float(margin)
    blockers: list[str] = []

    # 1. Symmetric model selection. The candidate's KTM-only best-val advantage is the cardinal
    #    fairness hole; learned baselines must get the same selection.
    if not seeds:
        blockers.append(BLOCKER_BASELINE_SELECTION)
    else:
        if any(s.baseline_checkpoint_policy != "best_val" for s in seeds):
            blockers.append(BLOCKER_BASELINE_SELECTION)
        if any(s.ktm_checkpoint_policy == "best_val" and s.baseline_checkpoint_policy != "best_val"
               for s in seeds):
            blockers.append(BLOCKER_ASYMMETRIC_VALIDATION)

    # 2. Budget must stay locked (matched optimizer steps) at every seed.
    if seeds and any(not s.comparison_locked for s in seeds):
        blockers.append(BLOCKER_BUDGET_NOT_MATCHED)

    # 3. Multi-seed stability.
    summary = seed_summary(seeds)
    if summary["n_seeds"] < int(min_seeds):
        blockers.append(BLOCKER_SINGLE_SEED)
    if summary["mean"] < margin:
        blockers.append(BLOCKER_MEAN_BELOW_MARGIN)
    # Statistical guard. With >=2 seeds the conservative normal-approx lower bound is binding and
    # must clear 0 (a high-variance win with a negative lower bound is blocked even if the mean
    # passes). With <2 seeds no spread can be estimated, so fall back to the supermajority
    # positivity rule — though such a case is already blocked by the single-seed check.
    if summary["n_seeds"] >= 2:
        if summary["lower_bound_95"] <= 0.0:
            blockers.append(BLOCKER_LOWER_BOUND)
    elif summary["positive_fraction"] < MIN_POSITIVE_SEED_FRACTION:
        blockers.append(BLOCKER_LOWER_BOUND)

    # 4. Architecture-affinity / generator-family generalization.
    if not families:
        blockers.append(BLOCKER_AFFINITY_NOT_TESTED)
    elif any(not f.recovery_allowed_for_family for f in families):
        blockers.append(BLOCKER_FAMILY_GENERALIZATION)

    blocker_reasons = sorted(set(blockers))
    recovery_allowed = len(blocker_reasons) == 0
    return {
        "schema": "kahlus.ktm_recovery_redteam.v1",
        "claim_status": "synthetic_recovery_candidate_under_review",
        "recovery_allowed": bool(recovery_allowed),
        "blocker_reasons": blocker_reasons,
        "margin": margin,
        "min_seeds": int(min_seeds),
        "selection_policy": SYMMETRIC_SELECTION_POLICY,
        "seed_summary": summary,
        "generator_families": [
            {
                "family": f.family,
                "ktm_mse": float(f.ktm_mse),
                "best_baseline": f.best_baseline,
                "best_baseline_mse": float(f.best_baseline_mse),
                "relative_improvement": float(f.relative_improvement),
                "recovery_allowed_for_family": f.recovery_allowed_for_family,
            }
            for f in families
        ],
        "per_seed": [
            {
                "seed": s.seed,
                "ktm_mse": float(s.ktm_mse),
                "best_baseline": s.best_baseline,
                "best_baseline_mse": float(s.best_baseline_mse),
                "relative_improvement": float(s.relative_improvement),
                "comparison_locked": bool(s.comparison_locked),
                "ktm_checkpoint_policy": s.ktm_checkpoint_policy,
                "baseline_checkpoint_policy": s.baseline_checkpoint_policy,
            }
            for s in seeds
        ],
    }
