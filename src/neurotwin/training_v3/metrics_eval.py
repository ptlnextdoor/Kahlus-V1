"""Evaluation metrics for the trained KTM (PROPOSED / SYNTHETIC ONLY).

Trajectory metrics on held-out episodes, a calibration-coverage check that exercises the
uncertainty head, and an honest KTM-vs-baselines comparison. The comparison is what keeps the
``synthetic_ktm_recovery`` scope blocked until a trained KTM genuinely beats strong baselines —
a decreasing loss alone is never enough.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from neurotwin.baseline_runner import regression_metrics
from neurotwin.models.ktm import TorchKTM
from neurotwin.transition_gym import TransitionGymBundle


def calibration_coverage(
    profile: np.ndarray, target: np.ndarray, variance: np.ndarray
) -> dict[str, Any]:
    """1-sigma empirical coverage of the predicted per-(episode,perturbation) variance."""

    se = (profile - target) ** 2  # (B, K, H, C)
    per = se.reshape(se.shape[0], se.shape[1], -1).mean(axis=2)  # (B, K)
    rmse = np.sqrt(per)
    sigma = np.sqrt(np.asarray(variance, dtype=np.float64))  # (B, K)
    coverage = float(np.mean(rmse <= sigma))
    finite = bool(np.isfinite(coverage) and np.isfinite(sigma).all() and np.isfinite(rmse).all())
    return {
        "nominal_1sigma": 0.6827,
        "empirical_coverage_1sigma": coverage,
        "mean_predicted_var": float(np.mean(variance)),
        "mean_rmse": float(np.mean(rmse)),
        "finite": finite,
    }


def evaluate_ktm(
    model: TorchKTM,
    bundle: TransitionGymBundle,
    episodes: Sequence[int],
    device: torch.device | str,
) -> dict[str, Any]:
    """Trajectory metrics + calibration for ``model`` on the given episode subset."""

    model.eval()
    idx = np.asarray(list(episodes), dtype=int)
    history = torch.from_numpy(np.asarray(bundle.history_eeg, dtype=np.float32)[idx]).to(device)
    target = np.asarray(bundle.response_eeg, dtype=np.float64)[idx]  # (B, K, H, C)
    with torch.no_grad():
        profile = model.predict_response_profile(history).detach().cpu().numpy().astype(np.float64)
        variance = model.predict_uncertainty(history).detach().cpu().numpy().astype(np.float64)
    metrics = regression_metrics(
        target.reshape(target.shape[0], -1), profile.reshape(profile.shape[0], -1)
    )
    return {
        "trajectory": metrics,
        "calibration": calibration_coverage(profile, target, variance),
        "n_episodes": int(idx.size),
    }


BASELINE_BUDGET_POLICY = "matched_optimizer_steps"


def ktm_vs_baselines(
    ktm_mse: float,
    baseline_metrics: dict[str, dict[str, float]],
    *,
    ktm_train_steps: int | None = None,
    baseline_train_steps: int | None = None,
    ktm_world_size: int = 1,
    ktm_global_batch_size: int | None = None,
    baseline_batch_size: Any = "full_batch",
    margin: float = 0.0,
) -> dict[str, Any]:
    """Compare KTM test MSE against the strongest baseline under a *locked* comparison.

    A bare ``ktm_mse < best_mse`` is not enough to claim recovery: a longer training budget or
    more parallel throughput can manufacture that gap. The recovery scope is only earned when the
    comparison is *locked* — the baselines trained for at least as many optimizer steps as the KTM
    (matched-optimizer-steps policy; world size is recorded but never used to inflate the baseline
    budget, which would overfit the tiny synthetic baselines) — **and** the KTM beats the strongest
    baseline by at least ``margin`` relative MSE. The full budget provenance is recorded so the
    comparison is auditable.
    """

    budget = {
        "ktm_train_steps": int(ktm_train_steps) if ktm_train_steps is not None else None,
        "ktm_world_size": int(ktm_world_size),
        "ktm_global_batch_size": (
            int(ktm_global_batch_size) if ktm_global_batch_size is not None else None
        ),
        "baseline_train_steps": (
            int(baseline_train_steps) if baseline_train_steps is not None else None
        ),
        "baseline_batch_size": baseline_batch_size,
        "baseline_budget_policy": BASELINE_BUDGET_POLICY,
    }
    budget_matched = bool(
        baseline_train_steps is not None
        and ktm_train_steps is not None
        and int(baseline_train_steps) >= int(ktm_train_steps)
    )
    # Same task/seed/data is guaranteed by the shared task builder + shared seed in the caller.
    comparison_locked = budget_matched
    margin = float(margin)

    baseline_mse = {model: float(vals["mse"]) for model, vals in baseline_metrics.items()}
    if not baseline_mse:
        return {
            "ktm_mse": float(ktm_mse),
            "best_baseline": None,
            "best_baseline_mse": None,
            "relative_improvement": None,
            "margin": margin,
            "budget_matched": budget_matched,
            "comparison_locked": comparison_locked,
            "same_task_seed_data": True,
            "budget": budget,
            "ktm_beats_baselines": False,
        }
    best_model = min(baseline_mse, key=baseline_mse.get)
    best_mse = baseline_mse[best_model]
    relative_improvement = (best_mse - float(ktm_mse)) / best_mse if best_mse > 0 else 0.0
    earned = bool(comparison_locked and relative_improvement >= margin)
    return {
        "ktm_mse": float(ktm_mse),
        "best_baseline": best_model,
        "best_baseline_mse": best_mse,
        "relative_improvement": float(relative_improvement),
        "margin": margin,
        "budget_matched": budget_matched,
        "comparison_locked": comparison_locked,
        "same_task_seed_data": True,
        "budget": budget,
        "ktm_beats_baselines": earned,
    }
