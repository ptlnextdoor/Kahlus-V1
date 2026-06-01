"""Shared scoring primitives used by benchmarks, eval, and training."""

from neurotwin.scoring.metrics import (
    RankingRow,
    bandpower_error,
    bootstrap_ci,
    mae,
    mse,
    pearsonr,
    r2_score,
    rank_models,
    regionwise_pearsonr,
    retrieval_accuracy,
    spectral_error,
    spearmanr,
)

__all__ = [
    "RankingRow",
    "bandpower_error",
    "bootstrap_ci",
    "mae",
    "mse",
    "pearsonr",
    "r2_score",
    "rank_models",
    "regionwise_pearsonr",
    "retrieval_accuracy",
    "spectral_error",
    "spearmanr",
]
