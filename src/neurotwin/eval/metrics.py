"""Compatibility re-export for legacy neurotwin.eval.metrics imports.

New code should import shared metrics from neurotwin.scoring.metrics. The eval
package keeps this shim so older callers do not get a surprise import break.
"""

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
