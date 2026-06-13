"""Metrics for the Transition Gym: trajectory forecasting + commutativity diagnostics."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from neurotwin.scoring.metrics import mse, pearsonr
from neurotwin.transition_gym.perturbation_library import PerturbationLibrary


def trajectory_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Forecast quality for flattened trajectory predictions."""

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "mse": mse(y_true, y_pred),
        "pearson_r": pearsonr(y_true.ravel(), y_pred.ravel()),
        "finite": bool(np.isfinite(y_pred).all()),
    }


def mean_commutator_gap(
    library: PerturbationLibrary,
    pairs: Sequence[tuple[str, str]],
    states: np.ndarray,
) -> float:
    """Mean AB-vs-BA gap over a set of ordered perturbation pairs.

    A value clearly above zero confirms the perturbation battery is non-commutative.
    """

    if not pairs:
        return 0.0
    gaps = [library.commutator_gap(a, b, states) for a, b in pairs]
    return float(np.mean(gaps))
