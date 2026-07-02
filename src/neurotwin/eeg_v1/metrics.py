from __future__ import annotations

import numpy as np


def smoothness_loss(y_hat: np.ndarray) -> float:
    """Second-difference smoothness penalty for future EEG windows."""

    arr = np.asarray(y_hat, dtype=np.float64)
    if arr.ndim < 3 or arr.shape[1] < 3:
        return 0.0
    second = arr[:, 2:, :] - 2.0 * arr[:, 1:-1, :] + arr[:, :-2, :]
    return float(np.sum(second**2))
