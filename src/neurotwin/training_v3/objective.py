"""KTM training objective + numerical guards (PROPOSED / SYNTHETIC ONLY).

The loss combines a trajectory MSE, a response-profile term (horizon-averaged MSE), and a
Gaussian negative log-likelihood that exercises the uncertainty head (calibration). Pure guard
helpers (finite check, loss-explosion detector) live here so they are unit-testable without a
training loop. No scientific claim is made by minimizing this loss.
"""

from __future__ import annotations

from collections import deque
import math

import torch

from neurotwin.training_v3.config import KTMTrainConfig


def ktm_loss(
    pred: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    cfg: KTMTrainConfig,
    *,
    profile_pred: torch.Tensor | None = None,
    profile_target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted (trajectory + profile + Gaussian NLL) loss.

    ``pred``/``target`` are ``(B, H, C)``; ``log_var`` is ``(B,)``. Returns the scalar loss and
    a JSON-able dict of its finite components. If full profile tensors ``(B, K, H, C)`` are
    supplied, the profile term trains the entire finite response profile; otherwise it falls back
    to the selected perturbation's horizon-averaged profile.
    """

    se = (pred - target) ** 2
    trajectory = se.mean()
    if profile_pred is not None and profile_target is not None:
        profile = ((profile_pred - profile_target) ** 2).mean()
    else:
        # Profile term: error of the horizon-averaged response (the C_K profile magnitude).
        profile = ((pred.mean(dim=1) - target.mean(dim=1)) ** 2).mean()
    # Gaussian NLL with a predicted per-sample variance (drives the uncertainty head).
    per_sample_se = se.reshape(se.shape[0], -1).mean(dim=1)
    nll = (0.5 * (log_var + per_sample_se / torch.exp(log_var))).mean()

    loss = cfg.w_traj * trajectory + cfg.w_profile * profile + cfg.w_nll * nll
    components = {
        "trajectory": float(trajectory.detach()),
        "profile": float(profile.detach()),
        "nll": float(nll.detach()),
        "total": float(loss.detach()),
    }
    return loss, components


def is_finite_loss(value: float) -> bool:
    return math.isfinite(float(value))


class LossExplosionGuard:
    """Aborts training when a loss spikes far above the recent running median.

    Keeps a sliding window of finite losses. Once warmed up, a new loss exceeding
    ``factor * median(window)`` is flagged as exploded. Median (not mean) so a single prior
    spike cannot desensitize the guard.
    """

    def __init__(self, factor: float, window: int = 16, warmup: int = 8) -> None:
        if factor <= 1.0:
            raise ValueError("factor must be > 1.0")
        self.factor = float(factor)
        self.warmup = int(warmup)
        self._window: deque[float] = deque(maxlen=int(window))

    def update(self, loss: float) -> bool:
        """Record ``loss``; return True iff it counts as an explosion (do not record spikes)."""

        if not is_finite_loss(loss):
            return True
        if len(self._window) >= self.warmup:
            ordered = sorted(self._window)
            median = ordered[len(ordered) // 2]
            if median > 0 and loss > self.factor * median:
                return True
        self._window.append(float(loss))
        return False
