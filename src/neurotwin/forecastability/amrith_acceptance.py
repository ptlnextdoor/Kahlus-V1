"""Amrith 127→128 overlap-trap acceptance — constitutionalized in M0."""
from __future__ import annotations

import numpy as np

L = 127
HORIZONS = (1, 2, 4, 8, 16, 32, 64)
RIDGE_ALPHA = 5.0


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    mat = xb.T @ xb + alpha * np.eye(xb.shape[1])
    return np.linalg.solve(mat, xb.T @ y)


def ridge_predict(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    return xb @ weights


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def run_amrith_acceptance(*, seed: int = 0) -> dict[str, object]:
    """Synthetic acceptance: persistence beats mean at h=1; overlap copy-trap is fake-good."""
    rng = np.random.default_rng(seed)
    n = 4000
    walks = np.cumsum(rng.normal(size=(n, L + max(HORIZONS) + 1)), axis=1).astype(np.float32)
    walks = walks - walks.mean(axis=1, keepdims=True)
    ctx = walks[:, :L]
    tgt = {h: walks[:, L + h - 1] for h in HORIZONS}
    last = ctx[:, -1]
    mean = ctx.mean(axis=1)
    pers_h1 = _mse(tgt[1], last)
    mean_h1 = _mse(tgt[1], mean)
    pers_hi = _mse(tgt[max(HORIZONS)], last)
    mean_hi = _mse(tgt[max(HORIZONS)], mean)

    n2 = 6000
    ar = np.zeros((n2, L + 1), dtype=np.float32)
    noise = rng.normal(size=(n2, L + 1)).astype(np.float32)
    for t in range(1, L + 1):
        ar[:, t] = 0.8 * ar[:, t - 1] + noise[:, t]
    ctx2, y2 = ar[:, :L], ar[:, L]
    split = n2 // 2
    weights = ridge_fit(ctx2[:split], y2[:split], RIDGE_ALPHA)
    ridge_mse = _mse(y2[split:], ridge_predict(weights, ctx2[split:]))
    mean_mse = _mse(y2[split:], ctx2[split:].mean(axis=1))

    seq2 = ar[:, : L + 1]
    xin, yout = seq2[:, :-1], seq2[:, 1:]
    cheat = np.concatenate([xin[:, 1:], xin[:, -1:]], axis=1)
    overlap_mse = _mse(yout, cheat)

    checks = {
        "persistence_beats_mean_h1": bool(pers_h1 < mean_h1),
        "persistence_error_grows_with_horizon": bool(pers_hi > pers_h1),
        "ridge_beats_mean_ar1": bool(np.isfinite(ridge_mse) and ridge_mse < 0.5 * mean_mse),
        "overlap_copy_fake_good": bool(overlap_mse < 0.2 * ridge_mse),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "metrics": {
            "persistence_h1_mse": pers_h1,
            "mean_h1_mse": mean_h1,
            "persistence_h64_mse": pers_hi,
            "mean_h64_mse": mean_hi,
            "ar1_ridge_mse": ridge_mse,
            "ar1_mean_mse": mean_mse,
            "overlap_copy_mse": overlap_mse,
        },
        "context_length": L,
        "horizons": list(HORIZONS),
    }
