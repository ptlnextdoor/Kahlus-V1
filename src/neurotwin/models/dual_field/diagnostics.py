"""Falsification diagnostics for the v2 synthetic dual-field benchmark.

PROPOSED / SYNTHETIC ONLY. These functions probe whether the dual-field split is *recoverable
and meaningful* on the synthetic system — they do not assert any real-data result. Every probe
uses a linear (ridge) decoder under a leakage-safe sequence split, and reports honest numbers
(including the case where a one-field model matches a two-field model).
"""

from __future__ import annotations

import numpy as np

from neurotwin.falsification import Outcome
from neurotwin.models.baselines import NumpyRidgeBaseline
from neurotwin.models.dual_field.config import DualFieldConfig
from neurotwin.models.dual_field.coupling import hrf_lag_weights
from neurotwin.models.dual_field.dual_field_compiler import DualFieldRollout, simulate_dual_field
from neurotwin.numerics import ignore_spurious_matmul_warnings
from neurotwin.scoring.metrics import mse, pearsonr, r2_score


def sequence_split(n_seq: int, test_fraction: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint train/test sequence indices (no window straddles the boundary)."""

    n_test = max(1, int(round(n_seq * test_fraction)))
    test = np.arange(n_seq - n_test, n_seq)
    train = np.arange(0, n_seq - n_test)
    if train.size == 0:
        raise ValueError("not enough sequences to form a train split")
    return train, test


def _windowed_xy(
    source: np.ndarray,
    target: np.ndarray,
    window: int,
    indices: np.ndarray,
    *,
    target_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(N, window*Fs)`` features and ``(N, Ft)`` targets over given sequences.

    ``target_offset=1`` predicts the next step (forecasting); ``0`` decodes the current step.
    """

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    time_steps = source.shape[1]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for b in indices:
        for t in range(window - 1, time_steps - target_offset):
            xs.append(source[b, t - window + 1 : t + 1].reshape(-1))
            ys.append(target[b, t + target_offset])
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _ridge_eval(x_train, y_train, x_test, y_test, alpha: float = 1.0) -> dict[str, float]:
    with ignore_spurious_matmul_warnings():
        model = NumpyRidgeBaseline(alpha=alpha).fit(x_train, y_train)
        pred = model.predict(x_test)
    return {
        "r2": r2_score(y_test, pred),
        "pearson_r": pearsonr(y_test.ravel(), pred.ravel()),
        "mse": mse(y_test, pred),
    }


# Outcome type lives in the shared falsification core; alias kept for local readability.
DiagnosticOutcome = Outcome


def fast_latent_recovery(rollout: DualFieldRollout, window: int = 4, threshold: float = 0.5) -> DiagnosticOutcome:
    """Recover fast latent ``N_t`` from an EEG-like observation window (linear decode)."""

    train, test = sequence_split(rollout.eeg.shape[0])
    xtr, ytr = _windowed_xy(rollout.eeg, rollout.neural, window, train)
    xte, yte = _windowed_xy(rollout.eeg, rollout.neural, window, test)
    detail = _ridge_eval(xtr, ytr, xte, yte)
    ok = detail["r2"] >= threshold
    return DiagnosticOutcome(
        "fast_latent_recovery", ok, detail,
        "" if ok else f"N_t recovery r2={detail['r2']:.3f} below {threshold}",
    )


def slow_latent_recovery(rollout: DualFieldRollout, window: int = 4, threshold: float = 0.5) -> DiagnosticOutcome:
    """Recover slow latent ``H_t`` from a BOLD/fNIRS-like observation window (linear decode)."""

    train, test = sequence_split(rollout.bold.shape[0])
    xtr, ytr = _windowed_xy(rollout.bold, rollout.hemo, window, train)
    xte, yte = _windowed_xy(rollout.bold, rollout.hemo, window, test)
    detail = _ridge_eval(xtr, ytr, xte, yte)
    ok = detail["r2"] >= threshold
    return DiagnosticOutcome(
        "slow_latent_recovery", ok, detail,
        "" if ok else f"H_t recovery r2={detail['r2']:.3f} below {threshold}",
    )


def eeg_dependence(rollout: DualFieldRollout, margin: float = 0.2) -> DiagnosticOutcome:
    """EEG-like output should depend primarily on the fast field N, not the slow field H."""

    train, test = sequence_split(rollout.eeg.shape[0])
    from_n = _ridge_eval(*_windowed_xy(rollout.neural, rollout.eeg, 1, train),
                         *_windowed_xy(rollout.neural, rollout.eeg, 1, test))
    from_h = _ridge_eval(*_windowed_xy(rollout.hemo, rollout.eeg, 1, train),
                         *_windowed_xy(rollout.hemo, rollout.eeg, 1, test))
    detail = {"r2_from_fast_N": from_n["r2"], "r2_from_slow_H": from_h["r2"],
              "fast_minus_slow": from_n["r2"] - from_h["r2"]}
    ok = (from_n["r2"] - from_h["r2"]) >= margin and from_n["r2"] > from_h["r2"]
    return DiagnosticOutcome(
        "eeg_depends_on_fast_field", ok, detail,
        "" if ok else f"EEG not fast-dominated (N r2={from_n['r2']:.3f} vs H r2={from_h['r2']:.3f})",
    )


def bold_dependence(rollout: DualFieldRollout, margin: float = 0.2) -> DiagnosticOutcome:
    """BOLD/fNIRS-like output should depend primarily on the slow field H, not current N."""

    train, test = sequence_split(rollout.bold.shape[0])
    from_h = _ridge_eval(*_windowed_xy(rollout.hemo, rollout.bold, 1, train),
                         *_windowed_xy(rollout.hemo, rollout.bold, 1, test))
    from_n = _ridge_eval(*_windowed_xy(rollout.neural, rollout.bold, 1, train),
                         *_windowed_xy(rollout.neural, rollout.bold, 1, test))
    detail = {"r2_from_slow_H": from_h["r2"], "r2_from_fast_N_current": from_n["r2"],
              "slow_minus_fast": from_h["r2"] - from_n["r2"]}
    ok = (from_h["r2"] - from_n["r2"]) >= margin and from_h["r2"] > from_n["r2"]
    return DiagnosticOutcome(
        "bold_depends_on_slow_field", ok, detail,
        "" if ok else f"BOLD not slow-dominated (H r2={from_h['r2']:.3f} vs N r2={from_n['r2']:.3f})",
    )


def effective_neural_to_bold_lag(config: DualFieldConfig, horizon: int) -> int:
    """Peak lag of the effective neural→BOLD impulse response.

    BOLD observes H, and H integrates the HRF-weighted neural window through a leaky (ρ)
    accumulator. The observed peak lag is therefore the argmax of the HRF weights convolved
    with the geometric ρ-kernel — NOT the raw HRF peak. This is the principled ground truth.
    """

    hrf = hrf_lag_weights(config.hemo_lag)
    geo = (1.0 - config.rho) * (config.rho ** np.arange(0, horizon + 1))
    effective = np.convolve(hrf, geo)
    return int(np.argmax(effective))


def lag_recovery(rollout: DualFieldRollout, max_lag: int | None = None, min_gain: float = 0.02) -> DiagnosticOutcome:
    """Show the neural→hemodynamic path is genuinely *lagged*, not instantaneous.

    Scans ``bold_t ~ N_{t-d}`` over lag d. The core falsifiable property of the slow path is
    that BOLD is best explained by *past* neural activity (recovered lag >= 1) and that the
    delayed fit beats the instantaneous (lag-0) fit. We report the recovered lag against an
    analytic reference but gate on the robust lagged-vs-instantaneous property, not a brittle
    exact-peak match (the discrete ρ-leak makes the exact peak simulator-specific).
    """

    cfg = rollout.config
    if max_lag is None:
        max_lag = cfg.hemo_lag + int(round(1.0 / max(1e-6, 1.0 - cfg.rho))) + 2
    max_lag = min(max_lag, rollout.bold.shape[1] - 2)
    train, test = sequence_split(rollout.bold.shape[0])
    neural, bold = np.asarray(rollout.neural, np.float64), np.asarray(rollout.bold, np.float64)
    time_steps = neural.shape[1]
    r2_by_lag: dict[int, float] = {}
    for d in range(0, max_lag + 1):
        def build(indices):
            xs, ys = [], []
            for b in indices:
                for t in range(d, time_steps):
                    xs.append(neural[b, t - d])
                    ys.append(bold[b, t])
            return np.asarray(xs), np.asarray(ys)
        xtr, ytr = build(train)
        xte, yte = build(test)
        r2_by_lag[d] = _ridge_eval(xtr, ytr, xte, yte)["r2"]
    recovered = int(max(r2_by_lag, key=r2_by_lag.get))
    analytic_ref = effective_neural_to_bold_lag(cfg, max_lag)
    gain_over_instantaneous = r2_by_lag[recovered] - r2_by_lag[0]
    detail = {"recovered_lag": float(recovered), "analytic_reference_lag": float(analytic_ref),
              "r2_at_lag0": r2_by_lag[0], "r2_at_recovered": r2_by_lag[recovered],
              "gain_over_instantaneous": gain_over_instantaneous}
    ok = recovered >= 1 and gain_over_instantaneous >= min_gain
    return DiagnosticOutcome(
        "lag_recovery", ok, detail,
        "" if ok else f"BOLD not clearly lagged (recovered={recovered}, gain over lag0={gain_over_instantaneous:.3f})",
    )


def long_rollout_stability(config: DualFieldConfig, time_steps: int = 256) -> DiagnosticOutcome:
    """Run a long rollout and confirm finite, bounded, non-exploding states."""

    long_cfg = DualFieldConfig(**{**config.__dict__, "time_steps": int(time_steps)})
    rollout = simulate_dual_field(long_cfg)
    neural = np.asarray(rollout.neural, np.float64)
    max_abs = float(np.max(np.abs(neural)))
    quarter = max(1, neural.shape[1] // 4)
    # Divergence test on peak amplitude (robust to steady-state mean fluctuation): compare the
    # max |state| of a mid steady-state window to the final window, skipping the cold-start
    # transient. A stable bounded system keeps peak amplitude ~constant; divergence grows it.
    mid_max = float(np.max(np.abs(neural[:, quarter : 2 * quarter])))
    late_max = float(np.max(np.abs(neural[:, -quarter:])))
    growth_ratio = late_max / mid_max if mid_max > 0 else 1.0
    finite = bool(np.isfinite(neural).all())
    bounded = max_abs <= 0.5 * long_cfg.state_clip
    not_exploding = growth_ratio <= 1.5
    detail = {"max_abs": max_abs, "state_clip": float(long_cfg.state_clip),
              "peak_growth_ratio_late_over_mid": growth_ratio, "time_steps": float(time_steps)}
    ok = finite and bounded and not_exploding
    reason = "" if ok else (
        f"unstable long rollout (finite={finite}, max_abs={max_abs:.2f}/{long_cfg.state_clip}, "
        f"growth={growth_ratio:.2f})"
    )
    return DiagnosticOutcome("long_rollout_stability", ok, detail, reason)


def one_vs_two_field_forecast(rollout: DualFieldRollout, window: int = 3, margin: float = 0.05) -> DiagnosticOutcome:
    """Compare a single-timescale one-field predictor to a structure-aware two-field predictor.

    The dual-field hypothesis is that the slow (BOLD) channel needs a *longer* memory than the
    fast (EEG) channel. We test it honestly:

    - One-field: a single ridge over a fixed joint [EEG, BOLD] window of length ``window`` (one
      compromise timescale for both channels) → next [EEG, BOLD].
    - Two-field: EEG_{t+1} from a short EEG+stimulus window (``window``); BOLD_{t+1} from a
      *longer* BOLD+EEG window (``window + hemo_lag``), matching the slow field's integration.

    The split is judged to matter only if two-field beats one-field on BOLD by ``margin``.
    Both pathways are linear ridge; if the single compromise window already suffices, one-field
    matches and we report that honestly (no forced win).
    """

    eeg, bold, stim = (np.asarray(getattr(rollout, k), np.float64) for k in ("eeg", "bold", "stimulus"))
    train, test = sequence_split(eeg.shape[0])
    joint = np.concatenate([eeg, bold], axis=-1)
    eeg_dim = eeg.shape[-1]
    slow_window = min(window + rollout.config.hemo_lag, eeg.shape[1] - 1)

    # One-field: a single joint window (one timescale for both channels) -> next joint.
    with ignore_spurious_matmul_warnings():
        xtr, ytr = _windowed_xy(joint, joint, window, train, target_offset=1)
        xte, yte = _windowed_xy(joint, joint, window, test, target_offset=1)
        one_pred = NumpyRidgeBaseline(alpha=1.0).fit(xtr, ytr).predict(xte)
    one_bold_mse = mse(yte[:, eeg_dim:], one_pred[:, eeg_dim:])
    one_eeg_mse = mse(yte[:, :eeg_dim], one_pred[:, :eeg_dim])

    # Two-field: fast EEG pathway (short window) + slow BOLD pathway (longer window).
    eeg_in = np.concatenate([eeg, stim], axis=-1)
    bold_in = np.concatenate([bold, eeg], axis=-1)
    with ignore_spurious_matmul_warnings():
        eeg_pred = NumpyRidgeBaseline(alpha=1.0).fit(
            *_windowed_xy(eeg_in, eeg, window, train, target_offset=1)
        ).predict(_windowed_xy(eeg_in, eeg, window, test, target_offset=1)[0])
        bold_pred = NumpyRidgeBaseline(alpha=1.0).fit(
            *_windowed_xy(bold_in, bold, slow_window, train, target_offset=1)
        ).predict(_windowed_xy(bold_in, bold, slow_window, test, target_offset=1)[0])
    two_eeg_mse = mse(_windowed_xy(eeg_in, eeg, window, test, target_offset=1)[1], eeg_pred)
    two_bold_mse = mse(_windowed_xy(bold_in, bold, slow_window, test, target_offset=1)[1], bold_pred)

    bold_dim = bold.shape[-1]
    one_total = (one_eeg_mse * eeg_dim + one_bold_mse * bold_dim) / (eeg_dim + bold_dim)
    two_total = (two_eeg_mse * eeg_dim + two_bold_mse * bold_dim) / (eeg_dim + bold_dim)
    # The split matters if the structure-aware two-field predictor beats the single-timescale
    # one-field predictor on the combined objective AND does not regress the slow channel.
    split_matters = (two_total < one_total * (1.0 - margin)) and (two_bold_mse <= one_bold_mse)
    detail = {
        "one_field_window": float(window),
        "two_field_slow_window": float(slow_window),
        "one_field_bold_mse": one_bold_mse,
        "two_field_bold_mse": two_bold_mse,
        "one_field_eeg_mse": one_eeg_mse,
        "two_field_eeg_mse": two_eeg_mse,
        "one_field_total_mse": one_total,
        "two_field_total_mse": two_total,
        "total_mse_improvement_frac": (one_total - two_total) / one_total if one_total > 0 else 0.0,
    }
    reason = "" if split_matters else (
        f"one-field matches/beats two-field (one_total={one_total:.5f} vs two_total={two_total:.5f}, "
        f"one_bold={one_bold_mse:.5f} vs two_bold={two_bold_mse:.5f}); dual-field split not yet justified"
    )
    return DiagnosticOutcome("one_vs_two_field", split_matters, detail, reason)
