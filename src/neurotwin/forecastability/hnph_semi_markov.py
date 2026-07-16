"""Cause-specific semi-Markov competing-risk nuisance comparator for HNPH B2.

This is the chief nuisance-only comparator the HNPH estimand is defined *relative
to*. The scientific quantity B2 measures is the residual predictive skill of
causal EEG *beyond* the strongest cheap physiological/temporal explanation, so a
weak comparator manufactures fake neural gain through its own approximation error
(the ``E[KL(p0 || q0)]`` term). It therefore receives serious, literature-grounded
engineering.

Design (grounded in the sleep-dynamics literature, Consensus 2026)
-----------------------------------------------------------------
Human sleep bouts are not memoryless, so a plain Markov chain is a poor null.
This comparator predicts the next-stable-transition outcome using only nuisance
information available at the forecast issue time:

- **State duration (sojourn).** Cause-specific discrete-time hazards indexed by
  elapsed epochs in the current macrostate. Non-geometric dwell times are the
  main reason semi-Markov beats Markov (Yang & Hursch 1973; Wang 2018 Weibull
  semi-Markov). Hazards are estimated empirically with Laplace smoothing rather
  than assuming a parametric family, which is more honest for a comparator.
- **Short prior-stage history.** The embedded transition structure is conditioned
  on the previous macrostate as well as the current one. Yetton 2018 found useful
  history is ~2 stages; adding more did not help, so second order is the design
  ceiling here.
- **Time of night.** Hazards vary with a coarse elapsed-night bin because REM
  propensity rises and NREM stability changes across the night (Kneib &
  Hennerfeind 2008 time-varying intensities). Clock time is labeled honestly as
  elapsed-night, never "circadian phase".

The outcome alphabet is the frozen five-way HNPH target
``{no_event, Wake, NREM, REM, Ambiguous}``. The comparator emits one calibrated
distribution per forecast anchor and never sees EEG. All fitting uses training
rows only; prediction consumes frozen fitted tables.

No heavy dependencies beyond numpy (already a base dep). No raw data access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

HNPH_SEMI_MARKOV_SCHEMA = "kahlus.hnph.semi_markov_competing_risk.v1"

# Frozen five-way HNPH outcome alphabet (matches label_reliability / baselines).
OUTCOME_ALPHABET: tuple[str, ...] = ("no_event", "Wake", "NREM", "REM", "Ambiguous")
_OUTCOME_INDEX = {name: i for i, name in enumerate(OUTCOME_ALPHABET)}


class SemiMarkovError(ValueError):
    """Raised when the comparator cannot be fit or applied coherently."""


@dataclass(frozen=True)
class SemiMarkovAnchor:
    """One forecast anchor's nuisance state at issue time, plus its observed outcome.

    ``outcome`` is an index into ``OUTCOME_ALPHABET`` (the adjudicated stable
    result within the lead band). It is used for fitting on training rows and for
    scoring on held-out rows; the comparator never reads it at prediction time.
    """

    current_state: str
    previous_state: str
    elapsed_epochs_in_state: int
    night_bin: int
    outcome: int

    def __post_init__(self) -> None:
        if self.elapsed_epochs_in_state < 0:
            raise SemiMarkovError("elapsed_epochs_in_state must be >= 0")
        if self.night_bin < 0:
            raise SemiMarkovError("night_bin must be >= 0")
        if not 0 <= self.outcome < len(OUTCOME_ALPHABET):
            raise SemiMarkovError(f"outcome must index {OUTCOME_ALPHABET}")


@dataclass(frozen=True)
class SemiMarkovComparator:
    """A fitted cause-specific semi-Markov competing-risk comparator.

    Predictions use hierarchical shrinkage: a fine cell's estimate is pulled
    toward its coarser parent by a weight set by the cell's own sample count, so
    a cell seen a few times leans on the parent and cannot overfit noise. This is
    what keeps the comparator from losing to a plain Markov null on structureless
    data (which would falsely inflate any downstream EEG gain).
    """

    schema: str
    n_outcomes: int
    dwell_bins: int
    night_bins: int
    laplace_alpha: float
    shrinkage_kappa: float
    # raw count tables at each granularity level (fine -> coarse)
    _full: dict[tuple[str, str, int, int], np.ndarray] = field(repr=False)
    _cp_night: dict[tuple[str, str, int], np.ndarray] = field(repr=False)
    _c_night: dict[tuple[str, int], np.ndarray] = field(repr=False)
    _current: dict[str, np.ndarray] = field(repr=False)
    _marginal_counts: np.ndarray = field(repr=False)

    def _shrink(self, counts: np.ndarray | None, parent: np.ndarray) -> np.ndarray:
        """Blend a level's counts toward its parent distribution by sample count."""
        if counts is None:
            return parent
        n = float(counts.sum())
        empirical = counts / n if n > 0 else parent
        weight = n / (n + self.shrinkage_kappa)
        return weight * empirical + (1.0 - weight) * parent

    def predict_one(self, anchor: SemiMarkovAnchor) -> np.ndarray:
        """Outcome distribution via marginal -> current -> (c,night) -> (c,p,night) -> full."""
        dwell = min(anchor.elapsed_epochs_in_state, self.dwell_bins - 1)
        night = min(anchor.night_bin, self.night_bins - 1)
        cur, prev = anchor.current_state, anchor.previous_state

        dist = _smoothed_distribution(self._marginal_counts, self.laplace_alpha)
        dist = self._shrink(self._current.get(cur), dist)
        dist = self._shrink(self._c_night.get((cur, night)), dist)
        dist = self._shrink(self._cp_night.get((cur, prev, night)), dist)
        dist = self._shrink(self._full.get((cur, prev, dwell, night)), dist)
        return dist / dist.sum()

    def predict(self, anchors: Sequence[SemiMarkovAnchor]) -> np.ndarray:
        if not anchors:
            raise SemiMarkovError("predict requires at least one anchor")
        return np.stack([self.predict_one(a) for a in anchors], axis=0)

    def summary(self) -> dict:
        return {
            "schema": self.schema,
            "n_outcomes": self.n_outcomes,
            "dwell_bins": self.dwell_bins,
            "night_bins": self.night_bins,
            "laplace_alpha": self.laplace_alpha,
            "shrinkage_kappa": self.shrinkage_kappa,
            "n_full_keys": len(self._full),
            "implementation": "cause_specific_semi_markov_competing_risk_hierarchical_shrinkage",
        }


def _smoothed_distribution(counts: np.ndarray, alpha: float) -> np.ndarray:
    total = counts.sum() + alpha * counts.size
    return (counts + alpha) / total


def fit_semi_markov_comparator(
    anchors: Sequence[SemiMarkovAnchor],
    *,
    dwell_bins: int = 6,
    night_bins: int = 4,
    laplace_alpha: float = 1.0,
    shrinkage_kappa: float = 40.0,
    n_outcomes: int = len(OUTCOME_ALPHABET),
) -> SemiMarkovComparator:
    """Fit the comparator from training anchors only.

    Stores raw counts at four granularity levels plus a global marginal. Prediction
    composes them with count-based hierarchical shrinkage (``shrinkage_kappa`` is
    the pseudo-count pulling each level toward its parent), so fine cells only take
    over once they have enough support to beat their coarser parent.
    """
    if not anchors:
        raise SemiMarkovError("fit requires at least one training anchor")
    if dwell_bins < 1 or night_bins < 1:
        raise SemiMarkovError("dwell_bins and night_bins must be >= 1")
    if laplace_alpha <= 0:
        raise SemiMarkovError("laplace_alpha must be > 0")
    if shrinkage_kappa <= 0:
        raise SemiMarkovError("shrinkage_kappa must be > 0")

    full: dict[tuple[str, str, int, int], np.ndarray] = {}
    cp_night: dict[tuple[str, str, int], np.ndarray] = {}
    c_night: dict[tuple[str, int], np.ndarray] = {}
    current: dict[str, np.ndarray] = {}
    marginal_counts = np.zeros(n_outcomes, dtype=np.float64)

    def _bump(store: dict, key, outcome: int) -> None:
        if key not in store:
            store[key] = np.zeros(n_outcomes, dtype=np.float64)
        store[key][outcome] += 1.0

    for a in anchors:
        dwell = min(a.elapsed_epochs_in_state, dwell_bins - 1)
        night = min(a.night_bin, night_bins - 1)
        _bump(full, (a.current_state, a.previous_state, dwell, night), a.outcome)
        _bump(cp_night, (a.current_state, a.previous_state, night), a.outcome)
        _bump(c_night, (a.current_state, night), a.outcome)
        _bump(current, a.current_state, a.outcome)
        marginal_counts[a.outcome] += 1.0

    return SemiMarkovComparator(
        schema=HNPH_SEMI_MARKOV_SCHEMA,
        n_outcomes=n_outcomes,
        dwell_bins=dwell_bins,
        night_bins=night_bins,
        laplace_alpha=laplace_alpha,
        shrinkage_kappa=shrinkage_kappa,
        _full=full,
        _cp_night=cp_night,
        _c_night=c_night,
        _current=current,
        _marginal_counts=marginal_counts,
    )


# --- Baseline comparators used to prove the semi-Markov model is actually strong ---


def persistence_distribution(anchor: SemiMarkovAnchor, n_outcomes: int = len(OUTCOME_ALPHABET)) -> np.ndarray:
    """Persistence null: predict 'no_event' (the current state persists)."""
    dist = np.full(n_outcomes, 1e-6, dtype=np.float64)
    dist[_OUTCOME_INDEX["no_event"]] = 1.0
    return dist / dist.sum()


def fit_first_order_markov(
    anchors: Sequence[SemiMarkovAnchor],
    *,
    laplace_alpha: float = 1.0,
    n_outcomes: int = len(OUTCOME_ALPHABET),
) -> dict[str, np.ndarray]:
    """First-order Markov null: outcome distribution conditioned on current state only."""
    if not anchors:
        raise SemiMarkovError("fit_first_order_markov requires anchors")
    counts: dict[str, np.ndarray] = {}
    for a in anchors:
        counts.setdefault(a.current_state, np.zeros(n_outcomes, dtype=np.float64))
        counts[a.current_state][a.outcome] += 1.0
    return {state: _smoothed_distribution(c, laplace_alpha) for state, c in counts.items()}


def cross_entropy_bits(distribution: np.ndarray, outcomes: np.ndarray, floor: float = 1e-12) -> float:
    """Mean per-anchor cross-entropy in bits (lower is better)."""
    distribution = np.clip(np.asarray(distribution, dtype=np.float64), floor, None)
    distribution = distribution / distribution.sum(axis=1, keepdims=True)
    picked = distribution[np.arange(len(outcomes)), outcomes]
    return float(-np.mean(np.log2(picked)))


def log_skill_gain_bits(
    model_distribution: np.ndarray,
    reference_distribution: np.ndarray,
    outcomes: np.ndarray,
) -> float:
    """Mean per-anchor log-skill gain of model over reference, in bits (higher = better)."""
    return cross_entropy_bits(reference_distribution, outcomes) - cross_entropy_bits(
        model_distribution, outcomes
    )
