"""Acceptance tests for the HNPH semi-Markov competing-risk comparator.

These implement the addendum's comparator-acceptance obligations on synthetic
worlds where the truth is known: the comparator must beat persistence and a
first-order Markov null when the data-generating process has dwell-dependent,
history-dependent, and time-of-night-dependent transition structure, and must
NOT beat them when the outcome is independent of the nuisance state (the null).
"""

from __future__ import annotations

import numpy as np
import pytest

from neurotwin.forecastability.hnph_semi_markov import (
    OUTCOME_ALPHABET,
    SemiMarkovAnchor,
    SemiMarkovError,
    cross_entropy_bits,
    fit_first_order_markov,
    fit_semi_markov_comparator,
    log_skill_gain_bits,
    persistence_distribution,
)

_IDX = {n: i for i, n in enumerate(OUTCOME_ALPHABET)}


def _generate_dwell_dependent_world(n: int, seed: int) -> list[SemiMarkovAnchor]:
    """Synthetic anchors where transition probability rises with dwell + time-of-night.

    NREM->REM ('REM' outcome) hazard increases with elapsed epochs in state and
    with later night bins, exactly the non-memoryless structure a semi-Markov
    model captures and a first-order Markov model cannot.
    """
    rng = np.random.default_rng(seed)
    states = ["Wake", "NREM", "REM"]
    anchors: list[SemiMarkovAnchor] = []
    for _ in range(n):
        current = rng.choice(states)
        previous = rng.choice(states)
        dwell = int(rng.integers(0, 6))
        night = int(rng.integers(0, 4))
        # Hazard of a REM transition rises with dwell and night; else mostly no_event.
        rem_hazard = 0.05 + 0.12 * dwell + 0.10 * night
        rem_hazard = min(rem_hazard, 0.95) if current == "NREM" else 0.02
        if rng.random() < rem_hazard:
            outcome = _IDX["REM"]
        elif rng.random() < 0.10:
            outcome = _IDX["Ambiguous"]
        else:
            outcome = _IDX["no_event"]
        anchors.append(SemiMarkovAnchor(current, previous, dwell, night, outcome))
    return anchors


def _generate_null_world(n: int, seed: int) -> list[SemiMarkovAnchor]:
    """Outcome independent of all nuisance state: no model can beat the marginal."""
    rng = np.random.default_rng(seed)
    states = ["Wake", "NREM", "REM"]
    anchors: list[SemiMarkovAnchor] = []
    for _ in range(n):
        outcome = int(rng.choice([_IDX["no_event"], _IDX["REM"], _IDX["Ambiguous"]], p=[0.6, 0.25, 0.15]))
        anchors.append(
            SemiMarkovAnchor(
                current_state=rng.choice(states),
                previous_state=rng.choice(states),
                elapsed_epochs_in_state=int(rng.integers(0, 6)),
                night_bin=int(rng.integers(0, 4)),
                outcome=outcome,
            )
        )
    return anchors


def _outcomes(anchors):
    return np.array([a.outcome for a in anchors], dtype=np.int64)


def test_beats_persistence_and_markov_on_structured_world():
    train = _generate_dwell_dependent_world(4000, seed=0)
    test = _generate_dwell_dependent_world(4000, seed=1)
    y = _outcomes(test)

    sm = fit_semi_markov_comparator(train)
    markov = fit_first_order_markov(train)

    sm_pred = sm.predict(test)
    persist_pred = np.stack([persistence_distribution(a) for a in test])
    markov_pred = np.stack([markov[a.current_state] for a in test])

    # Semi-Markov must have strictly positive log-skill gain over BOTH nulls,
    # because dwell + night structure is real and only it captures them.
    assert log_skill_gain_bits(sm_pred, persist_pred, y) > 0.05
    assert log_skill_gain_bits(sm_pred, markov_pred, y) > 0.01


def test_returns_no_gain_on_null_world():
    train = _generate_null_world(4000, seed=2)
    test = _generate_null_world(4000, seed=3)
    y = _outcomes(test)

    sm = fit_semi_markov_comparator(train)
    markov = fit_first_order_markov(train)
    sm_pred = sm.predict(test)
    markov_pred = np.stack([markov[a.current_state] for a in test])

    # On a world with no nuisance structure, semi-Markov gain over Markov must be
    # ~zero (within noise). A comparator that "wins" here is overfitting.
    gain = log_skill_gain_bits(sm_pred, markov_pred, y)
    assert abs(gain) < 0.02


def test_dwell_hazard_is_actually_learned():
    # Predicted REM probability must rise with dwell in the structured world.
    train = _generate_dwell_dependent_world(8000, seed=4)
    sm = fit_semi_markov_comparator(train)
    low = sm.predict_one(SemiMarkovAnchor("NREM", "Wake", 0, 2, _IDX["no_event"]))
    high = sm.predict_one(SemiMarkovAnchor("NREM", "Wake", 5, 2, _IDX["no_event"]))
    assert high[_IDX["REM"]] > low[_IDX["REM"]]


def test_backoff_handles_unseen_keys():
    train = _generate_dwell_dependent_world(500, seed=5)
    sm = fit_semi_markov_comparator(train)
    # An unseen exact key must still return a valid distribution via backoff.
    dist = sm.predict_one(SemiMarkovAnchor("REM", "REM", 5, 3, _IDX["no_event"]))
    assert dist.shape == (len(OUTCOME_ALPHABET),)
    assert dist.sum() == pytest.approx(1.0)
    assert np.all(dist > 0)


def test_scores_are_proper_and_bounded():
    train = _generate_dwell_dependent_world(1000, seed=6)
    sm = fit_semi_markov_comparator(train)
    y = _outcomes(train)
    ce = cross_entropy_bits(sm.predict(train), y)
    # Cross-entropy is non-negative and below the uniform-5-way bound log2(5).
    assert 0.0 <= ce <= np.log2(len(OUTCOME_ALPHABET))


def test_rejects_incoherent_inputs():
    with pytest.raises(SemiMarkovError):
        SemiMarkovAnchor("NREM", "Wake", -1, 0, 0)
    with pytest.raises(SemiMarkovError):
        SemiMarkovAnchor("NREM", "Wake", 0, 0, 99)
    with pytest.raises(SemiMarkovError):
        fit_semi_markov_comparator([])
    with pytest.raises(SemiMarkovError):
        fit_semi_markov_comparator(_generate_null_world(10, 0), laplace_alpha=0.0)
    with pytest.raises(SemiMarkovError):
        fit_semi_markov_comparator(_generate_null_world(10, 0), shrinkage_kappa=0.0)
