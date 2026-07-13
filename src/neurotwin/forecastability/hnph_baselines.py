"""Small classical HNPH baseline ladder for arbitrary context/target representations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neurotwin.repro import write_json


HNPH_CLASSICAL_BASELINE_SCHEMA = "kahlus.hnph.classical_baselines.v1"
_CHIEF_BASELINES = (
    "empirical_no_event_destination",
    "markov_transition_prior",
    "semi_markov_competing_risk",
    "validation_tuned_logistic_or_ridge_hazard",
    "validation_tuned_gbm_hazard",
)
_EEG_MODEL = "fixed_standard_eeg_features_plus_nuisance"


class HnphBaselineError(ValueError):
    """Raised when a table cannot support a person-held-out HNPH baseline comparison."""


@dataclass
class HnphClassicalTrial:
    """Precomputed causal features and labels; no raw windows or identity features enter models."""

    labels: np.ndarray
    subject_ids: np.ndarray
    splits: np.ndarray
    current_macrostates: np.ndarray
    semi_markov_keys: np.ndarray
    nuisance_features: np.ndarray
    eeg_features: np.ndarray
    n_classes: int = 4

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.subject_ids = np.asarray(self.subject_ids).astype(str)
        self.splits = np.asarray(self.splits).astype(str)
        self.current_macrostates = np.asarray(self.current_macrostates).astype(str)
        self.semi_markov_keys = np.asarray(self.semi_markov_keys).astype(str)
        self.nuisance_features = np.asarray(self.nuisance_features, dtype=np.float64)
        self.eeg_features = np.asarray(self.eeg_features, dtype=np.float64)
        n_rows = len(self.labels)
        arrays = (
            self.subject_ids,
            self.splits,
            self.current_macrostates,
            self.semi_markov_keys,
        )
        if n_rows == 0 or any(len(values) != n_rows for values in arrays):
            raise HnphBaselineError("all HNPH trial vectors must be non-empty and have the same length")
        if self.nuisance_features.ndim != 2 or self.eeg_features.ndim != 2:
            raise HnphBaselineError("nuisance_features and eeg_features must be two-dimensional")
        if self.nuisance_features.shape[0] != n_rows or self.eeg_features.shape[0] != n_rows:
            raise HnphBaselineError("feature matrices must have one row per target")
        if not np.isfinite(self.nuisance_features).all() or not np.isfinite(self.eeg_features).all():
            raise HnphBaselineError("classical HNPH feature matrices must be finite")
        if self.n_classes < 2 or np.any(self.labels < 0) or np.any(self.labels >= self.n_classes):
            raise HnphBaselineError("labels must be integer outcome categories within n_classes")
        if set(self.splits) != {"train", "validation", "test"}:
            raise HnphBaselineError("splits must contain exactly train, validation, and test rows")
        subjects_by_split = {
            split: set(self.subject_ids[self.splits == split])
            for split in ("train", "validation", "test")
        }
        if any(not values for values in subjects_by_split.values()):
            raise HnphBaselineError("each HNPH split must contain at least one subject")
        if subjects_by_split["train"] & subjects_by_split["validation"] or subjects_by_split["train"] & subjects_by_split["test"] or subjects_by_split["validation"] & subjects_by_split["test"]:
            raise HnphBaselineError("subjects must be disjoint across train, validation, and test")


@dataclass(frozen=True)
class HnphClassicalBaselineResult:
    selected_best_baseline: str
    validation_nll_by_model: dict[str, float]
    test_nll_by_model: dict[str, float]
    test_subject_balanced_log_skill_bits: float
    test_subject_bootstrap_lcb_95_bits: float
    bootstrap_replicates: int

    def to_dict(self) -> dict[str, Any]:
        return {"schema": HNPH_CLASSICAL_BASELINE_SCHEMA, **asdict(self)}


def run_hnph_classical_baselines(
    trial: HnphClassicalTrial,
    *,
    seed: int,
    bootstrap_replicates: int = 2000,
) -> HnphClassicalBaselineResult:
    """Select a nuisance-only chief baseline on validation, then evaluate EEG+B on held-out people."""

    if bootstrap_replicates <= 0:
        raise HnphBaselineError("bootstrap_replicates must be positive")
    masks = {split: trial.splits == split for split in ("train", "validation", "test")}
    _require_multiclass_trainability(trial.labels[masks["train"]])
    logistic_c = _select_logistic_c(trial, masks["train"], masks["validation"], seed)
    gbm_leaf = _select_gbm_leaf(trial, masks["train"], masks["validation"], seed)
    validation = _validation_probabilities(trial, masks, seed, logistic_c=logistic_c, gbm_leaf=gbm_leaf)
    validation_nll = {name: _nll(trial.labels[masks["validation"]], probability) for name, probability in validation.items()}
    selected = min(_CHIEF_BASELINES, key=lambda name: (validation_nll[name], name))
    test = _test_probabilities(trial, masks, seed, selected_logistic=logistic_c, selected_gbm=gbm_leaf)
    skill, lcb = _subject_balanced_log_skill(
        trial.labels[masks["test"]],
        test[_EEG_MODEL],
        test[selected],
        trial.subject_ids[masks["test"]],
        seed=seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    return HnphClassicalBaselineResult(
        selected_best_baseline=selected,
        validation_nll_by_model=validation_nll,
        test_nll_by_model={name: _nll(trial.labels[masks["test"]], probability) for name, probability in test.items()},
        test_subject_balanced_log_skill_bits=skill,
        test_subject_bootstrap_lcb_95_bits=lcb,
        bootstrap_replicates=bootstrap_replicates,
    )


def write_hnph_classical_baselines(path: str | Path, result: HnphClassicalBaselineResult) -> Path:
    """Write the hashable input artifact consumed by the HNPH feasibility gate."""

    return write_json(path, result.to_dict())


def _validation_probabilities(
    trial: HnphClassicalTrial,
    masks: dict[str, np.ndarray],
    seed: int,
    *,
    logistic_c: float,
    gbm_leaf: int,
) -> dict[str, np.ndarray]:
    train = masks["train"]
    validation = masks["validation"]
    return {
        "empirical_no_event_destination": _empirical_probabilities(trial.labels[train], int(np.sum(validation)), trial.n_classes),
        "markov_transition_prior": _conditional_probabilities(trial.labels[train], trial.current_macrostates[train], trial.current_macrostates[validation], trial.n_classes),
        "semi_markov_competing_risk": _conditional_probabilities(trial.labels[train], trial.semi_markov_keys[train], trial.semi_markov_keys[validation], trial.n_classes),
        "validation_tuned_logistic_or_ridge_hazard": _fit_logistic_predict(trial.nuisance_features[train], trial.labels[train], trial.nuisance_features[validation], trial.n_classes, logistic_c, seed),
        "validation_tuned_gbm_hazard": _fit_gbm_predict(trial.nuisance_features[train], trial.labels[train], trial.nuisance_features[validation], trial.n_classes, gbm_leaf, seed),
        _EEG_MODEL: _fit_logistic_predict(np.concatenate((trial.nuisance_features[train], trial.eeg_features[train]), axis=1), trial.labels[train], np.concatenate((trial.nuisance_features[validation], trial.eeg_features[validation]), axis=1), trial.n_classes, 1.0, seed),
    }


def _test_probabilities(
    trial: HnphClassicalTrial,
    masks: dict[str, np.ndarray],
    seed: int,
    *,
    selected_logistic: float,
    selected_gbm: int,
) -> dict[str, np.ndarray]:
    fit = masks["train"] | masks["validation"]
    test = masks["test"]
    return {
        "empirical_no_event_destination": _empirical_probabilities(trial.labels[fit], int(np.sum(test)), trial.n_classes),
        "markov_transition_prior": _conditional_probabilities(trial.labels[fit], trial.current_macrostates[fit], trial.current_macrostates[test], trial.n_classes),
        "semi_markov_competing_risk": _conditional_probabilities(trial.labels[fit], trial.semi_markov_keys[fit], trial.semi_markov_keys[test], trial.n_classes),
        "validation_tuned_logistic_or_ridge_hazard": _fit_logistic_predict(trial.nuisance_features[fit], trial.labels[fit], trial.nuisance_features[test], trial.n_classes, selected_logistic, seed),
        "validation_tuned_gbm_hazard": _fit_gbm_predict(trial.nuisance_features[fit], trial.labels[fit], trial.nuisance_features[test], trial.n_classes, selected_gbm, seed),
        _EEG_MODEL: _fit_logistic_predict(np.concatenate((trial.nuisance_features[fit], trial.eeg_features[fit]), axis=1), trial.labels[fit], np.concatenate((trial.nuisance_features[test], trial.eeg_features[test]), axis=1), trial.n_classes, 1.0, seed),
    }


def _select_logistic_c(trial: HnphClassicalTrial, train: np.ndarray, validation: np.ndarray, seed: int) -> float:
    candidates = (0.1, 1.0, 10.0)
    return min(
        candidates,
        key=lambda value: (
            _nll(trial.labels[validation], _fit_logistic_predict(trial.nuisance_features[train], trial.labels[train], trial.nuisance_features[validation], trial.n_classes, value, seed)),
            value,
        ),
    )


def _select_gbm_leaf(trial: HnphClassicalTrial, train: np.ndarray, validation: np.ndarray, seed: int) -> int:
    candidates = (3, 7)
    return min(
        candidates,
        key=lambda value: (
            _nll(trial.labels[validation], _fit_gbm_predict(trial.nuisance_features[train], trial.labels[train], trial.nuisance_features[validation], trial.n_classes, value, seed)),
            value,
        ),
    )


def _empirical_probabilities(labels: np.ndarray, n_rows: int, n_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64) + 0.5
    return np.broadcast_to(counts / counts.sum(), (n_rows, n_classes)).copy()


def _conditional_probabilities(labels: np.ndarray, train_keys: np.ndarray, test_keys: np.ndarray, n_classes: int) -> np.ndarray:
    global_probability = _empirical_probabilities(labels, 1, n_classes)[0]
    counts: dict[str, np.ndarray] = {}
    for key, label in zip(train_keys, labels, strict=True):
        counts.setdefault(str(key), np.full(n_classes, 0.5))[int(label)] += 1.0
    return np.asarray(
        [counts.get(str(key), global_probability * max(1, len(labels))).astype(np.float64) for key in test_keys],
        dtype=np.float64,
    ) / np.asarray(
        [counts.get(str(key), global_probability * max(1, len(labels))).sum() for key in test_keys],
        dtype=np.float64,
    )[:, None]


def _fit_logistic_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, n_classes: int, c_value: float, seed: int) -> np.ndarray:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, max_iter=500, random_state=seed))
    model.fit(x_train, y_train)
    return _complete_probability_matrix(model.predict_proba(x_test), model.classes_, n_classes)


def _fit_gbm_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, n_classes: int, max_leaf_nodes: int, seed: int) -> np.ndarray:
    model = HistGradientBoostingClassifier(max_iter=64, max_leaf_nodes=max_leaf_nodes, learning_rate=0.05, random_state=seed)
    model.fit(x_train, y_train)
    return _complete_probability_matrix(model.predict_proba(x_test), model.classes_, n_classes)


def _complete_probability_matrix(probability: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    result = np.full((len(probability), n_classes), 1e-12, dtype=np.float64)
    result[:, np.asarray(classes, dtype=np.int64)] = probability
    result /= result.sum(axis=1, keepdims=True)
    return result


def _nll(labels: np.ndarray, probability: np.ndarray) -> float:
    selected = np.clip(probability[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(selected)))


def _subject_balanced_log_skill(
    labels: np.ndarray,
    model_probability: np.ndarray,
    baseline_probability: np.ndarray,
    subject_ids: np.ndarray,
    *,
    seed: int,
    bootstrap_replicates: int,
) -> tuple[float, float]:
    model = np.clip(model_probability[np.arange(len(labels)), labels], 1e-12, 1.0)
    baseline = np.clip(baseline_probability[np.arange(len(labels)), labels], 1e-12, 1.0)
    per_anchor = np.log2(model / baseline)
    subjects = np.unique(subject_ids)
    per_subject = np.asarray([np.mean(per_anchor[subject_ids == subject]) for subject in subjects], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.mean(per_subject[rng.integers(0, len(per_subject), size=(bootstrap_replicates, len(per_subject)))], axis=1)
    return float(np.mean(per_subject)), float(np.quantile(samples, 0.05))


def _require_multiclass_trainability(labels: np.ndarray) -> None:
    if len(np.unique(labels)) < 2:
        raise HnphBaselineError("training split must contain at least two outcome categories")
