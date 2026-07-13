"""Small classical HNPH baseline ladder for arbitrary context/target representations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neurotwin.repro import write_json


HNPH_CLASSICAL_BASELINE_SCHEMA = "kahlus.hnph.classical_baselines.v2"
HNPH_PRIMARY_SOFT_OUTCOME_COUNT = 5
_SEMI_MARKOV_IMPLEMENTATION = "keyed_empirical_prior_diagnostic_placeholder"
_CHIEF_BASELINES = (
    "empirical_no_event_destination",
    "markov_transition_prior",
    "semi_markov_competing_risk",
    "validation_tuned_logistic_or_ridge_hazard",
    "validation_tuned_gbm_hazard",
)
_EEG_MODEL = "fixed_standard_eeg_features_plus_nuisance"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HnphBaselineError(ValueError):
    """Raised when a table cannot support a person-held-out HNPH baseline comparison."""


@dataclass
class HnphClassicalTrial:
    """Precomputed causal features with optional LOO soft targets for primary scoring.

    ``labels`` are retained only for the pre-existing hard-label diagnostic.
    Claim-mode target scoring instead requires 5-way
    ``leave_one_rater_out_soft_targets``.  The target is consumed directly by
    the likelihood scores and soft-weighted training rows, so a hard label is
    never substituted into validation or held-out scoring when it is present.
    """

    labels: np.ndarray
    subject_ids: np.ndarray
    splits: np.ndarray
    current_macrostates: np.ndarray
    semi_markov_keys: np.ndarray
    nuisance_features: np.ndarray
    eeg_features: np.ndarray
    n_classes: int = 4
    leave_one_rater_out_soft_targets: np.ndarray | None = None
    leave_one_rater_out_target_sha256: str | None = None

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.subject_ids = np.asarray(self.subject_ids).astype(str)
        self.splits = np.asarray(self.splits).astype(str)
        self.current_macrostates = np.asarray(self.current_macrostates).astype(str)
        self.semi_markov_keys = np.asarray(self.semi_markov_keys).astype(str)
        self.nuisance_features = np.asarray(self.nuisance_features, dtype=np.float64)
        self.eeg_features = np.asarray(self.eeg_features, dtype=np.float64)
        if self.leave_one_rater_out_soft_targets is not None:
            self.leave_one_rater_out_soft_targets = np.asarray(
                self.leave_one_rater_out_soft_targets,
                dtype=np.float64,
            )
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
        if self.leave_one_rater_out_soft_targets is not None:
            soft_targets = self.leave_one_rater_out_soft_targets
            if self.n_classes != HNPH_PRIMARY_SOFT_OUTCOME_COUNT:
                raise HnphBaselineError("leave-one-rater-out targets require the frozen five-way HNPH alphabet")
            if soft_targets.ndim != 2 or soft_targets.shape != (n_rows, self.n_classes):
                raise HnphBaselineError("leave-one-rater-out soft targets must have one five-way distribution per target")
            if not np.isfinite(soft_targets).all() or np.any(soft_targets < 0) or np.any(soft_targets.sum(axis=1) <= 0):
                raise HnphBaselineError("leave-one-rater-out soft targets must be finite non-empty distributions")
            if not isinstance(self.leave_one_rater_out_target_sha256, str) or not _SHA256.fullmatch(self.leave_one_rater_out_target_sha256):
                raise HnphBaselineError("leave-one-rater-out soft targets require a SHA-256 provenance hash")
            self.leave_one_rater_out_soft_targets = soft_targets / soft_targets.sum(axis=1, keepdims=True)
        elif self.leave_one_rater_out_target_sha256 is not None:
            raise HnphBaselineError("hard-label diagnostic trials cannot carry a primary-target provenance hash")
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

    @property
    def scoring_targets(self) -> np.ndarray:
        """Return primary soft labels when supplied, otherwise the hard diagnostic labels."""

        return self.leave_one_rater_out_soft_targets if self.leave_one_rater_out_soft_targets is not None else self.labels

    @property
    def scoring_rule(self) -> str:
        """Name the frozen scoring path used by this trial."""

        if self.leave_one_rater_out_soft_targets is not None:
            return "leave_one_rater_out_soft_cross_entropy_and_log_skill"
        return "hard_label_diagnostic_log_score"

    @property
    def claim_mode_primary_target_eligible(self) -> bool:
        """Whether this table has the required primary target, not a complete claim gate."""

        return self.leave_one_rater_out_soft_targets is not None


@dataclass(frozen=True)
class HnphClassicalBaselineResult:
    """Diagnostic baseline result; not sufficient evidence for a claim-mode comparison."""

    selected_best_baseline: str
    validation_nll_by_model: dict[str, float]
    test_nll_by_model: dict[str, float]
    test_subject_balanced_log_skill_bits: float
    test_subject_bootstrap_lcb_95_bits: float
    bootstrap_replicates: int
    scoring_rule: str = "hard_label_diagnostic_log_score"
    claim_mode_primary_target_eligible: bool = False
    primary_target_provenance_sha256: str | None = None
    chief_comparator_prediction_sha256: str | None = None
    semi_markov_competing_risk_implementation: str = _SEMI_MARKOV_IMPLEMENTATION
    claim_mode_comparator_eligible: bool = False
    claim_mode_max_t_inference_eligible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"schema": HNPH_CLASSICAL_BASELINE_SCHEMA, **asdict(self)}


def run_hnph_classical_baselines(
    trial: HnphClassicalTrial,
    *,
    seed: int,
    bootstrap_replicates: int = 2000,
) -> HnphClassicalBaselineResult:
    """Run a diagnostic ladder; its keyed semi-Markov row is not a claim-mode comparator."""

    if bootstrap_replicates <= 0:
        raise HnphBaselineError("bootstrap_replicates must be positive")
    masks = {split: trial.splits == split for split in ("train", "validation", "test")}
    targets = trial.scoring_targets
    _require_multiclass_trainability(targets[masks["train"]])
    logistic_c = _select_logistic_c(trial, masks["train"], masks["validation"], seed)
    gbm_leaf = _select_gbm_leaf(trial, masks["train"], masks["validation"], seed)
    validation = _validation_probabilities(trial, masks, seed, logistic_c=logistic_c, gbm_leaf=gbm_leaf)
    validation_nll = {name: _nll(targets[masks["validation"]], probability) for name, probability in validation.items()}
    selected = min(_CHIEF_BASELINES, key=lambda name: (validation_nll[name], name))
    test = _test_probabilities(trial, masks, seed, selected_logistic=logistic_c, selected_gbm=gbm_leaf)
    skill, lcb = _subject_balanced_log_skill(
        targets[masks["test"]],
        test[_EEG_MODEL],
        test[selected],
        trial.subject_ids[masks["test"]],
        seed=seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    return HnphClassicalBaselineResult(
        selected_best_baseline=selected,
        validation_nll_by_model=validation_nll,
        test_nll_by_model={name: _nll(targets[masks["test"]], probability) for name, probability in test.items()},
        test_subject_balanced_log_skill_bits=skill,
        test_subject_bootstrap_lcb_95_bits=lcb,
        bootstrap_replicates=bootstrap_replicates,
        scoring_rule=trial.scoring_rule,
        claim_mode_primary_target_eligible=trial.claim_mode_primary_target_eligible,
        primary_target_provenance_sha256=trial.leave_one_rater_out_target_sha256,
        chief_comparator_prediction_sha256=_probability_sha256(test[selected]),
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
    targets = trial.scoring_targets
    return {
        "empirical_no_event_destination": _empirical_probabilities(targets[train], int(np.sum(validation)), trial.n_classes),
        "markov_transition_prior": _conditional_probabilities(targets[train], trial.current_macrostates[train], trial.current_macrostates[validation], trial.n_classes),
        "semi_markov_competing_risk": _conditional_probabilities(targets[train], trial.semi_markov_keys[train], trial.semi_markov_keys[validation], trial.n_classes),
        "validation_tuned_logistic_or_ridge_hazard": _fit_logistic_predict(trial.nuisance_features[train], targets[train], trial.nuisance_features[validation], trial.n_classes, logistic_c, seed),
        "validation_tuned_gbm_hazard": _fit_gbm_predict(trial.nuisance_features[train], targets[train], trial.nuisance_features[validation], trial.n_classes, gbm_leaf, seed),
        _EEG_MODEL: _fit_logistic_predict(np.concatenate((trial.nuisance_features[train], trial.eeg_features[train]), axis=1), targets[train], np.concatenate((trial.nuisance_features[validation], trial.eeg_features[validation]), axis=1), trial.n_classes, 1.0, seed),
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
    targets = trial.scoring_targets
    return {
        "empirical_no_event_destination": _empirical_probabilities(targets[fit], int(np.sum(test)), trial.n_classes),
        "markov_transition_prior": _conditional_probabilities(targets[fit], trial.current_macrostates[fit], trial.current_macrostates[test], trial.n_classes),
        "semi_markov_competing_risk": _conditional_probabilities(targets[fit], trial.semi_markov_keys[fit], trial.semi_markov_keys[test], trial.n_classes),
        "validation_tuned_logistic_or_ridge_hazard": _fit_logistic_predict(trial.nuisance_features[fit], targets[fit], trial.nuisance_features[test], trial.n_classes, selected_logistic, seed),
        "validation_tuned_gbm_hazard": _fit_gbm_predict(trial.nuisance_features[fit], targets[fit], trial.nuisance_features[test], trial.n_classes, selected_gbm, seed),
        _EEG_MODEL: _fit_logistic_predict(np.concatenate((trial.nuisance_features[fit], trial.eeg_features[fit]), axis=1), targets[fit], np.concatenate((trial.nuisance_features[test], trial.eeg_features[test]), axis=1), trial.n_classes, 1.0, seed),
    }


def _select_logistic_c(trial: HnphClassicalTrial, train: np.ndarray, validation: np.ndarray, seed: int) -> float:
    candidates = (0.1, 1.0, 10.0)
    targets = trial.scoring_targets
    return min(
        candidates,
        key=lambda value: (
            _nll(targets[validation], _fit_logistic_predict(trial.nuisance_features[train], targets[train], trial.nuisance_features[validation], trial.n_classes, value, seed)),
            value,
        ),
    )


def _select_gbm_leaf(trial: HnphClassicalTrial, train: np.ndarray, validation: np.ndarray, seed: int) -> int:
    candidates = (3, 7)
    targets = trial.scoring_targets
    return min(
        candidates,
        key=lambda value: (
            _nll(targets[validation], _fit_gbm_predict(trial.nuisance_features[train], targets[train], trial.nuisance_features[validation], trial.n_classes, value, seed)),
            value,
        ),
    )


def _empirical_probabilities(targets: np.ndarray, n_rows: int, n_classes: int) -> np.ndarray:
    counts = _target_counts(targets, n_classes) + 0.5
    return np.broadcast_to(counts / counts.sum(), (n_rows, n_classes)).copy()


def _conditional_probabilities(targets: np.ndarray, train_keys: np.ndarray, test_keys: np.ndarray, n_classes: int) -> np.ndarray:
    global_probability = _empirical_probabilities(targets, 1, n_classes)[0]
    counts: dict[str, np.ndarray] = {}
    for key, target in zip(train_keys, _target_rows(targets, n_classes), strict=True):
        counts.setdefault(str(key), np.full(n_classes, 0.5, dtype=np.float64))[:] += target
    result = np.asarray([counts.get(str(key), global_probability) for key in test_keys], dtype=np.float64)
    return result / result.sum(axis=1, keepdims=True)


def _fit_logistic_predict(x_train: np.ndarray, targets: np.ndarray, x_test: np.ndarray, n_classes: int, c_value: float, seed: int) -> np.ndarray:
    x_fit, y_fit, weights = _classifier_training_rows(x_train, targets, n_classes)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            max_iter=500,
            random_state=seed,
            solver="saga" if weights is not None else "lbfgs",
        ),
    )
    if weights is None:
        model.fit(x_fit, y_fit)
    else:
        model.fit(x_fit, y_fit, logisticregression__sample_weight=weights)
    return _complete_probability_matrix(model.predict_proba(x_test), model.classes_, n_classes)


def _fit_gbm_predict(x_train: np.ndarray, targets: np.ndarray, x_test: np.ndarray, n_classes: int, max_leaf_nodes: int, seed: int) -> np.ndarray:
    x_fit, y_fit, weights = _classifier_training_rows(x_train, targets, n_classes)
    model = HistGradientBoostingClassifier(max_iter=64, max_leaf_nodes=max_leaf_nodes, learning_rate=0.05, random_state=seed)
    model.fit(x_fit, y_fit, sample_weight=weights)
    return _complete_probability_matrix(model.predict_proba(x_test), model.classes_, n_classes)


def _complete_probability_matrix(probability: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    result = np.full((len(probability), n_classes), 1e-12, dtype=np.float64)
    result[:, np.asarray(classes, dtype=np.int64)] = probability
    result /= result.sum(axis=1, keepdims=True)
    return result


def _nll(targets: np.ndarray, probability: np.ndarray) -> float:
    """Return hard-label NLL or soft-target cross entropy, depending on target shape."""

    target_rows = _target_rows(targets, probability.shape[1])
    normalized = _normalize_probability(probability)
    return float(-np.mean(np.sum(target_rows * np.log(normalized), axis=1)))


def _subject_balanced_log_skill(
    targets: np.ndarray,
    model_probability: np.ndarray,
    baseline_probability: np.ndarray,
    subject_ids: np.ndarray,
    *,
    seed: int,
    bootstrap_replicates: int,
) -> tuple[float, float]:
    if len(targets) != len(subject_ids) or model_probability.shape != baseline_probability.shape:
        raise HnphBaselineError("targets, probabilities, and subject IDs must align for log-skill")
    target_rows = _target_rows(targets, model_probability.shape[1])
    model = _normalize_probability(model_probability)
    baseline = _normalize_probability(baseline_probability)
    per_anchor = np.sum(target_rows * np.log2(model / baseline), axis=1)
    subjects = np.unique(subject_ids)
    per_subject = np.asarray([np.mean(per_anchor[subject_ids == subject]) for subject in subjects], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.mean(per_subject[rng.integers(0, len(per_subject), size=(bootstrap_replicates, len(per_subject)))], axis=1)
    return float(np.mean(per_subject)), float(np.quantile(samples, 0.05))


def _require_multiclass_trainability(targets: np.ndarray) -> None:
    if np.count_nonzero(_target_counts(targets, _target_class_count(targets)) > 0) < 2:
        raise HnphBaselineError("training split must contain at least two outcome categories")


def _target_class_count(targets: np.ndarray) -> int:
    array = np.asarray(targets)
    if array.ndim == 1:
        return int(np.max(array)) + 1 if len(array) else 0
    if array.ndim == 2:
        return int(array.shape[1])
    raise HnphBaselineError("targets must be hard labels or a two-dimensional soft-label distribution")


def _target_counts(targets: np.ndarray, n_classes: int) -> np.ndarray:
    array = np.asarray(targets)
    if array.ndim == 1:
        return np.bincount(array.astype(np.int64), minlength=n_classes).astype(np.float64)
    if array.ndim == 2 and array.shape[1] == n_classes:
        return np.sum(array, axis=0, dtype=np.float64)
    raise HnphBaselineError("targets must align with the HNPH outcome alphabet")


def _target_rows(targets: np.ndarray, n_classes: int) -> np.ndarray:
    array = np.asarray(targets)
    if array.ndim == 1:
        labels = array.astype(np.int64)
        if np.any(labels < 0) or np.any(labels >= n_classes):
            raise HnphBaselineError("hard labels must lie inside the HNPH outcome alphabet")
        return np.eye(n_classes, dtype=np.float64)[labels]
    if array.ndim == 2 and array.shape[1] == n_classes:
        if not np.isfinite(array).all() or np.any(array < 0) or np.any(array.sum(axis=1) <= 0):
            raise HnphBaselineError("soft targets must be finite non-empty distributions")
        return array / array.sum(axis=1, keepdims=True)
    raise HnphBaselineError("targets must align with the HNPH outcome alphabet")


def _classifier_training_rows(x_train: np.ndarray, targets: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rows = _target_rows(targets, n_classes)
    if np.asarray(targets).ndim == 1:
        return x_train, np.asarray(targets, dtype=np.int64), None
    repeated_x = np.repeat(x_train, n_classes, axis=0)
    labels = np.tile(np.arange(n_classes, dtype=np.int64), len(x_train))
    weights = rows.reshape(-1)
    keep = weights > 0
    if np.count_nonzero(np.unique(labels[keep])) < 2:
        raise HnphBaselineError("soft training targets must support at least two outcome categories")
    return repeated_x[keep], labels[keep], weights[keep]


def _normalize_probability(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0) or np.any(values.sum(axis=1) <= 0):
        raise HnphBaselineError("probability rows must be finite non-empty distributions")
    clipped = np.clip(values, 1e-12, 1.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


def _probability_sha256(probability: np.ndarray) -> str:
    """Bind the selected comparator's ordered float64 prediction table."""

    values = np.ascontiguousarray(_normalize_probability(probability), dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()
