"""Subject-cluster label-reproducibility references for HNPH construct validity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


LABEL_REPRODUCIBILITY_FAMILY_SCHEMA = "kahlus.hnph.label_reproducibility_family.v1"
HNPH_PRIMARY_OUTCOME_ALPHABET = ("no_event", "Wake", "NREM", "REM", "Ambiguous")
_PROBABILITY_FLOOR = 1e-12


class LabelReliabilityError(ValueError):
    """Raised when multi-rater/soft-label evidence cannot support a reproducibility reference."""


@dataclass(frozen=True)
class LabelReproducibilityReference:
    """Held-out-rater reproducibility, not a universal upper bound on model skill."""

    subject_balanced_log_skill_bits: float
    subject_bootstrap_lcb_95_bits: float
    subject_count: int
    independent_rater_count: int
    bootstrap_replicates: int
    method: str = "leave_one_rater_out_soft_label"
    probability_floor: float = _PROBABILITY_FLOOR
    interval_method: str = "subject_cluster_bootstrap_pointwise_one_sided_95"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LabelReproducibilityBandInput:
    """Raw per-rater targets for one named family cell (a lead band in B2)."""

    rater_target_probabilities: np.ndarray
    subject_ids: np.ndarray
    chief_comparator_probabilities: np.ndarray


@dataclass(frozen=True)
class LabelReproducibilityFamilyResult:
    """Joint subject-bootstrap max-t references over a preregistered family."""

    references_by_family_cell: Mapping[str, LabelReproducibilityReference]
    family_cell_ids: tuple[str, ...]
    subject_count: int
    independent_rater_count: int
    bootstrap_replicates: int
    max_t_critical_value: float
    max_t_passed: bool
    probability_floor: float = _PROBABILITY_FLOOR
    one_sided_confidence: float = 0.95
    method: str = "leave_one_rater_out_soft_target_subject_cluster_bootstrap_max_t"
    schema: str = LABEL_REPRODUCIBILITY_FAMILY_SCHEMA
    outcome_alphabet: tuple[str, ...] = HNPH_PRIMARY_OUTCOME_ALPHABET

    def __post_init__(self) -> None:
        if self.schema != LABEL_REPRODUCIBILITY_FAMILY_SCHEMA:
            raise LabelReliabilityError("label-reproducibility family schema is fixed")
        if self.outcome_alphabet != HNPH_PRIMARY_OUTCOME_ALPHABET:
            raise LabelReliabilityError("label-reproducibility family must use the frozen five-way outcome alphabet")
        if tuple(sorted(self.references_by_family_cell)) != self.family_cell_ids:
            raise LabelReliabilityError("family-cell references must match the ordered family-cell IDs")

    @property
    def references_by_band(self) -> Mapping[str, LabelReproducibilityReference]:
        """Compatibility view when each family cell is a lead band."""

        return self.references_by_family_cell

    @property
    def family_band_ids(self) -> tuple[str, ...]:
        """Compatibility view when each family cell is a lead band."""

        return self.family_cell_ids

    def to_dict(self) -> dict[str, object]:
        """Return a hashable public summary without rater labels or local paths."""

        return {
            "schema": self.schema,
            "method": self.method,
            "family_cell_ids": list(self.family_cell_ids),
            "subject_count": self.subject_count,
            "independent_rater_count": self.independent_rater_count,
            "bootstrap_replicates": self.bootstrap_replicates,
            "one_sided_confidence": self.one_sided_confidence,
            "probability_floor": self.probability_floor,
            "outcome_alphabet": list(self.outcome_alphabet),
            "max_t_critical_value": self.max_t_critical_value,
            "max_t_passed": self.max_t_passed,
            "references_by_family_cell": {
                family_cell: self.references_by_family_cell[family_cell].to_dict()
                for family_cell in self.family_cell_ids
            },
        }


def build_leave_one_rater_out_soft_targets(
    rater_labels: np.ndarray,
    held_out_rater_index: int,
) -> np.ndarray:
    """Build the frozen five-way consensus target without the scored rater.

    ``rater_labels`` is ``[anchor, rater]`` and uses ``-1`` for an absent
    annotation.  This helper deliberately does not infer labels from a hard
    consensus: the output remains a probability distribution.
    """

    labels = np.asarray(rater_labels)
    if labels.ndim != 2 or labels.shape[0] == 0 or labels.shape[1] < 3:
        raise LabelReliabilityError("leave-one-rater-out targets require [anchor, at least three raters] labels")
    if not isinstance(held_out_rater_index, int) or isinstance(held_out_rater_index, bool) or not 0 <= held_out_rater_index < labels.shape[1]:
        raise LabelReliabilityError("held-out rater index must name one available rater")
    try:
        integer_labels = labels.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise LabelReliabilityError("rater labels must be integer outcome categories") from exc
    if not np.array_equal(labels, integer_labels):
        raise LabelReliabilityError("rater labels must be integer outcome categories")
    remaining = np.delete(integer_labels, held_out_rater_index, axis=1)
    if np.any((remaining < -1) | (remaining >= len(HNPH_PRIMARY_OUTCOME_ALPHABET))):
        raise LabelReliabilityError("rater labels must be five-way categories or -1 for missing")
    target = np.zeros((labels.shape[0], len(HNPH_PRIMARY_OUTCOME_ALPHABET)), dtype=np.float64)
    for category in range(target.shape[1]):
        target[:, category] = np.sum(remaining == category, axis=1)
    totals = target.sum(axis=1)
    if np.any(totals <= 0):
        raise LabelReliabilityError("every anchor needs at least one non-held-out rater label")
    return target / totals[:, None]


def estimate_label_reproducibility_reference(
    held_out_rater_labels: np.ndarray,
    leave_one_rater_out_probabilities: np.ndarray,
    subject_ids: np.ndarray,
    chief_comparator_probabilities: np.ndarray,
    *,
    independent_rater_count: int,
    seed: int,
    bootstrap_replicates: int = 2000,
    probability_floor: float = _PROBABILITY_FLOOR,
) -> LabelReproducibilityReference:
    """Pointwise reference for one already-separated held-out rater.

    The family function below is the claim-mode path.  This remains useful for
    a local diagnostic or for inspecting a single rater before the joint family
    is built.
    """

    _validate_rater_count(independent_rater_count)
    subjects, per_subject = _single_rater_subject_skill(
        held_out_rater_labels,
        leave_one_rater_out_probabilities,
        subject_ids,
        chief_comparator_probabilities,
        probability_floor=probability_floor,
    )
    if bootstrap_replicates <= 0:
        raise LabelReliabilityError("bootstrap_replicates must be positive")
    rng = np.random.default_rng(seed)
    samples = np.mean(
        per_subject[rng.integers(0, len(per_subject), size=(bootstrap_replicates, len(per_subject)))],
        axis=1,
    )
    return LabelReproducibilityReference(
        subject_balanced_log_skill_bits=float(np.mean(per_subject)),
        subject_bootstrap_lcb_95_bits=float(np.quantile(samples, 0.05)),
        subject_count=len(subjects),
        independent_rater_count=independent_rater_count,
        bootstrap_replicates=bootstrap_replicates,
        probability_floor=probability_floor,
    )


def estimate_label_reproducibility_family(
    family_inputs: Mapping[str, LabelReproducibilityBandInput],
    *,
    seed: int,
    bootstrap_replicates: int = 2000,
    probability_floor: float = _PROBABILITY_FLOOR,
    one_sided_confidence: float = 0.95,
) -> LabelReproducibilityFamilyResult:
    """Compute a joint held-rater reference with subject-cluster max-t bounds.

    Each input contains all independent raters as ``[rater, anchor, five-way
    outcome]`` probabilities.  For each rater, the scored target is evaluated
    against the mean of every *other* rater and the same frozen nuisance
    comparator.  The bootstrap resamples subjects jointly across named bands.
    """

    if not isinstance(family_inputs, Mapping) or not family_inputs:
        raise LabelReliabilityError("label-reproducibility family requires at least one named family cell")
    if bootstrap_replicates < 2:
        raise LabelReliabilityError("simultaneous max-t requires at least two bootstrap replicates")
    if not np.isfinite(one_sided_confidence) or not 0 < one_sided_confidence < 1:
        raise LabelReliabilityError("one_sided_confidence must lie strictly between zero and one")
    _validate_probability_floor(probability_floor)

    if any(not isinstance(family_cell, str) for family_cell in family_inputs):
        raise LabelReliabilityError("family cells need string IDs")
    family_cell_ids = tuple(sorted(family_inputs))
    if any(not family_cell for family_cell in family_cell_ids) or len(set(family_cell_ids)) != len(family_cell_ids):
        raise LabelReliabilityError("family cells need unique non-empty IDs")
    per_subject_by_family_cell: dict[str, np.ndarray] = {}
    expected_subjects: tuple[str, ...] | None = None
    independent_rater_count: int | None = None
    for family_cell in family_cell_ids:
        value = family_inputs[family_cell]
        if not isinstance(value, LabelReproducibilityBandInput):
            raise LabelReliabilityError("each named family cell must provide LabelReproducibilityBandInput")
        subjects, per_subject, rater_count = _all_rater_subject_skill(value, probability_floor=probability_floor)
        subject_key = tuple(subjects.tolist())
        if expected_subjects is None:
            expected_subjects = subject_key
        elif subject_key != expected_subjects:
            raise LabelReliabilityError("all max-t family cells must contain the same subject IDs")
        if independent_rater_count is None:
            independent_rater_count = rater_count
        elif rater_count != independent_rater_count:
            raise LabelReliabilityError("all max-t family cells must contain the same independent rater count")
        per_subject_by_family_cell[family_cell] = per_subject

    assert expected_subjects is not None and independent_rater_count is not None
    per_subject_matrix = np.column_stack([per_subject_by_family_cell[family_cell] for family_cell in family_cell_ids])
    if not np.isfinite(per_subject_matrix).all():
        raise LabelReliabilityError("label-reproducibility subject scores must be finite")
    means = np.mean(per_subject_matrix, axis=0)
    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(
        0,
        per_subject_matrix.shape[0],
        size=(bootstrap_replicates, per_subject_matrix.shape[0]),
    )
    bootstrap_means = np.mean(per_subject_matrix[resample_indices], axis=1)
    standard_errors = np.std(bootstrap_means - means, axis=0, ddof=1)
    active = standard_errors > np.finfo(np.float64).eps
    studentized = np.zeros_like(bootstrap_means)
    if np.any(active):
        studentized[:, active] = (means[active] - bootstrap_means[:, active]) / standard_errors[active]
    critical_value = max(0.0, float(np.quantile(np.max(studentized, axis=1), one_sided_confidence)))
    lower_bounds = means - critical_value * standard_errors
    if not np.isfinite(critical_value) or not np.isfinite(lower_bounds).all():
        raise LabelReliabilityError("simultaneous max-t calculation produced non-finite bounds")
    references = {
        family_cell: LabelReproducibilityReference(
            subject_balanced_log_skill_bits=float(means[index]),
            subject_bootstrap_lcb_95_bits=float(lower_bounds[index]),
            subject_count=per_subject_matrix.shape[0],
            independent_rater_count=independent_rater_count,
            bootstrap_replicates=bootstrap_replicates,
            method="leave_one_rater_out_soft_label",
            probability_floor=probability_floor,
            interval_method="subject_cluster_bootstrap_max_t_one_sided",
        )
        for index, family_cell in enumerate(family_cell_ids)
    }
    return LabelReproducibilityFamilyResult(
        references_by_family_cell=references,
        family_cell_ids=family_cell_ids,
        subject_count=per_subject_matrix.shape[0],
        independent_rater_count=independent_rater_count,
        bootstrap_replicates=bootstrap_replicates,
        max_t_critical_value=critical_value,
        max_t_passed=True,
        probability_floor=probability_floor,
        one_sided_confidence=one_sided_confidence,
    )


def relative_to_label_reproducibility(
    model_log_skill_bits: float,
    reference: LabelReproducibilityReference,
) -> float | None:
    """Report a descriptive model/reference ratio; it is not a ceiling fraction."""

    denominator = reference.subject_balanced_log_skill_bits
    if not np.isfinite(model_log_skill_bits) or not np.isfinite(denominator) or denominator <= 0:
        return None
    return float(model_log_skill_bits / denominator)


def _all_rater_subject_skill(
    value: LabelReproducibilityBandInput,
    *,
    probability_floor: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    rater_targets = np.asarray(value.rater_target_probabilities, dtype=np.float64)
    subjects = np.asarray(value.subject_ids).astype(str)
    comparator = np.asarray(value.chief_comparator_probabilities, dtype=np.float64)
    if rater_targets.ndim != 3 or rater_targets.shape[0] < 3 or rater_targets.shape[1] == 0:
        raise LabelReliabilityError("family inputs require [at least three raters, anchor, five-way outcome] targets")
    if rater_targets.shape[2] != len(HNPH_PRIMARY_OUTCOME_ALPHABET):
        raise LabelReliabilityError("family inputs require the frozen five-way HNPH outcome alphabet")
    if len(subjects) != rater_targets.shape[1] or not np.all(subjects):
        raise LabelReliabilityError("each family anchor must have a subject ID")
    if comparator.shape != (rater_targets.shape[1], rater_targets.shape[2]):
        raise LabelReliabilityError("chief comparator probabilities must align with every family anchor")
    targets = _normalize_probability_cube(rater_targets, probability_floor)
    baseline = _normalize_probability_rows(comparator, probability_floor)
    score_by_rater = np.empty((rater_targets.shape[0], rater_targets.shape[1]), dtype=np.float64)
    total = np.sum(targets, axis=0)
    for rater in range(rater_targets.shape[0]):
        loo = _normalize_probability_rows(
            (total - targets[rater]) / (rater_targets.shape[0] - 1),
            probability_floor,
        )
        score_by_rater[rater] = np.sum(targets[rater] * np.log2(loo / baseline), axis=1)
    anchor_score = np.mean(score_by_rater, axis=0)
    unique_subjects = np.unique(subjects)
    per_subject = np.asarray(
        [np.mean(anchor_score[subjects == subject]) for subject in unique_subjects],
        dtype=np.float64,
    )
    if not np.isfinite(per_subject).all():
        raise LabelReliabilityError("held-rater log-skill must be finite")
    return unique_subjects, per_subject, rater_targets.shape[0]


def _single_rater_subject_skill(
    held_out_rater_labels: np.ndarray,
    leave_one_rater_out_probabilities: np.ndarray,
    subject_ids: np.ndarray,
    chief_comparator_probabilities: np.ndarray,
    *,
    probability_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(held_out_rater_labels, dtype=np.int64)
    probabilities = np.asarray(leave_one_rater_out_probabilities, dtype=np.float64)
    subjects = np.asarray(subject_ids).astype(str)
    comparator = np.asarray(chief_comparator_probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 2 or len(labels) == 0 or probabilities.shape[0] != len(labels):
        raise LabelReliabilityError("labels and leave-one-rater-out probabilities must align by anchor")
    if len(subjects) != len(labels) or not np.all(subjects):
        raise LabelReliabilityError("each label-reliability anchor must have a subject ID")
    if probabilities.shape[1] < 2 or comparator.shape != probabilities.shape:
        raise LabelReliabilityError("chief comparator probabilities must align with the soft-label outcome alphabet")
    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise LabelReliabilityError("held-out rater labels must lie inside the outcome alphabet")
    probability = _normalize_probability_rows(probabilities, probability_floor)
    baseline = _normalize_probability_rows(comparator, probability_floor)
    anchor_skill = np.log2(probability[np.arange(len(labels)), labels] / baseline[np.arange(len(labels)), labels])
    unique_subjects = np.unique(subjects)
    per_subject = np.asarray([np.mean(anchor_skill[subjects == subject]) for subject in unique_subjects], dtype=np.float64)
    if not np.isfinite(per_subject).all():
        raise LabelReliabilityError("held-rater log-skill must be finite")
    return unique_subjects, per_subject


def _validate_rater_count(independent_rater_count: int) -> None:
    if not isinstance(independent_rater_count, int) or isinstance(independent_rater_count, bool) or independent_rater_count < 3:
        raise LabelReliabilityError("label-reproducibility reference requires at least three independent raters")


def _validate_probability_floor(probability_floor: float) -> None:
    if not np.isfinite(probability_floor) or not 0 < probability_floor < 1:
        raise LabelReliabilityError("probability_floor must be finite and lie strictly between zero and one")


def _normalize_probability_rows(probability: np.ndarray, probability_floor: float) -> np.ndarray:
    _validate_probability_floor(probability_floor)
    if probability.ndim != 2 or not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability.sum(axis=1) <= 0):
        raise LabelReliabilityError("probabilities must be finite non-empty non-negative distributions")
    clipped = np.maximum(probability, probability_floor)
    return clipped / clipped.sum(axis=1, keepdims=True)


def _normalize_probability_cube(probability: np.ndarray, probability_floor: float) -> np.ndarray:
    if probability.ndim != 3 or not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability.sum(axis=2) <= 0):
        raise LabelReliabilityError("rater targets must be finite non-empty non-negative distributions")
    clipped = np.maximum(probability, probability_floor)
    return clipped / clipped.sum(axis=2, keepdims=True)
