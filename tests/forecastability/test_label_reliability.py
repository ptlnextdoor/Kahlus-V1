from __future__ import annotations

import unittest

import numpy as np

from neurotwin.forecastability import (
    HNPH_PRIMARY_OUTCOME_ALPHABET,
    LabelReliabilityError,
    LabelReproducibilityBandInput,
    LabelReproducibilityReference,
    build_leave_one_rater_out_soft_targets,
    estimate_label_reproducibility_family,
    estimate_label_reproducibility_reference,
    relative_to_label_reproducibility,
)


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    probabilities = np.full((len(labels), 4), 0.05)
    probabilities[np.arange(len(labels)), labels] = 0.85
    subjects = np.asarray(["s1"] * 4 + ["s2"] * 4)
    comparator = np.full((len(labels), 4), 0.25)
    return labels, probabilities, subjects, comparator


def _family_inputs() -> dict[str, LabelReproducibilityBandInput]:
    subjects = np.asarray(["s1"] * 4 + ["s2"] * 4 + ["s3"] * 4)
    base = np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1])
    rater_labels = np.vstack(
        (
            base,
            np.asarray([0, 1, 2, 3, 4, 0, 2, 2, 3, 4, 0, 2]),
            np.asarray([0, 1, 2, 4, 4, 0, 1, 2, 4, 4, 0, 1]),
        )
    )
    targets = np.eye(5, dtype=np.float64)[rater_labels]
    comparator = np.full((len(subjects), 5), 0.2)
    return {
        "B1": LabelReproducibilityBandInput(targets, subjects, comparator),
        "B2": LabelReproducibilityBandInput(targets[:, ::-1], subjects, comparator),
    }


class LabelReproducibilityTests(unittest.TestCase):
    def test_uses_the_same_chief_comparator_and_subject_bootstrap(self) -> None:
        labels, probabilities, subjects, comparator = _inputs()

        reference = estimate_label_reproducibility_reference(
            labels,
            probabilities,
            subjects,
            comparator,
            independent_rater_count=3,
            seed=7,
            bootstrap_replicates=100,
        )

        self.assertGreater(reference.subject_balanced_log_skill_bits, 0)
        self.assertTrue(np.isfinite(reference.subject_bootstrap_lcb_95_bits))
        self.assertEqual(reference.subject_count, 2)
        self.assertEqual(reference.independent_rater_count, 3)
        self.assertEqual(
            reference,
            estimate_label_reproducibility_reference(
                labels,
                probabilities,
                subjects,
                comparator,
                independent_rater_count=3,
                seed=7,
                bootstrap_replicates=100,
            ),
        )

    def test_requires_three_raters_and_reports_no_ratio_for_nonpositive_reference(self) -> None:
        labels, probabilities, subjects, comparator = _inputs()
        with self.assertRaisesRegex(LabelReliabilityError, "three independent raters"):
            estimate_label_reproducibility_reference(
                labels,
                probabilities,
                subjects,
                comparator,
                independent_rater_count=2,
                seed=7,
            )
        self.assertIsNone(
            relative_to_label_reproducibility(
                0.02,
                LabelReproducibilityReference(0.0, -0.01, 2, 3, 100),
            )
        )

    def test_builds_five_way_soft_target_without_the_held_out_rater(self) -> None:
        labels = np.asarray([[0, 1, 1], [2, 2, 3]])

        target = build_leave_one_rater_out_soft_targets(labels, 0)

        np.testing.assert_allclose(target, np.asarray([[0, 1, 0, 0, 0], [0, 0, 0.5, 0.5, 0]]))

    def test_joint_subject_cluster_max_t_family_is_deterministic_and_serializable(self) -> None:
        result = estimate_label_reproducibility_family(
            _family_inputs(),
            seed=17,
            bootstrap_replicates=200,
        )

        self.assertEqual(result.family_cell_ids, ("B1", "B2"))
        self.assertTrue(result.max_t_passed)
        self.assertEqual(result.outcome_alphabet, HNPH_PRIMARY_OUTCOME_ALPHABET)
        self.assertEqual(result.probability_floor, 1e-12)
        self.assertEqual(set(result.references_by_family_cell), {"B1", "B2"})
        self.assertTrue(all(np.isfinite(reference.subject_bootstrap_lcb_95_bits) for reference in result.references_by_family_cell.values()))
        self.assertEqual(
            result,
            estimate_label_reproducibility_family(_family_inputs(), seed=17, bootstrap_replicates=200),
        )
        payload = result.to_dict()
        self.assertEqual(payload["family_cell_ids"], ["B1", "B2"])
        self.assertTrue(payload["max_t_passed"])
        self.assertEqual(payload["outcome_alphabet"], list(HNPH_PRIMARY_OUTCOME_ALPHABET))
        self.assertEqual(payload["probability_floor"], 1e-12)

    def test_joint_max_t_rejects_a_family_without_the_same_subjects(self) -> None:
        family = _family_inputs()
        b2 = family["B2"]
        family["B2"] = LabelReproducibilityBandInput(
            b2.rater_target_probabilities,
            np.asarray(["s1"] * 4 + ["s2"] * 4 + ["s4"] * 4),
            b2.chief_comparator_probabilities,
        )

        with self.assertRaisesRegex(LabelReliabilityError, "same subject IDs"):
            estimate_label_reproducibility_family(family, seed=17, bootstrap_replicates=20)


if __name__ == "__main__":
    unittest.main()
