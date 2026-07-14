from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from neurotwin.forecastability import (
    HnphBaselineError,
    HnphClassicalTrial,
    run_hnph_classical_baselines,
    write_hnph_classical_baselines,
)


def _trial() -> HnphClassicalTrial:
    rng = np.random.default_rng(8)
    subjects = np.repeat([f"s{index:02d}" for index in range(15)], 12)
    splits = np.repeat(["train"] * 8 + ["validation"] * 3 + ["test"] * 4, 12)
    current = rng.integers(0, 3, size=len(subjects))
    bout_bucket = rng.integers(0, 3, size=len(subjects))
    nuisance = np.column_stack((current, bout_bucket, rng.normal(size=len(subjects))))
    eeg = rng.normal(size=(len(subjects), 3))
    labels = (current + bout_bucket + (eeg[:, 0] > 0).astype(int)) % 4
    return HnphClassicalTrial(
        labels=labels,
        subject_ids=subjects,
        splits=splits,
        current_macrostates=current,
        semi_markov_keys=np.asarray([f"{state}:{bucket}" for state, bucket in zip(current, bout_bucket, strict=True)]),
        nuisance_features=nuisance,
        eeg_features=eeg,
    )


def _soft_target_trial() -> HnphClassicalTrial:
    rng = np.random.default_rng(9)
    subjects = np.repeat([f"s{index:02d}" for index in range(15)], 12)
    splits = np.repeat(["train"] * 8 + ["validation"] * 3 + ["test"] * 4, 12)
    current = rng.integers(0, 3, size=len(subjects))
    bout_bucket = rng.integers(0, 3, size=len(subjects))
    nuisance = np.column_stack((current, bout_bucket, rng.normal(size=len(subjects))))
    eeg = rng.normal(size=(len(subjects), 3))
    soft_class = (current + bout_bucket + (eeg[:, 0] > 0).astype(int)) % 5
    soft_targets = np.full((len(subjects), 5), 0.02)
    soft_targets[np.arange(len(subjects)), soft_class] = 0.92
    return HnphClassicalTrial(
        labels=np.zeros(len(subjects), dtype=np.int64),
        subject_ids=subjects,
        splits=splits,
        current_macrostates=current,
        semi_markov_keys=np.asarray([f"{state}:{bucket}" for state, bucket in zip(current, bout_bucket, strict=True)]),
        nuisance_features=nuisance,
        eeg_features=eeg,
        n_classes=5,
        leave_one_rater_out_soft_targets=soft_targets,
        leave_one_rater_out_target_sha256="a" * 64,
    )


class HnphClassicalBaselineTests(unittest.TestCase):
    def test_runs_full_classical_ladder_on_subject_held_out_table(self) -> None:
        result = run_hnph_classical_baselines(_trial(), seed=4, bootstrap_replicates=200)

        self.assertIn(result.selected_best_baseline, result.validation_nll_by_model)
        self.assertIn("fixed_standard_eeg_features_plus_nuisance", result.test_nll_by_model)
        self.assertTrue(np.isfinite(result.test_subject_bootstrap_lcb_95_bits))
        self.assertEqual(
            result.semi_markov_competing_risk_implementation,
            "keyed_empirical_prior_diagnostic_placeholder",
        )
        self.assertFalse(result.claim_mode_comparator_eligible)
        self.assertFalse(result.claim_mode_max_t_inference_eligible)
        self.assertFalse(result.claim_mode_primary_target_eligible)
        self.assertIsNone(result.primary_target_provenance_sha256)
        self.assertRegex(result.chief_comparator_prediction_sha256 or "", r"^[0-9a-f]{64}$")
        self.assertEqual(result.scoring_rule, "hard_label_diagnostic_log_score")
        with tempfile.TemporaryDirectory() as tmp:
            artifact = write_hnph_classical_baselines(Path(tmp) / "baseline.json", result)
            self.assertTrue(artifact.exists())
            payload = json.loads(artifact.read_text())
            self.assertFalse(payload["claim_mode_comparator_eligible"])
            self.assertFalse(payload["claim_mode_max_t_inference_eligible"])

    def test_uses_five_way_leave_one_rater_out_soft_targets_for_validation_and_test_scoring(self) -> None:
        result = run_hnph_classical_baselines(_soft_target_trial(), seed=5, bootstrap_replicates=100)

        self.assertEqual(result.scoring_rule, "leave_one_rater_out_soft_cross_entropy_and_log_skill")
        self.assertTrue(result.claim_mode_primary_target_eligible)
        self.assertTrue(np.isfinite(result.test_subject_balanced_log_skill_bits))
        self.assertTrue(np.isfinite(result.test_subject_bootstrap_lcb_95_bits))
        self.assertFalse(result.claim_mode_comparator_eligible)
        self.assertFalse(result.claim_mode_max_t_inference_eligible)
        self.assertEqual(result.primary_target_provenance_sha256, "a" * 64)
        self.assertRegex(result.chief_comparator_prediction_sha256 or "", r"^[0-9a-f]{64}$")

    def test_rejects_soft_target_without_the_frozen_five_way_alphabet(self) -> None:
        trial = _trial()
        with self.assertRaisesRegex(HnphBaselineError, "five-way"):
            HnphClassicalTrial(
                labels=trial.labels,
                subject_ids=trial.subject_ids,
                splits=trial.splits,
                current_macrostates=trial.current_macrostates,
                semi_markov_keys=trial.semi_markov_keys,
                nuisance_features=trial.nuisance_features,
                eeg_features=trial.eeg_features,
                leave_one_rater_out_soft_targets=np.full((len(trial.labels), 4), 0.25),
            )

    def test_rejects_subject_overlap_before_model_fitting(self) -> None:
        trial = _trial()
        trial.splits[0] = "test"
        with self.assertRaisesRegex(HnphBaselineError, "disjoint"):
            HnphClassicalTrial(
                labels=trial.labels,
                subject_ids=trial.subject_ids,
                splits=trial.splits,
                current_macrostates=trial.current_macrostates,
                semi_markov_keys=trial.semi_markov_keys,
                nuisance_features=trial.nuisance_features,
                eeg_features=trial.eeg_features,
            )


if __name__ == "__main__":
    unittest.main()
