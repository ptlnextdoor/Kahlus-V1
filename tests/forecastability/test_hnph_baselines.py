from __future__ import annotations

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


class HnphClassicalBaselineTests(unittest.TestCase):
    def test_runs_full_classical_ladder_on_subject_held_out_table(self) -> None:
        result = run_hnph_classical_baselines(_trial(), seed=4, bootstrap_replicates=200)

        self.assertIn(result.selected_best_baseline, result.validation_nll_by_model)
        self.assertIn("fixed_standard_eeg_features_plus_nuisance", result.test_nll_by_model)
        self.assertTrue(np.isfinite(result.test_subject_bootstrap_lcb_95_bits))
        with tempfile.TemporaryDirectory() as tmp:
            artifact = write_hnph_classical_baselines(Path(tmp) / "baseline.json", result)
            self.assertTrue(artifact.exists())

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
