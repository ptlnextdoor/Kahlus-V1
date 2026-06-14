import math
import unittest

from neurotwin.training_v3.redteam import (
    BLOCKER_AFFINITY_NOT_TESTED,
    BLOCKER_ASYMMETRIC_VALIDATION,
    BLOCKER_BASELINE_SELECTION,
    BLOCKER_BUDGET_NOT_MATCHED,
    BLOCKER_FAMILY_GENERALIZATION,
    BLOCKER_LOWER_BOUND,
    BLOCKER_MEAN_BELOW_MARGIN,
    BLOCKER_SINGLE_SEED,
    GeneratorFamilyOutcome,
    SeedOutcome,
    recovery_redteam_gate,
    seed_summary,
)


def _seeds(rels, *, symmetric=True, locked=True):
    base_policy = "best_val" if symmetric else "final_step"
    return [
        SeedOutcome(
            seed=i,
            ktm_mse=0.000620,
            best_baseline="ssm_fallback",
            best_baseline_mse=0.000662,
            relative_improvement=float(r),
            comparison_locked=locked,
            ktm_checkpoint_policy="best_val",
            baseline_checkpoint_policy=base_policy,
        )
        for i, r in enumerate(rels)
    ]


def _families(rels):
    return [
        GeneratorFamilyOutcome(
            family=f"family_{i}",
            ktm_mse=0.000620,
            best_baseline="ssm_fallback",
            best_baseline_mse=0.000662,
            relative_improvement=float(r),
        )
        for i, r in enumerate(rels)
    ]


# Tight, comfortably-positive 5-seed win on the original family; both families generalize.
_GOOD_RELS = [0.063, 0.061, 0.065, 0.062, 0.064]
_GOOD_FAMILIES = [0.060, 0.055]


class RecoveryRedteamGateTests(unittest.TestCase):
    def test_full_pass_allows_recovery(self):
        gate = recovery_redteam_gate(
            seeds=_seeds(_GOOD_RELS), families=_families(_GOOD_FAMILIES)
        )
        self.assertTrue(gate["recovery_allowed"], gate["blocker_reasons"])
        self.assertEqual(gate["blocker_reasons"], [])

    def test_asymmetric_selection_blocks(self):
        gate = recovery_redteam_gate(
            seeds=_seeds(_GOOD_RELS, symmetric=False), families=_families(_GOOD_FAMILIES)
        )
        self.assertFalse(gate["recovery_allowed"])
        self.assertIn(BLOCKER_ASYMMETRIC_VALIDATION, gate["blocker_reasons"])
        self.assertIn(BLOCKER_BASELINE_SELECTION, gate["blocker_reasons"])

    def test_single_seed_blocks(self):
        gate = recovery_redteam_gate(seeds=_seeds([0.063]), families=_families(_GOOD_FAMILIES))
        self.assertFalse(gate["recovery_allowed"])
        self.assertIn(BLOCKER_SINGLE_SEED, gate["blocker_reasons"])

    def test_generator_family_failure_blocks(self):
        # Wins on the original family but collapses on the second (alternate) family.
        gate = recovery_redteam_gate(
            seeds=_seeds(_GOOD_RELS), families=_families([0.060, -0.02])
        )
        self.assertFalse(gate["recovery_allowed"])
        self.assertIn(BLOCKER_FAMILY_GENERALIZATION, gate["blocker_reasons"])

    def test_no_generator_families_blocks(self):
        gate = recovery_redteam_gate(seeds=_seeds(_GOOD_RELS), families=[])
        self.assertFalse(gate["recovery_allowed"])
        self.assertIn(BLOCKER_AFFINITY_NOT_TESTED, gate["blocker_reasons"])

    def test_mean_passes_but_lower_bound_negative_blocks(self):
        # mean = 0.08 >= margin, but high variance pushes the 95% lower bound below 0.
        rels = [0.20, 0.20, 0.20, 0.20, -0.40]
        summary = seed_summary(_seeds(rels))
        self.assertGreaterEqual(summary["mean"], 0.05)
        self.assertLessEqual(summary["lower_bound_95"], 0.0)
        gate = recovery_redteam_gate(seeds=_seeds(rels), families=_families(_GOOD_FAMILIES))
        self.assertFalse(gate["recovery_allowed"])
        self.assertIn(BLOCKER_LOWER_BOUND, gate["blocker_reasons"])
        self.assertNotIn(BLOCKER_MEAN_BELOW_MARGIN, gate["blocker_reasons"])

    def test_mean_below_margin_blocks(self):
        gate = recovery_redteam_gate(
            seeds=_seeds([0.01, 0.012, 0.009, 0.011, 0.01]), families=_families(_GOOD_FAMILIES)
        )
        self.assertFalse(gate["recovery_allowed"])
        self.assertIn(BLOCKER_MEAN_BELOW_MARGIN, gate["blocker_reasons"])

    def test_unmatched_budget_blocks(self):
        gate = recovery_redteam_gate(
            seeds=_seeds(_GOOD_RELS, locked=False), families=_families(_GOOD_FAMILIES)
        )
        self.assertFalse(gate["recovery_allowed"])
        self.assertIn(BLOCKER_BUDGET_NOT_MATCHED, gate["blocker_reasons"])

    def test_seed_summary_statistics(self):
        summary = seed_summary(_seeds([0.0, 0.1, 0.2]))
        self.assertEqual(summary["n_seeds"], 3)
        self.assertAlmostEqual(summary["mean"], 0.1, places=9)
        self.assertAlmostEqual(summary["min"], 0.0, places=9)
        self.assertAlmostEqual(summary["max"], 0.2, places=9)
        self.assertEqual(summary["n_positive"], 2)
        self.assertTrue(math.isfinite(summary["lower_bound_95"]))

    def test_empty_seeds_blocks_everything(self):
        gate = recovery_redteam_gate(seeds=[], families=[])
        self.assertFalse(gate["recovery_allowed"])
        self.assertIn(BLOCKER_BASELINE_SELECTION, gate["blocker_reasons"])
        self.assertIn(BLOCKER_SINGLE_SEED, gate["blocker_reasons"])
        self.assertIn(BLOCKER_AFFINITY_NOT_TESTED, gate["blocker_reasons"])


if __name__ == "__main__":
    unittest.main()
