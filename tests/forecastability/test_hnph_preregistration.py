from __future__ import annotations

from pathlib import Path
import re
import unittest

import yaml

from neurotwin.repro import hash_file


_ROOT = Path(__file__).resolve().parents[2]
_PROTOCOL = _ROOT / "configs" / "protocol" / "hnph_phase0_v0.3.yaml"
_ADDENDUM = _ROOT / "docs" / "research" / "hnph_b2_preregistration_addendum.md"


class HnphPreregistrationTests(unittest.TestCase):
    def test_v03_freezes_b2_threshold_comparator_and_label_validity_requirements(self) -> None:
        protocol = yaml.safe_load(_PROTOCOL.read_text(encoding="utf-8"))

        self.assertEqual(protocol["protocol_id"], "kahlus.hnph.phase0.v0.3")
        self.assertEqual(protocol["status"], "frozen_preregistration_before_claim_mode")
        self.assertTrue(_ADDENDUM.exists())
        match = re.search(r"\*\*Frozen v0\.3 protocol SHA-256:\*\* `([0-9a-f]{64})`", _ADDENDUM.read_text())
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group(1), hash_file(_PROTOCOL))
        self.assertEqual(protocol["b2_preregistration_addendum"]["document"], "docs/research/hnph_b2_preregistration_addendum.md")
        self.assertEqual(protocol["effect_threshold"]["epsilon_bits_per_anchor"], 0.02)
        self.assertEqual(protocol["effect_threshold"]["familywise_hypothesis_count"], 12)
        self.assertEqual(protocol["effect_threshold"]["sigma_source"], "training_folds_or_synthetic_only")
        self.assertEqual(
            protocol["target"]["b2_primary_outcome"]["alphabet"],
            ["no_event", "Wake", "NREM", "REM", "Ambiguous"],
        )
        self.assertEqual(protocol["target"]["b2_primary_outcome"]["ambiguous_handling"], "additional_scored_class")
        self.assertEqual(
            protocol["semi_markov_comparator_acceptance"]["required"],
            [
                "dwell_time_fit",
                "held_subject_calibration_and_brier_decomposition",
                "synthetic_no_eeg_dominance",
                "beats_simpler_markov_ladder",
                "nuisance_challenger_adequacy_audit",
            ],
        )
        self.assertTrue(protocol["label_construct_validity"]["required_before_h3"])
        self.assertEqual(protocol["label_construct_validity"]["minimum_independent_raters"], 3)
        self.assertEqual(protocol["label_construct_validity"]["reference_baseline"], "same_frozen_chief_nuisance_comparator")
        self.assertEqual(protocol["label_construct_validity"]["probability_floor"], 1.0e-12)
        self.assertEqual(
            protocol["label_construct_validity"]["outcome_alphabet"],
            ["no_event", "Wake", "NREM", "REM", "Ambiguous"],
        )
        self.assertEqual(protocol["control_suite"]["positive_control"], "synthetic_known_signal_pass")
        self.assertEqual(protocol["control_suite"]["nuisance_probe_control"], "nuisance_probe")
        self.assertEqual(protocol["control_suite"]["nuisance_probe_maximum_accuracy_above_chance"], 0.20)
        self.assertIn("preregistration_hash", protocol["evidence_artifact"]["required_fields"])
        self.assertIn("effect_size", protocol["evidence_artifact"]["required_fields"])
        self.assertIn("label_reproducibility_family", protocol["evidence_artifact"]["required_fields"])
        self.assertIn("label_rater_target_provenance_sha256", protocol["evidence_artifact"]["required_fields"])
        self.assertEqual(protocol["control_suite"]["positive_control"], "synthetic_known_signal_pass")
        self.assertEqual(protocol["control_suite"]["nuisance_probe_control"], "nuisance_probe")
        self.assertIn("pass_authorize_h3", protocol["evidence_artifact"]["frozen_stop_reasons"])


if __name__ == "__main__":
    unittest.main()
