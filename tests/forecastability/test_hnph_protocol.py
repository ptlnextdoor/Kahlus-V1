from __future__ import annotations

from pathlib import Path
import unittest

import yaml


_ROOT = Path(__file__).resolve().parents[2]
_PROTOCOL = _ROOT / "configs" / "protocol" / "hnph_phase0_v0.2.yaml"


class HnphProtocolTests(unittest.TestCase):
    def test_phase0_freeze_keeps_baseline_and_evidence_gates_machine_readable(self) -> None:
        protocol = yaml.safe_load(_PROTOCOL.read_text(encoding="utf-8"))

        self.assertEqual(protocol["program_position"]["phase0_role"], "neural_casp_baseline_evidence_ruler")
        self.assertEqual(protocol["program_position"]["downstream_flagship"], "passive_pci")
        self.assertFalse(protocol["program_position"]["hnph_results_are_passive_pci_evidence"])
        self.assertEqual(protocol["feasibility_gate"]["minimum_independent_subject_clusters"], 12)
        self.assertEqual(protocol["feasibility_gate"]["minimum_event_subjects"], 8)
        self.assertEqual(protocol["feasibility_gate"]["minimum_positive_primary_band_anchors"], 100)
        self.assertEqual(protocol["primary_endpoint"]["inference"]["bootstrap_replicates"], 2000)
        self.assertEqual(protocol["control_suite"]["nuisance_probe_maximum_accuracy_above_chance"], 0.20)
        self.assertEqual(protocol["evidence_artifact"]["required_outputs"], ["json", "markdown"])


if __name__ == "__main__":
    unittest.main()
