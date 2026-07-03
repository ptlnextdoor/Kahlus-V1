import json
import unittest
from pathlib import Path

from neurotwin.neurovisual import (
    REQUIRED_ONTOLOGY_FIELDS,
    build_condition_comparison_matrix,
    default_episode_profile,
    episode_profile_to_dict,
)


class NeurovisualOntologyTests(unittest.TestCase):
    def test_episode_profile_serializes_required_structured_fields(self):
        profile = default_episode_profile()
        payload = episode_profile_to_dict(profile)
        encoded = json.dumps(payload, sort_keys=True)

        self.assertIn("structured_history_h_t", payload)
        self.assertIn("not_diagnosis_notice", payload)
        self.assertIn("not a diagnosis", payload["not_diagnosis_notice"].lower())
        for field in REQUIRED_ONTOLOGY_FIELDS:
            self.assertIn(field, payload)
        self.assertIn("visual_field_location", encoded)
        self.assertNotIn("predicts_seizure", encoded)

    def test_condition_matrix_is_research_framing_not_diagnosis_engine(self):
        matrix = build_condition_comparison_matrix()
        names = {row["condition"] for row in matrix}

        self.assertIn("migraine_aura", names)
        self.assertIn("occipital_focal_aware_seizure_visual_aura", names)
        self.assertIn("medication_substance_metabolic_contributors", names)
        for row in matrix:
            self.assertIn("limits_of_inference", row)
            self.assertIn("claim_boundary", row)
            self.assertIn("not diagnostic", row["claim_boundary"].lower())

    def test_docs_state_side_branch_and_not_diagnosis_boundaries(self):
        repo = Path(__file__).resolve().parents[2]
        docs = [
            repo / "docs/research/kahlus_neurovisual_symptom_ontology.md",
            repo / "docs/research/kahlus_neurovisual_dataset_registry.md",
        ]
        for path in docs:
            text = path.read_text(encoding="utf-8").lower()
            self.assertIn("kahlus side branch", text)
            self.assertIn("not a diagnosis", text)


if __name__ == "__main__":
    unittest.main()
