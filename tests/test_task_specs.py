import unittest

from neurotwin.benchmarks.registry import competitor_registry
from neurotwin.benchmarks.task_specs import default_translation_tasks


class TaskSpecTests(unittest.TestCase):
    def test_required_v1_translation_tasks_are_present(self):
        task_ids = {task.task_id for task in default_translation_tasks()}

        self.assertIn("stimulus_past_fmri_to_future_fmri", task_ids)
        self.assertIn("eeg_meg_to_shared_latent_state", task_ids)
        self.assertIn("fmri_to_eeg_meg_spectral_proxy", task_ids)
        self.assertIn("anatomy_fmri_to_subject_conditioned_state", task_ids)
        self.assertIn("missing_modality_reconstruction", task_ids)
        self.assertIn("few_shot_subject_adaptation", task_ids)

    def test_competitor_registry_names_the_crowded_lanes(self):
        competitor_ids = {competitor.competitor_id for competitor in competitor_registry()}

        self.assertIn("tribe_v2", competitor_ids)
        self.assertIn("brainvista", competitor_ids)
        self.assertIn("brain_of", competitor_ids)
        self.assertIn("brainomni", competitor_ids)
        self.assertIn("brain_harmony", competitor_ids)
