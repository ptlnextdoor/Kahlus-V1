import unittest

from neurotwin.researchdock.data_card import build_researchdock_data_card
from neurotwin.researchdock.synthetic import make_synthetic_researchdock_sessions


class ResearchDockSyntheticTests(unittest.TestCase):
    def test_synthetic_generator_is_deterministic_and_covers_profiles(self):
        left = make_synthetic_researchdock_sessions(seed=7)
        right = make_synthetic_researchdock_sessions(seed=7)

        self.assertEqual(left, right)
        profiles = {session.metadata["profile"] for session in left}
        self.assertEqual(
            profiles,
            {
                "reward_responsive",
                "blunted_reward_response",
                "high_noise",
                "missing_pupil",
            },
        )

    def test_data_card_summarizes_safe_synthetic_sessions(self):
        sessions = make_synthetic_researchdock_sessions(seed=2)
        card = build_researchdock_data_card(sessions)

        self.assertEqual(card["dataset_id"], "researchdock_synthetic_v0")
        self.assertEqual(card["n_sessions"], 4)
        self.assertFalse(card["contains_pii"])
        self.assertFalse(card["contains_real_participant_data"])
        self.assertIn("missing_pupil", card["quality_flags"])


if __name__ == "__main__":
    unittest.main()
