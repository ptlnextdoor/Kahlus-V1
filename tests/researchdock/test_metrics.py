import unittest

import numpy as np

from neurotwin.researchdock.metrics import (
    compute_researchdock_metrics,
    response_profile_vector,
)
from neurotwin.researchdock.synthetic import make_synthetic_researchdock_sessions


class ResearchDockMetricTests(unittest.TestCase):
    def test_metrics_are_finite_for_synthetic_sessions(self):
        sessions = make_synthetic_researchdock_sessions(seed=11)
        rows = compute_researchdock_metrics(sessions)

        self.assertEqual(len(rows), 4)
        for row in rows:
            for key in (
                "reward_response_delta",
                "reaction_time_change",
                "effort_persistence_score",
                "pupil_response_amplitude",
                "pupil_recovery_slope",
                "ppg_hrv_proxy_mean",
                "task_accuracy_mean",
            ):
                self.assertIn(key, row)
                self.assertTrue(np.isfinite(row[key]), f"{key} was not finite in {row}")

    def test_missing_pupil_session_is_handled_without_nan_metrics(self):
        sessions = make_synthetic_researchdock_sessions(seed=3)
        missing = [session for session in sessions if "missing_pupil" in session.session_id][0]
        row = compute_researchdock_metrics((missing,))[0]

        self.assertEqual(row["pupil_sample_count"], 0)
        self.assertEqual(row["pupil_response_amplitude"], 0.0)
        self.assertEqual(row["reward_response_delta"], 0.0)
        self.assertIn("missing_pupil", row["quality_flags"])

    def test_response_profile_vector_is_stable_numeric_order(self):
        row = compute_researchdock_metrics(make_synthetic_researchdock_sessions(seed=5))[0]
        vector = response_profile_vector(row)

        self.assertEqual(vector.shape, (8,))
        self.assertTrue(np.isfinite(vector).all())
        self.assertEqual(vector.dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
