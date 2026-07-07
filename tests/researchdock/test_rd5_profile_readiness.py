import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurotwin.researchdock import metrics as researchdock_metrics
from neurotwin.researchdock.synthetic import make_synthetic_researchdock_sessions


class ResearchDockProfileReadinessTests(unittest.TestCase):
    def test_readiness_audit_does_not_cluster_synthetic_profiles(self):
        self.assertTrue(hasattr(researchdock_metrics, "audit_response_profile_readiness"))
        rows = researchdock_metrics.compute_researchdock_metrics(make_synthetic_researchdock_sessions(seed=17))
        audit = researchdock_metrics.audit_response_profile_readiness(rows, min_sessions=20)

        self.assertEqual(audit["readiness_scope"], "future_response_profile_clustering_readiness")
        self.assertFalse(audit["clustering_performed"])
        self.assertFalse(audit["ready_for_future_clustering"])
        self.assertNotIn("cluster_labels", audit)
        self.assertEqual(audit["n_metric_rows"], 4)
        self.assertTrue(audit["finite_profile_vectors"])
        self.assertIn("insufficient_sessions_for_future_clustering", audit["failure_reasons"])
        self.assertIn("missing_pupil_profiles_present", audit["failure_reasons"])

    def test_script_writes_profile_readiness_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_researchdock_synthetic.py",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--write-profile-readiness",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("profile_readiness=", result.stdout)
            json_path = Path(tmp) / "researchdock_profile_readiness.json"
            report_path = Path(tmp) / "researchdock_profile_readiness_report.md"
            self.assertTrue(json_path.exists())
            self.assertTrue(report_path.exists())
            audit = json.loads(json_path.read_text(encoding="utf-8"))
            report = report_path.read_text(encoding="utf-8")
            self.assertFalse(audit["clustering_performed"])
            self.assertNotIn("cluster_labels", audit)
            self.assertIn("No clustering was performed", report)
            self.assertIn("insufficient_sessions_for_future_clustering", report)


if __name__ == "__main__":
    unittest.main()
