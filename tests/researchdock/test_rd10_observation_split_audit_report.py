import tempfile
import unittest
from pathlib import Path

from neurotwin.researchdock.observation_model import (
    build_researchdock_observation_task,
    run_researchdock_observation_benchmark,
    write_researchdock_observation_artifacts,
)
from neurotwin.researchdock.synthetic import make_synthetic_researchdock_sessions


class ResearchDockRD10ObservationSplitAuditReportTests(unittest.TestCase):
    def test_observation_report_lists_split_audit_failure_reasons(self):
        task = build_researchdock_observation_task(make_synthetic_researchdock_sessions(seed=0), seed=0)
        result = run_researchdock_observation_benchmark(task)
        result["split"] = {
            **result["split"],
            "subject_overlap": True,
        }

        with tempfile.TemporaryDirectory() as tmp:
            write_researchdock_observation_artifacts(tmp, task=task, result=result)
            report = (Path(tmp) / "researchdock_observation_report.md").read_text(encoding="utf-8")

        self.assertIn("## Split Audit", report)
        self.assertIn("- leakage_passed: False", report)
        self.assertIn("## Split Audit Failures", report)
        self.assertIn("- subject overlap across train/test split", report)


if __name__ == "__main__":
    unittest.main()
