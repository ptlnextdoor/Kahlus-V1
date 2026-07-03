import tempfile
import unittest
from pathlib import Path

from neurotwin.stf import run_stf_synthetic_smoke, write_stf_smoke_artifacts


class STFSyntheticSmokeTests(unittest.TestCase):
    def test_smoke_writes_required_artifacts_and_passes_gate(self):
        payload = run_stf_synthetic_smoke(seed=0)

        self.assertTrue(payload["gate"]["scientific_claim_allowed"], payload["gate"]["failure_reasons"])
        self.assertFalse(payload["summary"]["a100_jobs_launched"])
        task_ids = {row["task_id"] for row in payload["baseline_rows"]}
        self.assertIn("shuffled_target_control", task_ids)
        self.assertIn("time_shifted_label_control", task_ids)

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_stf_smoke_artifacts(tmp, payload)
            for path in paths.values():
                self.assertTrue(Path(path).exists(), path)
            self.assertIn("tiny_ssm", paths["baseline_table"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
