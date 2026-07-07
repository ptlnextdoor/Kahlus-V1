import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurotwin.researchdock.public_datasets import (
    researchdock_public_dataset_reviews,
    write_researchdock_public_dataset_review,
)


class ResearchDockRD3PublicDatasetReviewTests(unittest.TestCase):
    def test_review_names_verified_dataset_candidates_without_loaders(self):
        reviews = {review.dataset_id: review for review in researchdock_public_dataset_reviews()}
        self.assertEqual(set(reviews), {"wesad", "deap", "seed"})
        for review in reviews.values():
            self.assertTrue(review.official_urls)
            self.assertTrue(review.source_notes)
            self.assertEqual(review.loader_status, "review_only_no_loader")
            self.assertIn("no raw participant data", review.boundary_notes)
            self.assertNotIn("diagnos", " ".join(review.researchdock_fit).lower())

    def test_dataset_mappings_capture_researchdock_relevance_and_terms(self):
        reviews = {review.dataset_id: review for review in researchdock_public_dataset_reviews()}
        self.assertIn("ppg_hrv_proxy", reviews["wesad"].researchdock_fields)
        self.assertIn("stress", reviews["wesad"].task_labels)
        self.assertIn("eeg", reviews["deap"].modalities)
        self.assertIn("valence_arousal_self_report", reviews["deap"].researchdock_fields)
        self.assertEqual(reviews["deap"].access_status, "historical_public_dataset_page_unavailable_at_review")
        self.assertIn("application_required", reviews["seed"].access_status)
        self.assertIn("eye_tracking", reviews["seed"].modalities)

    def test_writer_outputs_review_json_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_researchdock_public_dataset_review(tmp)
            review = json.loads(Path(paths["json"]).read_text())
            report = Path(paths["report"]).read_text()
        self.assertEqual(review["sprint"], "RD-3")
        self.assertEqual([row["dataset_id"] for row in review["datasets"]], ["wesad", "deap", "seed"])
        self.assertIn("No Loaders", report)
        self.assertIn("WESAD", report)
        self.assertIn("DEAP", report)
        self.assertIn("SEED", report)
        self.assertIn("not diagnosis", report.lower())

    def test_script_writes_rd3_public_dataset_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_researchdock_synthetic.py",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--write-public-dataset-review",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            self.assertIn("public_dataset_review=", result.stdout)
            self.assertIn("public_dataset_report=", result.stdout)
            review = json.loads((Path(tmp) / "researchdock_public_dataset_review.json").read_text())
        self.assertEqual(review["sprint"], "RD-3")
        self.assertTrue(all(row["loader_status"] == "review_only_no_loader" for row in review["datasets"]))


if __name__ == "__main__":
    unittest.main()
