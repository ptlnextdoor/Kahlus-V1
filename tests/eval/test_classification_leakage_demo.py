import tempfile
import unittest
from pathlib import Path

from neurotwin.adapters.synthetic import make_synthetic_event_batches, make_synthetic_recordings
from neurotwin.data.event_io import save_event_batches
from neurotwin.data.manifest_io import save_split_manifest
from neurotwin.data.split_manifest import build_split_manifest
from neurotwin.eval.paper_demos import PaperDemoConfig, run_leakage_demo


class ClassificationLeakageDemoTests(unittest.TestCase):
    def test_prepared_manifest_uses_moabb_dataset_label_and_claim_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "prepared"
            out_dir = root / "out"
            records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=2, modalities=("eeg",))
            batches = make_synthetic_event_batches(n_subjects=6, sessions_per_subject=2, modalities=("eeg",))
            split = build_split_manifest(records, policy="subject", seed=0)
            split_path = save_split_manifest(split, prepared / "split_manifest.json")
            event_path = save_event_batches(
                batches,
                prepared,
                manifest_metadata={"dataset": "moabb", "moabb_dataset": "BNCI2014_001"},
            )

            payload = run_leakage_demo(
                PaperDemoConfig(
                    task="motor_imagery_classification",
                    event_manifest=event_path,
                    split_manifest=split_path,
                    out_dir=out_dir,
                    window_length=8,
                    stride=8,
                    seeds=(0, 1, 2),
                )
            )
            artifact_presence = {
                "classification_leakage_demo.json": (out_dir / "classification_leakage_demo.json").exists(),
                "classification_leakage_demo.csv": (out_dir / "classification_leakage_demo.csv").exists(),
                "classification_leakage_figure.json": (out_dir / "classification_leakage_figure.json").exists(),
            }

        self.assertEqual(payload["dataset"], "moabb")
        self.assertEqual(payload["task"], "motor_imagery_classification")
        self.assertEqual(payload["paper_demo_gate"]["passed"], True)
        self.assertFalse(payload["claim_gate"]["bad_split_claim_allowed"])
        representative = payload["representative_seed_result"]["comparisons"]
        by_split = {row["split_id"]: row for row in representative}
        self.assertGreater(by_split["bad_segment_split"]["subject_overlap"], 0)
        self.assertEqual(by_split["correct_subject_heldout"]["subject_overlap"], 0)
        self.assertTrue(all(artifact_presence.values()), artifact_presence)


if __name__ == "__main__":
    unittest.main()
