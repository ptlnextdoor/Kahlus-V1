import unittest

from neurotwin.adapters.synthetic import make_synthetic_recordings
from neurotwin.data.leakage import check_manifest_leakage
from neurotwin.data.split_manifest import build_split_manifest


class SplitLeakageTests(unittest.TestCase):
    def test_subject_split_has_no_subject_overlap(self):
        records = make_synthetic_recordings(
            n_subjects=12,
            sessions_per_subject=2,
            modalities=("fmri", "eeg"),
            sites=("site-a", "site-b"),
            datasets=("synthetic_a", "synthetic_b"),
        )
        manifest = build_split_manifest(records, policy="subject", seed=13)
        report = check_manifest_leakage(manifest, keys=("subject_id",))

        self.assertTrue(report.passed, report.violations)
        self.assertEqual(manifest.policy, "subject")
        self.assertEqual(manifest.split_stage, "recording_manifest")
        self.assertEqual(len(manifest.record_hashes), len(records))

    def test_site_and_dataset_splits_block_group_leakage(self):
        records = make_synthetic_recordings(
            n_subjects=16,
            sessions_per_subject=1,
            modalities=("fmri", "eeg", "meg"),
            sites=("site-a", "site-b", "site-c", "site-d"),
            datasets=("synthetic_a", "synthetic_b", "synthetic_c"),
        )

        for policy, key in (("site", "site_id"), ("dataset", "dataset")):
            with self.subTest(policy=policy):
                manifest = build_split_manifest(records, policy=policy, seed=3)
                report = check_manifest_leakage(manifest, keys=(key,))
                self.assertTrue(report.passed, report.violations)

    def test_leakage_checker_catches_record_reuse(self):
        records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=1)
        manifest = build_split_manifest(records, policy="subject", seed=7)
        reused = manifest.train[0]
        manifest.val.append(reused)

        report = check_manifest_leakage(manifest, keys=("subject_id",))

        self.assertFalse(report.passed)
        self.assertTrue(any("record_id" in violation for violation in report.violations))
