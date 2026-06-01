import unittest

from neurotwin.adapters.synthetic import make_synthetic_recordings
from neurotwin.data.audit import audit_split_manifest
from neurotwin.data.split_manifest import build_split_manifest


class LeakageAuditTests(unittest.TestCase):
    def test_audit_catches_metadata_label_leak(self):
        records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=1)
        records[0].metadata["target_label"] = "class-a"
        manifest = build_split_manifest(records, policy="subject", seed=2)

        report = audit_split_manifest(manifest, policy="subject", forbidden_metadata_fields=("target_label",))

        self.assertFalse(report.passed)
        self.assertTrue(any("metadata" in violation for violation in report.violations))

    def test_audit_catches_overlapping_time_windows(self):
        records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=1)
        manifest = build_split_manifest(records, policy="subject", seed=2)
        train_record = manifest.train[0]
        test_record = manifest.test[0]
        test_record.metadata["source_record_id"] = train_record.record_id
        test_record.metadata["window_start"] = train_record.start_time
        test_record.metadata["window_end"] = train_record.end_time

        report = audit_split_manifest(manifest, policy="subject")

        self.assertFalse(report.passed)
        self.assertTrue(any("window overlap" in violation for violation in report.violations))
