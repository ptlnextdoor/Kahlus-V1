import json
import subprocess
import sys
import unittest
from pathlib import Path

from neurotwin.adapters.registry import dataset_registry
from neurotwin.upstreams import permissive_upstreams, upstream_registry


class RegistryTests(unittest.TestCase):
    def test_dataset_registry_names_core_data_sources(self):
        dataset_ids = {dataset.dataset_id for dataset in dataset_registry()}

        self.assertIn("openneuro_bids", dataset_ids)
        self.assertIn("dandi_nwb", dataset_ids)
        self.assertIn("moabb_eeg", dataset_ids)
        self.assertIn("tuh_eeg", dataset_ids)
        self.assertIn("paired_eeg_fmri", dataset_ids)

    def test_cli_implemented_dataset_adapters_are_not_marked_planned(self):
        registry = {dataset.dataset_id: dataset for dataset in dataset_registry()}

        for dataset_id in ("openneuro_bids", "moabb_eeg"):
            self.assertIn("implemented", registry[dataset_id].adapter_status)
            self.assertNotIn("planned", registry[dataset_id].adapter_status)

    def test_upstream_registry_tracks_reuse_and_quarantine(self):
        upstreams = {upstream.upstream_id: upstream for upstream in upstream_registry()}

        self.assertEqual(upstreams["mamba"].reuse_status, "permissive")
        self.assertEqual(upstreams["brainlm"].reuse_status, "restricted")
        self.assertIn("mamba", {upstream.upstream_id for upstream in permissive_upstreams()})
        self.assertNotIn("brainlm", {upstream.upstream_id for upstream in permissive_upstreams()})

    def test_lockfile_contains_verified_commits(self):
        lock = json.loads(Path("external/upstreams.lock.json").read_text())

        self.assertEqual(lock["upstreams"]["mamba"]["commit"], "a14b1dff0454a3bc27d9eb31355dc01e4b2490ec")
        self.assertEqual(lock["upstreams"]["braindecode"]["repo"], "https://github.com/braindecode/braindecode")

    def test_vendor_script_dry_run(self):
        result = subprocess.run(
            [sys.executable, "scripts/vendor_upstreams.py", "--dry-run", "--ids", "mamba"],
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("would clone mamba", result.stdout)
