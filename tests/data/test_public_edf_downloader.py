from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _load_downloader():
    path = Path(__file__).resolve().parents[2] / "scripts" / "fetch_kahlus_public_edf_datasets.py"
    spec = importlib.util.spec_from_file_location("fetch_kahlus_public_edf_datasets", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PublicEdfDownloaderTests(unittest.TestCase):
    def test_expected_raw_folders_match_loader_contract(self):
        mod = _load_downloader()
        self.assertEqual(mod.EXPECTED_FOLDERS, ("sleep-edf", "chb-mit", "eegmmi", "siena"))
        self.assertEqual([spec.folder for spec in mod.DATASETS.values()], list(mod.EXPECTED_FOLDERS))

    def test_select_records_and_sleep_hypnogram_matching(self):
        mod = _load_downloader()
        sleep_records = [
            "sleep-cassette/SC4001E0-PSG.edf",
            "sleep-cassette/SC4001EC-Hypnogram.edf",
            "sleep-cassette/SC4002E0-PSG.edf",
        ]
        self.assertEqual(
            mod.select_primary_records("sleep_edf_expanded", sleep_records, max_records=1, full=False),
            ["sleep-cassette/SC4001E0-PSG.edf"],
        )
        self.assertEqual(
            mod.match_sleep_hypnogram(
                "sleep-cassette/SC4001E0-PSG.edf",
                ["SC4001EC-Hypnogram.edf", "SC4001E0-PSG.edf"],
            ),
            "SC4001EC-Hypnogram.edf",
        )
        self.assertEqual(
            mod.select_primary_records(
                "chb_mit_physionet",
                ["chb01/chb01_01.edf", "chb01/chb01_01.edf.seizures"],
                max_records=4,
                full=False,
            ),
            ["chb01/chb01_01.edf"],
        )

    def test_urls_use_official_physionet_files_endpoint(self):
        mod = _load_downloader()
        url = mod.url_for(mod.DATASETS["eegmmi_physionet"], "S001/S001R03.edf")
        self.assertEqual(url, "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R03.edf")

    def test_refuses_repo_internal_output_by_default(self):
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=repo) as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/fetch_kahlus_public_edf_datasets.py",
                    "--out-root",
                    str(Path(tmp) / "raw"),
                    "--max-records-per-dataset",
                    "1",
                ],
                cwd=repo,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("refusing to write public raw EDF data inside repo", result.stderr)


if __name__ == "__main__":
    unittest.main()
