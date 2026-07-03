from __future__ import annotations

import unittest

from neurotwin.forecastability.m3 import fetch_chbmit_seizure_records, parse_chbmit_summary, tusz_source_audit


class ForecastabilityM3Tests(unittest.TestCase):
    def test_chbmit_summary_parser(self) -> None:
        parsed = parse_chbmit_summary(
            "\n".join(
                [
                    "File Name: chb01_03.edf",
                    "Number of Seizures in File: 1",
                    "Seizure Start Time: 2996 seconds",
                    "Seizure End Time: 3036 seconds",
                ]
            )
        )
        self.assertEqual(parsed["chb01_03.edf"], ((2996, 3036),))

    def test_chbmit_records_source(self) -> None:
        self.assertIn("chb01/chb01_03.edf", fetch_chbmit_seizure_records())

    def test_external_dataset_audit_is_honest(self) -> None:
        self.assertEqual(tusz_source_audit()["status"], "not_run_requires_external_tusz_access")


if __name__ == "__main__":
    unittest.main()
