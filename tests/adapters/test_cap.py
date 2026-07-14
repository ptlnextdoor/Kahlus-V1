from __future__ import annotations

import unittest

from neurotwin.adapters.cap import CAP_DATASET_ID, CapSealError, CapSealedManifest, build_cap_registry
from tests.forecastability.test_contracts import _record


class CapSealTests(unittest.TestCase):
    def test_seal_blocks_records_until_independent_custodian_opens_it(self) -> None:
        manifest = CapSealedManifest("a" * 64, "custodian", "evaluator")
        with self.assertRaisesRegex(CapSealError, "remain sealed"):
            build_cap_registry((), manifest=manifest)

    def test_opened_seal_accepts_only_cap_records(self) -> None:
        manifest = CapSealedManifest("a" * 64, "custodian", "evaluator", opened=True)
        cap_record = _record().__class__(**{**_record().__dict__, "dataset_id": CAP_DATASET_ID})
        registry = build_cap_registry((cap_record,), manifest=manifest)

        self.assertEqual(registry.records[0].dataset_id, CAP_DATASET_ID)


if __name__ == "__main__":
    unittest.main()
