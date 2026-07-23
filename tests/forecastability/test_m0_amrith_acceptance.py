from __future__ import annotations

import unittest

from neurotwin.forecastability.amrith_acceptance import run_amrith_acceptance


class AmrithM0AcceptanceTests(unittest.TestCase):
    def test_amrith_acceptance_all_checks(self) -> None:
        payload = run_amrith_acceptance(seed=1)
        self.assertTrue(payload["passed"])
        for check, ok in payload["checks"].items():
            self.assertTrue(ok, f"failed check: {check}")


if __name__ == "__main__":
    unittest.main()
