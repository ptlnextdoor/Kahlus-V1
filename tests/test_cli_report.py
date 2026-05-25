import os
import subprocess
import sys
import unittest


class CliReportTests(unittest.TestCase):
    def test_report_mentions_corrected_boss_fight_and_split_rules(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"

        result = subprocess.run(
            [sys.executable, "-m", "neurotwin.cli", "report", "--suite", "translation_smoke"],
            cwd=os.getcwd(),
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("Brain-OF", result.stdout)
        self.assertIn("TRIBE v2", result.stdout)
        self.assertIn("held-out subject/site/dataset", result.stdout)
        self.assertIn("missing-modality reconstruction", result.stdout)
