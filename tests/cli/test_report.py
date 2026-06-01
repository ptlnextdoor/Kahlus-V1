import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from neurotwin.benchmarks.reports import generate_run_report


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

    def test_run_report_does_not_reraise_artifact_read_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text('{"mse": 0.1}', encoding="utf-8")
            original_read_text = Path.read_text

            def fail_metrics_read(path: Path, *args: object, **kwargs: object) -> str:
                if path == metrics_path:
                    raise OSError("read blocked")
                return original_read_text(path, *args, **kwargs)

            with mock.patch.object(Path, "read_text", fail_metrics_read):
                report = generate_run_report(run_dir)

        self.assertIn("artifact_error=read_failed", report)
        self.assertIn("read blocked", report)
