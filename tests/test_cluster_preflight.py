import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurotwin.adapters.synthetic import make_synthetic_event_batches, make_synthetic_recordings
from neurotwin.data.event_io import save_event_batches
from neurotwin.data.manifest_io import save_split_manifest
from neurotwin.data.split_manifest import build_split_manifest
from neurotwin.runtime.preflight import materialize_cluster_config, run_cluster_preflight


class ClusterPreflightTests(unittest.TestCase):
    def _write_prepared(self, root: Path, n_time: int = 256) -> tuple[Path, Path]:
        records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=1, modalities=("eeg",))
        batches = make_synthetic_event_batches(
            n_subjects=6,
            sessions_per_subject=1,
            modalities=("eeg",),
            n_time=n_time,
        )
        split = build_split_manifest(records, policy="subject", seed=0)
        split_path = save_split_manifest(split, root / "split_manifest.json")
        event_path = save_event_batches(batches, root)
        return event_path, split_path

    def _write_config(self, root: Path, event_path: str, split_path: str, window_size: int = 128) -> Path:
        config = root / "config.yaml"
        config.write_text(
            "\n".join(
                [
                    "experiment: unit_cluster_preflight",
                    "dataset: moabb",
                    "task: neural_translation_v1",
                    "data:",
                    f"  event_manifest: {event_path}",
                    f"  split_manifest: {split_path}",
                    f"window_size: {window_size}",
                    f"stride: {window_size}",
                    "model:",
                    "  backbone: ssm_fallback",
                ]
            ),
            encoding="utf-8",
        )
        return config

    def test_preflight_rejects_placeholder_manifest_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "runs"
            run_root.mkdir()
            config = self._write_config(
                root,
                "/path/to/moabb_prepared/event_manifest.json",
                "/path/to/moabb_prepared/split_manifest.json",
            )

            report = run_cluster_preflight(config, run_root, require_prepared_windows=True)

            self.assertFalse(report.passed)
            self.assertTrue(any("placeholder" in violation for violation in report.violations))

    def test_preflight_rejects_relative_or_missing_manifest_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "runs"
            run_root.mkdir()
            config = self._write_config(root, "event_manifest.json", str(root / "missing_split_manifest.json"))

            report = run_cluster_preflight(config, run_root, require_prepared_windows=True)

            self.assertFalse(report.passed)
            self.assertTrue(any("event_manifest must be absolute" in violation for violation in report.violations))
            self.assertTrue(any("split_manifest does not exist" in violation for violation in report.violations))

    def test_preflight_rejects_zero_window_prepared_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "prepared"
            prepared.mkdir()
            run_root = root / "runs"
            run_root.mkdir()
            event_path, split_path = self._write_prepared(prepared, n_time=64)
            config = self._write_config(root, str(event_path), str(split_path), window_size=128)

            report = run_cluster_preflight(config, run_root, require_prepared_windows=True)

            self.assertFalse(report.passed)
            self.assertEqual(report.window_count, 0)
            self.assertTrue(any("zero windows" in violation for violation in report.violations))

    def test_preflight_accepts_nonzero_prepared_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "prepared"
            prepared.mkdir()
            run_root = root / "runs"
            run_root.mkdir()
            event_path, split_path = self._write_prepared(prepared, n_time=256)
            config = self._write_config(root, str(event_path), str(split_path), window_size=128)

            report = run_cluster_preflight(config, run_root, require_prepared_windows=True)

            self.assertTrue(report.passed, report.violations)
            self.assertGreater(report.window_count or 0, 0)
            for split_name in ("train", "val", "test"):
                self.assertGreater((report.window_counts_by_split or {})[split_name], 0)

    def test_preflight_accepts_expected_window_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "prepared"
            prepared.mkdir()
            run_root = root / "runs"
            run_root.mkdir()
            event_path, split_path = self._write_prepared(prepared, n_time=256)
            config = self._write_config(root, str(event_path), str(split_path), window_size=128)
            baseline = run_cluster_preflight(config, run_root, require_prepared_windows=True)

            report = run_cluster_preflight(
                config,
                run_root,
                require_prepared_windows=True,
                expect_window_count=baseline.window_count,
                expect_split_windows=baseline.window_counts_by_split,
            )

            self.assertTrue(report.passed, report.violations)

    def test_preflight_rejects_unexpected_window_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "prepared"
            prepared.mkdir()
            run_root = root / "runs"
            run_root.mkdir()
            event_path, split_path = self._write_prepared(prepared, n_time=256)
            config = self._write_config(root, str(event_path), str(split_path), window_size=128)

            report = run_cluster_preflight(
                config,
                run_root,
                require_prepared_windows=True,
                expect_window_count=999,
                expect_split_windows={"train": 999, "val": 999, "test": 999},
            )

            self.assertFalse(report.passed)
            self.assertTrue(any("expected window_count=999" in violation for violation in report.violations))
            self.assertTrue(any("expected split windows=" in violation for violation in report.violations))

    def test_materialize_config_writes_absolute_manifest_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "prepared"
            prepared.mkdir()
            event_path, split_path = self._write_prepared(prepared, n_time=256)
            template = self._write_config(
                root,
                "/path/to/moabb_prepared/event_manifest.json",
                "/path/to/moabb_prepared/split_manifest.json",
            )
            out = root / "outputs" / "configs" / "materialized.yaml"

            report = materialize_cluster_config(template, prepared, out)

            self.assertTrue(report.passed, report.violations)
            text = out.read_text(encoding="utf-8")
            self.assertIn(str(event_path), text)
            self.assertIn(str(split_path), text)
            self.assertNotIn("/path/to/", text)

    def test_materialize_config_rejects_tracked_config_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "prepared"
            prepared.mkdir()
            self._write_prepared(prepared, n_time=256)
            template = self._write_config(root, "/path/to/event_manifest.json", "/path/to/split_manifest.json")

            report = materialize_cluster_config(template, prepared, Path.cwd() / "configs" / "train" / "unit.yaml")

            self.assertFalse(report.passed)
            self.assertTrue(any("tracked configs" in violation for violation in report.violations))

    def test_cluster_preflight_cli_prints_machine_readable_output(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "prepared"
            prepared.mkdir()
            run_root = root / "runs"
            run_root.mkdir()
            event_path, split_path = self._write_prepared(prepared, n_time=256)
            config = self._write_config(root, str(event_path), str(split_path), window_size=128)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "cluster",
                    "preflight",
                    "--config",
                    str(config),
                    "--run-root",
                    str(run_root),
                    "--require-prepared-windows",
                    "--expect-window-count",
                    "12",
                    "--expect-split-windows",
                    "train:8,val:2,test:2",
                ],
                check=True,
                text=True,
                capture_output=True,
                env=env,
            )

            self.assertIn("preflight_passed=True", result.stdout)
            self.assertIn("window_count=", result.stdout)
            self.assertIn("window_counts_by_split=train:", result.stdout)

    def test_cluster_materialize_config_cli_prints_machine_readable_output(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "prepared"
            prepared.mkdir()
            event_path, split_path = self._write_prepared(prepared, n_time=256)
            template = self._write_config(
                root,
                "/path/to/moabb_prepared/event_manifest.json",
                "/path/to/moabb_prepared/split_manifest.json",
            )
            out = root / "outputs" / "configs" / "materialized.yaml"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "cluster",
                    "materialize-config",
                    "--template",
                    str(template),
                    "--prepared-root",
                    str(prepared),
                    "--out",
                    str(out),
                ],
                check=True,
                text=True,
                capture_output=True,
                env=env,
            )

            self.assertIn("materialize_config_passed=True", result.stdout)
            self.assertIn(f"event_manifest={event_path}", result.stdout)
            self.assertIn(f"split_manifest={split_path}", result.stdout)


if __name__ == "__main__":
    unittest.main()
