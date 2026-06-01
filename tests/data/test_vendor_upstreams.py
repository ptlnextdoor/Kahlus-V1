import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


def _load_vendor_module():
    module_path = Path("scripts/vendor_upstreams.py")
    spec = importlib.util.spec_from_file_location("vendor_upstreams", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load vendor_upstreams.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VendorUpstreamsTests(unittest.TestCase):
    def _configure_vendor_module(self, root: Path):
        vendor_upstreams = _load_vendor_module()
        lock_path = root / "upstreams.lock.json"
        vendor_dir = root / "vendor"
        lock_path.write_text(
            json.dumps(
                {
                    "upstreams": {
                        "mamba": {
                            "reuse_status": "permissive",
                            "repo": "https://example.invalid/mamba.git",
                            "commit": "abc123",
                        },
                        "restricted": {
                            "reuse_status": "restricted",
                            "repo": "https://example.invalid/restricted.git",
                            "commit": "def456",
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        vendor_upstreams.LOCK_PATH = lock_path
        vendor_upstreams.VENDOR_DIR = vendor_dir
        return vendor_upstreams

    def test_non_dry_run_clones_then_checks_out_missing_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            vendor_upstreams = self._configure_vendor_module(Path(tmp))
            with (
                mock.patch.object(sys, "argv", ["vendor_upstreams.py", "--ids", "mamba"]),
                mock.patch.object(vendor_upstreams.shutil, "which", return_value="/usr/bin/git"),
                mock.patch.object(vendor_upstreams.subprocess, "run") as run,
            ):
                self.assertEqual(vendor_upstreams.main(), 0)

            target = vendor_upstreams.VENDOR_DIR / "mamba"
            run.assert_has_calls(
                [
                    mock.call(
                        ["/usr/bin/git", "clone", "--filter=blob:none", "https://example.invalid/mamba.git", str(target)],
                        check=True,
                        timeout=vendor_upstreams.GIT_TIMEOUT_SECONDS,
                    ),
                    mock.call(
                        ["/usr/bin/git", "-C", str(target), "checkout", "abc123"],
                        check=True,
                        timeout=vendor_upstreams.GIT_TIMEOUT_SECONDS,
                    ),
                ]
            )

    def test_non_dry_run_fetches_then_checks_out_existing_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            vendor_upstreams = self._configure_vendor_module(Path(tmp))
            target = vendor_upstreams.VENDOR_DIR / "mamba"
            target.mkdir(parents=True)
            with (
                mock.patch.object(sys, "argv", ["vendor_upstreams.py", "--ids", "mamba"]),
                mock.patch.object(vendor_upstreams.shutil, "which", return_value="/usr/bin/git"),
                mock.patch.object(vendor_upstreams.subprocess, "run") as run,
            ):
                self.assertEqual(vendor_upstreams.main(), 0)

            run.assert_has_calls(
                [
                    mock.call(
                        ["/usr/bin/git", "-C", str(target), "fetch", "--depth", "1", "origin", "abc123"],
                        check=True,
                        timeout=vendor_upstreams.GIT_TIMEOUT_SECONDS,
                    ),
                    mock.call(
                        ["/usr/bin/git", "-C", str(target), "checkout", "abc123"],
                        check=True,
                        timeout=vendor_upstreams.GIT_TIMEOUT_SECONDS,
                    ),
                ]
            )

    def test_restricted_upstream_is_skipped_without_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            vendor_upstreams = self._configure_vendor_module(Path(tmp))
            stdout = io.StringIO()
            with (
                redirect_stdout(stdout),
                mock.patch.object(sys, "argv", ["vendor_upstreams.py", "--ids", "restricted"]),
                mock.patch.object(vendor_upstreams.subprocess, "run") as run,
            ):
                self.assertEqual(vendor_upstreams.main(), 0)

        run.assert_not_called()
        self.assertIn("skip restricted", stdout.getvalue())

    def test_unknown_upstream_id_fails_before_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            vendor_upstreams = self._configure_vendor_module(Path(tmp))
            with (
                mock.patch.object(sys, "argv", ["vendor_upstreams.py", "--ids", "missing"]),
                mock.patch.object(vendor_upstreams.subprocess, "run") as run,
                self.assertRaises(SystemExit) as raised,
            ):
                vendor_upstreams.main()

        run.assert_not_called()
        self.assertIn("Unknown upstream id: missing", str(raised.exception))

    def test_git_timeout_mentions_upstream_operation_and_duration(self):
        vendor_upstreams = _load_vendor_module()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "mamba"
            target.mkdir()
            with mock.patch.object(
                vendor_upstreams.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(cmd=["git"], timeout=vendor_upstreams.GIT_TIMEOUT_SECONDS),
            ), mock.patch.object(vendor_upstreams.shutil, "which", return_value="/usr/bin/git"):
                with self.assertRaises(SystemExit) as raised:
                    vendor_upstreams._clone_or_checkout("mamba", "https://example.invalid/mamba.git", "abc123", target)

        message = str(raised.exception)
        self.assertIn("mamba", message)
        self.assertIn("fetch", message)
        self.assertIn(str(vendor_upstreams.GIT_TIMEOUT_SECONDS), message)
