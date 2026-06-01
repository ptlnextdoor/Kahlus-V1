import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.artifacts.helpers import assert_runner_archive, copy_repo_to_temp_git


class RunnerBundleArtifactTests(unittest.TestCase):
    def test_copy_repo_to_temp_git_ignores_untracked_files(self):
        sentinel = Path("__untracked_packaging_sentinel__.txt")
        self.assertFalse(sentinel.exists(), sentinel)
        try:
            sentinel.write_text("must not be copied\n", encoding="utf-8")
            with tempfile.TemporaryDirectory() as tmp:
                tmp_repo = copy_repo_to_temp_git(tmp)
                self.assertFalse((tmp_repo / sentinel.name).exists())
        finally:
            sentinel.unlink(missing_ok=True)

    def test_package_bundle_uses_head_archive_and_dirty_guard(self):
        script = Path("scripts/package_run_bundle.sh").read_text(encoding="utf-8")

        self.assertIn("git archive", script)
        self.assertIn("Refusing to package a dirty worktree", script)
        self.assertIn("ALLOW_DIRTY_BUNDLE", script)
        self.assertIn("BUNDLE_METADATA.txt", script)
        for excluded in (".git", ".context", "outputs", "runs", "*.pt", "*.npy", "*.npz"):
            self.assertIn(f"--exclude='{excluded}'", script)

    def test_package_runner_bundle_smokes_real_archive(self):
        if shutil.which("shasum") is None:
            self.skipTest("shasum is required for runner bundle verification")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_repo = copy_repo_to_temp_git(tmp)
            result = subprocess.run(
                ["bash", "scripts/package_runner_bundle.sh"],
                cwd=tmp_repo,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

            archives = sorted((tmp_repo / "outputs").glob("neurotwin-a100-runner-*.tar.gz"))
            self.assertEqual(len(archives), 1)
            assert_runner_archive(self, archives[0], Path(tmp) / "extract")
