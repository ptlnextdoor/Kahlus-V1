import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from tests.artifact_helpers import assert_runner_archive, copy_repo_to_temp_git


class HandoffZipArtifactTests(unittest.TestCase):
    def test_package_a100_handoff_zip_smokes_real_archive(self):
        if shutil.which("shasum") is None:
            self.skipTest("shasum is required for handoff bundle verification")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_repo = copy_repo_to_temp_git(tmp)
            result = subprocess.run(
                ["bash", "scripts/package_a100_handoff_zip.sh"],
                cwd=tmp_repo,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

            zips = sorted((tmp_repo / "outputs").glob("neurotwin-a100-handoff-*.zip"))
            runners = sorted((tmp_repo / "outputs").glob("neurotwin-a100-runner-*.tar.gz"))
            self.assertEqual(len(zips), 1)
            self.assertEqual(len(runners), 1)

            zip_path = zips[0]
            extract_root = Path(tmp) / "handoff"
            with zipfile.ZipFile(zip_path, "r") as archive:
                names = set(archive.namelist())
                archive.extractall(extract_root)

            roots = {name.split("/", 1)[0] for name in names if name}
            self.assertEqual(len(roots), 1)
            root = roots.pop()
            runner_name = root.replace("handoff", "runner")
            expected = {
                f"{root}/COMMIT_HASH.txt",
                f"{root}/README_HANDOFF.md",
                f"{root}/SHA256SUMS",
                f"{root}/{runner_name}.tar.gz",
            }
            self.assertEqual(names, expected)

            handoff_root = extract_root / root
            readme = (handoff_root / "README_HANDOFF.md").read_text(encoding="utf-8")
            for required in (
                "minimal practical code visibility",
                "not cryptographic source secrecy",
                "scp",
                "Raspberry Pi",
                "sha256sum -c SHA256SUMS",
                "Primary Docker 6-GPU path",
                "bash scripts/run_docker_6gpu.sh",
                '--gpus "device=0,1,2,3,4,5"',
                "conda env create -f environment-a100.yml",
                "bash scripts/run_smoke.sh outputs/smoke",
                "bash scripts/run_full.sh /path/to/shared/persistent/neurotwin",
                "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
                '--gpus "device=<gpu_id>"',
                "/workspace/repo",
                "/raid/scratch/$USER/neurotwin-",
                "python -m neurotwin.cli eval audit",
                "python -m neurotwin.cli cluster materialize-config",
                "python -m neurotwin.cli cluster preflight",
                "torchrun --standalone --nproc_per_node=6",
                "python -m neurotwin.cli report",
                "Expected Outputs",
                "Known Limitations",
                "bash scripts/package_a100_evidence_bundle.sh",
                "MOABB task labels are intentionally removed",
            ):
                self.assertIn(required, readme)
            for forbidden in (
                "git clone",
                "<PRIVATE_REPO_URL>",
                "bash scripts/package_runner_bundle.sh",
                "bash scripts/package_run_bundle.sh",
            ):
                self.assertNotIn(forbidden, readme)

            checksum = subprocess.run(
                ["shasum", "-a", "256", "-c", "SHA256SUMS"],
                cwd=handoff_root,
                text=True,
                capture_output=True,
            )
            self.assertEqual(checksum.returncode, 0, checksum.stderr + checksum.stdout)
            assert_runner_archive(self, handoff_root / f"{runner_name}.tar.gz", Path(tmp) / "nested")
