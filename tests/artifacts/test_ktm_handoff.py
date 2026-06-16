import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

RUNNER_REQUIRED = {
    "COMMIT_HASH.txt",
    "SHA256SUMS",
    "BUNDLE_MANIFEST.txt",
    "BUNDLE_METADATA.txt",
    "AGENT_RUNBOOK.md",
    "README_HANDOFF.md",
    "README_KTM_HANDOFF.md.in",
    "README_RUN.md",
    "environment-ktm-a100.yml",
    "pyproject.toml",
    "configs/train/ktm_a100_micro.yaml",
    "configs/train/ktm_recovery_point_objective.yaml",
    "configs/train/ktm_recovery_capacity_smoke.yaml",
    "configs/train/ktm_synthetic_smoke.yaml",
    "scripts/_bootstrap.py",
    "scripts/run_ktm_failure_analysis.py",
    "scripts/run_ktm_train.py",
    "scripts/run_ktm_smoke.sh",
    "scripts/docker_ktm_inner.sh",
    "scripts/run_docker_ktm.sh",
    "scripts/docker_gpu_preflight.py",
    "scripts/package_ktm_evidence_bundle.py",
    "scripts/slurm/train_ktm_a100.sh",
    "scripts/slurm/_train_ktm_a100_inner.sh",
    "src/neurotwin/training_v3/trainer.py",
    "src/neurotwin/models/ktm/torch_ktm.py",
}
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".ckpt", ".npy", ".npz", ".pem", ".key", ".pyc")


class KtmHandoffTests(unittest.TestCase):
    def setUp(self):
        if shutil.which("shasum") is None and shutil.which("sha256sum") is None:
            self.skipTest("a sha256 checker is required")
        self.repo = Path.cwd()

    def test_handoff_zip_shape_and_runner_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            env = {**os.environ, "OUT_DIR": str(out)}
            result = subprocess.run(
                ["bash", "scripts/package_ktm_a100_handoff_zip.sh"],
                cwd=self.repo, text=True, capture_output=True, env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

            zips = sorted(out.glob("kahlus-ktm-a100-handoff-*.zip"))
            self.assertEqual(len(zips), 1)
            extract = Path(tmp) / "x"
            with zipfile.ZipFile(zips[0]) as archive:
                names = set(archive.namelist())
                archive.extractall(extract)
            roots = {n.split("/", 1)[0] for n in names}
            self.assertEqual(len(roots), 1)
            root = roots.pop()
            self.assertEqual(
                names,
                {
                    f"{root}/COMMIT_HASH.txt",
                    f"{root}/README_HANDOFF.md",
                    f"{root}/SHA256SUMS",
                    f"{root}/{root.replace('handoff', 'runner')}.tar.gz",
                },
            )

            # README content contract: MOABB-not-applicable + Sprint 3D config + honest 7/8 GPU.
            readme = (extract / root / "README_HANDOFF.md").read_text()
            self.assertIn("MOABB audit is not applicable", readme)
            self.assertIn("configs/train/ktm_recovery_point_objective.yaml", readme)
            self.assertIn("configs/train/ktm_recovery_capacity_smoke.yaml", readme)
            self.assertIn("nproc_per_node=7", readme)
            self.assertIn("nproc_per_node=8", readme)
            self.assertIn("--expected-gpus 7", readme)
            self.assertIn("synthetic_ktm_recovery", readme)
            self.assertIn("AGENT_RUNBOOK.md", readme)

            # Extract + verify the runner tarball.
            runner_tar = extract / root / f"{root.replace('handoff', 'runner')}.tar.gz"
            with tarfile.open(runner_tar, "r:gz") as tar:
                members = tar.getnames()
                try:
                    tar.extractall(extract / "runner", filter="data")
                except TypeError:
                    tar.extractall(extract / "runner")
            runner_root = members[0].split("/", 1)[0]
            present = {m.split("/", 1)[1] for m in members if "/" in m}
            for rel in RUNNER_REQUIRED:
                self.assertIn(rel, present, rel)
            for m in members:
                self.assertFalse(m.endswith(FORBIDDEN_SUFFIXES), m)
                self.assertNotIn("__pycache__", m)
                self.assertFalse("/tests/" in m, m)
            runner = extract / "runner" / runner_root
            runner_text = "\n".join(
                (runner / name).read_text(encoding="utf-8")
                for name in ("AGENT_RUNBOOK.md", "README_RUN.md", "README_HANDOFF.md")
            )
            self.assertIn("KTM_CONFIG=configs/train/ktm_recovery_point_objective.yaml", runner_text)
            self.assertIn("configs/train/ktm_recovery_capacity_smoke.yaml", runner_text)
            self.assertIn("--expected-gpus 7", runner_text)

            checker = "shasum" if shutil.which("shasum") else "sha256sum"
            cmd = ["shasum", "-a", "256", "-c", "SHA256SUMS"] if checker == "shasum" \
                else ["sha256sum", "-c", "SHA256SUMS"]
            verify = subprocess.run(cmd, cwd=runner, text=True, capture_output=True)
            self.assertEqual(verify.returncode, 0, verify.stderr + verify.stdout)

    def test_evidence_bundle_includes_run_files_excludes_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            persistent = Path(tmp) / "persistent"
            run_dir = persistent / "runs" / "ktm_micro_sweep"
            run_dir.mkdir(parents=True)
            for name in ("metrics.json", "baseline_table.json", "baseline_table.csv",
                         "evidence_gate.json", "model_card.json", "data_card.json",
                         "run_config.json", "failure_reasons.json", "environment.json",
                         "gpu_preflight.json"):
                (run_dir / name).write_text("{}", encoding="utf-8")
            # secrets / checkpoints that MUST be excluded
            (run_dir / "checkpoint.pt").write_text("x", encoding="utf-8")
            (run_dir / ".env").write_text("SECRET=1", encoding="utf-8")
            (run_dir / "id.pem").write_text("key", encoding="utf-8")
            (persistent / "logs").mkdir()
            (persistent / "logs" / "kahlus-ktm-docker-20260614.log").write_text("log", encoding="utf-8")

            zip_path = Path(tmp) / "evidence.zip"
            result = subprocess.run(
                ["python3", "scripts/package_ktm_evidence_bundle.py",
                 str(persistent), str(zip_path), "kahlus-ktm-evidence", str(self.repo), "deadbeef"],
                cwd=self.repo, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

            with zipfile.ZipFile(zip_path) as archive:
                names = archive.namelist()
            rels = {n.split("/", 1)[1] for n in names if "/" in n}
            self.assertIn("run/metrics.json", rels)
            self.assertIn("run/environment.json", rels)
            self.assertIn("run/gpu_preflight.json", rels)
            self.assertIn("logs/kahlus-ktm-docker-20260614.log", rels)
            self.assertIn("README_SEND_TO_FRIEND.md", rels)
            self.assertIn("handoff-SHA256SUMS", rels)
            # exclusions
            self.assertFalse(any(n.endswith("checkpoint.pt") for n in names), names)
            self.assertFalse(any(n.endswith(".env") for n in names), names)
            self.assertFalse(any(n.endswith(".pem") for n in names), names)


if __name__ == "__main__":
    unittest.main()
