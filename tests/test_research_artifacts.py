import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path


class ResearchArtifactTests(unittest.TestCase):
    def _copy_repo_to_temp_git(self, tmp: str) -> Path:
        repo_root = Path.cwd()
        files = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.splitlines()

        tmp_repo = Path(tmp) / "repo"
        tmp_repo.mkdir()
        for rel in files:
            source = repo_root / rel
            if not source.is_file():
                continue
            target = tmp_repo / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        subprocess.run(["git", "init", "-q"], cwd=tmp_repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_repo, check=True)
        subprocess.run(["git", "config", "user.name", "Bundle Test"], cwd=tmp_repo, check=True)
        subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_repo, check=True)
        subprocess.run(["git", "add", "."], cwd=tmp_repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "bundle test"], cwd=tmp_repo, check=True)
        return tmp_repo

    def _assert_runner_archive(self, archive: Path, extract_root: Path) -> Path:
        with tarfile.open(archive, "r:gz") as tar:
            names = set(tar.getnames())
            try:
                tar.extractall(path=extract_root, filter="data")
            except TypeError:
                tar.extractall(path=extract_root)

        roots = {name.split("/", 1)[0] for name in names if name}
        self.assertEqual(len(roots), 1)
        root = roots.pop()

        required = {
            "BUNDLE_MANIFEST.txt",
            "BUNDLE_METADATA.txt",
            "COMMIT_HASH.txt",
            "README.md",
            "README_RUN.md",
            "SHA256SUMS",
            "configs/train/moabb_a100_smoke.yaml",
            "configs/train/prepared_synthetic_debug.yaml",
            "environment-a100.yml",
            "pyproject.toml",
            "requirements/cluster-a100.txt",
            "scripts/lib/moabb_prepare_common.sh",
            "scripts/package_runner_bundle.sh",
            "scripts/prepare_moabb_benchmark.sh",
            "scripts/run_full.sbatch",
            "scripts/run_full.sh",
            "scripts/run_smoke.sh",
            "scripts/slurm/_train_a100_inner.sh",
            "scripts/train_a100_inner.sh",
            "src/neurotwin/data/__init__.py",
            "src/neurotwin/data/windows.py",
        }
        for rel in required:
            self.assertIn(f"{root}/{rel}", names)

        forbidden_parts = {
            ".context",
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
            "graphify-out",
            "outputs",
            "runs",
            "tests",
        }
        forbidden_suffixes = (".ckpt", ".npy", ".npz", ".pt", ".pth", ".pyc")
        for name in names:
            rel = name.split("/", 1)[1] if "/" in name else ""
            parts = Path(rel).parts
            self.assertFalse(any(part.startswith("._") for part in parts), name)
            self.assertFalse(forbidden_parts.intersection(parts), name)
            self.assertFalse(rel.startswith("docs/paper/"), name)
            self.assertFalse(rel.startswith("docs/research/"), name)
            self.assertFalse(rel.endswith(forbidden_suffixes), name)

        bundle_root = extract_root / root
        manifest = (bundle_root / "BUNDLE_MANIFEST.txt").read_text(encoding="utf-8").splitlines()
        payload = sorted(
            path.relative_to(bundle_root).as_posix()
            for path in bundle_root.rglob("*")
            if path.is_file() and path.name not in {"BUNDLE_MANIFEST.txt", "SHA256SUMS"}
        )
        self.assertEqual(manifest, payload)

        checksum = subprocess.run(
            ["shasum", "-a", "256", "-c", "SHA256SUMS"],
            cwd=bundle_root,
            text=True,
            capture_output=True,
        )
        self.assertEqual(checksum.returncode, 0, checksum.stderr + checksum.stdout)
        return bundle_root

    def test_a100_h100_configs_scripts_and_paper_docs_exist(self):
        required = [
            "configs/train/moabb_debug.yaml",
            "configs/train/moabb_smoke_locked.yaml",
            "configs/train/prepared_synthetic_multitask_debug.yaml",
            "configs/train/moabb_a100.yaml",
            "configs/train/moabb_a100_smoke.yaml",
            "configs/train/neurotwin_v1_a100.yaml",
            "configs/train/moabb_h100.yaml",
            "configs/train/bids_debug.yaml",
            "configs/train/neurotwin_v1_h100.yaml",
            "configs/eval/neural_translation_v1.yaml",
            "scripts/slurm/train_a100.sh",
            "scripts/slurm/_train_a100_inner.sh",
            "scripts/slurm/eval_a100.sh",
            "scripts/slurm/sweep_a100.sh",
            "scripts/slurm/train_h100.sh",
            "scripts/slurm/eval_h100.sh",
            "scripts/slurm/sweep_h100.sh",
            "scripts/prepare_moabb_smoke.sh",
            "scripts/prepare_moabb_benchmark.sh",
            "scripts/cluster/chapman_a100_first_run.sh",
            "scripts/cluster/runpod_a100_rehearsal.sh",
            "scripts/run_smoke.sh",
            "scripts/run_full.sh",
            "scripts/run_full.sbatch",
            "scripts/package_run_bundle.sh",
            "scripts/package_a100_handoff_zip.sh",
            "scripts/package_runner_bundle.sh",
            "scripts/train_a100_inner.sh",
            "README_RUN.md",
            "environment-a100.yml",
            "requirements/cluster-a100.txt",
            "docs/CLAIMS.md",
            "docs/A100_RUNBOOK.md",
            "docs/CHAPMAN_A100_QUICKSTART.md",
            "docs/RUNPOD_A100_REHEARSAL.md",
            "docs/H100_RUNBOOK.md",
            "docs/paper/outline.md",
            "docs/paper/limitations.md",
        ]

        for path in required:
            with self.subTest(path=path):
                self.assertTrue(Path(path).exists(), path)

    def test_claims_doc_blocks_forbidden_claims(self):
        claims = Path("docs/CLAIMS.md").read_text(encoding="utf-8")

        self.assertIn("Do not claim", claims)
        self.assertIn("clinical digital twin", claims)
        self.assertIn("Synthetic smoke tests validate plumbing only", claims)
        self.assertIn("tribe_style", claims)
        self.assertIn("clean-room approximation", claims)
        self.assertIn("Do not describe it as exact TRIBE v2", claims)

    def test_moabb_scripts_and_cluster_configs_use_benchmark_windows(self):
        smoke = Path("scripts/prepare_moabb_smoke.sh").read_text(encoding="utf-8")
        benchmark = Path("scripts/prepare_moabb_benchmark.sh").read_text(encoding="utf-8")
        a100 = Path("configs/train/moabb_a100.yaml").read_text(encoding="utf-8")
        a100_smoke = Path("configs/train/moabb_a100_smoke.yaml").read_text(encoding="utf-8")
        h100 = Path("configs/train/moabb_h100.yaml").read_text(encoding="utf-8")

        for script in (smoke, benchmark):
            self.assertIn('WINDOW_LENGTH="${WINDOW_LENGTH:-128}"', script)
            self.assertIn('STRIDE="${STRIDE:-128}"', script)
            self.assertIn("--require-windows", script)
        for config in (a100, a100_smoke, h100):
            self.assertIn("window_size: 128", config)
            self.assertIn("stride: 128", config)

    def test_a100_slurm_scripts_require_safe_inputs(self):
        train = Path("scripts/slurm/train_a100.sh").read_text(encoding="utf-8")
        inner = Path("scripts/slurm/_train_a100_inner.sh").read_text(encoding="utf-8")
        eval_script = Path("scripts/slurm/eval_a100.sh").read_text(encoding="utf-8")

        self.assertIn("Refusing to run the generic placeholder config", train)
        self.assertIn("_train_a100_inner.sh", train)
        self.assertIn("cluster preflight", inner)
        self.assertLess(inner.index("cluster preflight"), inner.index("torchrun"))
        self.assertIn("--require-cuda", inner)
        self.assertIn("--require-prepared-windows", inner)
        self.assertIn("Refusing to run default/synthetic eval", eval_script)
        self.assertNotIn("python -m neurotwin.cli eval --suite", eval_script)

    def test_operator_run_bundle_files_are_self_contained(self):
        readme = Path("README_RUN.md").read_text(encoding="utf-8")
        run_full = Path("scripts/run_full.sh").read_text(encoding="utf-8")
        run_full_sbatch = Path("scripts/run_full.sbatch").read_text(encoding="utf-8")
        environment = Path("environment-a100.yml").read_text(encoding="utf-8")

        for required in (
            "Operator Workflow",
            "sha256sum -c SHA256SUMS",
            "conda env create -f environment-a100.yml",
            "python -m pip install -e '.[moabb,cluster]'",
            "Raspberry Pi Handoff Path",
            "Use the Raspberry Pi only as a Chapman-network bridge",
            "neurotwin-a100-runner-<short_sha>.tar.gz",
            "scp neurotwin-a100-runner-<short_sha>.tar.gz",
            "scp /tmp/neurotwin-a100-runner-<short_sha>.tar.gz",
            "tar -xzf ~/neurotwin-a100-runner-<short_sha>.tar.gz",
            "bash scripts/run_smoke.sh",
            "bash scripts/run_full.sh",
            "1x A100 80GB",
            "128G",
            "MOABB `BNCI2014_001`",
            "Expected Full Outputs",
            "Success Condition",
            "Resume And Safe Rerun",
        ):
            self.assertIn(required, readme)
        for developer_only in (
            "bash scripts/package_runner_bundle.sh",
            "bash scripts/package_run_bundle.sh",
            "git clone",
            "<PRIVATE_REPO_URL>",
            "clean committed checkout",
            "packaging machine",
            "full-source bundle",
        ):
            self.assertNotIn(developer_only, readme)
        self.assertIn("outputs/configs/moabb_a100.materialized.yaml", run_full)
        self.assertIn("EXPECTED_WINDOW_COUNT", run_full)
        self.assertIn("EXPECTED_TRAIN_WINDOWS", run_full)
        self.assertIn("cluster materialize-config", run_full)
        self.assertIn("--expect-window-count", run_full)
        self.assertIn("--expect-split-windows", run_full)
        self.assertIn("SBATCH_PARTITION", run_full)
        self.assertIn("SBATCH_ACCOUNT", run_full)
        self.assertIn("SBATCH_QOS", run_full)
        self.assertIn("RUN_LOG_DIR", run_full)
        self.assertIn("--output \"$RUN_LOG_DIR/neurotwin-a100-full-%j.out\"", run_full)
        self.assertIn("--error \"$RUN_LOG_DIR/neurotwin-a100-full-%j.err\"", run_full)
        self.assertIn("/Users|/Users/*", run_full)
        self.assertIn("/path/to|/path/to/*|/absolute|/absolute/*", run_full)
        self.assertIn("Persistent root must not be inside the checkout", run_full)
        self.assertIn("REPO_ROOT", run_full_sbatch)
        self.assertNotIn('dirname "${BASH_SOURCE[0]}"', run_full_sbatch)
        self.assertIn("/tmp|/tmp/*|/private/tmp", run_full)
        self.assertNotIn("scripts/slurm/train_a100.sh", run_full_sbatch)
        self.assertNotIn("\nsbatch ", run_full_sbatch)
        self.assertIn("scripts/train_a100_inner.sh", run_full_sbatch)
        for dependency in ("python=3.10", "pytorch-cuda=12.1", "moabb", "mne-bids", "scikit-learn"):
            self.assertIn(dependency, environment)

    def test_chapman_first_run_launcher_contains_required_sequence(self):
        launcher = Path("scripts/cluster/chapman_a100_first_run.sh").read_text(encoding="utf-8")

        self.assertIn("scripts/run_full.sh", launcher)
        self.assertIn("exec", launcher)
        self.assertNotIn("moabb_a100_chapman.yaml", launcher)
        self.assertNotIn("sbatch scripts/slurm/train_a100.sh", launcher)

    def test_runpod_rehearsal_is_budget_gated(self):
        script = Path("scripts/cluster/runpod_a100_rehearsal.sh").read_text(encoding="utf-8")
        doc = Path("docs/RUNPOD_A100_REHEARSAL.md").read_text(encoding="utf-8")

        self.assertIn("RUNPOD_MAX_BUDGET_USD", script)
        self.assertIn("must be <= 5", script)
        self.assertIn("A100", script)
        self.assertIn("run_full.sbatch", script)
        self.assertIn("runpod_rehearsal_passed=True", script)
        self.assertIn("$5", doc)
        self.assertIn("not a scientific result", doc)

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
            tmp_repo = self._copy_repo_to_temp_git(tmp)
            result = subprocess.run(
                ["bash", "scripts/package_runner_bundle.sh"],
                cwd=tmp_repo,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

            archives = sorted((tmp_repo / "outputs").glob("neurotwin-a100-runner-*.tar.gz"))
            self.assertEqual(len(archives), 1)
            self._assert_runner_archive(archives[0], Path(tmp) / "extract")

    def test_package_a100_handoff_zip_smokes_real_archive(self):
        if shutil.which("shasum") is None:
            self.skipTest("shasum is required for handoff bundle verification")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_repo = self._copy_repo_to_temp_git(tmp)
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
                "conda env create -f environment-a100.yml",
                "bash scripts/run_smoke.sh outputs/smoke",
                "bash scripts/run_full.sh /path/to/shared/persistent/neurotwin",
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
            self._assert_runner_archive(handoff_root / f"{runner_name}.tar.gz", Path(tmp) / "nested")

    def test_moabb_benchmark_script_blocks_slurm_tmp_fallback(self):
        env = dict(os.environ)
        env.pop("NEUROTWIN_DATA", None)
        env["SLURM_JOB_ID"] = "unit-test"

        result = subprocess.run(
            ["bash", "scripts/prepare_moabb_benchmark.sh"],
            text=True,
            capture_output=True,
            env=env,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NEUROTWIN_DATA must be set", result.stderr + result.stdout)
