import os
import subprocess
import unittest
from pathlib import Path


class ResearchArtifactTests(unittest.TestCase):
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
            "What",
            "The friend running Chapman does not need GitHub access",
            "Raspberry Pi Handoff Path",
            "Use the Raspberry Pi only as a Chapman-network bridge",
            "bash scripts/package_runner_bundle.sh",
            "neurotwin-a100-runner-<short_sha>.tar.gz",
            "scp outputs/neurotwin-a100-runner-<short_sha>.tar.gz",
            "scp /tmp/neurotwin-a100-runner-<short_sha>.tar.gz",
            "tar -xzf ~/neurotwin-a100-runner-<short_sha>.tar.gz",
            "minimal practical code visibility",
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

    def test_package_runner_bundle_is_minimal_and_manifested(self):
        script = Path("scripts/package_runner_bundle.sh").read_text(encoding="utf-8")

        self.assertIn("neurotwin-a100-runner-$SHORT_SHA", script)
        self.assertIn("git archive", script)
        self.assertIn("Refusing to package a dirty worktree", script)
        self.assertIn("ALLOW_DIRTY_RUNNER_BUNDLE", script)
        self.assertIn("COMMIT_HASH.txt", script)
        self.assertIn("BUNDLE_MANIFEST.txt", script)
        self.assertIn("SHA256SUMS", script)
        self.assertIn("configs/train/moabb_a100_smoke.yaml", script)
        self.assertIn("configs/train/prepared_synthetic_debug.yaml", script)
        self.assertIn("scripts/run_smoke.sh", script)
        self.assertIn("scripts/run_full.sh", script)
        self.assertIn("scripts/run_full.sbatch", script)
        self.assertIn("scripts/train_a100_inner.sh", script)
        self.assertIn("scripts/prepare_moabb_benchmark.sh", script)
        self.assertIn("src", script)
        for excluded in (
            ".git",
            ".context",
            "tests",
            "docs/research",
            "docs/paper",
            "graphify-out",
            "outputs",
            "runs",
            "*.pt",
            "*.npy",
            "*.npz",
        ):
            self.assertIn(f"--exclude='{excluded}'", script)

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
