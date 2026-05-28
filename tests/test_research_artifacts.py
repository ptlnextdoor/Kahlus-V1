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
            "scripts/run_smoke.sh",
            "scripts/run_full.sh",
            "scripts/run_full.sbatch",
            "scripts/package_run_bundle.sh",
            "README_RUN.md",
            "environment-a100.yml",
            "requirements/cluster-a100.txt",
            "docs/CLAIMS.md",
            "docs/A100_RUNBOOK.md",
            "docs/CHAPMAN_A100_QUICKSTART.md",
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
            "git clone https://github.com/ptlnextdoor/Kahlus-V1.git",
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
        self.assertIn("/tmp|/tmp/*|/private/tmp", run_full)
        self.assertNotIn("scripts/slurm/train_a100.sh", run_full_sbatch)
        self.assertNotIn("\nsbatch ", run_full_sbatch)
        for dependency in ("python=3.10", "pytorch-cuda=12.1", "moabb", "mne-bids", "scikit-learn"):
            self.assertIn(dependency, environment)

    def test_chapman_first_run_launcher_contains_required_sequence(self):
        launcher = Path("scripts/cluster/chapman_a100_first_run.sh").read_text(encoding="utf-8")

        self.assertIn("prepare_moabb_benchmark.sh", launcher)
        self.assertIn("EXPECTED_WINDOW_COUNT", launcher)
        self.assertIn("window_count != expected", launcher)
        self.assertIn("moabb_a100_chapman.yaml", launcher)
        self.assertIn("cluster preflight", launcher)
        self.assertIn("train --dry-run", launcher)
        self.assertIn("sbatch scripts/slurm/train_a100.sh", launcher)

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
