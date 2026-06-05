import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class ArtifactDocsContractsTests(unittest.TestCase):
    def _run_docker_launcher_dry_run(self, env_overrides: dict[str, str]) -> dict[str, str]:
        with tempfile.TemporaryDirectory() as tmp:
            persistent = Path(tmp) / "persistent"
            env = dict(os.environ)
            env.pop("CUDA_VISIBLE_DEVICES", None)
            env.pop("CONTAINER_CUDA_VISIBLE_DEVICES", None)
            env.update(
                {
                    "NEUROTWIN_DOCKER_DRY_RUN": "1",
                    "NEUROTWIN_ALLOW_LOCAL_PERSISTENT_ROOT": "1",
                    "DOCKER_RUN_ID": "unit-test",
                    **env_overrides,
                }
            )
            result = subprocess.run(
                ["bash", "scripts/run_docker_6gpu.sh", str(persistent)],
                text=True,
                capture_output=True,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            values: dict[str, str] = {}
            for line in (persistent / "docker_run.env").read_text(encoding="utf-8").splitlines():
                key, value = line.split("=", 1)
                values[key] = value
            return values

    def test_docker_launcher_uses_container_local_cuda_visible_devices(self):
        values = self._run_docker_launcher_dry_run({"CUDA_VISIBLE_DEVICES": "2,3,4,5,6,7"})

        self.assertEqual(values["HOST_GPU_IDS"], "0,1,2,3,4,5")
        self.assertEqual(values["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5")

    def test_docker_launcher_default_and_diagnostic_cuda_visible_devices(self):
        default_values = self._run_docker_launcher_dry_run({})
        diagnostic_values = self._run_docker_launcher_dry_run(
            {"GPU_COUNT": "1", "NPROC_PER_NODE": "1", "HOST_GPU_IDS": "7"}
        )

        self.assertEqual(default_values["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5")
        self.assertEqual(diagnostic_values["HOST_GPU_IDS"], "7")
        self.assertEqual(diagnostic_values["CUDA_VISIBLE_DEVICES"], "0")

    def test_docker_launcher_honors_explicit_container_cuda_visible_devices(self):
        values = self._run_docker_launcher_dry_run(
            {
                "GPU_COUNT": "2",
                "NPROC_PER_NODE": "2",
                "HOST_GPU_IDS": "4,5",
                "CUDA_VISIBLE_DEVICES": "4,5",
                "CONTAINER_CUDA_VISIBLE_DEVICES": "0,1",
            }
        )

        self.assertEqual(values["HOST_GPU_IDS"], "4,5")
        self.assertEqual(values["CUDA_VISIBLE_DEVICES"], "0,1")

    def test_a100_h100_configs_scripts_and_paper_docs_exist(self):
        required = [
            "configs/train/moabb_debug.yaml",
            "configs/train/moabb_smoke_locked.yaml",
            "configs/train/prepared_synthetic_multitask_debug.yaml",
            "configs/train/moabb_a100.yaml",
            "configs/train/moabb_a100_smoke.yaml",
            "configs/train/neurotwin_v1_a100.yaml",
            "configs/train/algonauts_real_stimulus_debug.yaml",
            "configs/train/algonauts_pair_operator_debug.yaml",
            "configs/train/algonauts_pair_operator_full.yaml",
            "configs/train/algonauts_pair_operator_ablation_array.yaml",
            "configs/train/synthetic_field_compiler_debug.yaml",
            "configs/train/algonauts_field_compiler_debug.yaml",
            "configs/train/moabb_field_compiler_debug.yaml",
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
            "scripts/run_docker_6gpu.sh",
            "scripts/docker_a100_inner.sh",
            "scripts/docker_gpu_preflight.py",
            "scripts/run_full.sh",
            "scripts/run_full.sbatch",
            "scripts/package_a100_evidence_bundle.sh",
            "scripts/package_a100_evidence_bundle.py",
            "scripts/package_run_bundle.sh",
            "scripts/package_a100_handoff_zip.sh",
            "scripts/package_runner_bundle.sh",
            "scripts/render_a100_handoff_readme.py",
            "scripts/train_a100_inner.sh",
            "Dockerfile.a100",
            "README_HANDOFF.md.in",
            "README_AGENT_DEPLOY.md",
            "README_RUN.md",
            "environment-a100.yml",
            "requirements/cluster-a100.txt",
            "docs/CLAIMS.md",
            "docs/A100_RUNBOOK.md",
            "docs/research/neurotwin_nfc_research_dossier.pdf",
            "docs/research/neurotwin_nfc_research_dossier.tex",
            "docs/research/nfc_implementation_plan.md",
            "docs/research/nfc_architecture_decision_log.md",
            "docs/research/nfc_falsification_criteria.md",
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
        self.assertIn("Neural Field Compiler", claims)
        self.assertIn("Pair-Operator is an ablation", claims)
        self.assertIn("tribe_style", claims)
        self.assertIn("clean-room approximation", claims)
        self.assertIn("Do not describe it as exact TRIBE v2", claims)
        self.assertIn("real video/audio/text encoders", claims)

    def test_tribe_style_does_not_become_required_dependency(self):
        forbidden = (
            "tribev2",
            "neuralset",
            "neuraltrain",
            "huggingface_hub",
            "moviepy",
            "x_transformers",
            "gtts",
            "langdetect",
            "spacy",
            "julius",
            "levenshtein",
        )
        checked = {
            "pyproject.toml": Path("pyproject.toml").read_text(encoding="utf-8").lower(),
            "environment-a100.yml": Path("environment-a100.yml").read_text(encoding="utf-8").lower(),
            "requirements/cluster-a100.txt": Path("requirements/cluster-a100.txt").read_text(encoding="utf-8").lower(),
        }

        for path, text in checked.items():
            for package in forbidden:
                with self.subTest(path=path, package=package):
                    self.assertNotIn(package, text)

    def test_a100_runbook_separates_fast_and_heavy_lanes(self):
        runbook = Path("docs/A100_RUNBOOK.md").read_text(encoding="utf-8")

        self.assertIn("Fast Iteration Lane", runbook)
        self.assertIn("Heavy 6-GPU Lane", runbook)
        self.assertIn("--ntasks-per-node=6 --gres=gpu:a100:6", runbook)
        self.assertIn("MOABB EEG is expected to skip `tribe_style`", runbook)
        self.assertIn("Do not retry failed multi-GPU runs blindly", runbook)

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
        finalizer = Path("src/neurotwin/reports/finalize.py").read_text(encoding="utf-8")
        eval_script = Path("scripts/slurm/eval_a100.sh").read_text(encoding="utf-8")

        self.assertIn("Refusing to run the generic placeholder config", train)
        self.assertIn("_train_a100_inner.sh", train)
        self.assertIn("#SBATCH --ntasks-per-node=6", train)
        self.assertIn("#SBATCH --gres=gpu:a100:6", train)
        self.assertIn("cluster preflight", inner)
        self.assertLess(inner.index("cluster preflight"), inner.index("torchrun"))
        self.assertIn("--require-cuda", inner)
        self.assertIn("--require-prepared-windows", inner)
        for required in (
            "run finalize",
            "--paper-mode-dir",
            "--event-manifest",
            "--split-manifest",
            "--seeds 0 1 2",
        ):
            self.assertIn(required, inner)
        for required in (
            "prepared_baseline_suite.json",
            "seed_aggregate.json",
            "seed_aggregate.csv",
            "baseline_failures.json",
            "paper_mode_gate.json",
            "run_leakage_demo",
            "run_identity_probe",
            "write_final_prepared_evidence_gate",
            "generate_model_card_report",
            "copy_prepared_eval_audit",
            "paper_mode_artifacts_unavailable",
        ):
            self.assertIn(required, finalizer)
        self.assertIn("Refusing to run default/synthetic eval", eval_script)
        self.assertNotIn("python -m neurotwin.cli eval --suite", eval_script)

    def test_docker_6gpu_runner_contains_required_sequence(self):
        launcher = Path("scripts/run_docker_6gpu.sh").read_text(encoding="utf-8")
        inner = Path("scripts/docker_a100_inner.sh").read_text(encoding="utf-8")
        finalizer = Path("src/neurotwin/reports/finalize.py").read_text(encoding="utf-8")
        preflight = Path("scripts/docker_gpu_preflight.py").read_text(encoding="utf-8")

        for required in (
            "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
            '--gpus "\\"device=${HOST_GPU_IDS}\\""',
            "--ipc=host",
            "--shm-size=64g",
            "--ulimit memlock=-1",
            "--ulimit stack=67108864",
            "-v \"$REPO_ROOT\":/workspace/repo",
            "/raid/scratch/$USER/neurotwin-<short_sha>",
            "HOST_GPU_IDS=${HOST_GPU_IDS:-${2:-0,1,2,3,4,5}}",
            "GPU_COUNT=${GPU_COUNT:-6}",
            "NPROC_PER_NODE=${NPROC_PER_NODE:-$GPU_COUNT}",
            "CONTAINER_CUDA_VISIBLE_DEVICES=0",
            "DOCKER_RUN_ID=${DOCKER_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}",
            "DOCKER_LOG_PATH=${DOCKER_LOG_PATH:-\"$RUN_LOG_DIR/neurotwin-a100-docker-$DOCKER_RUN_ID.log\"}",
            "A100_RUN_PAPER_MODE_IN_FULL=${A100_RUN_PAPER_MODE_IN_FULL:-0}",
            "docker_run.env",
            "tee -a \"$DOCKER_LOG_PATH\"",
            "-e CUDA_VISIBLE_DEVICES=\"$CONTAINER_CUDA_VISIBLE_DEVICES\"",
            "-e NCCL_DEBUG=\"${NCCL_DEBUG:-INFO}\"",
            "-e DOCKER_LOG_PATH=\"$DOCKER_LOG_PATH\"",
            "-e A100_RUN_PAPER_MODE_IN_FULL=\"$A100_RUN_PAPER_MODE_IN_FULL\"",
            "bash scripts/docker_a100_inner.sh",
        ):
            self.assertIn(required, launcher)
        self.assertNotIn("python - <<", launcher)
        self.assertNotIn('if [[ -n "${CUDA_VISIBLE_DEVICES:-}"', launcher)
        self.assertNotIn("CONTAINER_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES", launcher)
        for required in (
            "python -m pip install -e \".[moabb,cluster]\"",
            "python scripts/docker_gpu_preflight.py \"$PERSISTENT_ROOT/gpu_preflight.json\"",
            "bash scripts/run_smoke.sh outputs/smoke",
            "bash scripts/prepare_moabb_benchmark.sh",
            "python -m neurotwin.cli eval audit",
            "python -m neurotwin.cli cluster materialize-config",
            "python -m neurotwin.cli cluster preflight",
            "--require-cuda",
            "torchrun --standalone --nproc_per_node=\"$NPROC_PER_NODE\"",
            "python -m neurotwin.cli run finalize",
            "--paper-mode-dir",
            "--event-manifest",
            "--split-manifest",
            "--seeds 0 1 2",
            "A100_RUN_PAPER_MODE_IN_FULL",
            "paper_mode_artifacts_unavailable",
            "bash scripts/package_a100_evidence_bundle.sh",
        ):
            self.assertIn(required, inner)
        for required in (
            "prepared_baseline_suite.json",
            "seed_aggregate.json",
            "seed_aggregate.csv",
            "baseline_failures.json",
            "paper_mode_gate.json",
            "run_leakage_demo",
            "run_identity_probe",
            "write_final_prepared_evidence_gate",
            "generate_model_card_report",
            "copy_prepared_eval_audit",
        ):
            self.assertIn(required, finalizer)
        self.assertNotIn('A100_REQUIRE_PAPER_MODE_GATE=1', inner)
        for required in (
            "torch.cuda.device_count()",
            "expected_gpu_count",
            "visible_gpu_count",
            "docker_image",
            "HOST_GPU_IDS",
            "CUDA_VISIBLE_DEVICES",
            "NPROC_PER_NODE",
            "Expected exactly",
        ):
            self.assertIn(required, preflight)

    def test_agent_deploy_docs_and_dockerfile_are_6gpu_first(self):
        doc = Path("README_AGENT_DEPLOY.md").read_text(encoding="utf-8")
        dockerfile = Path("Dockerfile.a100").read_text(encoding="utf-8")

        for required in (
            "automated deployment agent",
            'docker run --rm --gpus "\\"device=${HOST_GPU_IDS}\\""',
            "device_count",
            "not exactly `6`, stop",
            "HOST_GPU_IDS=0,1,2,3,4,5",
            "CONTAINER_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5",
            "GPU_COUNT=6",
            "NPROC_PER_NODE=6",
            "--shm-size=64g",
            "--ulimit memlock=-1",
            "NCCL_DEBUG=INFO",
            "bash scripts/run_docker_6gpu.sh",
            "torchrun --standalone --nproc_per_node=6",
            "DOCKER_LOG_PATH",
            "run/docker_run.env",
            "current Docker log",
            "One-GPU Diagnostic Only",
            "not the requested 6-GPU handoff run",
            "torch.cuda.set_device(local_rank)",
            "DistributedDataParallel",
            "dependency/runtime image helper",
            "does not hide source code",
        ):
            self.assertIn(required, doc)
        self.assertNotIn("<timestamp>", doc)
        self.assertIn("FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel", dockerfile)
        self.assertIn("not a source-hiding image", dockerfile)
        self.assertIn("COPY src ./src", dockerfile)
        self.assertIn("python -m pip install -e '.[moabb,cluster]'", dockerfile)

    def test_operator_run_bundle_files_are_self_contained(self):
        readme = Path("README_RUN.md").read_text(encoding="utf-8")
        run_full = Path("scripts/run_full.sh").read_text(encoding="utf-8")
        run_full_sbatch = Path("scripts/run_full.sbatch").read_text(encoding="utf-8")
        environment = Path("environment-a100.yml").read_text(encoding="utf-8")

        for required in (
            "Operator Workflow",
            "sha256sum -c SHA256SUMS",
            "Primary Docker 6-GPU Path",
            "README_AGENT_DEPLOY.md",
            "Dockerfile.a100",
            "bash scripts/run_docker_6gpu.sh",
            "scripts/docker_a100_inner.sh",
            "conda env create -f environment-a100.yml",
            "python -m pip install -e '.[moabb,cluster]'",
            "Docker Fallback",
            "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
            '--gpus "\\"device=${HOST_GPU_IDS}\\""',
            '--gpus "\\"device=<host_gpu_id>\\""',
            "--ipc=host --shm-size=64g",
            "HOST_GPU_IDS=0,1,2,3,4,5",
            "GPU_COUNT=6",
            "NPROC_PER_NODE=6",
            "CONTAINER_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5",
            "/raid/scratch/$USER/neurotwin-<short_sha>",
            "Raspberry Pi Handoff Path",
            "Use the Raspberry Pi only as a Chapman-network bridge",
            "neurotwin-a100-runner-<short_sha>.tar.gz",
            "scp neurotwin-a100-runner-<short_sha>.tar.gz",
            "scp /tmp/neurotwin-a100-runner-<short_sha>.tar.gz",
            "tar -xzf ~/neurotwin-a100-runner-<short_sha>.tar.gz",
            "bash scripts/run_smoke.sh",
            "bash scripts/run_full.sh",
            "torchrun --standalone --nproc_per_node=1",
            "bash scripts/package_a100_evidence_bundle.sh",
            "$NEUROTWIN_DATA/gpu_preflight.json",
            "$NEUROTWIN_DATA/docker_run.env",
            "neurotwin-a100-docker-<generated>.log",
            "MOABB task labels are intentionally not persisted",
            "1x A100 80GB",
            "6x A100 80GB",
            "128G",
            "MOABB `BNCI2014_001`",
            "Expected Full Outputs",
            "Known Limitations",
            "Success Condition",
            "Resume And Safe Rerun",
        ):
            self.assertIn(required, readme)
        self.assertNotIn("<timestamp>", readme)
        self.assertNotIn("The expanded Docker host command is", readme)
        self.assertNotIn("docker run --rm -it", readme)
        self.assertNotIn("bash scripts/docker_a100_inner.sh", readme)
        self.assertNotIn("python -m neurotwin.cli eval audit", readme)
        self.assertNotIn("python -m neurotwin.cli cluster materialize-config", readme)
        self.assertNotIn("python -m neurotwin.cli cluster preflight", readme)
        self.assertNotIn("python -m neurotwin.cli report", readme)
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
        self.assertNotIn("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel bash", readme)
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
        for dependency in ("python=3.10", "pytorch-cuda=12.1", "moabb", "mne", "scikit-learn"):
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
