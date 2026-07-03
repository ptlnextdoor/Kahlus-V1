from __future__ import annotations

import json
import os
import shutil
import shlex
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path


def _copy_current_worktree_to_clean_git(tmp: str) -> Path:
    source = Path.cwd()
    target = Path(tmp) / "repo"
    ignore = shutil.ignore_patterns(
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "graphify-out",
        "outputs",
        "runs",
    )
    shutil.copytree(source, target, ignore=ignore)
    subprocess.run(["git", "init", "-q"], cwd=target, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=target, check=True)
    subprocess.run(["git", "config", "user.name", "A100 Test"], cwd=target, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=target, check=True)
    subprocess.run(["git", "add", "."], cwd=target, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "a100 7x package test"], cwd=target, check=True)
    return target


class KahlusA100SevenGpuHandoffTests(unittest.TestCase):
    def test_package_contains_7x_a100_handoff_contract_and_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_current_worktree_to_clean_git(tmp)
            metadata_file = repo / "src" / "neurotwin" / "researchdock" / ".DS_Store"
            metadata_file.write_text("local metadata must not ship\n", encoding="utf-8")
            model_artifact = repo / "src" / "neurotwin" / "researchdock" / "trained_encoder.safetensors"
            model_artifact.write_bytes(b"model weights must not ship\n")
            sklearn_artifact = repo / "src" / "neurotwin" / "benchmarks" / "baseline_cache.joblib"
            sklearn_artifact.write_bytes(b"serialized baseline cache must not ship\n")
            subprocess.run(["git", "add", "-f", str(metadata_file.relative_to(repo))], cwd=repo, check=True)
            subprocess.run(["git", "add", "-f", str(model_artifact.relative_to(repo))], cwd=repo, check=True)
            subprocess.run(["git", "add", "-f", str(sklearn_artifact.relative_to(repo))], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "tracked forbidden artifact fixtures"], cwd=repo, check=True)
            out_dir = Path(tmp) / "out"
            result = subprocess.run(
                [sys.executable, "scripts/package_kahlus_a100_7x_handoff.py", "--out-dir", str(out_dir)],
                cwd=repo,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            zips = sorted(out_dir.glob("kahlus-a100-7x-handoff-*.zip"))
            self.assertEqual(len(zips), 1)

            extract = Path(tmp) / "extract"
            with zipfile.ZipFile(zips[0], "r") as archive:
                names = set(archive.namelist())
                archive.extractall(extract)
            roots = {name.split("/", 1)[0] for name in names if name}
            self.assertEqual(len(roots), 1)
            root_name = roots.pop()
            root = extract / root_name

            for rel in (
                "COMMIT_HASH.txt",
                "CLEAN_WORKTREE.txt",
                "A100_HANDOFF_MANIFEST.json",
                "README_A100_7X_HANDOFF.md",
                "SHA256SUMS",
            ):
                self.assertIn(f"{root_name}/{rel}", names)

            runner_archives = [name for name in names if name.endswith(".tar.gz")]
            self.assertEqual(len(runner_archives), 1)
            manifest = json.loads((root / "A100_HANDOFF_MANIFEST.json").read_text(encoding="utf-8"))
            readme = (root / "README_A100_7X_HANDOFF.md").read_text(encoding="utf-8")
            self.assertEqual(manifest["expected_gpu_count"], 7)
            self.assertEqual(manifest["gpu_label"], "7xA100")
            self.assertIn("--nproc_per_node=7", manifest["ddp_torchrun_command"])
            self.assertNotIn("--require-pass", manifest["ddp_torchrun_command"])
            self.assertIn("nvidia-smi", manifest["gpu_count_verification_command"])
            self.assertIn('"7"', manifest["gpu_count_verification_command"])
            self.assertIn("run_stf_chb_mit_smoke.py", manifest["stf_cluster_public_smoke_command"])
            self.assertIn("${CHB_MIT_ROOT}", manifest["stf_cluster_public_smoke_command"])
            self.assertIn("--expected-gpus 7", manifest["audit_command"])
            self.assertIn("python3 -m unittest discover -s tests/researchdock -v", manifest["cpu_smoke_command"])
            self.assertIn("tests/stf", manifest["stf_cpu_smoke_command"])
            self.assertIn("run_stf_synthetic_smoke.py", manifest["stf_synthetic_smoke_command"])
            self.assertIn("fetch_chb_mit_smoke_subset.py", manifest["stf_subset_fetch_command"])
            self.assertIn("run_stf_public_data_audit.py", manifest["stf_public_audit_command"])
            self.assertIn("run_stf_chb_mit_smoke.py", manifest["stf_public_smoke_command"])
            self.assertEqual(manifest["runner_checksum_command"], "shasum -a 256 -c RUNNER_SHA256SUMS")
            self.assertEqual(
                manifest["runner_self_smoke_command"],
                "PYTHONPATH=src python3 scripts/smoke_a100_runner.py",
            )
            self.assertIn("scripts/package_a100_evidence_bundle.py", manifest["evidence_bundle_writer"])
            self.assertIn("scripts/audit_ktm_a100_evidence.py", manifest["audit_script"])
            self.assertIn("7xA100", readme)
            self.assertIn("--nproc_per_node=7", readme)
            self.assertIn("## Runner Checksum", readme)
            self.assertIn("## STF Local Checks", readme)
            self.assertIn("run_stf_chb_mit_smoke.py", readme)
            self.assertIn("RUNNER_SHA256SUMS", readme)
            self.assertIn("## Runner Self-Smoke", readme)
            self.assertIn("scripts/smoke_a100_runner.py", readme)
            self.assertIn("nvidia-smi", readme)
            self.assertIn("## STF Public Smoke On Cluster", readme)
            self.assertIn("CHB_MIT_ROOT", readme)
            self.assertNotIn("8xA100", readme)
            self.assertNotIn("8x A100", readme)

            clean_proof = (root / "CLEAN_WORKTREE.txt").read_text(encoding="utf-8")
            self.assertIn("clean_worktree=true", clean_proof)
            self.assertIn((root / "COMMIT_HASH.txt").read_text(encoding="utf-8").strip(), clean_proof)

            with tarfile.open(root / Path(runner_archives[0]).name, "r:gz") as tar:
                runner_names = set(tar.getnames())
                runner_extract = Path(tmp) / "runner_extract"
                tar.extractall(runner_extract, filter="data")
            self.assertIn("runner/COMMIT_HASH.txt", runner_names)
            self.assertIn("runner/scripts/_bootstrap.py", runner_names)
            self.assertIn("runner/scripts/smoke_a100_runner.py", runner_names)
            self.assertIn("runner/scripts/fetch_chb_mit_smoke_subset.py", runner_names)
            self.assertIn("runner/scripts/run_stf_chb_mit_smoke.py", runner_names)
            self.assertIn("runner/scripts/run_stf_public_data_audit.py", runner_names)
            self.assertIn("runner/scripts/run_stf_synthetic_smoke.py", runner_names)
            self.assertIn("runner/scripts/package_a100_evidence_bundle.py", runner_names)
            self.assertIn("runner/scripts/audit_ktm_a100_evidence.py", runner_names)
            self.assertIn("runner/configs/train/moabb_a100_smoke.yaml", runner_names)
            self.assertIn("runner/src/neurotwin/cli.py", runner_names)
            self.assertIn("runner/src/neurotwin/stf/chb_mit.py", runner_names)
            self.assertIn("runner/tests/stf/test_stf_chb_mit_smoke.py", runner_names)
            self.assertIn("runner/docs/research/kahlus_stf_public_dataset_review.md", runner_names)
            self.assertIn("runner/src/neurotwin/training/command.py", runner_names)
            runner_checksum = subprocess.run(
                ["shasum", "-a", "256", "-c", "RUNNER_SHA256SUMS"],
                cwd=runner_extract / "runner",
                text=True,
                capture_output=True,
            )
            self.assertEqual(runner_checksum.returncode, 0, runner_checksum.stderr + runner_checksum.stdout)
            cpu_smoke_args = shlex.split(manifest["cpu_smoke_command"])
            cpu_smoke_env = {**os.environ}
            while cpu_smoke_args and "=" in cpu_smoke_args[0] and not cpu_smoke_args[0].startswith("-"):
                key, value = cpu_smoke_args.pop(0).split("=", 1)
                cpu_smoke_env[key] = value
            runner_cpu_smoke = subprocess.run(
                cpu_smoke_args,
                cwd=runner_extract / "runner",
                env=cpu_smoke_env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(runner_cpu_smoke.returncode, 0, runner_cpu_smoke.stderr + runner_cpu_smoke.stdout)
            cli_train_help = subprocess.run(
                [sys.executable, "-m", "neurotwin.cli", "train", "--help"],
                cwd=runner_extract / "runner",
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )
            self.assertEqual(cli_train_help.returncode, 0, cli_train_help.stderr + cli_train_help.stdout)
            self.assertIn("--config", cli_train_help.stdout)
            self.assertIn("--dry-run", cli_train_help.stdout)
            train_dry_run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "train",
                    "--dry-run",
                    "--config",
                    "configs/train/moabb_a100_smoke.yaml",
                ],
                cwd=runner_extract / "runner",
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )
            self.assertEqual(train_dry_run.returncode, 0, train_dry_run.stderr + train_dry_run.stdout)
            runner_self_smoke = subprocess.run(
                [sys.executable, "scripts/smoke_a100_runner.py"],
                cwd=runner_extract / "runner",
                text=True,
                capture_output=True,
            )
            self.assertEqual(runner_self_smoke.returncode, 0, runner_self_smoke.stderr + runner_self_smoke.stdout)
            self.assertIn("runner_self_smoke_passed=true", runner_self_smoke.stdout)
            self.assertIn("a100_jobs_launched=false", runner_self_smoke.stdout)
            audit_help = subprocess.run(
                [sys.executable, "scripts/audit_ktm_a100_evidence.py", "--help"],
                cwd=runner_extract / "runner",
                text=True,
                capture_output=True,
            )
            self.assertEqual(audit_help.returncode, 0, audit_help.stderr + audit_help.stdout)
            self.assertIn("--expected-gpus", audit_help.stdout)
            self.assertIn("expected visible GPU count", audit_help.stdout)
            evidence_root = Path(tmp) / "persistent"
            (evidence_root / "runs" / "moabb_a100_smoke").mkdir(parents=True)
            evidence_zip = Path(tmp) / "runner-evidence.zip"
            evidence_bundle = subprocess.run(
                [
                    sys.executable,
                    "scripts/package_a100_evidence_bundle.py",
                    str(evidence_root),
                    str(evidence_zip),
                    "runner-self-smoke-evidence",
                    str(runner_extract / "runner"),
                    manifest["commit_hash"],
                ],
                cwd=runner_extract / "runner",
                text=True,
                capture_output=True,
            )
            self.assertEqual(evidence_bundle.returncode, 0, evidence_bundle.stderr + evidence_bundle.stdout)
            self.assertTrue(evidence_zip.exists())
            with zipfile.ZipFile(evidence_zip, "r") as archive:
                evidence_names = set(archive.namelist())
                evidence_readme = archive.read("runner-self-smoke-evidence/README_HANDOFF.md").decode("utf-8")
            self.assertIn("runner-self-smoke-evidence/README_HANDOFF.md", evidence_names)
            self.assertIn("runner-self-smoke-evidence/README_SEND_TO_FRIEND.md", evidence_names)
            self.assertIn("runner-self-smoke-evidence/handoff-SHA256SUMS", evidence_names)
            self.assertIn("not a scientific result", evidence_readme)
            self.assertIn("clinical claim", evidence_readme)
            self.assertNotIn("8xA100", evidence_readme)
            self.assertNotIn("6-GPU", evidence_readme)
            audit_out = Path(tmp) / "runner-audit"
            evidence_audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_ktm_a100_evidence.py",
                    "--evidence",
                    str(evidence_zip),
                    "--out-dir",
                    str(audit_out),
                    "--expected-gpus",
                    "7",
                    "--allow-missing-logs",
                ],
                cwd=runner_extract / "runner",
                text=True,
                capture_output=True,
            )
            self.assertEqual(evidence_audit.returncode, 2, evidence_audit.stderr + evidence_audit.stdout)
            self.assertIn("verdict=fail", evidence_audit.stdout)
            audit_json = audit_out / "a100_evidence_audit.json"
            audit_md = audit_out / "a100_evidence_report.md"
            self.assertTrue(audit_json.exists())
            self.assertTrue(audit_md.exists())
            audit_payload = json.loads(audit_json.read_text(encoding="utf-8"))
            self.assertEqual(audit_payload["verdict"], "fail")
            self.assertIn("required_file_missing", {item["code"] for item in audit_payload["findings"]})
            for name in names | runner_names:
                self.assertFalse(
                    name.endswith(
                        (
                            ".pt",
                            ".pth",
                            ".ckpt",
                            ".npy",
                            ".npz",
                            ".safetensors",
                            ".onnx",
                            ".pkl",
                            ".pickle",
                            ".joblib",
                            ".h5",
                            ".hdf5",
                            ".pb",
                            ".tflite",
                            ".pem",
                            ".key",
                        )
                    ),
                    name,
                )
                self.assertNotIn(".DS_Store", Path(name).parts, name)
                self.assertNotIn(".env", Path(name).parts, name)
                self.assertNotIn("raw_private", name.lower())

            checksum = subprocess.run(
                ["shasum", "-a", "256", "-c", "SHA256SUMS"],
                cwd=root,
                text=True,
                capture_output=True,
            )
            self.assertEqual(checksum.returncode, 0, checksum.stderr + checksum.stdout)

    def test_package_refuses_dirty_worktree(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_current_worktree_to_clean_git(tmp)
            (repo / "dirty.txt").write_text("untracked\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/package_kahlus_a100_7x_handoff.py",
                    "--out-dir",
                    str(Path(tmp) / "out"),
                ],
                cwd=repo,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("clean worktree", result.stderr + result.stdout)

    def test_package_refuses_tracked_symlink_in_runner_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_current_worktree_to_clean_git(tmp)
            outside = Path(tmp) / "outside-runner-source.txt"
            outside.write_text("outside content must not be dereferenced into runner\n", encoding="utf-8")
            link = repo / "src" / "neurotwin" / "researchdock" / "linked_outside_source.py"
            link.symlink_to(outside)
            subprocess.run(["git", "add", "-f", str(link.relative_to(repo))], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "tracked symlink fixture"], cwd=repo, check=True)

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/package_kahlus_a100_7x_handoff.py",
                    "--out-dir",
                    str(Path(tmp) / "out"),
                ],
                cwd=repo,
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("symlink", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
