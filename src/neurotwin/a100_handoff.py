"""Local 7xA100 handoff package builder.

The builder prepares a sendable package but never launches cluster jobs. It
requires a clean git worktree so the handoff can honestly name one exact commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from typing import Iterable


DEFAULT_GPU_COUNT = 7
FORBIDDEN_SUFFIXES = (
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
    ".pyc",
)
FORBIDDEN_PARTS = {".git", ".env", ".ds_store", "__macosx", "__pycache__", "graphify-out", "outputs", "runs"}
FORBIDDEN_MARKERS = ("secret", "password", "passwd", "api_key", "apikey", "ssh_key", "raw_private")

RUNNER_PATHS = (
    "pyproject.toml",
    "environment-a100.yml",
    "requirements/cluster-a100.txt",
    "configs/train/kahlus_multidataset_a100_evidence.yaml",
    "configs/train/moabb_a100.yaml",
    "configs/train/moabb_a100_smoke.yaml",
    "docs/research/kahlus_stf_public_dataset_review.md",
    "scripts/_bootstrap.py",
    "scripts/audit_ktm_a100_evidence.py",
    "scripts/docker_gpu_preflight.py",
    "scripts/fetch_kahlus_public_edf_datasets.py",
    "scripts/fetch_chb_mit_smoke_subset.py",
    "scripts/package_a100_evidence_bundle.py",
    "scripts/prepare_multidataset_a100_evidence.sh",
    "scripts/run_stf_chb_mit_smoke.py",
    "scripts/run_stf_public_data_audit.py",
    "scripts/run_stf_synthetic_smoke.py",
    "scripts/run_researchdock_synthetic.py",
    "scripts/smoke_a100_runner.py",
    "src/neurotwin/__init__.py",
    "src/neurotwin/a100_audit",
    "src/neurotwin/adapters",
    "src/neurotwin/benchmarks",
    "src/neurotwin/cli.py",
    "src/neurotwin/config.py",
    "src/neurotwin/config_types.py",
    "src/neurotwin/contracts",
    "src/neurotwin/data",
    "src/neurotwin/doctor.py",
    "src/neurotwin/eval",
    "src/neurotwin/gates",
    "src/neurotwin/models",
    "src/neurotwin/numerics.py",
    "src/neurotwin/reports",
    "src/neurotwin/repro.py",
    "src/neurotwin/researchdock",
    "src/neurotwin/runtime",
    "src/neurotwin/scoring",
    "src/neurotwin/stf",
    "src/neurotwin/training",
    "src/neurotwin/transition_gym",
    "src/neurotwin/upstreams.py",
    "tests/researchdock",
    "tests/data",
    "tests/stf",
)


class A100HandoffError(RuntimeError):
    """Raised when a handoff package would be misleading or unsafe."""


@dataclass(frozen=True)
class A100HandoffPackage:
    zip_path: Path
    root_name: str
    commit_hash: str
    runner_tarball: str
    manifest: dict[str, object]


def package_kahlus_a100_7x_handoff(repo_root: str | Path, out_dir: str | Path) -> A100HandoffPackage:
    return package_kahlus_a100_handoff(repo_root, out_dir, gpu_count=DEFAULT_GPU_COUNT)


def package_kahlus_a100_handoff(repo_root: str | Path, out_dir: str | Path, *, gpu_count: int = DEFAULT_GPU_COUNT) -> A100HandoffPackage:
    if gpu_count < 1:
        raise A100HandoffError("gpu_count must be at least 1")
    repo = Path(repo_root).resolve()
    out = Path(out_dir).resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain")
    if status.strip():
        raise A100HandoffError("A100 handoff requires a clean worktree before packaging")

    short = commit[:7]
    root_name = f"kahlus-a100-{gpu_count}x-handoff-{short}"
    runner_name = f"kahlus-a100-{gpu_count}x-runner-{short}.tar.gz"
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / f"{root_name}.zip"
    manifest = _manifest(commit, runner_name, gpu_count=gpu_count)

    with tempfile.TemporaryDirectory() as tmp:
        stage_root = Path(tmp) / root_name
        stage_root.mkdir(parents=True)
        runner_tarball = stage_root / runner_name

        _write_runner_tarball(repo, runner_tarball, commit)
        (stage_root / "COMMIT_HASH.txt").write_text(commit + "\n", encoding="utf-8")
        (stage_root / "CLEAN_WORKTREE.txt").write_text(
            f"clean_worktree=true\ncommit={commit}\nstatus_command=git status --porcelain\n",
            encoding="utf-8",
        )
        (stage_root / "A100_HANDOFF_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (stage_root / "README_A100_7X_HANDOFF.md").write_text(_readme(manifest), encoding="utf-8")
        (stage_root / "README_RUN.md").write_text(_readme_run(manifest), encoding="utf-8")
        _write_checksums(stage_root, "SHA256SUMS")
        _assert_safe_tree(stage_root)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(p for p in stage_root.rglob("*") if p.is_file()):
                archive.write(path, f"{root_name}/{path.relative_to(stage_root).as_posix()}")

    return A100HandoffPackage(
        zip_path=zip_path,
        root_name=root_name,
        commit_hash=commit,
        runner_tarball=runner_name,
        manifest=manifest,
    )


def _manifest(commit: str, runner_name: str, *, gpu_count: int) -> dict[str, object]:
    gpu_label = f"{gpu_count}xA100"
    return {
        "package": f"kahlus_a100_{gpu_count}x_handoff",
        "evidence_package": "kahlus-multidataset-a100-evidence",
        "commit_hash": commit,
        "clean_worktree_required": True,
        "expected_gpu_count": gpu_count,
        "gpu_label": gpu_label,
        "runner_tarball": runner_name,
        "cpu_smoke_command": "PYTHONPATH=src python3 -m unittest discover -s tests/researchdock -v",
        "stf_cpu_smoke_command": "PYTHONPATH=src python3 -m unittest discover -s tests/stf -v",
        "stf_synthetic_smoke_command": (
            "PYTHONPATH=src python3 scripts/run_stf_synthetic_smoke.py "
            "--out-dir /tmp/kahlus_stf_smoke --seed 0"
        ),
        "stf_subset_fetch_command": (
            "PYTHONPATH=src python3 scripts/fetch_chb_mit_smoke_subset.py "
            "--dataset chb_mit_physionet --out-root /tmp/kahlus_chbmit_smoke_subset "
            "--patients 2 --records-per-patient 2"
        ),
        "public_edf_smoke_fetch_command": (
            "python3 scripts/fetch_kahlus_public_edf_datasets.py "
            "--out-root /data/kahlus/raw --max-records-per-dataset 16"
        ),
        "public_edf_full_fetch_command": (
            "python3 scripts/fetch_kahlus_public_edf_datasets.py "
            "--out-root /data/kahlus/raw --full"
        ),
        "stf_public_audit_command": (
            "PYTHONPATH=src python3 scripts/run_stf_public_data_audit.py "
            "--dataset chb_mit_physionet --data-root /tmp/kahlus_chbmit_smoke_subset "
            "--out-dir /tmp/kahlus_stf_chbmit_audit"
        ),
        "stf_public_smoke_command": (
            "PYTHONPATH=src python3 scripts/run_stf_chb_mit_smoke.py "
            "--dataset chb_mit_physionet --data-root /tmp/kahlus_chbmit_smoke_subset "
            "--out-dir /tmp/kahlus_stf_chbmit_smoke --max-records 4 "
            "--max-samples-per-record 900000 --max-channels 8"
        ),
        "runner_checksum_command": "shasum -a 256 -c RUNNER_SHA256SUMS",
        "runner_self_smoke_command": "PYTHONPATH=src python3 scripts/smoke_a100_runner.py",
        "gpu_count_verification_command": (
            f"test \"$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')\" = \"{gpu_count}\""
        ),
        "multidataset_prepare_command": (
            "MULTIDATASET_ROOT=/data/kahlus/raw PYTHONPATH=src "
            "bash scripts/prepare_multidataset_a100_evidence.sh /data/kahlus/prepared/kahlus_multidataset_a100_evidence"
        ),
        "multidataset_dry_run_command": (
            "PYTHONPATH=src python3 -m neurotwin.cli train --dry-run "
            "--config /data/kahlus/prepared/kahlus_multidataset_a100_evidence/configs/kahlus_multidataset_a100_evidence.materialized.yaml"
        ),
        "multidataset_ddp_torchrun_command": (
            f"torchrun --standalone --nproc_per_node={gpu_count} -m neurotwin.cli train "
            "--config /data/kahlus/prepared/kahlus_multidataset_a100_evidence/configs/kahlus_multidataset_a100_evidence.materialized.yaml"
        ),
        "ddp_torchrun_command": (
            f"torchrun --standalone --nproc_per_node={gpu_count} -m neurotwin.cli train "
            "--config configs/train/moabb_a100_smoke.yaml"
        ),
        "stf_cluster_public_smoke_command": (
            "PYTHONPATH=src python3 scripts/run_stf_chb_mit_smoke.py "
            "--dataset chb_mit_physionet --data-root ${CHB_MIT_ROOT} "
            "--out-dir ${STF_OUT_DIR:-/tmp/kahlus_stf_chbmit_smoke} "
            "--max-records 4 --max-samples-per-record 900000 --max-channels 8"
        ),
        "evidence_bundle_writer": "python3 scripts/package_a100_evidence_bundle.py",
        "audit_script": "python3 scripts/audit_ktm_a100_evidence.py",
        "raw_data_archive_policy": (
            "Raw EDFs are not included in the code handoff zip. Use the official PhysioNet downloader, "
            "or pass --include-raw-data-root to scripts/package_kahlus_a100_7x_handoff.py to create a separate large raw-data archive."
        ),
        "audit_command": (
            "PYTHONPATH=src python3 scripts/audit_ktm_a100_evidence.py "
            f"--evidence <returned-evidence.zip> --out-dir <audit-out> --expected-gpus {gpu_count}"
        ),
        "claim_boundary": "infrastructure_handoff_only_no_clinical_or_model_superiority_claims",
        "exclusions": [
            "no secrets",
            "no checkpoints unless explicitly allowed",
            "no raw private participant data",
            "no A100 job launched by this package command",
        ],
    }


def _readme(manifest: dict[str, object]) -> str:
    return (
        f"# Kahlus {manifest['gpu_label']} Handoff\n\n"
        f"This package is a local handoff artifact for a {manifest['gpu_label']} cluster. Packaging it does not launch "
        "A100, Slurm, Docker, or torchrun jobs.\n\n"
        f"- commit: {manifest['commit_hash']}\n"
        f"- GPU label: {manifest['gpu_label']}\n"
        f"- expected_gpu_count: {manifest['expected_gpu_count']}\n"
        f"- runner tarball: {manifest['runner_tarball']}\n\n"
        "## CPU Smoke\n\n"
        "```bash\n"
        f"{manifest['cpu_smoke_command']}\n"
        "```\n\n"
        "## STF Local Checks\n\n"
        "```bash\n"
        f"{manifest['stf_cpu_smoke_command']}\n"
        f"{manifest['stf_synthetic_smoke_command']}\n"
        f"{manifest['stf_subset_fetch_command']}\n"
        f"{manifest['stf_public_audit_command']}\n"
        f"{manifest['stf_public_smoke_command']}\n"
        "```\n\n"
        "## Runner Checksum\n\n"
        "After extracting the runner tarball, verify its internal manifest first:\n\n"
        "```bash\n"
        f"{manifest['runner_checksum_command']}\n"
        "```\n\n"
        "## Runner Self-Smoke\n\n"
        "After extracting the runner tarball, verify local package/audit plumbing with:\n\n"
        "```bash\n"
        f"{manifest['runner_self_smoke_command']}\n"
        "```\n\n"
        "## DDP Command\n\n"
        "Verify the allocation is honestly 7 GPUs before using any torchrun command:\n\n"
        "```bash\n"
        f"{manifest['gpu_count_verification_command']}\n"
        "```\n\n"
        "```bash\n"
        f"{manifest['ddp_torchrun_command']}\n"
        "```\n\n"
        "## Multi-Dataset Evidence Package\n\n"
        "Fetch official public EDF roots from PhysioNet outside the repo. For a quick smoke subset:\n\n"
        "```bash\n"
        f"{manifest['public_edf_smoke_fetch_command']}\n"
        "```\n\n"
        "For the full public EDF download:\n\n"
        "```bash\n"
        f"{manifest['public_edf_full_fetch_command']}\n"
        "```\n\n"
        "`MULTIDATASET_ROOT` should contain exactly these reusable subdirectories: "
        "`sleep-edf`, `chb-mit`, `eegmmi`, and `siena`. You may alternatively set `SLEEP_EDF_ROOT`, "
        "`CHB_MIT_ROOT`, `EEGMMI_ROOT`, and `SIENA_ROOT` individually.\n\n"
        "**Do not delete raw dataset roots after a run.** Treat them as a persistent shared cache so future "
        "A100 runs can reuse the same downloads. Delete only derived outputs under the prepared output directory "
        "when you intentionally want to rerun preprocessing.\n\n"
        "The code handoff zip does not include raw EDF data. If a raw bundle is explicitly needed, create a "
        "separate archive with `scripts/package_kahlus_a100_7x_handoff.py --include-raw-data-root <path>`; "
        "full raw EDF bundles may be tens of GB.\n\n"
        "```bash\n"
        f"{manifest['multidataset_prepare_command']}\n"
        f"{manifest['multidataset_dry_run_command']}\n"
        "```\n\n"
        f"The full multi-dataset run uses the same honest {manifest['expected_gpu_count']}-GPU allocation label:\n\n"
        "```bash\n"
        f"{manifest['multidataset_ddp_torchrun_command']}\n"
        "```\n\n"
        "## STF Public Smoke On Cluster\n\n"
        "Set `CHB_MIT_ROOT` to an external CHB-MIT root. Do not place raw EDF files in the repo.\n\n"
        "```bash\n"
        f"{manifest['stf_cluster_public_smoke_command']}\n"
        "```\n\n"
        "## Evidence Return\n\n"
        "After the cluster run, write a small evidence bundle with:\n\n"
        "```bash\n"
        f"{manifest['evidence_bundle_writer']} <persistent-root> <output.zip> <evidence-name> <repo-root> <commit>\n"
        "```\n\n"
        "Audit returned evidence with:\n\n"
        "```bash\n"
        f"{manifest['audit_command']}\n"
        "```\n\n"
        "## Boundaries\n\n"
        "- Infrastructure handoff only.\n"
        "- No clinical, recovery, diagnosis, treatment, or model-superiority claim.\n"
        "- No secrets, checkpoints, raw arrays, or raw private participant data are included.\n"
    )


def _readme_run(manifest: dict[str, object]) -> str:
    return (
        "# README_RUN: Kahlus Multi-Dataset A100 Evidence\n\n"
        "## What This Run Does\n\n"
        "Prepares public EDF EEG datasets into Kahlus event manifests, runs leakage audits and brutal baselines, "
        f"then provides a materialized config for a {manifest['gpu_label']} torchrun training job.\n\n"
        "## Persistent Dataset Rule\n\n"
        "**Never delete the raw dataset directory.** Keep Sleep-EDF, CHB-MIT, EEGMMI, and Siena under a persistent "
        "cluster path such as `/data/kahlus/raw` or `/scratch/kahlus/raw`. Future runs should reuse those files. "
        "Only delete prepared outputs under `/data/kahlus/prepared/...` when intentionally rerunning preprocessing.\n\n"
        "## Setup\n\n"
        "```bash\n"
        "tar -xzf kahlus-a100-7x-runner-*.tar.gz\n"
        "cd runner\n"
        "python3 -m pip install --upgrade pip\n"
        "python3 -m pip install -e '.[moabb,cluster]'\n"
        "```\n\n"
        "Install CUDA PyTorch first if the cluster image does not already provide it.\n\n"
        "## Dataset Layout\n\n"
        "```bash\n"
        "# Official PhysioNet download, smoke subset:\n"
        f"{manifest['public_edf_smoke_fetch_command']}\n"
        "# Or full download:\n"
        f"{manifest['public_edf_full_fetch_command']}\n"
        "export MULTIDATASET_ROOT=/data/kahlus/raw\n"
        "# Expected reusable subdirs:\n"
        "# $MULTIDATASET_ROOT/sleep-edf\n"
        "# $MULTIDATASET_ROOT/chb-mit\n"
        "# $MULTIDATASET_ROOT/eegmmi\n"
        "# $MULTIDATASET_ROOT/siena\n"
        "```\n\n"
        "Official PhysioNet sources are the provenance ground truth. Kaggle mirrors are acceptable only as exact "
        "verified mirrors, never as the source of truth. Do not commit raw EDF files. The code handoff zip does "
        "not include raw EDF data; use a separate raw-data archive only when explicitly requested.\n\n"
        "## Smoke / Prepare Command\n\n"
        "```bash\n"
        "export PREPARED=/data/kahlus/prepared/kahlus_multidataset_a100_evidence\n"
        "MAX_RECORDS_PER_DATASET=16 MAX_WINDOWS_PER_SPLIT=128 \\\n"
        "  MULTIDATASET_ROOT=$MULTIDATASET_ROOT PYTHONPATH=src \\\n"
        "  bash scripts/prepare_multidataset_a100_evidence.sh $PREPARED\n"
        "```\n\n"
        "## Full Prepare Command\n\n"
        "```bash\n"
        "export PREPARED=/data/kahlus/prepared/kahlus_multidataset_a100_evidence_full\n"
        "MAX_RECORDS_PER_DATASET=999999 MAX_WINDOWS_PER_SPLIT=4096 \\\n"
        "  MULTIDATASET_ROOT=$MULTIDATASET_ROOT PYTHONPATH=src \\\n"
        "  bash scripts/prepare_multidataset_a100_evidence.sh $PREPARED\n"
        "```\n\n"
        "## Dry Run\n\n"
        "```bash\n"
        "PYTHONPATH=src python3 -m neurotwin.cli train --dry-run \\\n"
        "  --config $PREPARED/configs/kahlus_multidataset_a100_evidence.materialized.yaml\n"
        "```\n\n"
        f"## Full {manifest['gpu_label']} Run\n\n"
        "```bash\n"
        f"{manifest['gpu_count_verification_command']}\n"
        f"torchrun --standalone --nproc_per_node={manifest['expected_gpu_count']} -m neurotwin.cli train \\\n"
        "  --config $PREPARED/configs/kahlus_multidataset_a100_evidence.materialized.yaml\n"
        "```\n\n"
        "## Expected Outputs\n\n"
        "```text\n"
        "$PREPARED/event_manifest.json\n"
        "$PREPARED/split_manifest.json\n"
        "$PREPARED/multidataset_evidence_gate.json\n"
        "$PREPARED/baseline_suite/prepared_baseline_suite.json\n"
        "$PREPARED/configs/kahlus_multidataset_a100_evidence.materialized.yaml\n"
        "runs/<run-id>/summary.json\n"
        "runs/<run-id>/metrics.json\n"
        "```\n\n"
        "## Success Condition\n\n"
        "Preparation succeeds when `gate_passed=True`, eval audit passes, and the baseline suite writes "
        "`prepared_baseline_suite.json`. Training succeeds when `runs/<run-id>/summary.json` exists and reports "
        "completed prepared training.\n\n"
        "## Resume / Rerun\n\n"
        "Reuse the same raw dataset root. Rerun preprocessing into a new `$PREPARED` directory, or delete only the "
        "old prepared directory. Do not delete `/data/kahlus/raw` or equivalent raw dataset storage.\n\n"
        "## Boundaries\n\n"
        "No clinical seizure prediction, sleep diagnosis, treatment prediction, recovery claim, or brain foundation "
        "model claim is allowed from this handoff alone.\n"
    )


def _write_runner_tarball(repo: Path, tarball: Path, commit: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        runner_root = Path(tmp) / "runner"
        runner_root.mkdir(parents=True)
        (runner_root / "COMMIT_HASH.txt").write_text(commit + "\n", encoding="utf-8")
        (runner_root / "README_A100_7X_RUNNER.md").write_text(
            "Run only after local checks pass and the requested A100 allocation is confirmed.\n",
            encoding="utf-8",
        )
        (runner_root / "README_RUN.md").write_text(
            "Fetch public EDFs with scripts/fetch_kahlus_public_edf_datasets.py from official PhysioNet sources. "
            "Keep raw datasets in a persistent cluster directory. Never delete that raw dataset cache; delete only "
            "prepared outputs when rerunning preprocessing. See the top-level README_RUN.md before launch.\n",
            encoding="utf-8",
        )
        for rel in RUNNER_PATHS:
            source = repo / rel
            if not source.exists():
                raise A100HandoffError(f"required handoff source is missing: {rel}")
            _copy_source(source, runner_root / rel)
        _write_checksums(runner_root, "RUNNER_SHA256SUMS")
        _assert_safe_tree(runner_root)
        with tarfile.open(tarball, "w:gz") as tar:
            for path in sorted(p for p in runner_root.rglob("*") if p.is_file()):
                tar.add(path, arcname=f"runner/{path.relative_to(runner_root).as_posix()}")


def _copy_source(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise A100HandoffError(f"symlink is not allowed in handoff source: {source}")
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            rel = path.relative_to(source)
            if path.is_symlink():
                raise A100HandoffError(f"symlink is not allowed in handoff source: {path}")
            if not path.is_file():
                continue
            if _path_forbidden(rel) or _path_forbidden(path.relative_to(source.parents[0])):
                continue
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        return
    if _path_forbidden(source.name):
        raise A100HandoffError(f"required source has forbidden name: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_checksums(root: Path, filename: str) -> None:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != filename):
        rows.append(f"{sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
    (root / filename).write_text("\n".join(rows) + "\n", encoding="utf-8")


def _assert_safe_tree(root: Path) -> None:
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if path.is_symlink():
            raise A100HandoffError(f"symlink is not allowed in handoff package: {rel}")
        if path.is_file() and _path_forbidden(rel):
            raise A100HandoffError(f"forbidden file path in handoff package: {rel}")


def _path_forbidden(path: str | Path) -> bool:
    rel = Path(path)
    parts = tuple(part for part in rel.parts if part not in ("", "."))
    lowered = [part.lower() for part in parts]
    if any(part in FORBIDDEN_PARTS for part in lowered):
        return True
    if any(part.startswith(".env.") for part in lowered):
        return True
    text = "/".join(lowered)
    if any(marker in text for marker in FORBIDDEN_MARKERS):
        return True
    return rel.name.lower().endswith(FORBIDDEN_SUFFIXES)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True)
    return result.stdout.strip()
