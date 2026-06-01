#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import re
import shutil
import sys
import tempfile
import zipfile

try:
    from render_a100_handoff_readme import render_handoff_readme
except ModuleNotFoundError:
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from render_a100_handoff_readme import render_handoff_readme


RUN_FILES = (
    "summary.json",
    "metrics.json",
    "metrics.csv",
    "metrics.jsonl",
    "config.yaml",
    "environment.json",
    "split_manifest.json",
)
PREPARED_FILES = (
    "eval_audit.json",
    "data_manifest.json",
    "event_manifest.json",
    "split_manifest.json",
    "leakage_report.json",
)
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
DOCKER_LOG_PATTERN = re.compile(r"^neurotwin-a100-docker-[A-Za-z0-9_.:-]+\.log$")


@dataclass(frozen=True)
class EvidenceBundleConfig:
    persistent_root: Path
    zip_path: Path
    evidence_name: str
    repo_root: Path
    full_sha: str

    @property
    def run_dir(self) -> Path:
        return self.persistent_root / "runs" / "moabb_a100_smoke"

    @property
    def prepared_dir(self) -> Path:
        return self.persistent_root / "prepared" / "moabb_benchmark"

    @property
    def logs_dir(self) -> Path:
        return self.persistent_root / "logs"


def is_forbidden(path: Path) -> bool:
    name = path.name
    lower = name.lower()
    if name == "pw.txt" or lower == ".env" or lower.startswith(".env."):
        return True
    if lower.endswith((".pem", ".key", ".pt", ".pth", ".ckpt", ".npy", ".npz", ".tar.gz", ".zip")):
        return True
    secret_markers = ("password", "passwd", "secret", "api_key", "apikey", "ssh_key", "wandb")
    return any(marker in lower for marker in secret_markers)


def copy_file(source: Path, destination: Path) -> bool:
    if not source.is_file() or is_forbidden(source):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def copy_tree_files(source_root: Path, destination_root: Path) -> None:
    if not source_root.is_dir():
        return
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        rel = source.relative_to(source_root)
        if any(part.startswith(".") and part != ".gitkeep" for part in rel.parts):
            continue
        copy_file(source, destination_root / rel)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or is_forbidden(path):
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_env_file(path: Path) -> dict[str, str]:
    if not path.is_file() or is_forbidden(path):
        return {}
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            values[key] = value.strip()
    return values


def find_nested_string(payload: dict[str, Any], path: tuple[str, ...]) -> str | None:
    value: object = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
        return str(value).strip()
    return None


def is_safe_job_id(value: str) -> bool:
    return bool(JOB_ID_PATTERN.fullmatch(value))


def current_slurm_job_id(run_root: Path) -> str | None:
    candidate_paths = (
        ("run", "slurm", "job_id"),
        ("slurm", "job_id"),
        ("slurm_job_id",),
        ("job_id",),
    )
    for filename in ("environment.json", "summary.json"):
        payload = load_json(run_root / filename)
        for path in candidate_paths:
            job_id = find_nested_string(payload, path)
            if job_id:
                return job_id if is_safe_job_id(job_id) else None
    return None


def safe_child(root: Path, filename: str) -> Path | None:
    root_resolved = root.resolve()
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        return None
    return candidate


def safe_resolved_path(root: Path, candidate: Path) -> Path | None:
    root_resolved = root.resolve()
    try:
        candidate_resolved = candidate.resolve()
        candidate_resolved.relative_to(root_resolved)
    except (OSError, ValueError):
        return None
    return candidate_resolved


def copy_current_run_logs(source_root: Path, destination_root: Path, job_id: str | None) -> None:
    if not job_id or not is_safe_job_id(job_id) or not source_root.is_dir():
        return
    for suffix in (".out", ".err"):
        filename = f"neurotwin-a100-full-{job_id}{suffix}"
        source = safe_child(source_root, filename)
        destination = safe_child(destination_root, filename)
        if source is not None and destination is not None:
            copy_file(source, destination)


def current_docker_log_path(root: Path, run_root: Path) -> Path | None:
    docker_env = load_env_file(root / "docker_run.env")
    candidates = [
        docker_env.get("DOCKER_LOG_PATH"),
        find_nested_string(load_json(run_root / "environment.json"), ("run", "container", "docker_log_path")),
        find_nested_string(load_json(run_root / "summary.json"), ("run", "container", "docker_log_path")),
    ]
    for raw_path in candidates:
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.is_absolute():
            return path
    return None


def copy_current_docker_log(source_root: Path, destination_root: Path, log_path: Path | None) -> None:
    if log_path is None or not source_root.is_dir():
        return
    source = safe_resolved_path(source_root, log_path)
    if source is None or not source.is_file() or not DOCKER_LOG_PATTERN.fullmatch(source.name):
        return
    destination = safe_child(destination_root, source.name)
    if destination is not None:
        copy_file(source, destination)


def write_readmes(config: EvidenceBundleConfig, root: Path) -> None:
    (root / "COMMIT_HASH.txt").write_text(config.full_sha + "\n", encoding="utf-8")
    handoff_source = config.repo_root / "README_HANDOFF.md"
    if handoff_source.is_file() and not is_forbidden(handoff_source):
        shutil.copy2(handoff_source, root / "README_HANDOFF.md")
    else:
        short_sha = config.full_sha[:7]
        render_handoff_readme(
            config.repo_root / "README_HANDOFF.md.in",
            root / "README_HANDOFF.md",
            full_sha=config.full_sha,
            short_sha=short_sha,
            runner_name=f"neurotwin-a100-runner-{short_sha}",
            persistent_root_example=f"/raid/scratch/$USER/neurotwin-{short_sha}",
        )
    (root / "README_SEND_TO_FRIEND.md").write_text(
        "# Sendable NeuroTwin A100 Evidence\n\n"
        "Send this zip back after an A100 run. It includes small review artifacts only: summaries, metrics, "
        "tables, figures, prepared manifests/audits, Docker GPU preflight proof, Docker run metadata, "
        "current-run logs, the source commit, and checksums.\n\n"
        "It intentionally excludes checkpoints, raw prepared arrays, runner tarballs, zip artifacts, passwords, "
        "API keys, SSH keys, `.env*` files, and other secret-looking files. Keep large checkpoints on the cluster "
        "unless they are requested explicitly.\n\n"
        "Verify the bundle after extraction with:\n\n"
        "```bash\nshasum -a 256 -c handoff-SHA256SUMS\n```\n",
        encoding="utf-8",
    )


def write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "handoff-SHA256SUMS"):
        rel = path.relative_to(root).as_posix()
        rows.append(f"{sha256(path.read_bytes()).hexdigest()}  {rel}")
    (root / "handoff-SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def package_evidence_bundle(config: EvidenceBundleConfig) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        stage_root = Path(tmp) / config.evidence_name
        stage_root.mkdir(parents=True)

        for rel in RUN_FILES:
            copy_file(config.run_dir / rel, stage_root / "run" / rel)
        copy_file(config.persistent_root / "gpu_preflight.json", stage_root / "run" / "gpu_preflight.json")
        copy_file(config.persistent_root / "docker_run.env", stage_root / "run" / "docker_run.env")
        copy_tree_files(config.run_dir / "tables", stage_root / "run" / "tables")
        copy_tree_files(config.run_dir / "figures", stage_root / "run" / "figures")
        for rel in PREPARED_FILES:
            copy_file(config.prepared_dir / rel, stage_root / "prepared" / rel)
        copy_current_run_logs(config.logs_dir, stage_root / "logs", current_slurm_job_id(config.run_dir))
        copy_current_docker_log(config.logs_dir, stage_root / "logs", current_docker_log_path(config.persistent_root, config.run_dir))
        write_readmes(config, stage_root)
        write_checksums(stage_root)

        for included in stage_root.rglob("*"):
            if included.is_file() and is_forbidden(included):
                raise SystemExit(f"forbidden file staged for evidence bundle: {included.relative_to(stage_root)}")

        config.zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(config.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(p for p in stage_root.rglob("*") if p.is_file()):
                archive.write(path, f"{config.evidence_name}/{path.relative_to(stage_root).as_posix()}")
    return config.zip_path


def _usage() -> str:
    return (
        "usage: package_a100_evidence_bundle.py "
        "/shared/persistent/neurotwin output.zip evidence-name repo-root full-sha"
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 5:
        print(_usage(), file=sys.stderr)
        return 2
    persistent_root, zip_path, evidence_name, repo_root, full_sha = args
    config = EvidenceBundleConfig(
        persistent_root=Path(persistent_root).resolve(),
        zip_path=Path(zip_path).resolve(),
        evidence_name=evidence_name,
        repo_root=Path(repo_root).resolve(),
        full_sha=full_sha,
    )
    package_evidence_bundle(config)
    print(f"evidence_zip={config.zip_path}")
    print(f"evidence_zip_sha256={sha256(config.zip_path.read_bytes()).hexdigest()}")
    print(f"commit={config.full_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
