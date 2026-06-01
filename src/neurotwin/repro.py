from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import random
import shlex
import subprocess
import tempfile
from typing import Any

import numpy as np
import torch
import yaml


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def manifest_hash(records: Any) -> str:
    return stable_hash(records)


def hash_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit(repo_root: str | Path = ".") -> str | None:
    return resolve_source_commit(repo_root)["commit"]


def resolve_source_commit(repo_root: str | Path = ".") -> dict[str, Any]:
    """Resolve a source commit from git, falling back to COMMIT_HASH.txt."""

    root = Path(repo_root)
    if (root / ".git").exists():
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                text=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError):
            result = None
        if result is not None:
            commit = result.stdout.strip()
            if commit:
                return {"commit": commit, "source": "git", "source_commit_missing": False}

    fallback = _read_commit_hash_file(root)
    return {
        "commit": fallback,
        "source": "COMMIT_HASH.txt" if fallback else None,
        "source_commit_missing": True,
    }


def _read_commit_hash_file(repo_root: Path) -> str | None:
    candidates = [repo_root / "COMMIT_HASH.txt"]
    cwd_candidate = Path.cwd() / "COMMIT_HASH.txt"
    if cwd_candidate not in candidates:
        candidates.append(cwd_candidate)
    parent_candidate = repo_root.parent / "COMMIT_HASH.txt"
    if parent_candidate not in candidates:
        candidates.append(parent_candidate)
    for path in candidates:
        try:
            commit = path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
        except (OSError, IndexError):
            continue
        if commit:
            return commit
    return None


def capture_run_metadata(argv: list[str] | tuple[str, ...] | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    environ = os.environ if env is None else env
    captured_argv = list(os.sys.argv if argv is None else argv)
    sanitized_argv = _sanitize_argv(captured_argv)
    slurm = {
        "job_id": environ.get("SLURM_JOB_ID"),
        "job_name": environ.get("SLURM_JOB_NAME"),
        "node_id": environ.get("SLURM_NODEID"),
        "node_list": environ.get("SLURM_JOB_NODELIST") or environ.get("SLURM_NODELIST"),
        "proc_id": environ.get("SLURM_PROCID"),
        "local_id": environ.get("SLURM_LOCALID"),
    }
    container = {
        "detected": _container_detected(environ),
        "type": _container_type(environ),
        "name": (
            environ.get("APPTAINER_NAME")
            or environ.get("SINGULARITY_NAME")
            or environ.get("CONTAINER_NAME")
            or environ.get("HOSTNAME")
        ),
        "docker_image": environ.get("DOCKER_IMAGE"),
    }
    distributed = {
        "local_rank": environ.get("LOCAL_RANK"),
        "rank": environ.get("RANK"),
        "world_size": environ.get("WORLD_SIZE"),
        "cuda_visible_devices": environ.get("CUDA_VISIBLE_DEVICES"),
        "nccl_debug": environ.get("NCCL_DEBUG"),
    }
    if slurm["job_id"]:
        mode = "slurm"
    elif container["detected"]:
        mode = "container"
    else:
        mode = "direct"
    return {
        "mode": mode,
        "argv": sanitized_argv,
        "command": " ".join(shlex.quote(part) for part in sanitized_argv),
        "slurm": slurm,
        "container": container,
        "distributed": distributed,
    }


def _sanitize_argv(argv: list[str]) -> list[str]:
    sanitized: list[str] = []
    redact_next = False
    secret_markers = ("password", "passwd", "secret", "token", "api-key", "apikey", "access-key", "private-key")
    for arg in argv:
        lowered = arg.lower()
        normalized = lowered.replace("_", "-")
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        if normalized.startswith("--") and any(marker in normalized for marker in secret_markers):
            if "=" in arg:
                key, _value = arg.split("=", 1)
                sanitized.append(f"{key}=<redacted>")
            else:
                sanitized.append(arg)
                redact_next = True
            continue
        sanitized.append(arg)
    return sanitized


def _container_detected(env: dict[str, str]) -> bool:
    if any(env.get(key) for key in ("APPTAINER_CONTAINER", "SINGULARITY_CONTAINER", "CONTAINER", "container")):
        return True
    return env is os.environ and Path("/.dockerenv").exists()


def _container_type(env: dict[str, str]) -> str | None:
    if env.get("APPTAINER_CONTAINER") or env.get("APPTAINER_NAME"):
        return "apptainer"
    if env.get("SINGULARITY_CONTAINER") or env.get("SINGULARITY_NAME"):
        return "singularity"
    if env is os.environ and Path("/.dockerenv").exists():
        return "docker"
    if env.get("CONTAINER") or env.get("container"):
        return env.get("CONTAINER") or env.get("container")
    return None


def cuda_metadata() -> dict[str, Any]:
    available = torch.cuda.is_available()
    count = torch.cuda.device_count()
    names: list[str] = []
    if available:
        for index in range(count):
            try:
                names.append(torch.cuda.get_device_name(index))
            except (AssertionError, RuntimeError):
                names.append("")
    current_device_name = None
    if available and count:
        try:
            current_device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        except (AssertionError, RuntimeError):
            current_device_name = names[0] if names else None
    return {
        "available": available,
        "device_count": count,
        "device_names": names,
        "device_name": current_device_name,
        "version": torch.version.cuda,
        "nccl_version": _nccl_version(),
    }


def _nccl_version() -> str | None:
    try:
        version = torch.cuda.nccl.version()  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError):
        return None
    return ".".join(str(part) for part in version) if isinstance(version, tuple) else str(version)


def capture_environment(
    repo_root: str | Path = ".",
    argv: list[str] | tuple[str, ...] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    source_commit = resolve_source_commit(repo_root)
    cuda = cuda_metadata()
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": os.sys.executable,
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": cuda["available"],
            "cuda_device_count": cuda["device_count"],
            "cuda_device_name": cuda["device_name"],
            "cuda_device_names": cuda["device_names"],
            "cuda_version": cuda["version"],
            "torch_cuda_version": cuda["version"],
            "nccl_version": cuda["nccl_version"],
        },
        "git": {
            "commit": source_commit["commit"],
            "source": source_commit["source"],
            "source_commit_missing": source_commit["source_commit_missing"],
        },
        "source_commit_missing": source_commit["source_commit_missing"],
        "run": capture_run_metadata(argv=argv, env=env),
    }


def checkpoint_manifest(run_dir: str | Path) -> list[dict[str, Any]]:
    run_path = Path(run_dir)
    if not run_path.exists():
        return []
    manifest: list[dict[str, Any]] = []
    for path in sorted(run_path.glob("checkpoint*.pt"), key=lambda item: item.name):
        if not path.is_file():
            continue
        manifest.append(
            {
                "filename": path.name,
                "path": path.relative_to(run_path).as_posix(),
                "size": path.stat().st_size,
                "sha256": hash_file(path),
            }
        )
    return manifest


def create_run_dir(root: str | Path = "runs", run_id: str | None = None) -> Path:
    root_path = Path(root)
    is_default_root = str(root_path) in {"runs", "./runs"}
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root_path / run_id
    try:
        _ensure_run_dir_writable(run_dir)
        return run_dir
    except OSError:
        if not is_default_root:
            raise
        fallback_root = Path(tempfile.gettempdir()) / "neurotwin-runs"
        fallback = fallback_root / run_id
        _ensure_run_dir_writable(fallback)
        return fallback


def _ensure_run_dir_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / f".write_probe_{os.getpid()}"
    probe.write_text("ok", encoding="utf-8")
    try:
        probe.unlink()
    except FileNotFoundError:
        pass


def snapshot_config(config: dict[str, Any], run_dir: str | Path) -> Path:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    path = run_path / "config.yaml"
    path.write_text(yaml.safe_dump(_jsonable(config), sort_keys=True), encoding="utf-8")
    return path


def write_json(path: str | Path, payload: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def append_jsonl(path: str | Path, payload: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")
    return out


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
