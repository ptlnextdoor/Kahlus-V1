from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import random
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
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def capture_environment(repo_root: str | Path = ".") -> dict[str, Any]:
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": os.sys.executable,
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
        },
        "git": {
            "commit": get_git_commit(repo_root),
        },
    }


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
