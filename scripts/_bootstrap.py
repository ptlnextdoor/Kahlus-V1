from __future__ import annotations

from pathlib import Path
import sys


def ensure_src_import_path(anchor: str | Path) -> Path:
    """Allow direct script execution without mutating imports at module load."""
    repo_root = Path(anchor).resolve().parents[1]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return repo_root


def ensure_scripts_import_path(anchor: str | Path) -> Path:
    """Allow lazy sibling script imports from module-based test loaders."""
    scripts_root = Path(anchor).resolve().parent
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    return scripts_root
