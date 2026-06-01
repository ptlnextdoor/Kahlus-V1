#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess  # nosec B404 - bounded local git invocation with fixed argv shape and timeout.
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "external" / "upstreams.lock.json"
VENDOR_DIR = REPO_ROOT / "external" / "vendor"
GIT_TIMEOUT_SECONDS = 300


def main() -> int:
    parser = argparse.ArgumentParser(description="Vendor pinned upstream research repos.")
    parser.add_argument("--ids", nargs="*", default=None, help="Specific upstream IDs to vendor.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without cloning.")
    parser.add_argument(
        "--include-restricted",
        action="store_true",
        help="Allow mixed/restricted upstreams. Default vendors permissive entries only.",
    )
    args = parser.parse_args()

    lock = json.loads(LOCK_PATH.read_text())
    selected_ids = args.ids or sorted(lock["upstreams"])
    for upstream_id in selected_ids:
        if upstream_id not in lock["upstreams"]:
            raise SystemExit(f"Unknown upstream id: {upstream_id}")
        spec = lock["upstreams"][upstream_id]
        if spec["reuse_status"] != "permissive" and not args.include_restricted:
            print(f"skip {upstream_id}: reuse_status={spec['reuse_status']}")
            continue
        target = VENDOR_DIR / upstream_id
        if args.dry_run:
            print(f"would clone {upstream_id} from {spec['repo']} at {spec['commit']} -> {target}")
            continue
        _clone_or_checkout(upstream_id, spec["repo"], spec["commit"], target)
    return 0


def _clone_or_checkout(upstream_id: str, repo: str, commit: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    git = _git_executable()
    if target.exists():
        _run_git(upstream_id, "fetch", [git, "-C", str(target), "fetch", "--depth", "1", "origin", commit])
    else:
        _run_git(upstream_id, "clone", [git, "clone", "--filter=blob:none", repo, str(target)])
    _run_git(upstream_id, "checkout", [git, "-C", str(target), "checkout", commit])


def _git_executable() -> str:
    git = shutil.which("git")
    if git is None:
        raise SystemExit("git executable not found")
    return git


def _run_git(upstream_id: str, operation: str, command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, timeout=GIT_TIMEOUT_SECONDS)  # nosec B603 - command is built from an absolute git path and pinned lockfile args.
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            f"Timed out while running git {operation} for upstream {upstream_id} after {GIT_TIMEOUT_SECONDS}s"
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
