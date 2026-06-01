#!/usr/bin/env bash
set -euo pipefail
export COPYFILE_DISABLE=1

usage() {
  echo "usage: scripts/package_a100_evidence_bundle.sh /shared/persistent/neurotwin [output-dir]" >&2
}

if (($# < 1 || $# > 2)); then
  usage
  exit 2
fi

PERSISTENT_ROOT=$1
OUT_DIR=${2:-outputs}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -d "$PERSISTENT_ROOT" ]]; then
  echo "Persistent root does not exist: $PERSISTENT_ROOT" >&2
  exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to package the evidence bundle." >&2
  exit 2
fi

if [[ -f "$REPO_ROOT/COMMIT_HASH.txt" ]]; then
  FULL_SHA="$(head -n 1 "$REPO_ROOT/COMMIT_HASH.txt" | tr -d '[:space:]')"
else
  FULL_SHA="$(cd "$REPO_ROOT" && git rev-parse HEAD)"
fi
SHORT_SHA="${FULL_SHA:0:7}"
EVIDENCE_NAME="neurotwin-a100-results-$SHORT_SHA-evidence"
EVIDENCE_ZIP="$OUT_DIR/$EVIDENCE_NAME.zip"

mkdir -p "$OUT_DIR"

python3 - "$PERSISTENT_ROOT" "$EVIDENCE_ZIP" "$EVIDENCE_NAME" "$REPO_ROOT" "$FULL_SHA" <<'PY'
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import re
import shutil
import sys
import tempfile
import zipfile


persistent_root = Path(sys.argv[1]).resolve()
zip_path = Path(sys.argv[2]).resolve()
evidence_name = sys.argv[3]
repo_root = Path(sys.argv[4]).resolve()
full_sha = sys.argv[5]

run_dir = persistent_root / "runs" / "moabb_a100_smoke"
prepared_dir = persistent_root / "prepared" / "moabb_benchmark"
logs_dir = persistent_root / "logs"

run_files = (
    "summary.json",
    "metrics.json",
    "metrics.csv",
    "metrics.jsonl",
    "config.yaml",
    "environment.json",
    "split_manifest.json",
)
prepared_files = (
    "eval_audit.json",
    "data_manifest.json",
    "event_manifest.json",
    "split_manifest.json",
    "leakage_report.json",
)
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _is_forbidden(path: Path) -> bool:
    name = path.name
    lower = name.lower()
    if name == "pw.txt" or lower == ".env" or lower.startswith(".env."):
        return True
    if lower.endswith((".pem", ".key", ".pt", ".pth", ".ckpt", ".npy", ".npz", ".tar.gz", ".zip")):
        return True
    secret_markers = ("password", "passwd", "secret", "api_key", "apikey", "ssh_key", "wandb")
    return any(marker in lower for marker in secret_markers)


def _copy_file(source: Path, destination: Path) -> bool:
    if not source.is_file() or _is_forbidden(source):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _copy_tree_files(source_root: Path, destination_root: Path) -> None:
    if not source_root.is_dir():
        return
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        rel = source.relative_to(source_root)
        if any(part.startswith(".") and part != ".gitkeep" for part in rel.parts):
            continue
        _copy_file(source, destination_root / rel)


def _load_json(path: Path) -> dict:
    if not path.is_file() or _is_forbidden(path):
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _find_nested_string(payload: dict, path: tuple[str, ...]) -> str | None:
    value: object = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
        return str(value).strip()
    return None


def _is_safe_job_id(value: str) -> bool:
    return bool(JOB_ID_PATTERN.fullmatch(value))


def _current_slurm_job_id(run_root: Path) -> str | None:
    candidate_paths = (
        ("run", "slurm", "job_id"),
        ("slurm", "job_id"),
        ("slurm_job_id",),
        ("job_id",),
    )
    for filename in ("environment.json", "summary.json"):
        payload = _load_json(run_root / filename)
        for path in candidate_paths:
            job_id = _find_nested_string(payload, path)
            if job_id:
                return job_id if _is_safe_job_id(job_id) else None
    return None


def _safe_child(root: Path, filename: str) -> Path | None:
    root_resolved = root.resolve()
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        return None
    return candidate


def _copy_current_run_logs(source_root: Path, destination_root: Path, job_id: str | None) -> None:
    if not job_id or not _is_safe_job_id(job_id) or not source_root.is_dir():
        return
    for suffix in (".out", ".err"):
        filename = f"neurotwin-a100-full-{job_id}{suffix}"
        source = _safe_child(source_root, filename)
        destination = _safe_child(destination_root, filename)
        if source is not None and destination is not None:
            _copy_file(source, destination)


def _write_readmes(root: Path) -> None:
    (root / "COMMIT_HASH.txt").write_text(full_sha + "\n", encoding="utf-8")
    handoff_source = repo_root / "README_HANDOFF.md"
    if handoff_source.is_file() and not _is_forbidden(handoff_source):
        shutil.copy2(handoff_source, root / "README_HANDOFF.md")
    else:
        (root / "README_HANDOFF.md").write_text(
            "# NeuroTwin A100 Handoff\n\n"
            f"Evidence bundle for commit `{full_sha}`.\n\n"
            "This run is an A100 infrastructure validation only. It is meant to show that the runner can "
            "prepare MOABB manifests, pass leakage/window audits, see CUDA, train for the configured smoke "
            "steps, and write reviewable metrics and reports.\n\n"
            "It is not a scientific result, not a 3-seed paper-mode report, and not evidence of clinical or "
            "model-superiority claims. MOABB task labels are intentionally not persisted in prepared event "
            "metadata.\n",
            encoding="utf-8",
        )
    (root / "README_SEND_TO_FRIEND.md").write_text(
        "# Sendable NeuroTwin A100 Evidence\n\n"
        "Send this zip back after an A100 run. It includes small review artifacts only: summaries, metrics, "
        "tables, figures, prepared manifests/audits, Docker GPU preflight proof, logs, the source commit, "
        "and checksums.\n\n"
        "It intentionally excludes checkpoints, raw prepared arrays, runner tarballs, zip artifacts, passwords, "
        "API keys, SSH keys, `.env*` files, and other secret-looking files. Keep large checkpoints on the cluster "
        "unless they are requested explicitly.\n\n"
        "Verify the bundle after extraction with:\n\n"
        "```bash\nshasum -a 256 -c handoff-SHA256SUMS\n```\n",
        encoding="utf-8",
    )


def _write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "handoff-SHA256SUMS"):
        rel = path.relative_to(root).as_posix()
        rows.append(f"{sha256(path.read_bytes()).hexdigest()}  {rel}")
    (root / "handoff-SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


with tempfile.TemporaryDirectory() as tmp:
    stage_root = Path(tmp) / evidence_name
    stage_root.mkdir(parents=True)

    for rel in run_files:
        _copy_file(run_dir / rel, stage_root / "run" / rel)
    _copy_file(persistent_root / "gpu_preflight.json", stage_root / "run" / "gpu_preflight.json")
    _copy_tree_files(run_dir / "tables", stage_root / "run" / "tables")
    _copy_tree_files(run_dir / "figures", stage_root / "run" / "figures")
    for rel in prepared_files:
        _copy_file(prepared_dir / rel, stage_root / "prepared" / rel)
    _copy_current_run_logs(logs_dir, stage_root / "logs", _current_slurm_job_id(run_dir))
    _write_readmes(stage_root)
    _write_checksums(stage_root)

    for included in stage_root.rglob("*"):
        if included.is_file() and _is_forbidden(included):
            raise SystemExit(f"forbidden file staged for evidence bundle: {included.relative_to(stage_root)}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in stage_root.rglob("*") if p.is_file()):
            archive.write(path, f"{evidence_name}/{path.relative_to(stage_root).as_posix()}")

print(f"evidence_zip={zip_path}")
print(f"evidence_zip_sha256={sha256(zip_path.read_bytes()).hexdigest()}")
print(f"commit={full_sha}")
PY
