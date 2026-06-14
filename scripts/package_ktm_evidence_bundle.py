#!/usr/bin/env python3
"""Package the small, sendable evidence bundle after a Kahlus v3 KTM A100 micro-sweep.

SYNTHETIC ONLY. Collects the run's review artifacts (JSON/CSV bundle + environment + preflight +
logs) into a checksummed zip for sending back. It intentionally EXCLUDES checkpoints, raw arrays, and
secrets. There is no MOABB/prepared/real-data section — the KTM run is synthetic.

usage: scripts/package_ktm_evidence_bundle.py <persistent_root> <zip_path> <evidence_name> <repo_root> <full_sha>
"""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import zipfile

RUN_FILES = (
    "metrics.json",
    "baseline_table.json",
    "baseline_table.csv",
    "evidence_gate.json",
    "model_card.json",
    "data_card.json",
    "run_config.json",
    "failure_reasons.json",
    "environment.json",
    "gpu_preflight.json",
    # Progress + failure forensics so a run that dies partway is still recoverable + debuggable.
    "progress.jsonl",
    "run_status.json",
    "failure_report.json",
)
FORBIDDEN_SUFFIXES = (".pem", ".key", ".pt", ".pth", ".ckpt", ".npy", ".npz", ".tar.gz", ".zip")
FORBIDDEN_MARKERS = ("password", "passwd", "secret", "api_key", "apikey", "ssh_key", "wandb", "token")
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
LOG_PATTERN = re.compile(r"^kahlus-ktm-.*\.log$")

README_SEND = """# Sendable Kahlus v3 KTM A100 Evidence (SYNTHETIC)

Small review artifacts from a synthetic KTM micro-sweep: metrics, baseline table, evidence gate,
model/data cards, run config, failure reasons, environment, GPU preflight proof, and logs.

It excludes checkpoints, raw arrays, runner/zip artifacts, passwords, API keys, SSH keys, `.env*`
files, and W&B tokens. Checkpoints stay on the cluster unless explicitly requested.

This is infrastructure validation only — `synthetic_ktm_training_harness` may pass;
`synthetic_ktm_recovery` stays blocked unless KTM beats baselines under locked metrics. No real-data,
clinical, consciousness, Orch-OR, or model-superiority claim. **MOABB audit is not applicable;
synthetic split/data-card checks are used instead.**

Verify after extraction:

```bash
shasum -a 256 -c handoff-SHA256SUMS
```
"""


def _is_forbidden(name: str) -> bool:
    lower = name.lower()
    if lower in {"pw.txt", ".env"} or lower.startswith(".env."):
        return True
    if lower.endswith(FORBIDDEN_SUFFIXES) or lower.startswith("checkpoint"):
        return True
    return any(marker in lower for marker in FORBIDDEN_MARKERS)


def _copy(src: Path, dest: Path) -> None:
    if not src.is_file() or src.is_symlink() or _is_forbidden(src.name):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def main(argv: list[str]) -> int:
    if len(argv) != 6:
        print(__doc__, file=sys.stderr)
        return 2
    persistent_root = Path(argv[1]).resolve()
    zip_path = Path(argv[2]).resolve()
    evidence_name = argv[3]
    _repo_root = Path(argv[4]).resolve()
    full_sha = argv[5]

    run_id = os.environ.get("KTM_RUN_ID", "ktm_micro_sweep")
    if not SAFE_NAME.match(run_id):
        run_id = "ktm_micro_sweep"
    run_dir = persistent_root / "runs" / run_id
    logs_dir = persistent_root / "logs"

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / evidence_name
        for rel in RUN_FILES:
            # gpu_preflight.json may live in the run dir or the persistent root.
            src = run_dir / rel
            if not src.is_file() and rel == "gpu_preflight.json":
                src = persistent_root / rel
            _copy(src, stage / "run" / rel)
        if logs_dir.is_dir():
            for log in sorted(logs_dir.glob("*.log")):
                if LOG_PATTERN.match(log.name):
                    _copy(log, stage / "logs" / log.name)

        (stage / "COMMIT_HASH.txt").write_text(full_sha + "\n", encoding="utf-8")
        (stage / "README_SEND_TO_FRIEND.md").write_text(README_SEND, encoding="utf-8")

        checksums = []
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            digest = sha256(path.read_bytes()).hexdigest()
            checksums.append(f"{digest}  {path.relative_to(stage).as_posix()}")
        (stage / "handoff-SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")

        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(p for p in stage.rglob("*") if p.is_file()):
                archive.write(path, f"{evidence_name}/{path.relative_to(stage).as_posix()}")

    print(f"evidence_zip={zip_path}")
    print(f"evidence_zip_sha256={sha256(zip_path.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
