"""KTM A100 evidence intake auditor (read-only, SYNTHETIC lane).

Consumes a returned KTM A100 evidence *folder or .zip* and produces a structured verdict about
whether the bundle is complete, secret-safe, GPU-consistent, claim-safe, reproducible, and
scientifically interpretable. It is a pure consumer: it NEVER runs A100/cluster jobs, never changes
training or model architecture, and never manufactures recovery/model-superiority claims. It only
reads what a run already produced and reports whether the evidence supports a narrow synthetic claim.

The bundle shape it expects (written by ``scripts/package_ktm_evidence_bundle.py``)::

    <bundle_root>/
      run/{metrics,baseline_table,evidence_gate,model_card,data_card,run_config,
           failure_reasons,environment,gpu_preflight,run_status,failure_report}.json
      run/baseline_table.csv
      run/progress.jsonl
      logs/kahlus-ktm-*.log
      COMMIT_HASH.txt
      README_SEND_TO_FRIEND.md
      handoff-SHA256SUMS
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import tempfile
from typing import Any
import zipfile

from neurotwin.gates import NARROW_CLAIM_SCOPES
from neurotwin.repro import hash_file

# --- severities -----------------------------------------------------------------------------
FAIL = "fail"
WARN = "warn"
INFO = "info"
_VERDICT_RANK = {INFO: 0, WARN: 1, FAIL: 2}

# --- expected bundle contents ---------------------------------------------------------------
REQUIRED_RUN_FILES: tuple[str, ...] = (
    "metrics.json",
    "baseline_table.json",
    "baseline_table.csv",
    "evidence_gate.json",
    "model_card.json",
    "data_card.json",
    "run_config.json",
    "failure_reasons.json",
    "environment.json",
)
OPTIONAL_RUN_FILES: tuple[str, ...] = (
    "gpu_preflight.json",
    "run_status.json",
    "progress.jsonl",
    "failure_report.json",
)
CHECKSUM_FILENAMES: tuple[str, ...] = ("handoff-SHA256SUMS", "SHA256SUMS")

# --- secret / checkpoint guard (mirrors scripts/package_ktm_evidence_bundle.py) --------------
FORBIDDEN_SUFFIXES: tuple[str, ...] = (".pem", ".key", ".pt", ".pth", ".ckpt", ".npy", ".npz", ".tar.gz", ".zip")
FORBIDDEN_MARKERS: tuple[str, ...] = ("password", "passwd", "secret", "api_key", "apikey", "ssh_key", "wandb", "token")

# --- claim-safety guard ---------------------------------------------------------------------
# Affirmative broad-claim keywords. Only scanned inside *claim-bearing* fields (keys hinting at a
# claim/scope/status/verdict). Free-text disclaimer fields are skipped on purpose: a model card
# legitimately says "no clinical or control claims", and a blunt keyword scan would false-fail it.
BROAD_CLAIM_KEYWORDS: tuple[str, ...] = (
    "clinical",
    "diagnos",
    "consciousness",
    "orch-or",
    "orch or",
    "orchor",
    "brain control",
    "brain-control",
    "treatment",
    "patient",
    "sota",
    "state-of-the-art",
    "real eeg",
    "real-data",
    "real data",
)
CLAIM_FIELD_HINTS: tuple[str, ...] = ("claim", "scope", "status", "verdict", "assertion", "conclusion")
DISCLAIMER_FIELD_HINTS: tuple[str, ...] = (
    "limitation",
    "note",
    "disclaimer",
    "caveat",
    "readme",
    "description",
    "summary",
    "reason",
    "blocked",
)

# Keys the unified evidence gate dossier must carry.
GATE_REQUIRED_KEYS: tuple[str, ...] = (
    "branch",
    "dataset",
    "split_audit_passed",
    "baseline_table_present",
    "finite_metrics",
    "calibration_checked",
    "claim_scope",
    "scientific_claim_allowed",
    "failure_reasons",
)


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "code": self.code, "message": self.message}


@dataclass
class AuditResult:
    verdict: str
    findings: list[Finding]
    commit_hash: str | None
    run_status: str | None
    run_completed: bool | None
    environment: dict[str, Any]
    metrics: dict[str, Any]
    baseline_comparison: dict[str, Any]
    gate: dict[str, Any]
    recovery: dict[str, Any]
    missing_files: list[str]
    secret_scan: dict[str, Any]
    checksums: dict[str, Any]
    failure_report: dict[str, Any] | None
    next_action: str
    source: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["findings"] = [f.to_dict() for f in self.findings]
        payload["schema"] = "kahlus.a100_evidence_audit.v1"
        payload["claim_status"] = "synthetic_audit_only"
        return payload


# ============================================================================================
# Intake
# ============================================================================================
def _safe_extract(zip_path: Path, dest: Path) -> Path:
    """Extract a zip into ``dest`` rejecting absolute paths and ``..`` traversal (zip-slip)."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            normalized = Path(member)
            if normalized.is_absolute() or ".." in normalized.parts:
                raise ValueError(f"unsafe path in evidence zip: {member!r}")
            target = (dest / member).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"zip member escapes extraction dir: {member!r}")
        archive.extractall(dest)
    return _find_bundle_root(dest)


def _find_bundle_root(base: Path) -> Path:
    """Locate the bundle root (the dir holding COMMIT_HASH.txt / run/), descending one wrapper dir."""
    if (base / "COMMIT_HASH.txt").exists() or (base / "run").is_dir():
        return base
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        inner = subdirs[0]
        if (inner / "COMMIT_HASH.txt").exists() or (inner / "run").is_dir():
            return inner
    return base


def _run_dir(root: Path) -> Path:
    run = root / "run"
    return run if run.is_dir() else root


def _load_json(path: Path) -> tuple[Any | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    except OSError as exc:  # pragma: no cover - unreadable file
        return None, f"unreadable: {exc}"


# ============================================================================================
# Secret / checkpoint guard
# ============================================================================================
def _is_forbidden_name(name: str) -> bool:
    lower = name.lower()
    if lower in {"pw.txt", ".env"} or lower.startswith(".env."):
        return True
    if lower.endswith(FORBIDDEN_SUFFIXES) or lower.startswith("checkpoint"):
        return True
    return any(marker in lower for marker in FORBIDDEN_MARKERS)


def scan_secrets(root: Path) -> tuple[list[str], list[Finding]]:
    """Flag any checkpoint/secret/raw-array file that should never be inside a sendable bundle."""
    offenders: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if _is_forbidden_name(path.name):
            offenders.append(rel)
    findings = [
        Finding(FAIL, "secret_or_checkpoint", f"forbidden file inside evidence bundle: {rel}")
        for rel in offenders
    ]
    return offenders, findings


# ============================================================================================
# Checksums
# ============================================================================================
def verify_checksums(root: Path) -> tuple[dict[str, Any], list[Finding]]:
    sums_path: Path | None = None
    for name in CHECKSUM_FILENAMES:
        candidate = root / name
        if candidate.is_file():
            sums_path = candidate
            break

    if sums_path is None:
        return (
            {"present": False, "verified": 0, "mismatched": [], "missing": []},
            [Finding(WARN, "checksums_absent", "no handoff-SHA256SUMS/SHA256SUMS file to verify against")],
        )

    mismatched: list[str] = []
    missing: list[str] = []
    verified = 0
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        expected_hex, rel = parts
        target = root / rel
        if not target.is_file():
            missing.append(rel)
            continue
        if hash_file(target).lower() != expected_hex.lower():
            mismatched.append(rel)
        else:
            verified += 1

    findings: list[Finding] = []
    for rel in mismatched:
        findings.append(Finding(FAIL, "checksum_mismatch", f"checksum mismatch: {rel}"))
    for rel in missing:
        findings.append(Finding(FAIL, "checksum_missing_file", f"checksummed file missing from bundle: {rel}"))
    summary = {
        "present": True,
        "file": sums_path.name,
        "verified": verified,
        "mismatched": mismatched,
        "missing": missing,
    }
    return summary, findings


# ============================================================================================
# Required / optional file presence
# ============================================================================================
def check_required_files(
    root: Path, *, allow_missing_logs: bool
) -> tuple[list[str], dict[str, Any], list[Finding]]:
    run = _run_dir(root)
    findings: list[Finding] = []
    missing: list[str] = []
    unparseable: list[str] = []

    for name in REQUIRED_RUN_FILES:
        path = run / name
        if not path.is_file():
            missing.append(name)
            findings.append(Finding(FAIL, "required_file_missing", f"required evidence file missing: {name}"))
            continue
        if name.endswith(".json"):
            _, err = _load_json(path)
            if err:
                unparseable.append(name)
                findings.append(Finding(FAIL, "required_file_unparseable", f"required file {name}: {err}"))

    present_optional: list[str] = []
    for name in OPTIONAL_RUN_FILES:
        path = run / name
        if path.is_file():
            present_optional.append(name)
        else:
            # Forensic extras are nice-to-have. A clean completed run legitimately lacks
            # failure_report.json, so absence is informational, not a warning.
            findings.append(Finding(INFO, "optional_file_absent", f"optional evidence file absent: {name}"))

    log_files = sorted(p.name for p in (root / "logs").glob("*.log")) if (root / "logs").is_dir() else []
    if not log_files and not allow_missing_logs:
        findings.append(Finding(WARN, "logs_absent", "no logs/*.log present (pass allow_missing_logs to silence)"))

    inventory = {
        "required_present": [n for n in REQUIRED_RUN_FILES if n not in missing],
        "optional_present": present_optional,
        "logs": log_files,
        "unparseable": unparseable,
    }
    return missing, inventory, findings


# ============================================================================================
# Environment audit
# ============================================================================================
def audit_environment(
    env: dict[str, Any] | None,
    preflight: dict[str, Any] | None,
    expected_gpus: int,
) -> tuple[dict[str, Any], list[Finding]]:
    findings: list[Finding] = []
    if not isinstance(env, dict):
        return (
            {"available": False},
            [Finding(FAIL, "environment_unreadable", "environment.json absent or not an object")],
        )

    torch_meta = env.get("torch", {}) if isinstance(env.get("torch"), dict) else {}
    run_meta = env.get("run", {}) if isinstance(env.get("run"), dict) else {}
    dist = run_meta.get("distributed", {}) if isinstance(run_meta.get("distributed"), dict) else {}
    container = run_meta.get("container", {}) if isinstance(run_meta.get("container"), dict) else {}

    gpu_count = torch_meta.get("cuda_device_count")
    summary = {
        "available": True,
        "gpu_count": gpu_count,
        "gpu_names": torch_meta.get("cuda_device_names"),
        "cuda_visible_devices": dist.get("cuda_visible_devices"),
        "world_size": dist.get("world_size"),
        "local_rank": dist.get("local_rank"),
        "rank": dist.get("rank"),
        "torch_version": torch_meta.get("version"),
        "torch_cuda_version": torch_meta.get("torch_cuda_version") or torch_meta.get("cuda_version"),
        "nccl_version": torch_meta.get("nccl_version"),
        "docker_image": container.get("docker_image"),
        "run_mode": run_meta.get("mode"),
        "expected_gpus": expected_gpus,
    }

    if gpu_count is None:
        findings.append(Finding(WARN, "gpu_count_unknown", "environment.json has no torch.cuda_device_count"))
    elif int(gpu_count) != int(expected_gpus):
        findings.append(
            Finding(FAIL, "gpu_count_mismatch", f"visible GPUs {gpu_count} != expected {expected_gpus}")
        )

    if isinstance(preflight, dict):
        summary["preflight_passed"] = preflight.get("passed")
        summary["preflight_visible_gpu_count"] = preflight.get("visible_gpu_count")
        summary["preflight_visible_gpu_names"] = preflight.get("visible_gpu_names")
        if preflight.get("passed") is False:
            findings.append(Finding(FAIL, "preflight_failed", "gpu_preflight.json reports passed=false"))
        visible = preflight.get("visible_gpu_count")
        if visible is not None and int(visible) != int(expected_gpus):
            findings.append(
                Finding(
                    FAIL,
                    "preflight_gpu_mismatch",
                    f"gpu_preflight visible_gpu_count {visible} != expected {expected_gpus}",
                )
            )

    return summary, findings


# ============================================================================================
# Metric audit
# ============================================================================================
def audit_metrics(
    metrics: dict[str, Any] | None,
    gate: dict[str, Any] | None,
    run_status: dict[str, Any] | None,
    failure_report: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], list[Finding]]:
    findings: list[Finding] = []
    metrics = metrics if isinstance(metrics, dict) else {}
    gate = gate if isinstance(gate, dict) else {}

    comparison = metrics.get("ktm_vs_baselines", {}) if isinstance(metrics.get("ktm_vs_baselines"), dict) else {}
    beats = comparison.get("ktm_beats_baselines")

    harness_passed = bool(gate.get("scientific_claim_allowed")) and gate.get("claim_scope") in {
        "synthetic_ktm_training_harness",
        "synthetic_ktm_recovery",
    }
    recovery_allowed = bool(metrics.get("recovery_claim_allowed"))

    # Run completion: failure_report present (or run_status failed) means the run did not complete.
    status_value = (run_status or {}).get("status") if isinstance(run_status, dict) else None
    run_failed = bool(failure_report) or status_value == "failed"
    run_completed = (not run_failed) if (status_value or failure_report is not None or metrics) else None

    metric_summary = {
        "ktm_mse": comparison.get("ktm_mse"),
        "best_baseline": comparison.get("best_baseline"),
        "best_baseline_mse": comparison.get("best_baseline_mse"),
        "ktm_beats_baselines": beats,
        "loss_decreased": metrics.get("loss_decreased"),
        "val_mse_before": metrics.get("val_mse_before"),
        "val_mse_after": metrics.get("val_mse_after"),
        "best_val_mse": metrics.get("best_val_mse"),
        "harness_gate_passed": harness_passed,
        "recovery_claim_allowed": recovery_allowed,
        "run_status": status_value,
        "run_completed": run_completed,
    }

    if metrics.get("loss_decreased") is False:
        findings.append(Finding(WARN, "loss_not_decreased", "training loss did not decrease over the run"))
    if run_failed:
        findings.append(Finding(WARN, "run_failed", f"run did not complete (status={status_value!r}, failure_report present={bool(failure_report)})"))

    return metric_summary, comparison, findings


# ============================================================================================
# Claim audit (the heart of "claim-safe")
# ============================================================================================
def _scan_claim_text(obj: Any, key: str | None, source: str, out: list[Finding]) -> None:
    if isinstance(obj, dict):
        for child_key, value in obj.items():
            _scan_claim_text(value, str(child_key), source, out)
    elif isinstance(obj, list):
        for item in obj:
            _scan_claim_text(item, key, source, out)
    elif isinstance(obj, str) and key:
        key_lower = key.lower()
        if any(hint in key_lower for hint in DISCLAIMER_FIELD_HINTS):
            return
        if not any(hint in key_lower for hint in CLAIM_FIELD_HINTS):
            return
        value_lower = obj.lower()
        for keyword in BROAD_CLAIM_KEYWORDS:
            if keyword in value_lower:
                out.append(
                    Finding(
                        FAIL,
                        "broad_claim",
                        f"{source}: claim field {key!r} asserts broad keyword {keyword!r}: {obj!r}",
                    )
                )


def audit_claims(
    gate: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    model_card: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[Finding]]:
    findings: list[Finding] = []
    gate = gate if isinstance(gate, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    model_card = model_card if isinstance(model_card, dict) else {}

    comparison = metrics.get("ktm_vs_baselines", {}) if isinstance(metrics.get("ktm_vs_baselines"), dict) else {}
    beats = bool(comparison.get("ktm_beats_baselines"))

    claim_scope = gate.get("claim_scope")
    scientific_claim_allowed = bool(gate.get("scientific_claim_allowed"))

    # 1. Evidence gate must carry the full unified dossier.
    missing_keys = [k for k in GATE_REQUIRED_KEYS if k not in gate]
    if missing_keys:
        findings.append(
            Finding(FAIL, "gate_schema_incomplete", f"evidence gate missing required fields: {missing_keys}")
        )

    # 2. Claim scope must be one of the narrow synthetic/audit scopes.
    if claim_scope is not None and claim_scope not in NARROW_CLAIM_SCOPES:
        findings.append(
            Finding(FAIL, "claim_scope_too_broad", f"claim scope {claim_scope!r} not in narrow allowlist")
        )

    # 3. Recovery may not be asserted unless KTM actually beat baselines.
    recovery_asserted = (
        bool(metrics.get("recovery_claim_allowed"))
        or bool(model_card.get("recovery_claim_allowed"))
        or claim_scope == "synthetic_ktm_recovery"
    )
    if recovery_asserted and not beats:
        findings.append(
            Finding(
                FAIL,
                "unearned_recovery_claim",
                "synthetic_ktm_recovery asserted but KTM did not beat baselines on locked metrics",
            )
        )

    # 4. Affirmative broad claims hiding in claim-bearing fields (disclaimers are skipped).
    _scan_claim_text(gate, None, "evidence_gate.json", findings)
    _scan_claim_text(metrics, None, "metrics.json", findings)
    _scan_claim_text(model_card, None, "model_card.json", findings)

    recovery_summary = {
        "claim_scope": claim_scope,
        "scientific_claim_allowed": scientific_claim_allowed,
        "recovery_asserted": recovery_asserted,
        "ktm_beats_baselines": beats,
        "recovery_claim_safe": not recovery_asserted or beats,
        "gate_failure_reasons": gate.get("failure_reasons", []),
    }
    return recovery_summary, findings


# ============================================================================================
# Orchestration
# ============================================================================================
def _next_action(verdict: str, findings: list[Finding]) -> str:
    codes = {f.code for f in findings if f.severity == FAIL}
    if "secret_or_checkpoint" in codes:
        return "DO NOT forward this bundle: strip checkpoints/secrets on the cluster and re-export the evidence."
    if codes & {"checksum_mismatch", "checksum_missing_file"}:
        return "Checksum integrity failed; request a fresh, intact re-export before trusting any metric."
    if codes & {"required_file_missing", "required_file_unparseable"}:
        return "Bundle is incomplete; request a re-export that includes the missing/valid evidence files."
    if codes & {"gpu_count_mismatch", "preflight_failed", "preflight_gpu_mismatch"}:
        return "GPU topology does not match --expected-gpus; confirm cluster allocation and re-run."
    if codes & {"unearned_recovery_claim", "claim_scope_too_broad", "broad_claim", "gate_schema_incomplete"}:
        return "Block the claim: the bundle asserts more than the evidence supports. Do not record as a result."
    warn_codes = {f.code for f in findings if f.severity == WARN}
    if "run_failed" in warn_codes:
        return "Run did not complete; inspect failure_report.json and re-run before claiming harness readiness."
    if verdict == WARN:
        return "Bundle is claim-safe but has gaps (see warnings); usable as synthetic harness evidence with caveats."
    return "Bundle is complete and claim-safe; safe to record as synthetic_ktm_training_harness evidence."


def _audit_root(root: Path, expected_gpus: int, allow_missing_logs: bool, source: dict[str, Any]) -> AuditResult:
    run = _run_dir(root)
    findings: list[Finding] = []

    # Intake: presence + parse.
    missing, inventory, presence_findings = check_required_files(root, allow_missing_logs=allow_missing_logs)
    findings.extend(presence_findings)

    # Secret/checkpoint guard.
    offenders, secret_findings = scan_secrets(root)
    findings.extend(secret_findings)

    # Checksums + commit hash.
    checksum_summary, checksum_findings = verify_checksums(root)
    findings.extend(checksum_findings)
    commit_path = root / "COMMIT_HASH.txt"
    commit_hash = commit_path.read_text(encoding="utf-8").strip() if commit_path.is_file() else None
    if commit_hash is None:
        findings.append(Finding(WARN, "commit_hash_absent", "COMMIT_HASH.txt missing from bundle"))

    # Load the structured artifacts (None if missing/invalid; downstream audits degrade gracefully).
    metrics, _ = _load_json(run / "metrics.json")
    gate, _ = _load_json(run / "evidence_gate.json")
    model_card, _ = _load_json(run / "model_card.json")
    environment, _ = _load_json(run / "environment.json")
    preflight, _ = _load_json(run / "gpu_preflight.json")
    if preflight is None:
        preflight_alt, _ = _load_json(root / "gpu_preflight.json")
        preflight = preflight_alt
    run_status, _ = _load_json(run / "run_status.json")
    failure_report, _ = _load_json(run / "failure_report.json")

    env_summary, env_findings = audit_environment(environment, preflight, expected_gpus)
    findings.extend(env_findings)

    metric_summary, comparison, metric_findings = audit_metrics(metrics, gate, run_status, failure_report)
    findings.extend(metric_findings)

    recovery_summary, claim_findings = audit_claims(gate, metrics, model_card)
    findings.extend(claim_findings)

    # Commit-hash cross-check against environment.git.commit.
    env_commit = ((environment or {}).get("git", {}) or {}).get("commit") if isinstance(environment, dict) else None
    if commit_hash and env_commit and commit_hash != env_commit:
        findings.append(
            Finding(WARN, "commit_hash_divergence", f"COMMIT_HASH.txt {commit_hash} != environment git commit {env_commit}")
        )

    verdict = INFO
    for finding in findings:
        if _VERDICT_RANK[finding.severity] > _VERDICT_RANK[verdict]:
            verdict = finding.severity
    verdict = {INFO: "pass", WARN: WARN, FAIL: FAIL}[verdict]

    failure_summary = None
    if isinstance(failure_report, dict):
        failure_summary = {
            "status": failure_report.get("status"),
            "phase": failure_report.get("phase"),
            "error_type": failure_report.get("error_type"),
            "error": failure_report.get("error"),
        }

    return AuditResult(
        verdict=verdict,
        findings=findings,
        commit_hash=commit_hash,
        run_status=metric_summary.get("run_status"),
        run_completed=metric_summary.get("run_completed"),
        environment=env_summary,
        metrics=metric_summary,
        baseline_comparison=comparison,
        gate=gate if isinstance(gate, dict) else {},
        recovery=recovery_summary,
        missing_files=missing,
        secret_scan={"clean": not offenders, "offenders": offenders},
        checksums=checksum_summary,
        failure_report=failure_summary,
        next_action=_next_action(verdict, findings),
        source={**source, "inventory": inventory},
    )


def audit_evidence(
    evidence_path: str | Path,
    *,
    expected_gpus: int = 7,
    allow_missing_logs: bool = False,
) -> AuditResult:
    """Audit a returned KTM A100 evidence folder or .zip. Read-only; never runs cluster work."""
    path = Path(evidence_path)

    if path.is_file() and path.name.lower().endswith(".zip"):
        with tempfile.TemporaryDirectory(prefix="ktm_a100_audit_") as tmp:
            try:
                root = _safe_extract(path, Path(tmp))
            except (zipfile.BadZipFile, ValueError) as exc:
                return _intake_failure(f"could not extract evidence zip {path.name}: {exc}", str(path), "zip")
            return _audit_root(
                root,
                expected_gpus=expected_gpus,
                allow_missing_logs=allow_missing_logs,
                source={"kind": "zip", "path": str(path)},
            )

    if path.is_dir():
        root = _find_bundle_root(path)
        return _audit_root(
            root,
            expected_gpus=expected_gpus,
            allow_missing_logs=allow_missing_logs,
            source={"kind": "folder", "path": str(path)},
        )

    return _intake_failure(f"evidence path is not a folder or .zip: {path}", str(path), "unknown")


def _intake_failure(message: str, path: str, kind: str) -> AuditResult:
    finding = Finding(FAIL, "intake_failed", message)
    return AuditResult(
        verdict=FAIL,
        findings=[finding],
        commit_hash=None,
        run_status=None,
        run_completed=None,
        environment={"available": False},
        metrics={},
        baseline_comparison={},
        gate={},
        recovery={},
        missing_files=list(REQUIRED_RUN_FILES),
        secret_scan={"clean": None, "offenders": []},
        checksums={"present": False},
        failure_report=None,
        next_action="Bundle could not be opened; request a valid evidence folder or zip.",
        source={"kind": kind, "path": path},
    )


# ============================================================================================
# Markdown report
# ============================================================================================
_VERDICT_BADGE = {"pass": "PASS", WARN: "WARN", FAIL: "FAIL"}


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else "—"
    return str(value)


def render_report_md(result: AuditResult) -> str:
    r = result
    lines: list[str] = []
    lines.append("# KTM A100 Evidence Audit (SYNTHETIC)")
    lines.append("")
    lines.append(f"**Verdict:** {_VERDICT_BADGE.get(r.verdict, r.verdict.upper())}")
    lines.append("")
    lines.append(f"- Source: `{r.source.get('path', '')}` ({r.source.get('kind', 'unknown')})")
    lines.append(f"- Commit hash: `{_fmt(r.commit_hash)}`")
    lines.append(f"- Run completed: {_fmt(r.run_completed)} (status: {_fmt(r.run_status)})")
    lines.append(f"- Next action: {r.next_action}")
    lines.append("")

    fails = [f for f in r.findings if f.severity == FAIL]
    warns = [f for f in r.findings if f.severity == WARN]
    lines.append("## Findings")
    lines.append(f"- FAIL: {len(fails)}  WARN: {len(warns)}")
    for finding in fails:
        lines.append(f"  - **FAIL** [{finding.code}] {finding.message}")
    for finding in warns:
        lines.append(f"  - WARN [{finding.code}] {finding.message}")
    if not fails and not warns:
        lines.append("  - none")
    lines.append("")

    env = r.environment
    lines.append("## GPU / Environment")
    lines.append(f"- Expected GPUs: {_fmt(env.get('expected_gpus'))}  Visible GPUs: {_fmt(env.get('gpu_count'))}")
    lines.append(f"- GPU names: {_fmt(env.get('gpu_names'))}")
    lines.append(f"- CUDA_VISIBLE_DEVICES: {_fmt(env.get('cuda_visible_devices'))}")
    lines.append(f"- WORLD_SIZE: {_fmt(env.get('world_size'))}  RANK: {_fmt(env.get('rank'))}  LOCAL_RANK: {_fmt(env.get('local_rank'))}")
    lines.append(f"- torch: {_fmt(env.get('torch_version'))}  cuda: {_fmt(env.get('torch_cuda_version'))}  nccl: {_fmt(env.get('nccl_version'))}")
    lines.append(f"- Docker image: {_fmt(env.get('docker_image'))}  run mode: {_fmt(env.get('run_mode'))}")
    if "preflight_passed" in env:
        lines.append(f"- Preflight passed: {_fmt(env.get('preflight_passed'))} (visible {_fmt(env.get('preflight_visible_gpu_count'))})")
    lines.append("")

    m = r.metrics
    lines.append("## Metrics")
    lines.append(f"- KTM MSE: {_fmt(m.get('ktm_mse'))}")
    lines.append(f"- Best baseline: {_fmt(m.get('best_baseline'))} (MSE {_fmt(m.get('best_baseline_mse'))})")
    lines.append(f"- KTM beats baselines: {_fmt(m.get('ktm_beats_baselines'))}")
    lines.append(f"- Loss decreased: {_fmt(m.get('loss_decreased'))} ({_fmt(m.get('val_mse_before'))} -> {_fmt(m.get('val_mse_after'))})")
    lines.append(f"- Harness gate passed: {_fmt(m.get('harness_gate_passed'))}")
    lines.append("")

    g = r.gate
    lines.append("## Evidence Gate")
    lines.append(f"- Branch: {_fmt(g.get('branch'))}  Dataset: {_fmt(g.get('dataset'))}")
    lines.append(f"- Claim scope: {_fmt(g.get('claim_scope'))}")
    lines.append(f"- Scientific claim allowed: {_fmt(g.get('scientific_claim_allowed'))}")
    lines.append(f"- Gate failure reasons: {_fmt(g.get('failure_reasons'))}")
    lines.append("")

    rec = r.recovery
    lines.append("## Recovery Claim")
    lines.append(f"- Recovery asserted: {_fmt(rec.get('recovery_asserted'))}")
    lines.append(f"- Recovery claim safe: {_fmt(rec.get('recovery_claim_safe'))}")
    lines.append("")

    lines.append("## Integrity")
    sec = r.secret_scan
    lines.append(f"- Secret/checkpoint scan clean: {_fmt(sec.get('clean'))}")
    if sec.get("offenders"):
        lines.append(f"  - offenders: {_fmt(sec.get('offenders'))}")
    chk = r.checksums
    lines.append(f"- Checksums present: {_fmt(chk.get('present'))} verified: {_fmt(chk.get('verified'))}")
    if chk.get("mismatched"):
        lines.append(f"  - mismatched: {_fmt(chk.get('mismatched'))}")
    lines.append(f"- Missing required files: {_fmt(r.missing_files)}")
    lines.append("")

    if r.failure_report:
        fr = r.failure_report
        lines.append("## Failure Report")
        lines.append(f"- status: {_fmt(fr.get('status'))}  phase: {_fmt(fr.get('phase'))}")
        lines.append(f"- error: {_fmt(fr.get('error_type'))}: {_fmt(fr.get('error'))}")
        lines.append("")

    return "\n".join(lines) + "\n"
