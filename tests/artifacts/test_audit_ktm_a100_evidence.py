"""Tests for the KTM A100 evidence intake auditor (synthetic fixtures only)."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from neurotwin.a100_audit import audit_evidence, render_report_md


def _gate(**overrides):
    base = {
        "schema": "kahlus.unified_evidence_gate.v1",
        "branch": "v3",
        "dataset": "ktm_training_synthetic",
        "split_audit_passed": True,
        "baseline_table_present": True,
        "finite_metrics": True,
        "calibration_checked": True,
        "claim_scope": "synthetic_ktm_training_harness",
        "scientific_claim_allowed": True,
        "failure_reasons": [],
    }
    base.update(overrides)
    return base


def _metrics(**overrides):
    base = {
        "schema": "kahlus.ktm_training_metrics.v1",
        "branch": "v3",
        "claim_status": "synthetic_training_harness_only",
        "loss_decreased": True,
        "val_mse_before": 2.0,
        "val_mse_after": 1.0,
        "best_val_mse": 1.0,
        "ktm_vs_baselines": {
            "ktm_mse": 1.0,
            "best_baseline": "ridge",
            "best_baseline_mse": 0.9,
            "ktm_beats_baselines": False,
        },
        "baseline_metrics": {"ridge": {"mse": 0.9}},
        "recovery_claim_allowed": False,
    }
    base.update(overrides)
    return base


def _model_card(**overrides):
    base = {
        "schema": "kahlus.ktm_model_card.v1",
        "branch": "v3",
        "model": "TorchKTM",
        "claim_status": "synthetic_training_harness_only",
        "claim_scope": "synthetic_ktm_training_harness",
        "scientific_claim_allowed": True,
        "recovery_scope": "synthetic_ktm_recovery",
        "recovery_claim_allowed": False,
        "ktm_beats_baselines": False,
        # Disclaimer text deliberately mentions clinical / real EEG / control — the auditor must NOT
        # treat negated limitation text as a broad claim.
        "limitations": ["Synthetic Transition Gym only; no real EEG, no clinical or control claims."],
    }
    base.update(overrides)
    return base


def _environment(gpu_count=8):
    return {
        "captured_at": "2026-06-13T00:00:00+00:00",
        "torch": {
            "version": "2.6.0",
            "cuda_available": True,
            "cuda_device_count": gpu_count,
            "cuda_device_name": "NVIDIA A100",
            "cuda_device_names": ["NVIDIA A100"] * max(gpu_count, 0),
            "cuda_version": "12.4",
            "torch_cuda_version": "12.4",
            "nccl_version": "2.21.5",
        },
        "git": {"commit": "deadbeef", "source": "git", "source_commit_missing": False},
        "source_commit_missing": False,
        "run": {
            "mode": "container",
            "container": {"docker_image": "pytorch/pytorch:2.6.0-cuda12.4"},
            "distributed": {
                "local_rank": "0",
                "rank": "0",
                "world_size": str(gpu_count),
                "cuda_visible_devices": ",".join(str(i) for i in range(gpu_count)),
            },
        },
    }


def _preflight(gpu_count=8):
    return {
        "passed": True,
        "expected_gpu_count": gpu_count,
        "visible_gpu_count": gpu_count,
        "visible_gpu_names": ["NVIDIA A100"] * gpu_count,
        "cuda_available": True,
        "nccl_version": "2.21.5",
    }


def _write_sums(root: Path) -> None:
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "handoff-SHA256SUMS"):
        digest = sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    (root / "handoff-SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_evidence(
    base: Path,
    *,
    gpu_count: int = 8,
    gate: dict | None = None,
    metrics: dict | None = None,
    model_card: dict | None = None,
    drop: tuple[str, ...] = (),
    extra: dict[str, str] | None = None,
    write_sums: bool = True,
) -> Path:
    """Build a synthetic KTM A100 evidence bundle folder; returns the bundle root."""
    root = base / "kahlus-ktm-evidence"
    run = root / "run"
    run.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)

    files: dict[str, str] = {
        "run/metrics.json": json.dumps(metrics or _metrics()),
        "run/baseline_table.json": json.dumps(
            {
                "rows": [
                    {"model_id": "ridge", "mse": 0.9, "status": "completed"},
                    {"model_id": "ktm_torch", "mse": 1.0, "status": "completed"},
                ],
                "ranking": [],
                "ktm_vs_baselines": {"best_baseline": "ridge", "ktm_beats_baselines": False},
            }
        ),
        "run/baseline_table.csv": "model_id,mse,status\nridge,0.9,completed\nktm_torch,1.0,completed\n",
        "run/evidence_gate.json": json.dumps(gate or _gate()),
        "run/model_card.json": json.dumps(model_card or _model_card()),
        "run/data_card.json": json.dumps({"schema": "kahlus.transition_gym_data_card.v1", "branch": "v3"}),
        "run/run_config.json": json.dumps({"branch": "v3", "dataset": "ktm_training_synthetic"}),
        "run/failure_reasons.json": json.dumps(
            {"training": [], "baselines": [], "aborted": False, "recovery_blocked_reasons": []}
        ),
        "run/environment.json": json.dumps(_environment(gpu_count)),
        "run/gpu_preflight.json": json.dumps(_preflight(gpu_count)),
        "run/run_status.json": json.dumps({"status": "training_complete", "completed_steps": 100, "total_steps": 100}),
        "run/progress.jsonl": json.dumps({"event": "run_started"}) + "\n" + json.dumps({"event": "eval"}) + "\n",
        "COMMIT_HASH.txt": "deadbeef\n",
        "README_SEND_TO_FRIEND.md": "# Sendable Kahlus v3 KTM A100 Evidence (SYNTHETIC)\n",
        "logs/kahlus-ktm-test.log": "log line\n",
    }
    if extra:
        files.update(extra)
    for rel in drop:
        files.pop(rel, None)

    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    if write_sums:
        _write_sums(root)
    return root


def _zip_bundle(root: Path, zip_path: Path) -> Path:
    name = root.name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            archive.write(path, f"{name}/{path.relative_to(root).as_posix()}")
    return zip_path


def _codes(result, severity=None):
    return {f.code for f in result.findings if severity is None or f.severity == severity}


class AuditKtmA100EvidenceTests(unittest.TestCase):
    def test_valid_folder_passes_and_disclaimer_text_is_not_a_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_evidence(Path(tmp))
            result = audit_evidence(root, expected_gpus=8)
            self.assertEqual(result.verdict, "pass", _codes(result))
            self.assertNotIn("broad_claim", _codes(result))  # limitation text must not false-fail
            self.assertTrue(result.secret_scan["clean"])
            self.assertEqual(result.commit_hash, "deadbeef")

    def test_valid_zip_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_evidence(Path(tmp))
            zip_path = _zip_bundle(root, Path(tmp) / "evidence.zip")
            result = audit_evidence(zip_path, expected_gpus=8)
            self.assertEqual(result.verdict, "pass", _codes(result))

    def test_missing_required_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_evidence(Path(tmp), drop=("run/metrics.json",))
            result = audit_evidence(root, expected_gpus=8)
            self.assertEqual(result.verdict, "fail")
            self.assertIn("required_file_missing", _codes(result, "fail"))
            self.assertIn("metrics.json", result.missing_files)

    def test_checkpoint_file_in_bundle_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_evidence(Path(tmp), extra={"run/checkpoint_best.pt": "weights"})
            result = audit_evidence(root, expected_gpus=8)
            self.assertEqual(result.verdict, "fail")
            self.assertIn("secret_or_checkpoint", _codes(result, "fail"))

    def test_env_and_key_and_pem_files_fail(self):
        for secret in ("run/.env", "run/id.pem", "run/server.key"):
            with self.subTest(secret=secret), tempfile.TemporaryDirectory() as tmp:
                root = make_evidence(Path(tmp), extra={secret: "x"})
                result = audit_evidence(root, expected_gpus=8)
                self.assertEqual(result.verdict, "fail")
                self.assertIn("secret_or_checkpoint", _codes(result, "fail"))
                self.assertFalse(result.secret_scan["clean"])

    def test_expected_gpu_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_evidence(Path(tmp), gpu_count=8)
            result = audit_evidence(root, expected_gpus=6)
            self.assertEqual(result.verdict, "fail")
            self.assertIn("gpu_count_mismatch", _codes(result, "fail"))

    def test_recovery_claim_without_baseline_win_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics = _metrics(recovery_claim_allowed=True)  # but ktm_beats_baselines stays False
            root = make_evidence(Path(tmp), metrics=metrics)
            result = audit_evidence(root, expected_gpus=8)
            self.assertEqual(result.verdict, "fail")
            self.assertIn("unearned_recovery_claim", _codes(result, "fail"))

    def test_broad_claim_scope_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_evidence(Path(tmp), gate=_gate(claim_scope="clinical_digital_twin"))
            result = audit_evidence(root, expected_gpus=8)
            self.assertEqual(result.verdict, "fail")
            self.assertIn("claim_scope_too_broad", _codes(result, "fail"))

    def test_broad_claim_keyword_in_claim_field_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics = _metrics(claim_status="clinical_grade_diagnosis_ready")
            root = make_evidence(Path(tmp), metrics=metrics)
            result = audit_evidence(root, expected_gpus=8)
            self.assertEqual(result.verdict, "fail")
            self.assertIn("broad_claim", _codes(result, "fail"))

    def test_checksum_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_evidence(Path(tmp))
            (root / "run" / "metrics.json").write_text("{}\n", encoding="utf-8")  # tamper after sums
            result = audit_evidence(root, expected_gpus=8)
            self.assertEqual(result.verdict, "fail")
            self.assertIn("checksum_mismatch", _codes(result, "fail"))

    def test_reports_are_written_by_cli(self):
        repo = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = make_evidence(Path(tmp))
            out = Path(tmp) / "out"
            env = dict(os.environ, PYTHONPATH="src")
            result = subprocess.run(
                [sys.executable, "scripts/audit_ktm_a100_evidence.py",
                 "--evidence", str(root), "--out-dir", str(out), "--expected-gpus", "8"],
                cwd=repo, env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((out / "a100_evidence_audit.json").is_file())
            self.assertTrue((out / "a100_evidence_report.md").is_file())
            payload = json.loads((out / "a100_evidence_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["verdict"], "pass")
            self.assertIn("# KTM A100 Evidence Audit", (out / "a100_evidence_report.md").read_text(encoding="utf-8"))

    def test_cli_help(self):
        repo = Path.cwd()
        env = dict(os.environ, PYTHONPATH="src")
        result = subprocess.run(
            [sys.executable, "scripts/audit_ktm_a100_evidence.py", "--help"],
            cwd=repo, env=env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--evidence", result.stdout)

    def test_render_report_contains_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_evidence(Path(tmp))
            md = render_report_md(audit_evidence(root, expected_gpus=8))
            for section in ("## Findings", "## GPU / Environment", "## Metrics", "## Evidence Gate", "## Recovery Claim"):
                self.assertIn(section, md)


if __name__ == "__main__":
    unittest.main()
