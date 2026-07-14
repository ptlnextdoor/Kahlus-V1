from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neurotwin.benchmarks.baseline_suite import _run_task_models
from neurotwin.data.split_manifest import RecordingRecord, SplitManifest, build_split_manifest
from neurotwin.eeg_v1.dataset import build_future_forecasting_task, make_synthetic_eeg_v1_dataset


RUNNER_IDS = ("persistence", "linear_ridge", "gbm", "mlp", "tcn", "transformer", "neurotwin")
DISPLAY_IDS = {"linear_ridge": "ridge"}
REQUIRED_ROWS = ("persistence", "ridge", "gbm", "mlp", "tcn", "transformer", "neurotwin")


def run_m0_gate(
    out_dir: str | Path,
    *,
    seed: int = 0,
    train_steps: int = 2,
    enforce_clean_worktree: bool = True,
    n_subjects: int = 8,
    n_time: int = 48,
    n_channels: int = 4,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "environment.json", _environment_payload())
    _write_json(out / "git_state.json", _git_state())

    round_hashes = []
    for name in ("round_1", "round_2"):
        table = _run_one(
            out / name,
            seed=seed,
            train_steps=train_steps,
            n_subjects=n_subjects,
            n_time=n_time,
            n_channels=n_channels,
        )
        round_hashes.append(_sha256_file(table))
    shutil.copyfile(out / "round_1" / "baseline_table.csv", out / "baseline_table.csv")

    git_state = _read_json(out / "git_state.json")
    table_rows = _read_csv_rows(out / "baseline_table.csv")
    present = tuple(row["model_id"] for row in table_rows)
    missing = sorted(set(REQUIRED_ROWS) - set(present))
    bit_stable = len(set(round_hashes)) == 1
    clean_ok = bool(git_state["clean_worktree"]) or not enforce_clean_worktree
    gate = {
        "milestone": "M0",
        "gate_passed": bool(bit_stable and not missing and clean_ok),
        "bit_stable_baseline_table": bit_stable,
        "baseline_table_sha256_round_1": round_hashes[0],
        "baseline_table_sha256_round_2": round_hashes[1],
        "required_rows": list(REQUIRED_ROWS),
        "missing_rows": missing,
        "clean_worktree_required": enforce_clean_worktree,
        "clean_worktree": bool(git_state["clean_worktree"]),
        "stop_reason": "M0 gate reached; do not proceed to M1 until this report is reviewed.",
    }
    _write_json(out / "m0_gate_report.json", gate)
    _write_report(out / "M0_EVIDENCE_REPORT.md", gate, table_rows)
    return gate


def _run_one(
    out: Path,
    *,
    seed: int,
    train_steps: int,
    n_subjects: int,
    n_time: int,
    n_channels: int,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    _set_determinism(seed)
    dataset = make_synthetic_eeg_v1_dataset(seed=seed, n_subjects=n_subjects, n_time=n_time, n_channels=n_channels)
    task = build_future_forecasting_task(dataset, window_length=8, forecast_horizon=1, stride=2)
    manifest_paths = _freeze_manifests(out / "manifests", tuple(dataset.records), seed=seed)
    payload = _run_task_models(task, seed=seed, train_steps=train_steps, model_ids=RUNNER_IDS)
    _write_json(out / "task_payload.json", payload)
    _write_json(out / "manifest_index.json", manifest_paths)
    table = _write_baseline_table(out / "baseline_table.csv", payload["metrics_by_model"], payload["ranking"])
    return table


def _freeze_manifests(out: Path, records: tuple[RecordingRecord, ...], *, seed: int) -> dict[str, dict[str, str]]:
    out.mkdir(parents=True, exist_ok=True)
    dataset_records = records + tuple(_as_external_dataset_record(record) for record in records)
    manifests = {
        "patient_held_out": build_split_manifest(records, policy="subject", seed=seed),
        "site_held_out": build_split_manifest(records, policy="site", seed=seed),
        "dataset_held_out": build_split_manifest(dataset_records, policy="dataset", seed=seed),
    }
    index = {}
    for name, manifest in manifests.items():
        path = out / f"{name}.json"
        _write_json(path, _manifest_payload(manifest))
        index[name] = {"path": f"manifests/{path.name}", "sha256": _sha256_file(path)}
    return index


def _as_external_dataset_record(record: RecordingRecord) -> RecordingRecord:
    payload = asdict(record)
    payload["record_id"] = f"{record.record_id}_external"
    payload["dataset"] = f"{record.dataset}_external"
    payload["path"] = f"synthetic://{payload['record_id']}"
    payload["metadata"] = {**record.metadata, "synthetic_external_manifest_only": True}
    return RecordingRecord(**payload)


def _manifest_payload(manifest: SplitManifest) -> dict[str, Any]:
    return {
        "policy": manifest.policy,
        "seed": manifest.seed,
        "split_stage": manifest.split_stage,
        "notes": manifest.notes,
        "record_hashes": dict(sorted(manifest.record_hashes.items())),
        "train": [asdict(record) for record in manifest.train],
        "val": [asdict(record) for record in manifest.val],
        "test": [asdict(record) for record in manifest.test],
    }


def _write_baseline_table(path: Path, metrics_by_model: dict[str, dict[str, float]], ranking: list[dict[str, Any]]) -> Path:
    rank_by_model = {row["model_id"]: int(row["rank"]) for row in ranking}
    rows = []
    for model_id, metrics in metrics_by_model.items():
        display_id = DISPLAY_IDS.get(model_id, model_id)
        rows.append(
            {
                "model_id": display_id,
                "rank": rank_by_model.get(model_id, 999),
                "mse": _fmt(metrics["mse"]),
                "mae": _fmt(metrics["mae"]),
                "pearsonr": _fmt(metrics["pearsonr"]),
                "r2": _fmt(metrics["r2"]),
            }
        )
    rows.sort(key=lambda row: (int(row["rank"]), row["model_id"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model_id", "rank", "mse", "mae", "pearsonr", "r2"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_report(path: Path, gate: dict[str, Any], rows: list[dict[str, str]]) -> None:
    lines = [
        "# Kahlus Forecastability Trial 0 - M0 Evidence Report",
        "",
        f"Gate passed: `{gate['gate_passed']}`",
        f"Bit-stable baseline table: `{gate['bit_stable_baseline_table']}`",
        f"Clean worktree required: `{gate['clean_worktree_required']}`",
        f"Clean worktree observed: `{gate['clean_worktree']}`",
        f"Missing required rows: `{', '.join(gate['missing_rows']) if gate['missing_rows'] else 'none'}`",
        "",
        "## Baseline Ranking",
        "",
        "| rank | model | MSE | MAE | Pearson r | R2 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['rank']} | {row['model_id']} | {row['mse']} | {row['mae']} | {row['pearsonr']} | {row['r2']} |")
    lines.extend(
        [
            "",
            "## Gate Discipline",
            "",
            "M0 stops here. M1 should not start until the worktree is clean and this harness is accepted as the ground-truth evaluator.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _environment_payload() -> dict[str, Any]:
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except Exception as exc:  # noqa: BLE001 - environment capture should not fail the run.
        sklearn_version = f"unavailable: {exc}"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn_version,
    }


def _git_state() -> dict[str, Any]:
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "status_short": _git("status", "--short"),
        "clean_worktree": _git("status", "--short") == "",
    }


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001 - non-git runs are still useful as evidence payloads.
        return ""


def _set_determinism(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _fmt(value: float) -> str:
    return f"{float(value):.12g}"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
