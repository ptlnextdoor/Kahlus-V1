from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from neurotwin.eeg_v1.dataset import EEGV1Dataset, _benchmark_status, _dataset_sampling_rate_hz, _future_windows
from neurotwin.eeg_v1.gates import build_eeg_v1_adaptation_gate
from neurotwin.models.baselines import NumpyRidgeBaseline
from neurotwin.repro import write_json
from neurotwin.scoring.metrics import mae, mse, pearsonr, r2_score, rank_models


REQUIRED_ADAPTATION_CHECKSUM_ARTIFACTS = (
    "adaptation_baseline_gap_summary.json",
    "adaptation_dataset_summary.json",
    "adaptation_evidence_gate.json",
    "adaptation_failure_reasons.json",
    "adaptation_metrics.json",
    "adaptation_report.md",
    "adaptation_run_config.json",
    "adaptation_split_audit.json",
    "adaptation_subject_baseline_gap_summary.json",
    "adaptation_subject_metrics.csv",
    "adaptation_table.csv",
    "adaptation_target_scale_context.json",
    "adaptation_verification.json",
)


@dataclass(frozen=True)
class SubjectAdaptationSplit:
    subject_id: str
    support_x: np.ndarray
    support_y: np.ndarray
    query_x: np.ndarray
    query_y: np.ndarray


@dataclass(frozen=True)
class FewShotAdaptationTask:
    dataset_id: str
    pretrain_x: np.ndarray
    pretrain_y: np.ndarray
    subject_splits: dict[str, SubjectAdaptationSplit]
    window_length: int
    forecast_horizon: int
    support_windows: int
    source: str
    benchmark_status: str
    sampling_rate_hz: float | None

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(sorted(self.subject_splits))

    @property
    def query_windows(self) -> int:
        return int(sum(split.query_x.shape[0] for split in self.subject_splits.values()))


def build_fewshot_adaptation_task(
    dataset: EEGV1Dataset,
    *,
    window_length: int = 8,
    forecast_horizon: int = 1,
    support_windows: int = 5,
) -> FewShotAdaptationTask:
    if support_windows < 1:
        raise ValueError("support_windows must be >= 1")
    by_record = {batch.recording_id: batch for batch in dataset.batches}
    pretrain_xs: list[np.ndarray] = []
    pretrain_ys: list[np.ndarray] = []
    for record in dataset.split_manifest.train:
        batch = by_record.get(record.record_id)
        if batch is None:
            continue
        x, y, _ = _future_windows(batch.signal, window_length, forecast_horizon)
        if x.size:
            pretrain_xs.append(x)
            pretrain_ys.append(y)
    subject_windows: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for record in dataset.split_manifest.test:
        batch = by_record.get(record.record_id)
        if batch is None:
            continue
        x, y, _ = _future_windows(batch.signal, window_length, forecast_horizon)
        if x.size:
            subject_windows.setdefault(record.subject_id, []).append((x, y))
    splits: dict[str, SubjectAdaptationSplit] = {}
    for subject_id, parts in subject_windows.items():
        x = np.concatenate([part[0] for part in parts], axis=0).astype(np.float32)
        y = np.concatenate([part[1] for part in parts], axis=0).astype(np.float32)
        if x.shape[0] <= support_windows:
            continue
        splits[subject_id] = SubjectAdaptationSplit(
            subject_id=subject_id,
            support_x=x[:support_windows],
            support_y=y[:support_windows],
            query_x=x[support_windows:],
            query_y=y[support_windows:],
        )
    if not pretrain_xs or not splits:
        raise ValueError("few-shot adaptation task requires train subjects and held-out query windows")
    return FewShotAdaptationTask(
        dataset_id=dataset.dataset_id,
        pretrain_x=np.concatenate(pretrain_xs, axis=0).astype(np.float32),
        pretrain_y=np.concatenate(pretrain_ys, axis=0).astype(np.float32),
        subject_splits=splits,
        window_length=int(window_length),
        forecast_horizon=int(forecast_horizon),
        support_windows=int(support_windows),
        source=dataset.source,
        benchmark_status=_benchmark_status(dataset),
        sampling_rate_hz=_dataset_sampling_rate_hz(dataset),
    )


def run_fewshot_adaptation(
    task: FewShotAdaptationTask,
    *,
    seed: int = 0,
    pretrain_steps: int = 20,
    adapt_steps: int = 10,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    base = _TinySequenceForecaster(task.pretrain_x.shape[-1], hidden_dim=24)
    _train_model(base, task.pretrain_x, task.pretrain_y, steps=pretrain_steps, trainable="all")

    method_subject_rows: list[dict[str, Any]] = []
    predictions: dict[str, list[np.ndarray]] = {
        "support_persistence": [],
        "support_ridge": [],
        "linear_probe": [],
        "bottleneck_adapter": [],
        "full_finetune": [],
    }
    truths: list[np.ndarray] = []
    for subject_id, split in task.subject_splits.items():
        truths.append(split.query_y)
        per_method = {
            "support_persistence": _predict_persistence(split.query_x),
            "support_ridge": _predict_support_ridge(split),
            "linear_probe": _predict_linear_probe(base, split, steps=adapt_steps, seed=seed),
            "bottleneck_adapter": _predict_adapter(base, split, steps=adapt_steps, seed=seed),
            "full_finetune": _predict_full_finetune(base, split, steps=adapt_steps, seed=seed),
        }
        for method, pred in per_method.items():
            predictions[method].append(pred)
            metrics = _metric_bundle(split.query_y, pred)
            method_subject_rows.append({"subject_id": subject_id, "method": method, **metrics, "n_query": split.query_y.shape[0]})

    y_true = np.concatenate(truths, axis=0)
    metrics_by_method = {
        method: _metric_bundle(y_true, np.concatenate(parts, axis=0))
        for method, parts in predictions.items()
    }
    ranking = [
        {"method": row.model_id, "metric": row.metric, "value": row.value, "rank": row.rank}
        for row in rank_models(metrics_by_method, metric="mse", higher_is_better=False)
    ]
    table = [
        {"method": method, **metrics_by_method[method], "rank": _rank_for(method, ranking)}
        for method in ("support_persistence", "support_ridge", "linear_probe", "bottleneck_adapter", "full_finetune")
    ]
    return {
        "schema": "kahlus.eeg_v1_fewshot_adaptation.v1",
        "dataset": task.dataset_id,
        "source": task.source,
        "benchmark_status": task.benchmark_status,
        "seed": int(seed),
        "pretrain_steps": int(pretrain_steps),
        "adapt_steps": int(adapt_steps),
        "support_windows": int(task.support_windows),
        "query_windows": int(task.query_windows),
        "methods": [row["method"] for row in table],
        "metrics_by_method": metrics_by_method,
        "adaptation_ranking": ranking,
        "adaptation_table": table,
        "subject_metrics": method_subject_rows,
        "claim_boundary": "benchmark_readiness_only_no_adapter_superiority_claim",
    }


def write_fewshot_adaptation_artifacts(
    out_dir: str | Path,
    *,
    task: FewShotAdaptationTask,
    result: dict[str, Any],
    split_audit: dict[str, Any],
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    finite = all(
        np.isfinite(float(value))
        for metrics in result["metrics_by_method"].values()
        for value in metrics.values()
    )
    gate = build_eeg_v1_adaptation_gate(
        dataset=task.dataset_id,
        split_audit_passed=bool(split_audit["leakage_passed"]),
        finite_metrics=finite,
        baseline_table_present=bool(result["adaptation_table"]),
        support_windows=task.support_windows,
        query_windows=task.query_windows,
    )
    dataset_summary = _adaptation_dataset_summary(task, result)
    target_scale = _adaptation_target_scale_context(task, result)
    baseline_gap = _adaptation_baseline_gap_summary(result)
    subject_baseline_gap = _adaptation_subject_baseline_gap_summary(result)
    paths = {
        "metrics": write_json(out / "adaptation_metrics.json", result),
        "evidence_gate": write_json(out / "adaptation_evidence_gate.json", gate),
        "run_config": write_json(
            out / "adaptation_run_config.json",
            {
                "dataset": task.dataset_id,
                "data_source": task.source,
                "benchmark_status": task.benchmark_status,
                "seed": int(result["seed"]),
                "pretrain_steps": int(result["pretrain_steps"]),
                "adapt_steps": int(result["adapt_steps"]),
                "window_length": task.window_length,
                "forecast_horizon": task.forecast_horizon,
                "sampling_rate_hz": task.sampling_rate_hz,
                "support_windows": task.support_windows,
                "query_windows": int(result["query_windows"]),
                "methods": result["methods"],
                "claim_scope": gate["claim_scope"],
                "implementation_note": "bottleneck_adapter is a local adapter baseline, not an imported LoRA implementation",
            },
        ),
        "dataset_summary": write_json(out / "adaptation_dataset_summary.json", dataset_summary),
        "target_scale_context": write_json(out / "adaptation_target_scale_context.json", target_scale),
        "baseline_gap_summary": write_json(out / "adaptation_baseline_gap_summary.json", baseline_gap),
        "subject_baseline_gap_summary": write_json(
            out / "adaptation_subject_baseline_gap_summary.json",
            subject_baseline_gap,
        ),
        "verification": write_json(
            out / "adaptation_verification.json",
            _adaptation_verification_payload(out_dir),
        ),
        "split_audit": write_json(out / "adaptation_split_audit.json", split_audit),
        "failure_reasons": write_json(
            out / "adaptation_failure_reasons.json",
            {
                "gate_failures": list(gate["failure_reasons"]),
                "split_audit_failures": list(split_audit["failure_reasons"]),
            },
        ),
    }
    paths["adaptation_table"] = _write_rows(out / "adaptation_table.csv", result["adaptation_table"])
    paths["subject_metrics"] = _write_rows(out / "adaptation_subject_metrics.csv", result["subject_metrics"])
    report_path = out / "adaptation_report.md"
    report_path.write_text(
        _format_adaptation_report(
            result,
            gate,
            dataset_summary,
            split_audit,
            target_scale,
            baseline_gap,
            subject_baseline_gap,
        ),
        encoding="utf-8",
    )
    paths["report"] = report_path
    paths["checksum_manifest"] = _write_adaptation_checksum_manifest(
        out / "adaptation_checksum_manifest.json",
        [paths[key] for key in sorted(paths)],
    )
    return paths


def audit_adaptation_checksum_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    root = Path(artifact_dir)
    manifest_path = root / "adaptation_checksum_manifest.json"
    failure_reasons: list[str] = []
    if not manifest_path.exists():
        return {
            "schema": "kahlus.eeg_v1_fewshot_adaptation_checksum_audit.v1",
            "passed": False,
            "artifact_dir": str(root),
            "manifest": str(manifest_path),
            "artifacts_checked": 0,
            "failure_reasons": ["missing_manifest:adaptation_checksum_manifest.json"],
        }

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "schema": "kahlus.eeg_v1_fewshot_adaptation_checksum_audit.v1",
            "passed": False,
            "artifact_dir": str(root),
            "manifest": str(manifest_path),
            "artifacts_checked": 0,
            "failure_reasons": ["invalid_manifest_json:adaptation_checksum_manifest.json"],
        }

    if manifest.get("schema") != "kahlus.eeg_v1_fewshot_adaptation_checksums.v1":
        failure_reasons.append("unsupported_manifest_schema")
    if manifest.get("algorithm") != "sha256":
        failure_reasons.append("unsupported_checksum_algorithm")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        artifacts = []
        failure_reasons.append("invalid_artifacts_list")

    checked = 0
    manifest_paths: set[str] = set()
    required_manifest_paths = set(REQUIRED_ADAPTATION_CHECKSUM_ARTIFACTS)
    for row in artifacts:
        if not isinstance(row, dict):
            failure_reasons.append("invalid_artifact_row")
            continue
        rel_path = row.get("path")
        expected_bytes = row.get("bytes")
        expected_digest = row.get("sha256")
        if not isinstance(rel_path, str) or not rel_path or "/" in rel_path or "\\" in rel_path:
            failure_reasons.append("invalid_artifact_path")
            continue
        if rel_path in manifest_paths:
            failure_reasons.append(f"duplicate_manifest_entry:{rel_path}")
        manifest_paths.add(rel_path)
        if rel_path not in required_manifest_paths:
            failure_reasons.append(f"unexpected_manifest_entry:{rel_path}")
        artifact_path = root / rel_path
        if not artifact_path.exists():
            failure_reasons.append(f"missing_artifact:{rel_path}")
            continue
        payload = artifact_path.read_bytes()
        checked += 1
        if expected_bytes != len(payload):
            failure_reasons.append(f"bytes_mismatch:{rel_path}")
        if expected_digest != hashlib.sha256(payload).hexdigest():
            failure_reasons.append(f"checksum_mismatch:{rel_path}")

    for required_path in REQUIRED_ADAPTATION_CHECKSUM_ARTIFACTS:
        if required_path not in manifest_paths:
            failure_reasons.append(f"missing_manifest_entry:{required_path}")
    _validate_adaptation_verification_sidecar(root, failure_reasons)

    return {
        "schema": "kahlus.eeg_v1_fewshot_adaptation_checksum_audit.v1",
        "passed": not failure_reasons,
        "artifact_dir": str(root),
        "manifest": str(manifest_path),
        "artifacts_checked": checked,
        "failure_reasons": failure_reasons,
    }


def _validate_adaptation_verification_sidecar(root: Path, failure_reasons: list[str]) -> None:
    sidecar_path = root / "adaptation_verification.json"
    if not sidecar_path.exists():
        failure_reasons.append("missing_artifact:adaptation_verification.json")
        return
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        failure_reasons.append("invalid_verification_json")
        return
    if sidecar.get("schema") != "kahlus.eeg_v1_fewshot_adaptation_verification.v1":
        failure_reasons.append("invalid_verification_schema")
    if sidecar.get("execution_lane") != "local_cpu_or_single_process_only":
        failure_reasons.append("invalid_verification_execution_lane")
    if sidecar.get("a100_jobs_launched") is not False:
        failure_reasons.append("invalid_verification_a100_jobs_launched")
    if sidecar.get("checksum_manifest") != "adaptation_checksum_manifest.json":
        failure_reasons.append("invalid_verification_checksum_manifest")
    expected_command = (
        "PYTHONPATH=src python3 scripts/audit_eeg_v1_adaptation_checksums.py "
        f"--artifact-dir {root}"
    )
    if sidecar.get("checksum_audit_command") != expected_command:
        failure_reasons.append("invalid_verification_checksum_audit_command")


class _TinySequenceForecaster(nn.Module):
    def __init__(self, channels: int, hidden_dim: int = 24) -> None:
        super().__init__()
        self.encoder = nn.GRU(channels, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, _ = self.encoder(x)
        return self.readout(latent)


class _AdapterSequenceForecaster(nn.Module):
    def __init__(self, base: _TinySequenceForecaster, bottleneck: int = 8) -> None:
        super().__init__()
        self.encoder = base.encoder
        self.readout = base.readout
        hidden = base.readout.in_features
        self.adapter = nn.Sequential(nn.Linear(hidden, bottleneck), nn.Tanh(), nn.Linear(bottleneck, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, _ = self.encoder(x)
        adapted = latent + self.adapter(latent)
        return self.readout(adapted)


def _train_model(model: nn.Module, x: np.ndarray, y: np.ndarray, *, steps: int, trainable: str) -> None:
    if trainable == "readout":
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("readout.")
    elif trainable == "adapter":
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("adapter.")
    else:
        for param in model.parameters():
            param.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters for adaptation method")
    optimizer = torch.optim.AdamW(params, lr=2e-2)
    loss_fn = nn.MSELoss()
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    model.train()
    for _ in range(max(1, int(steps))):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xt), yt)
        loss.backward()
        optimizer.step()


def _predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(x, dtype=torch.float32)).detach().cpu().numpy()
    return pred.astype(np.float64)


def _clone_base(base: _TinySequenceForecaster) -> _TinySequenceForecaster:
    clone = _TinySequenceForecaster(base.readout.out_features, hidden_dim=base.readout.in_features)
    clone.load_state_dict(base.state_dict())
    return clone


def _predict_persistence(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).copy()


def _predict_support_ridge(split: SubjectAdaptationSplit) -> np.ndarray:
    model = NumpyRidgeBaseline(alpha=1.0).fit(_flat(split.support_x), _flat(split.support_y))
    pred = model.predict(_flat(split.query_x))
    return pred.reshape(split.query_y.shape).astype(np.float64)


def _predict_linear_probe(base: _TinySequenceForecaster, split: SubjectAdaptationSplit, *, steps: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed + 11)
    model = _clone_base(base)
    _train_model(model, split.support_x, split.support_y, steps=steps, trainable="readout")
    return _predict(model, split.query_x)


def _predict_adapter(base: _TinySequenceForecaster, split: SubjectAdaptationSplit, *, steps: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed + 17)
    model = _AdapterSequenceForecaster(_clone_base(base))
    _train_model(model, split.support_x, split.support_y, steps=steps, trainable="adapter")
    return _predict(model, split.query_x)


def _predict_full_finetune(base: _TinySequenceForecaster, split: SubjectAdaptationSplit, *, steps: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed + 23)
    model = _clone_base(base)
    _train_model(model, split.support_x, split.support_y, steps=steps, trainable="all")
    return _predict(model, split.query_x)


def _flat(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64).reshape(arr.shape[0], -1)


def _metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "pearsonr": pearsonr(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def _rank_for(method: str, ranking: list[dict[str, Any]]) -> int:
    for row in ranking:
        if row["method"] == method:
            return int(row["rank"])
    return 0


def _adaptation_dataset_summary(task: FewShotAdaptationTask, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": task.dataset_id,
        "data_source": task.source,
        "benchmark_status": task.benchmark_status,
        "n_pretrain_windows": int(task.pretrain_x.shape[0]),
        "n_adaptation_subjects": int(len(task.subject_splits)),
        "support_windows_per_subject": int(task.support_windows),
        "n_support_windows": int(task.support_windows * len(task.subject_splits)),
        "n_query_windows": int(task.query_windows),
        "window_length": int(task.window_length),
        "forecast_horizon": int(task.forecast_horizon),
        "sampling_rate_hz": task.sampling_rate_hz,
        "n_channels": int(task.pretrain_x.shape[-1]),
        "method_count": int(len(result["methods"])),
    }


def _adaptation_target_scale_context(task: FewShotAdaptationTask, result: dict[str, Any]) -> dict[str, Any]:
    y_true = np.concatenate([split.query_y for split in task.subject_splits.values()], axis=0).astype(np.float64)
    target_std = float(np.std(y_true))
    target_variance = float(np.var(y_true))
    target_units = "normalized_eeg_fixture_units" if task.source == "synthetic_fixture" else "dataset_units_unknown_or_preprocessed"
    return {
        "target_units": target_units,
        "scale_note": "MSE is reported in squared target units; compare it to target variance before interpreting magnitude.",
        "target_mean": float(np.mean(y_true)),
        "target_std": target_std,
        "target_variance": target_variance,
        "target_min": float(np.min(y_true)),
        "target_max": float(np.max(y_true)),
        "methods": {
            method: _scale_context_for_method(metrics, target_std=target_std, target_variance=target_variance)
            for method, metrics in result["metrics_by_method"].items()
        },
    }


def _scale_context_for_method(metrics: dict[str, float], *, target_std: float, target_variance: float) -> dict[str, float | None]:
    method_mse = metrics.get("mse")
    if method_mse is None or not np.isfinite(float(method_mse)):
        return {"rmse": None, "mse_relative_to_target_variance": None, "rmse_relative_to_target_std": None}
    rmse = float(np.sqrt(float(method_mse)))
    return {
        "rmse": rmse,
        "mse_relative_to_target_variance": float(method_mse) / target_variance if target_variance > 0 else None,
        "rmse_relative_to_target_std": rmse / target_std if target_std > 0 else None,
    }


def _adaptation_baseline_gap_summary(result: dict[str, Any]) -> dict[str, Any]:
    metrics_by_method = result["metrics_by_method"]
    support_methods = ("support_persistence", "support_ridge")
    adaptation_methods = ("linear_probe", "bottleneck_adapter", "full_finetune")
    best_support = min(support_methods, key=lambda method: float(metrics_by_method[method]["mse"]))
    best_adaptation = min(adaptation_methods, key=lambda method: float(metrics_by_method[method]["mse"]))
    best_support_mse = float(metrics_by_method[best_support]["mse"])
    best_adaptation_mse = float(metrics_by_method[best_adaptation]["mse"])
    delta = best_adaptation_mse - best_support_mse
    return {
        "best_support_baseline": best_support,
        "best_support_baseline_mse": best_support_mse,
        "best_adaptation_method": best_adaptation,
        "best_adaptation_method_mse": best_adaptation_mse,
        "adaptation_vs_best_support_baseline_mse_delta": float(delta),
        "adaptation_beats_best_support_baseline": bool(delta < 0.0),
        "interpretation": "Baselines are results; adaptation methods must beat the best support baseline before any adapter-win discussion.",
    }


def _adaptation_subject_baseline_gap_summary(result: dict[str, Any]) -> dict[str, Any]:
    support_methods = ("support_persistence", "support_ridge")
    adaptation_methods = ("linear_probe", "bottleneck_adapter", "full_finetune")
    by_subject: dict[str, dict[str, dict[str, Any]]] = {}
    for row in result.get("subject_metrics", ()):
        by_subject.setdefault(str(row["subject_id"]), {})[str(row["method"])] = row

    subjects: list[dict[str, Any]] = []
    for subject_id in sorted(by_subject):
        rows = by_subject[subject_id]
        best_support = min(support_methods, key=lambda method: float(rows[method]["mse"]))
        best_adaptation = min(adaptation_methods, key=lambda method: float(rows[method]["mse"]))
        best_support_mse = float(rows[best_support]["mse"])
        best_adaptation_mse = float(rows[best_adaptation]["mse"])
        delta = best_adaptation_mse - best_support_mse
        subjects.append(
            {
                "subject_id": subject_id,
                "n_query": int(rows[best_support]["n_query"]),
                "best_support_baseline": best_support,
                "best_support_baseline_mse": best_support_mse,
                "best_adaptation_method": best_adaptation,
                "best_adaptation_method_mse": best_adaptation_mse,
                "adaptation_vs_best_support_baseline_mse_delta": float(delta),
                "adaptation_beats_best_support_baseline": bool(delta < 0.0),
            }
        )
    wins = sum(1 for row in subjects if row["adaptation_beats_best_support_baseline"])
    return {
        "n_subjects": len(subjects),
        "subjects_where_adaptation_beats_best_support_baseline": int(wins),
        "subjects": subjects,
        "interpretation": "Subject-level adapter wins require beating each subject's best support baseline first.",
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    names = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in names})
    return path


def _format_adaptation_report(
    result: dict[str, Any],
    gate: dict[str, Any],
    dataset_summary: dict[str, Any],
    split_audit: dict[str, Any],
    target_scale: dict[str, Any],
    baseline_gap: dict[str, Any],
    subject_baseline_gap: dict[str, Any],
) -> str:
    gate_criteria = gate["gate_criteria"]
    subject_metric_rows = len(result.get("subject_metrics", ()))
    lines = [
        "# Kahlus v1 Few-Shot Adaptation Report",
        "",
        f"- dataset: {result['dataset']}",
        f"- source: {result['source']}",
        f"- benchmark_status: {result['benchmark_status']}",
        f"- claim_scope: {gate['claim_scope']}",
        f"- scientific_claim_allowed: {gate['scientific_claim_allowed']}",
        f"- support_windows: {result['support_windows']}",
        f"- query_windows: {result['query_windows']}",
        "",
        "## Artifact Index",
        "",
        "| artifact | purpose |",
        "| --- | --- |",
        *_adaptation_artifact_index_rows(),
        "",
        "## Checksum Audit",
        "",
        "- checksum_manifest: adaptation_checksum_manifest.json",
        "- verification_sidecar: adaptation_verification.json",
        "- command: `PYTHONPATH=src python3 scripts/audit_eeg_v1_adaptation_checksums.py --artifact-dir <artifact-dir>`",
        "- `<artifact-dir>` is the directory containing this report and the listed adaptation artifacts.",
        "",
        "## Run Config",
        "",
        f"- seed: {result['seed']}",
        f"- pretrain_steps: {result['pretrain_steps']}",
        f"- adapt_steps: {result['adapt_steps']}",
        f"- window_length: {dataset_summary['window_length']}",
        f"- forecast_horizon: {dataset_summary['forecast_horizon']}",
        f"- sampling_rate_hz: {_report_value(dataset_summary.get('sampling_rate_hz'))}",
        f"- support_windows: {result['support_windows']}",
        f"- query_windows: {result['query_windows']}",
        "",
        "## Dataset Summary",
        "",
        f"- n_pretrain_windows: {dataset_summary['n_pretrain_windows']}",
        f"- n_adaptation_subjects: {dataset_summary['n_adaptation_subjects']}",
        f"- support_windows_per_subject: {dataset_summary['support_windows_per_subject']}",
        f"- n_support_windows: {dataset_summary['n_support_windows']}",
        f"- n_query_windows: {dataset_summary['n_query_windows']}",
        f"- window_length: {dataset_summary['window_length']}",
        f"- forecast_horizon: {dataset_summary['forecast_horizon']}",
        f"- sampling_rate_hz: {_report_value(dataset_summary.get('sampling_rate_hz'))}",
        f"- n_channels: {dataset_summary['n_channels']}",
        f"- method_count: {dataset_summary['method_count']}",
        "",
        "## Target Scale Context",
        "",
        f"- target_units: {target_scale['target_units']}",
        f"- target_std: {_report_value(target_scale.get('target_std'))}",
        f"- target_variance: {_report_value(target_scale.get('target_variance'))}",
        f"- scale_note: {target_scale['scale_note']}",
        "",
        "| method | rmse | mse_relative_to_target_variance | rmse_relative_to_target_std |",
        "| --- | --- | --- | --- |",
        *_target_scale_rows(target_scale.get("methods", {})),
        "",
        "## Baseline Gap Summary",
        "",
        f"- best_support_baseline: {baseline_gap['best_support_baseline']}",
        f"- best_support_baseline_mse: {_report_value(baseline_gap.get('best_support_baseline_mse'))}",
        f"- best_adaptation_method: {baseline_gap['best_adaptation_method']}",
        f"- best_adaptation_method_mse: {_report_value(baseline_gap.get('best_adaptation_method_mse'))}",
        f"- adaptation_vs_best_support_baseline_mse_delta: {_report_value(baseline_gap.get('adaptation_vs_best_support_baseline_mse_delta'))}",
        f"- adaptation_beats_best_support_baseline: {baseline_gap['adaptation_beats_best_support_baseline']}",
        f"- interpretation: {baseline_gap['interpretation']}",
        "",
        "## Per-Subject Baseline Gap Summary",
        "",
        f"- n_subjects: {subject_baseline_gap['n_subjects']}",
        f"- subjects_where_adaptation_beats_best_support_baseline: {subject_baseline_gap['subjects_where_adaptation_beats_best_support_baseline']}",
        f"- interpretation: {subject_baseline_gap['interpretation']}",
        "",
        "| subject_id | best_support_baseline | best_adaptation_method | adaptation_vs_best_support_baseline_mse_delta | adaptation_beats_best_support_baseline |",
        "| --- | --- | --- | --- | --- |",
        *_subject_baseline_gap_rows(subject_baseline_gap.get("subjects", ())),
        "",
        "## Metric Breakdown Summary",
        "",
        f"- adaptation_subject_rows: {subject_metric_rows}",
        "- detailed_sidecars: adaptation_subject_metrics.csv",
        "",
        "## Split Audit",
        "",
        f"- split_type: {split_audit['split_type']}",
        f"- leakage_passed: {split_audit['leakage_passed']}",
        f"- subject_overlap: {split_audit['subject_overlap']}",
        f"- window_overlap: {split_audit['window_overlap']}",
        f"- failure_reasons: {len(split_audit['failure_reasons'])}",
        "",
        "## Evidence Gate Criteria",
        "",
        f"- min_support_windows: {gate_criteria['min_support_windows']}",
        f"- min_query_windows: {gate_criteria['min_query_windows']}",
        f"- requires_split_audit_passed: {gate_criteria['requires_split_audit_passed']}",
        f"- requires_baseline_table_present: {gate_criteria['requires_baseline_table_present']}",
        f"- requires_finite_metrics: {gate_criteria['requires_finite_metrics']}",
        f"- requires_calibration_checked: {gate_criteria['requires_calibration_checked']}",
        f"- allowed_claim_scope: {gate_criteria['allowed_claim_scope']}",
        "",
        "## Method Order",
        "",
        "| order | method | group |",
        "| --- | --- | --- |",
        *_method_order_rows(result["methods"]),
        "",
        "## Ranking",
        "",
        "| rank | method | mse |",
        "| --- | --- | --- |",
    ]
    for row in result["adaptation_ranking"]:
        lines.append(f"| {row['rank']} | {row['method']} | {float(row['value']):.6g} |")
    if gate["failure_reasons"]:
        lines.extend(
            [
                "",
                "## Evidence Gate Failures",
                "",
                *[f"- {reason}" for reason in gate["failure_reasons"]],
            ]
        )
    if split_audit["failure_reasons"]:
        lines.extend(
            [
                "",
                "## Split Audit Failures",
                "",
                *[f"- {reason}" for reason in split_audit["failure_reasons"]],
            ]
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- Baselines are listed first and are part of the result.",
            "- `bottleneck_adapter` is a local lightweight adapter baseline, not a LoRA claim.",
            "- This report does not claim adapter superiority, clinical utility, diagnosis, treatment, foundation-model status, or SOTA.",
        ]
    )
    if result.get("benchmark_status") == "local_manifest_not_public_hbn_benchmark":
        lines.append("- HBN-EEG inputs came from a local manifest; not a public HBN benchmark result.")
    return "\n".join(lines) + "\n"


def _adaptation_artifact_index_rows() -> list[str]:
    rows = [
        ("adaptation_metrics.json", "aggregate metrics, ranking, and subject metrics payload"),
        ("adaptation_table.csv", "method-level ranking table"),
        ("adaptation_subject_metrics.csv", "subject-level metrics by method"),
        ("adaptation_evidence_gate.json", "benchmark-readiness gate decision and criteria"),
        ("adaptation_run_config.json", "replay-critical run configuration"),
        ("adaptation_dataset_summary.json", "bounded dataset and window geometry summary"),
        ("adaptation_split_audit.json", "subject-held-out leakage audit"),
        ("adaptation_failure_reasons.json", "gate and split-audit failure sidecar"),
        ("adaptation_target_scale_context.json", "target-scale context for normalized MSE interpretation"),
        ("adaptation_baseline_gap_summary.json", "best support baseline vs best adaptation method gap"),
        ("adaptation_subject_baseline_gap_summary.json", "per-subject support-baseline vs adapter gap"),
        ("adaptation_verification.json", "local verification lane and checksum-audit command"),
        ("adaptation_checksum_manifest.json", "SHA-256 manifest for the emitted evidence artifacts"),
    ]
    return [f"| {artifact} | {purpose} |" for artifact, purpose in rows]


def _adaptation_verification_payload(out_dir: str | Path) -> dict[str, Any]:
    return {
        "schema": "kahlus.eeg_v1_fewshot_adaptation_verification.v1",
        "execution_lane": "local_cpu_or_single_process_only",
        "a100_jobs_launched": False,
        "checksum_manifest": "adaptation_checksum_manifest.json",
        "checksum_audit_command": (
            "PYTHONPATH=src python3 scripts/audit_eeg_v1_adaptation_checksums.py "
            f"--artifact-dir {out_dir}"
        ),
    }


def _write_adaptation_checksum_manifest(path: Path, artifacts: list[Path]) -> Path:
    rows = []
    for artifact in sorted(artifacts, key=lambda item: item.name):
        payload = artifact.read_bytes()
        rows.append(
            {
                "path": artifact.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return write_json(
        path,
        {
            "schema": "kahlus.eeg_v1_fewshot_adaptation_checksums.v1",
            "algorithm": "sha256",
            "artifacts": rows,
        },
    )


def _method_order_rows(methods: list[str]) -> list[str]:
    baseline_methods = {"support_persistence", "support_ridge"}
    return [
        f"| {idx} | {method} | {'baseline' if method in baseline_methods else 'adaptation'} |"
        for idx, method in enumerate(methods, start=1)
    ]


def _target_scale_rows(methods: dict[str, dict[str, float | None]]) -> list[str]:
    return [
        (
            f"| {method} | {_report_value(values.get('rmse'))} | "
            f"{_report_value(values.get('mse_relative_to_target_variance'))} | "
            f"{_report_value(values.get('rmse_relative_to_target_std'))} |"
        )
        for method, values in methods.items()
    ]


def _subject_baseline_gap_rows(rows: Any) -> list[str]:
    return [
        (
            f"| {row['subject_id']} | {row['best_support_baseline']} | "
            f"{row['best_adaptation_method']} | "
            f"{_report_value(row.get('adaptation_vs_best_support_baseline_mse_delta'))} | "
            f"{row['adaptation_beats_best_support_baseline']} |"
        )
        for row in rows
    ]


def _report_value(value: object) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.6g}" if np.isfinite(numeric) else ""
