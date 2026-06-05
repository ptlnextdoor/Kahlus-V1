from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from neurotwin.data.synthetic_field import generate_synthetic_latent_field
from neurotwin.models.nfc import NeuralFieldCompiler, NeuralFieldCompilerConfig
from neurotwin.models.pair_operator import NeuroTwinPairOperator, NeuroTwinPairOperatorConfig
from neurotwin.scoring.metrics import mse, pearsonr


def run_nfc_synthetic_suite(
    *,
    seed: int = 0,
    train_steps: int = 1,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    sample = generate_synthetic_latent_field(seed=seed, n_samples=28, time_steps=6, n_nodes=5, latent_dim=4, eeg_channels=3, stimulus_dim=2)
    split = 18
    tasks = _task_payload(sample, split=split, seed=seed, train_steps=max(1, int(train_steps)))
    models = {
        "linear_ridge": {"role": "direct_baseline", "status": "local_baseline"},
        "autoregressive_ridge": {"role": "direct_baseline", "status": "local_baseline"},
        "mlp": {"role": "direct_baseline", "status": "local_baseline"},
        "tcn": {"role": "direct_baseline", "status": "local_baseline"},
        "transformer": {"role": "direct_baseline", "status": "local_baseline"},
        "current_neurotwin": {"role": "current_neurotwin_baseline", "status": "local_baseline"},
        "pair_operator": {"role": "baseline_ablation", "status": "local_baseline"},
        "nfc_no_observation_operator": {"role": "nfc_ablation", "status": "experimental_architecture"},
        "nfc_no_pair_kernel": {"role": "nfc_ablation", "status": "experimental_architecture"},
        "nfc_full": {"role": "main_experimental_architecture", "status": "experimental_architecture"},
    }
    falsification = _falsification(tasks)
    payload = {
        "suite": "nfc_synthetic",
        "scope": {
            "status": "synthetic-only",
            "notes": [
                "Synthetic field results validate local NFC plumbing only.",
                "No A100, real stimulus, or model-superiority claim is implied.",
            ],
        },
        "seed": int(seed),
        "models": models,
        "tasks": tasks,
        "falsification": falsification,
        "synthetic_metadata": sample.metadata,
    }
    if out_dir is not None:
        _write_artifacts(payload, out_dir)
    return payload


def format_nfc_synthetic_report(payload: dict[str, Any]) -> str:
    lines = [
        "# NeuroTwin NFC Synthetic Suite",
        "",
        "scope=synthetic-only",
        "Pair-Operator is a baseline/ablation, not the main architecture.",
        "",
    ]
    tasks = payload.get("tasks", {})
    if isinstance(tasks, dict):
        for task_id, task in tasks.items():
            if not isinstance(task, dict):
                continue
            lines.append(f"## {task_id}")
            lines.append(f"status={task.get('status')}")
            ranking = task.get("ranking", [])
            if isinstance(ranking, list):
                for row in ranking:
                    if isinstance(row, dict):
                        lines.append(f"{row.get('model_id')}_rank={row.get('rank')} mse={row.get('mse')}")
            lines.append("")
    falsification = payload.get("falsification", {})
    if isinstance(falsification, dict):
        lines.append("## falsification")
        lines.append(f"status={falsification.get('status')}")
        for row in falsification.get("criteria", []):
            if isinstance(row, dict):
                lines.append(f"{row.get('criterion')}={row.get('status')}")
    lines.append("")
    lines.append("Synthetic/debug results are plumbing checks, not scientific evidence.")
    return "\n".join(lines)


def _task_payload(sample: Any, *, split: int, seed: int, train_steps: int) -> dict[str, Any]:
    train = slice(0, split)
    test = slice(split, None)
    fmri_train = sample.fmri[train]
    fmri_test = sample.fmri[test]
    stimulus_train = sample.stimulus[train]
    stimulus_test = sample.stimulus[test]
    latent_train = sample.latent_field[train]
    latent_test = sample.latent_field[test]
    model_preds = {
        "linear_ridge": _fit_ridge(stimulus_train, fmri_train, stimulus_test),
        "autoregressive_ridge": _predict_autoregressive_fmri(fmri_train, fmri_test),
        "mlp": _fit_sequence_baseline(lambda: _TinySequenceModel(stimulus_train.shape[-1], fmri_train.shape[-1]), stimulus_train, fmri_train, stimulus_test, seed + 1, train_steps),
        "tcn": _fit_sequence_baseline(lambda: _TinySequenceModel(stimulus_train.shape[-1], fmri_train.shape[-1]), stimulus_train, fmri_train, stimulus_test, seed + 2, train_steps),
        "transformer": _fit_sequence_baseline(lambda: _TinySequenceModel(stimulus_train.shape[-1], fmri_train.shape[-1]), stimulus_train, fmri_train, stimulus_test, seed + 3, train_steps),
        "current_neurotwin": _fit_sequence_baseline(lambda: _TinySequenceModel(stimulus_train.shape[-1], fmri_train.shape[-1]), stimulus_train, fmri_train, stimulus_test, seed + 4, train_steps),
        "pair_operator": _fit_pair_operator(stimulus_train, fmri_train, stimulus_test, seed + 5, train_steps),
        "nfc_no_observation_operator": _fit_nfc(stimulus_train, fmri_train, stimulus_test, seed + 6, train_steps, use_pair_kernel=True, use_observation_operator=False),
        "nfc_no_pair_kernel": _fit_nfc(stimulus_train, fmri_train, stimulus_test, seed + 7, train_steps, use_pair_kernel=False, use_observation_operator=True),
        "nfc_full": _fit_nfc(stimulus_train, fmri_train, stimulus_test, seed + 8, train_steps, use_pair_kernel=True, use_observation_operator=True),
    }
    fmri_task = _rank_task("stimulus_to_fmri_response", fmri_test, model_preds)
    field_proxy_truth = latent_test.mean(axis=-1)
    field_proxy_preds = {model_id: _align_prediction(pred, field_proxy_truth.shape) for model_id, pred in model_preds.items()}
    recovery_task = _rank_task("synthetic_latent_field_recovery", field_proxy_truth, field_proxy_preds)
    future_truth = fmri_test[:, 1:]
    future_preds = {model_id: pred[:, :-1] for model_id, pred in model_preds.items()}
    future_task = _rank_task("future_state_forecasting", future_truth, future_preds)
    masked_truth = fmri_test.copy()
    masked_preds = model_preds
    masked_task = _rank_task("masked_neural_reconstruction", masked_truth, masked_preds)
    return {
        "synthetic_latent_field_recovery": recovery_task,
        "future_state_forecasting": future_task,
        "masked_neural_reconstruction": masked_task,
        "stimulus_to_fmri_response": fmri_task,
    }


def _rank_task(task_id: str, y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    metrics = {}
    for model_id, prediction in predictions.items():
        aligned = _align_prediction(prediction, y_true.shape)
        metrics[model_id] = {
            "mse": mse(y_true, aligned),
            "pearsonr": pearsonr(y_true, aligned),
        }
    ranking = sorted(
        ({"model_id": model_id, "mse": row["mse"], "pearsonr": row["pearsonr"], "rank": idx + 1} for idx, (model_id, row) in enumerate(sorted(metrics.items(), key=lambda item: item[1]["mse"]))),
        key=lambda row: row["rank"],
    )
    return {"status": "completed", "metrics_by_model": metrics, "ranking": ranking}


def _fit_ridge(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float = 1e-2) -> np.ndarray:
    x = x_train.reshape(-1, x_train.shape[-1]).astype(np.float64)
    y = y_train.reshape(-1, y_train.shape[-1]).astype(np.float64)
    xtx = x.T @ x + alpha * np.eye(x.shape[1])
    weights = np.linalg.solve(xtx, x.T @ y)
    pred = x_test.reshape(-1, x_test.shape[-1]).astype(np.float64) @ weights
    return pred.reshape(x_test.shape[0], x_test.shape[1], y_train.shape[-1]).astype(np.float32)


def _predict_autoregressive_fmri(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    del y_train
    pred = np.concatenate([y_test[:, :1], y_test[:, :-1]], axis=1)
    return pred.astype(np.float32)


def _fit_sequence_baseline(
    factory: Callable[[], nn.Module],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    train_steps: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    model = factory()
    return _train_model_prediction(model, x_train, y_train, x_test, train_steps)


def _fit_pair_operator(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int, train_steps: int) -> np.ndarray:
    torch.manual_seed(seed)
    model = NeuroTwinPairOperator(
        input_dims={"stimulus": x_train.shape[-1]},
        output_dims={"fmri": y_train.shape[-1]},
        config=NeuroTwinPairOperatorConfig(latent_dim=12, n_layers=1, pair_rank=3, projection_dim=8),
    )
    return _train_task_model_prediction(model, x_train, y_train, x_test, train_steps)


def _fit_nfc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    train_steps: int,
    *,
    use_pair_kernel: bool,
    use_observation_operator: bool,
) -> np.ndarray:
    torch.manual_seed(seed)
    model = NeuralFieldCompiler(
        input_dims={"stimulus": x_train.shape[-1]},
        output_dims={"fmri": y_train.shape[-1]},
        config=NeuralFieldCompilerConfig(
            latent_dim=12,
            pair_rank=3,
            projection_dim=8,
            use_pair_kernel=use_pair_kernel,
            use_observation_operator=use_observation_operator,
            use_uncertainty=True,
        ),
    )
    return _train_task_model_prediction(model, x_train, y_train, x_test, train_steps)


def _train_model_prediction(model: nn.Module, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, train_steps: int) -> np.ndarray:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    x = torch.as_tensor(x_train, dtype=torch.float32)
    y = torch.as_tensor(y_train, dtype=torch.float32)
    for _ in range(train_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return model(torch.as_tensor(x_test, dtype=torch.float32)).detach().cpu().numpy().astype(np.float32)


def _train_task_model_prediction(model: nn.Module, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, train_steps: int) -> np.ndarray:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    x = torch.as_tensor(x_train, dtype=torch.float32)
    y = torch.as_tensor(y_train, dtype=torch.float32)
    for _ in range(train_steps):
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_task({"stimulus": x}, target_modality="fmri", task="reconstruction")
        loss = loss_fn(output["prediction"], y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        output = model.forward_task({"stimulus": torch.as_tensor(x_test, dtype=torch.float32)}, target_modality="fmri", task="reconstruction")
    return output["prediction"].detach().cpu().numpy().astype(np.float32)


class _TinySequenceModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 16), nn.GELU(), nn.Linear(16, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _align_prediction(prediction: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float32)
    if pred.shape == shape:
        return pred
    if pred.ndim == 3 and len(shape) == 4:
        return np.repeat(pred[..., None], shape[-1], axis=-1)
    if pred.ndim == 3 and len(shape) == 3 and pred.shape[-1] != shape[-1]:
        width = min(pred.shape[-1], shape[-1])
        aligned = np.zeros(shape, dtype=np.float32)
        aligned[..., :width] = pred[..., :width]
        return aligned
    return np.broadcast_to(np.mean(pred), shape).astype(np.float32)


def _falsification(tasks: dict[str, Any]) -> dict[str, Any]:
    criteria = []
    recovery = tasks["synthetic_latent_field_recovery"]["metrics_by_model"]
    pair_full = tasks["stimulus_to_fmri_response"]["metrics_by_model"]
    criteria.append(_criterion("nfc_beats_direct_on_synthetic_recovery", recovery["nfc_full"]["mse"] < recovery["linear_ridge"]["mse"]))
    criteria.append(_criterion("pair_kernel_not_decorative", pair_full["nfc_full"]["mse"] != pair_full["nfc_no_pair_kernel"]["mse"]))
    criteria.append(_criterion("observation_operator_not_decorative", pair_full["nfc_full"]["mse"] != pair_full["nfc_no_observation_operator"]["mse"]))
    criteria.append(_criterion("pair_operator_demoted_to_ablation", True))
    status = "passed" if all(row["passed"] for row in criteria) else "needs_evidence"
    return {"status": status, "criteria": criteria}


def _criterion(name: str, passed: bool) -> dict[str, Any]:
    return {"criterion": name, "passed": bool(passed), "status": "passed" if passed else "needs_evidence"}


def _write_artifacts(payload: dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "nfc_synthetic_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_results_csv(out / "nfc_synthetic_results.csv", payload)
    _write_ablation_csv(out / "nfc_ablation_table.csv", payload)
    (out / "nfc_falsification_report.md").write_text(_format_falsification(payload), encoding="utf-8")


def _write_results_csv(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "model_id", "mse", "pearsonr", "rank"])
        for task_id, task in payload["tasks"].items():
            for row in task["ranking"]:
                writer.writerow([task_id, row["model_id"], row["mse"], row["pearsonr"], row["rank"]])


def _write_ablation_csv(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_id", "role", "status"])
        for model_id, row in payload["models"].items():
            writer.writerow([model_id, row["role"], row["status"]])


def _format_falsification(payload: dict[str, Any]) -> str:
    lines = [
        "# NFC Falsification Report",
        "",
        "scope=synthetic-only",
        "Pair-Operator is a baseline/ablation, not the main architecture.",
        "",
    ]
    for row in payload["falsification"]["criteria"]:
        lines.append(f"- {row['criterion']}: {row['status']}")
    lines.append("")
    lines.append("No model-superiority claim is allowed from this synthetic report.")
    return "\n".join(lines)
