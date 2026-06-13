"""Shared local baseline runner for the v2 dual-field and v3 Transition Gym tasks.

Reuses the existing baselines (``models/baselines.py``), torch baselines
(``models/torch_models.py``), and metrics (``scoring/metrics.py``). Runs a fixed set of
baselines on a leakage-safe synthetic regression task and emits the artifact contract:
``metrics.json``, ``baseline_table.{csv,json}``, ``evidence_gate.json`` (unified gate),
and ``run_config.json``. Failures are captured honestly, never hidden.

PROPOSED / SYNTHETIC ONLY. No A100. No scientific-superiority claim — calibration is not
computed here, so the evidence gate correctly blocks any claim.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from neurotwin.gates import evaluate_gate, write_evidence_gate
from neurotwin.models.baselines import NumpyRidgeBaseline, TorchMLPBaseline
from neurotwin.models.torch_models import TinySSMBaseline, TinyTransformerBaseline
from neurotwin.numerics import ignore_spurious_matmul_warnings
from neurotwin.repro import set_global_seed, write_json
from neurotwin.scoring.metrics import mae, mse, pearsonr, r2_score, rank_models

DEFAULT_MODELS: tuple[str, ...] = (
    "ridge",
    "autoregressive_ridge",
    "mlp",
    "transformer",
    "ssm_fallback",
    "nfc",
)


@dataclass(frozen=True)
class RegressionTask:
    """A leakage-safe synthetic regression task consumed by the runner.

    ``x_*`` are sequence arrays ``(n, window, channels_in)``; ``y_*`` are 2D targets
    ``(n, channels_out)``. Train/test rows come from disjoint source sequences/episodes.
    """

    name: str
    branch: str
    dataset: str
    claim_scope: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def shapes(self) -> dict[str, list[int]]:
        return {
            "x_train": list(self.x_train.shape),
            "y_train": list(self.y_train.shape),
            "x_test": list(self.x_test.shape),
            "y_test": list(self.y_test.shape),
        }


@dataclass
class BaselineRunResult:
    metrics_by_model: dict[str, dict[str, float]]
    ranking: list[dict[str, Any]]
    failure_reasons: list[str]
    evidence_gate: dict[str, Any]
    seed: int


def _flatten_window(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(x.shape[0], -1)


def _last_step(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[:, -1, :]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "pearson_r": pearsonr(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()),
    }


def _fit_ridge(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    model = NumpyRidgeBaseline(alpha=alpha).fit(x_train, y_train)
    return model.predict(x_test)


def _train_torch(model: nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int, sequence: bool) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(max(1, int(steps))):
        optimizer.zero_grad()
        out = model(x)
        pred = out[:, -1, :] if sequence and out.ndim == 3 else out
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()


def _predict_torch(model: nn.Module, x: torch.Tensor, sequence: bool) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(x)
    pred = out[:, -1, :] if sequence and out.ndim == 3 else out
    return pred.detach().cpu().numpy().astype(np.float64)


def _run_single_model(model_id: str, task: RegressionTask, train_steps: int, seed: int) -> np.ndarray:
    """Return a prediction array for ``model_id`` on ``task.x_test`` or raise on skip."""

    channels_in = task.x_train.shape[-1]
    channels_out = task.y_train.shape[-1]

    if model_id == "ridge":
        return _fit_ridge(_last_step(task.x_train), task.y_train, _last_step(task.x_test), alpha=1.0)
    if model_id == "autoregressive_ridge":
        return _fit_ridge(_flatten_window(task.x_train), task.y_train, _flatten_window(task.x_test), alpha=1.0)
    if model_id == "mlp":
        set_global_seed(seed)
        model = TorchMLPBaseline(task.x_train.shape[1] * channels_in, channels_out)
        xt = torch.tensor(_flatten_window(task.x_train), dtype=torch.float32)
        yt = torch.tensor(np.asarray(task.y_train, dtype=np.float64), dtype=torch.float32)
        _train_torch(model, xt, yt, train_steps, sequence=False)
        return _predict_torch(model, torch.tensor(_flatten_window(task.x_test), dtype=torch.float32), sequence=False)
    if model_id in {"transformer", "ssm_fallback"}:
        set_global_seed(seed)
        if model_id == "transformer":
            model = TinyTransformerBaseline(channels_in, channels_out, latent_dim=32, n_heads=4, n_layers=1)
        else:
            model = TinySSMBaseline(channels_in, channels_out, latent_dim=32, n_layers=1)
        xt = torch.tensor(np.asarray(task.x_train, dtype=np.float64), dtype=torch.float32)
        yt = torch.tensor(np.asarray(task.y_train, dtype=np.float64), dtype=torch.float32)
        _train_torch(model, xt, yt, train_steps, sequence=True)
        return _predict_torch(model, torch.tensor(np.asarray(task.x_test, dtype=np.float64), dtype=torch.float32), sequence=True)
    if model_id == "nfc":
        # The Neural Field Compiler has a bespoke train/eval path and is intentionally not
        # run as a flat-regression baseline here. We confirm importability and skip honestly.
        try:
            from neurotwin.models.nfc import NeuralFieldCompiler  # noqa: F401
        except Exception as exc:  # pragma: no cover - import environment dependent
            raise RuntimeError(f"nfc baseline unavailable: {exc}") from exc
        raise RuntimeError(
            "nfc baseline skipped: NeuralFieldCompiler is importable but has a bespoke "
            "train/eval path and is not wired into the shared flat-regression runner"
        )
    raise RuntimeError(f"unknown baseline model_id: {model_id!r}")


def dual_field_regression_task(config: Any = None, *, window: int = 4) -> RegressionTask:
    """Build a leakage-safe next-step EEG forecasting task from the v2 dual-field system.

    Windows from a given sequence stay within a single train/test split (split by sequence
    index), so no window straddles the boundary.
    """

    from neurotwin.models.dual_field import DualFieldConfig, simulate_dual_field

    config = config or DualFieldConfig()
    rollout = simulate_dual_field(config)
    eeg = np.asarray(rollout.eeg, dtype=np.float64)  # (B, T, C)
    n_seq, time_steps, channels = eeg.shape
    if window >= time_steps:
        raise ValueError("window must be smaller than time_steps")

    def _windows(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for b in indices:
            for t in range(window - 1, time_steps - 1):
                xs.append(eeg[b, t - window + 1 : t + 1])
                ys.append(eeg[b, t + 1])
        return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)

    n_test = max(1, int(round(n_seq * 0.25)))
    test_seq = np.arange(n_seq - n_test, n_seq)
    train_seq = np.arange(0, n_seq - n_test)
    x_train, y_train = _windows(train_seq)
    x_test, y_test = _windows(test_seq)
    return RegressionTask(
        name="dual_field_eeg_forecasting",
        branch="v2",
        dataset="dual_field_synthetic",
        claim_scope="synthetic_dual_field_recovery",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        metadata={"window": int(window), "channels": int(channels), **rollout.metadata},
    )


def transition_gym_regression_task(config: Any = None) -> RegressionTask:
    """Build a response-profile forecasting task from the v3 Transition Gym.

    Inputs are history EEG windows; targets are the flattened response profile. Train/test
    use the gym's own leakage-checked episode splits.
    """

    from neurotwin.transition_gym import SyntheticWorldConfig, build_transition_gym

    config = config or SyntheticWorldConfig()
    bundle = build_transition_gym(config)
    x = np.asarray(bundle.history_eeg, dtype=np.float64)  # (E, L, C)
    response = np.asarray(bundle.response_eeg, dtype=np.float64)  # (E, K, H, C)
    y = response.reshape(response.shape[0], -1)  # (E, K*H*C)
    train = np.asarray(bundle.splits.train_episodes, dtype=int)
    test = np.asarray(bundle.splits.test_episodes, dtype=int)
    return RegressionTask(
        name="transition_gym_response_forecasting",
        branch="v3",
        dataset="transition_gym_synthetic",
        claim_scope="synthetic_transition_gym",
        x_train=x[train],
        y_train=y[train],
        x_test=x[test],
        y_test=y[test],
        metadata={
            "mean_commutator_gap": bundle.metadata["mean_commutator_gap"],
            "perturbation_battery_K": bundle.data_card["perturbation_battery_K"],
            "horizon_H": bundle.data_card["horizon_H"],
        },
    )


def run_baselines(
    task: RegressionTask,
    *,
    models: Sequence[str] = DEFAULT_MODELS,
    train_steps: int = 60,
    seed: int = 0,
    split_audit_passed: bool = True,
) -> BaselineRunResult:
    """Run baselines on a task and build the (claim-blocking) evidence gate."""

    set_global_seed(seed)
    metrics_by_model: dict[str, dict[str, float]] = {}
    failure_reasons: list[str] = []

    with ignore_spurious_matmul_warnings():
        for model_id in models:
            try:
                pred = _run_single_model(model_id, task, train_steps, seed)
                if pred.shape != task.y_test.shape:
                    raise ValueError(f"prediction shape {pred.shape} != target {task.y_test.shape}")
                model_metrics = _metrics(task.y_test, pred)
                if not all(np.isfinite(v) for v in model_metrics.values()):
                    raise ValueError("non-finite metric produced")
                metrics_by_model[model_id] = model_metrics
            except Exception as exc:
                failure_reasons.append(f"{model_id}: {exc}")

    if metrics_by_model:
        ranking_rows = rank_models(metrics_by_model, metric="mse", higher_is_better=False)
        ranking = [
            {"model_id": row.model_id, "metric": row.metric, "value": row.value, "rank": row.rank}
            for row in ranking_rows
        ]
    else:
        ranking = []
        failure_reasons.append("baseline ranking unavailable: no model produced finite metrics")

    finite_metrics = bool(metrics_by_model) and all(
        all(np.isfinite(v) for v in m.values()) for m in metrics_by_model.values()
    )
    gate = evaluate_gate(
        branch=task.branch,
        dataset=task.dataset,
        split_audit_passed=split_audit_passed,
        baseline_table_present=bool(metrics_by_model),
        finite_metrics=finite_metrics,
        # The shared runner does not compute calibration, so a scientific claim must be
        # blocked here. This is the correct, honest default for a baseline sweep.
        calibration_checked=False,
        claim_scope=task.claim_scope,
    )
    return BaselineRunResult(
        metrics_by_model=metrics_by_model,
        ranking=ranking,
        failure_reasons=failure_reasons,
        evidence_gate=gate,
        seed=int(seed),
    )


def write_run_artifacts(
    out_dir: str | Path,
    task: RegressionTask,
    result: BaselineRunResult,
    *,
    models: Sequence[str] = DEFAULT_MODELS,
    train_steps: int = 60,
) -> dict[str, Path]:
    """Persist metrics.json, baseline_table.{csv,json}, evidence_gate.json, run_config.json."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics_path = write_json(
        out / "metrics.json",
        {
            "task": task.name,
            "branch": task.branch,
            "dataset": task.dataset,
            "seed": result.seed,
            "metrics_by_model": result.metrics_by_model,
            "ranking": result.ranking,
            "failure_reasons": result.failure_reasons,
        },
    )

    table_rows = [
        {
            "model_id": model_id,
            "mse": m["mse"],
            "mae": m["mae"],
            "r2": m["r2"],
            "pearson_r": m["pearson_r"],
            "status": "completed",
        }
        for model_id, m in result.metrics_by_model.items()
    ]
    table_json_path = write_json(out / "baseline_table.json", {"rows": table_rows, "ranking": result.ranking})
    table_csv_path = out / "baseline_table.csv"
    fieldnames = ["model_id", "mse", "mae", "r2", "pearson_r", "status"]
    with table_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in table_rows:
            writer.writerow(row)

    gate_path = write_evidence_gate(out / "evidence_gate.json", result.evidence_gate)

    run_config_path = write_json(
        out / "run_config.json",
        {
            "task": task.name,
            "branch": task.branch,
            "dataset": task.dataset,
            "claim_scope": task.claim_scope,
            "seed": result.seed,
            "requested_models": list(models),
            "train_steps": int(train_steps),
            "shapes": task.shapes(),
            "task_metadata": task.metadata,
        },
    )

    return {
        "metrics": metrics_path,
        "baseline_table_json": table_json_path,
        "baseline_table_csv": table_csv_path,
        "evidence_gate": gate_path,
        "run_config": run_config_path,
    }
