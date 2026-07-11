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
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from neurotwin.models.dual_field import DualFieldConfig
    from neurotwin.transition_gym import SyntheticWorldConfig

from neurotwin.gates import evaluate_gate, write_evidence_gate
from neurotwin.models.baselines import NumpyRidgeBaseline, TorchMLPBaseline
from neurotwin.models.torch_models import TinyGRUBaseline, TinyTransformerBaseline
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
    # Optional validation split. When present, learned baselines can be selected on val MSE — the
    # same best-validation selection the KTM uses — so the recovery comparison is symmetric.
    x_val: np.ndarray | None = None
    y_val: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_val(self) -> bool:
        return self.x_val is not None and self.y_val is not None

    def shapes(self) -> dict[str, list[int]]:
        shapes = {
            "x_train": list(self.x_train.shape),
            "y_train": list(self.y_train.shape),
            "x_test": list(self.x_test.shape),
            "y_test": list(self.y_test.shape),
        }
        if self.has_val():
            shapes["x_val"] = list(self.x_val.shape)
            shapes["y_val"] = list(self.y_val.shape)
        return shapes


@dataclass
class BaselineRunResult:
    metrics_by_model: dict[str, dict[str, float]]
    ranking: list[dict[str, Any]]
    failure_reasons: list[str]
    evidence_gate: dict[str, Any]
    seed: int
    # Symmetric-selection bookkeeping. ``metrics_by_model`` carries the *selected* metrics (best-val
    # when ``select_best_val`` and a val split is present, else final-step); ``final_metrics_by_model``
    # always carries the final-step metrics as a secondary record. ``checkpoint_policy_by_model`` is
    # per-model ("best_val" | "final_step" | "fitted"); ``selection_policy`` summarizes the run.
    final_metrics_by_model: dict[str, dict[str, float]] = field(default_factory=dict)
    checkpoint_policy_by_model: dict[str, str] = field(default_factory=dict)
    selection_policy: str = "final_step"


def _flatten_window(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(x.shape[0], -1)


def _last_step(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[:, -1, :]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Canonical metric bundle (mse/mae/r2/pearson_r) for a regression prediction."""

    return {
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "pearson_r": pearsonr(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()),
    }


def retrieval_knn_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, k: int = 5) -> np.ndarray:
    """Dependency-free retrieval baseline: mean target of the k nearest train rows by L2."""

    xt = np.asarray(x_train, dtype=np.float64).reshape(x_train.shape[0], -1)
    xe = np.asarray(x_test, dtype=np.float64).reshape(x_test.shape[0], -1)
    y_train = np.asarray(y_train, dtype=np.float64)
    k = max(1, min(int(k), xt.shape[0]))
    preds = [y_train[np.argsort(np.linalg.norm(xt - q, axis=1))[:k]].mean(axis=0) for q in xe]
    return np.asarray(preds, dtype=np.float64)


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


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _train_torch_select(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    steps: int,
    sequence: bool,
    eval_every: int = 10,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
    """Train, snapshotting the lowest-val-MSE state. Returns ``(best_state, final_state, best_step)``.

    This is the baseline-side analogue of the KTM trainer's best-validation checkpoint selection,
    so a learned baseline is reported at the same kind of checkpoint the KTM is — symmetric, not
    final-step-vs-best-val.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    best_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_step = 0
    n = max(1, int(steps))
    every = max(1, int(eval_every))
    for step in range(1, n + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x)
        pred = out[:, -1, :] if sequence and out.ndim == 3 else out
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        if step % every == 0 or step == n:
            model.eval()
            with torch.no_grad():
                vout = model(x_val)
                vpred = vout[:, -1, :] if sequence and vout.ndim == 3 else vout
                vmse = float(((vpred - y_val) ** 2).mean().detach())
            if vmse < best_val:
                best_val = vmse
                best_step = step
                best_state = _clone_state(model)
    final_state = _clone_state(model)
    if best_state is None:
        best_state, best_step = final_state, n
    return best_state, final_state, best_step


def _predict_state(model: nn.Module, state: dict[str, torch.Tensor], x: torch.Tensor, sequence: bool) -> np.ndarray:
    model.load_state_dict(state)
    return _predict_torch(model, x, sequence)


def _f32(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(arr, dtype=np.float64), dtype=torch.float32)


def _asarray64(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)


_NON_ITERATIVE = frozenset({"ridge", "autoregressive_ridge", "retrieval_knn"})
_TORCH_BASELINES = frozenset({"mlp", "transformer", "ssm_fallback"})


def _build_torch_baseline(model_id: str, task: RegressionTask, seed: int):
    """Single source for a learned baseline: ``(model, x_preprocessor, sequence_flag)``.

    Both the final-step (:func:`_run_single_model`) and best-val (:func:`_run_single_model_selected`)
    paths construct their model + tensors through this factory, so each baseline's architecture and
    hyperparameters live in exactly one place. ``mlp`` flattens the window; the sequence models keep it.
    """

    set_global_seed(seed)
    channels_in = task.x_train.shape[-1]
    channels_out = task.y_train.shape[-1]
    if model_id == "mlp":
        return TorchMLPBaseline(task.x_train.shape[1] * channels_in, channels_out), _flatten_window, False
    if model_id == "transformer":
        return TinyTransformerBaseline(channels_in, channels_out, latent_dim=32, n_heads=4, n_layers=1), _asarray64, True
    if model_id == "ssm_fallback":
        return TinyGRUBaseline(channels_in, channels_out, latent_dim=32, n_layers=1), _asarray64, True
    raise RuntimeError(f"unknown torch baseline model_id: {model_id!r}")


def _run_single_model(model_id: str, task: RegressionTask, train_steps: int, seed: int) -> np.ndarray:
    """Return a final-step prediction array for ``model_id`` on ``task.x_test`` or raise on skip."""

    if model_id == "retrieval_knn":
        return retrieval_knn_predict(task.x_train, task.y_train, task.x_test)
    if model_id == "ridge":
        return _fit_ridge(_last_step(task.x_train), task.y_train, _last_step(task.x_test), alpha=1.0)
    if model_id == "autoregressive_ridge":
        return _fit_ridge(_flatten_window(task.x_train), task.y_train, _flatten_window(task.x_test), alpha=1.0)
    if model_id in _TORCH_BASELINES:
        model, xprep, sequence = _build_torch_baseline(model_id, task, seed)
        _train_torch(model, _f32(xprep(task.x_train)), _f32(task.y_train), train_steps, sequence=sequence)
        return _predict_torch(model, _f32(xprep(task.x_test)), sequence=sequence)
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


def _run_single_model_selected(
    model_id: str, task: RegressionTask, train_steps: int, seed: int, eval_every: int = 10
) -> dict[str, Any]:
    """Best-val variant: returns best-val + final-step test predictions and the checkpoint policy.

    Non-iterative baselines (ridge / retrieval) have no checkpoints, so both predictions are their
    single fitted result and the policy is ``fitted``. Learned baselines reuse the same factory and
    are reported at their lowest-val-MSE checkpoint — symmetric with the KTM.
    """

    if model_id in _NON_ITERATIVE:
        pred = _run_single_model(model_id, task, train_steps, seed)
        return {"pred_best_val": pred, "pred_final": pred, "checkpoint_policy": "fitted", "best_step": 0}
    if model_id not in _TORCH_BASELINES:
        # nfc (or anything unknown) takes the same honest skip/raise as the final-step path.
        _run_single_model(model_id, task, train_steps, seed)
        raise RuntimeError(f"unknown baseline model_id: {model_id!r}")  # pragma: no cover

    model, xprep, sequence = _build_torch_baseline(model_id, task, seed)
    best_state, final_state, best_step = _train_torch_select(
        model, _f32(xprep(task.x_train)), _f32(task.y_train),
        _f32(xprep(task.x_val)), _f32(task.y_val), train_steps, sequence=sequence, eval_every=eval_every,
    )
    xe = _f32(xprep(task.x_test))
    return {
        "pred_best_val": _predict_state(model, best_state, xe, sequence),
        "pred_final": _predict_state(model, final_state, xe, sequence),
        "checkpoint_policy": "best_val",
        "best_step": int(best_step),
    }


def selected_model_predictions(
    model_id: str, task: RegressionTask, train_steps: int, seed: int, eval_every: int = 10
) -> dict[str, Any]:
    """Public best-val + final-step test predictions + checkpoint policy for ``model_id``.

    A stable surface over the internal selected-baseline path so out-of-module callers (e.g. the
    failure-analysis diagnostic) need not reach a private helper. Reuses — never reimplements or
    weakens — the baseline training/selection path.
    """

    return _run_single_model_selected(model_id, task, train_steps, seed, eval_every)


def dual_field_regression_task(config: "DualFieldConfig | None" = None, *, window: int = 4) -> RegressionTask:
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


def transition_gym_regression_task(config: "SyntheticWorldConfig | None" = None) -> RegressionTask:
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
    val = np.asarray(bundle.splits.val_episodes, dtype=int)
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
        x_val=x[val],
        y_val=y[val],
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
    select_best_val: bool = False,
    val_eval_every: int = 10,
) -> BaselineRunResult:
    """Run baselines on a task and build the (claim-blocking) evidence gate.

    ``select_best_val`` enables symmetric best-validation selection for *learned* baselines (the
    same selection the KTM uses). It requires the task to carry a validation split; otherwise the
    runner falls back to final-step metrics and records that honestly. ``metrics_by_model`` always
    carries the *selected* metrics that the recovery comparison must use.
    """

    set_global_seed(seed)
    use_best_val = bool(select_best_val and task.has_val())
    metrics_by_model: dict[str, dict[str, float]] = {}
    final_metrics_by_model: dict[str, dict[str, float]] = {}
    checkpoint_policy_by_model: dict[str, str] = {}
    failure_reasons: list[str] = []

    with ignore_spurious_matmul_warnings():
        for model_id in models:
            try:
                if use_best_val:
                    res = _run_single_model_selected(model_id, task, train_steps, seed, val_eval_every)
                    pred_best, pred_final = res["pred_best_val"], res["pred_final"]
                    for pred in (pred_best, pred_final):
                        if pred.shape != task.y_test.shape:
                            raise ValueError(f"prediction shape {pred.shape} != target {task.y_test.shape}")
                    selected_metrics = regression_metrics(task.y_test, pred_best)
                    final_metrics = regression_metrics(task.y_test, pred_final)
                    if not all(np.isfinite(v) for v in selected_metrics.values()):
                        raise ValueError("non-finite metric produced")
                    metrics_by_model[model_id] = selected_metrics
                    final_metrics_by_model[model_id] = final_metrics
                    checkpoint_policy_by_model[model_id] = res["checkpoint_policy"]
                else:
                    pred = _run_single_model(model_id, task, train_steps, seed)
                    if pred.shape != task.y_test.shape:
                        raise ValueError(f"prediction shape {pred.shape} != target {task.y_test.shape}")
                    model_metrics = regression_metrics(task.y_test, pred)
                    if not all(np.isfinite(v) for v in model_metrics.values()):
                        raise ValueError("non-finite metric produced")
                    metrics_by_model[model_id] = model_metrics
                    final_metrics_by_model[model_id] = model_metrics
                    checkpoint_policy_by_model[model_id] = (
                        "fitted" if model_id in _NON_ITERATIVE else "final_step"
                    )
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
        final_metrics_by_model=final_metrics_by_model,
        checkpoint_policy_by_model=checkpoint_policy_by_model,
        selection_policy="symmetric_best_val" if use_best_val else "final_step",
    )


BASELINE_TABLE_FIELDS: tuple[str, ...] = ("model_id", "mse", "mae", "r2", "pearson_r", "status")


def baseline_table_rows(metrics_by_model: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """Canonical baseline-table rows from a per-model metric mapping."""

    return [
        {"model_id": model_id, "mse": m["mse"], "mae": m["mae"], "r2": m["r2"],
         "pearson_r": m["pearson_r"], "status": "completed"}
        for model_id, m in metrics_by_model.items()
    ]


def write_baseline_table(
    out_dir: str | Path,
    rows: Sequence[dict[str, Any]],
    ranking: Sequence[dict[str, Any]],
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the canonical baseline_table.{json,csv}; the single owner of that schema.

    ``extra`` merges extra keys into the JSON payload only (the CSV stays the flat table), so
    callers can annotate the JSON without forking the schema.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"rows": list(rows), "ranking": list(ranking)}
    if extra:
        payload.update(extra)
    json_path = write_json(out / "baseline_table.json", payload)
    csv_path = out / "baseline_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BASELINE_TABLE_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in BASELINE_TABLE_FIELDS})
    return {"baseline_table_json": json_path, "baseline_table_csv": csv_path}


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

    table_paths = write_baseline_table(
        out, baseline_table_rows(result.metrics_by_model), result.ranking
    )
    table_json_path = table_paths["baseline_table_json"]
    table_csv_path = table_paths["baseline_table_csv"]

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
