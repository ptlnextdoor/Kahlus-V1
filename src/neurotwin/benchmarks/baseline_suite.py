from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypedDict

import numpy as np
import torch
from torch import nn

from neurotwin.contracts.paper_mode import CANONICAL_REQUIRED_SEEDS
from neurotwin.data.prepared_tasks import SupervisedWindowTask
from neurotwin.models.baselines import NumpyRidgeBaseline, TorchMLPBaseline, TorchTCNBaseline
from neurotwin.models.torch_models import (
    NeuralStateSpaceTranslator,
    NeuralStateSpaceTranslatorConfig,
    TinySSMBaseline,
    TinyTransformerBaseline,
)
from neurotwin.models.tribe_style import TribeStyleStimulusEncoder
from neurotwin.scoring.metrics import (
    bandpower_error,
    bootstrap_ci,
    mae,
    mse,
    pearsonr,
    r2_score,
    rank_models,
    regionwise_pearsonr,
    spectral_error,
    spearmanr,
)


@dataclass(frozen=True)
class BaselineFailure:
    model_id: str
    task_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"model_id": self.model_id, "task_id": self.task_id, "reason": self.reason}


class ScopePayload(TypedDict):
    status: str
    notes: list[str]


class RankingPayload(TypedDict):
    model_id: str
    metric: str
    value: float
    rank: int


class AggregateRankPayload(TypedDict):
    model_id: str
    mean_rank: float
    tasks_ranked: int


class AggregatePayload(TypedDict):
    selection_metric: str
    higher_is_better: bool
    aggregate_rank: list[AggregateRankPayload]


class TaskPayload(TypedDict):
    status: str
    source_modality: str
    target_modality: str
    metrics_by_model: dict[str, dict[str, float]]
    ranking: list[RankingPayload]
    failures: list[dict[str, str]]
    notes: list[str]


class PaperModeContractPayload(TypedDict):
    required_seeds: list[int]
    require_ci: bool
    notes: list[str]


class BaselineSuitePayload(TypedDict):
    scope: ScopePayload
    tasks: dict[str, TaskPayload]
    aggregate: AggregatePayload
    seed: int
    seeds: list[int]
    benchmark_contract: PaperModeContractPayload
    baseline_catalog: list[dict[str, object]]
    baseline_failures: list[dict[str, str]]


class PreparedTaskPayload(TypedDict, total=False):
    status: str
    source_modality: str
    target_modality: str
    metrics: dict[str, float]
    metrics_by_model: dict[str, dict[str, float]]
    ranking: list[RankingPayload]
    failures: list[dict[str, str]]
    notes: list[str]


class PreparedDataPayload(TypedDict):
    event_manifest: str
    split_manifest: str
    event_summary: dict[str, object]
    window_length: int
    stride: int
    skipped_tasks: list[dict[str, str]]


class PreparedPaperModeContractPayload(TypedDict):
    required_seeds: list[int]
    observed_seeds: list[int]
    require_ci: bool
    gate_status: str


class PreparedAggregateRankPayload(AggregateRankPayload, total=False):
    std_rank: float
    n_seeds: int


class PreparedAggregatePayload(TypedDict):
    selection_metric: str
    higher_is_better: bool
    aggregate_rank: list[PreparedAggregateRankPayload]


class PreparedBaselineSuitePayload(TypedDict):
    scope: ScopePayload
    tasks: dict[str, PreparedTaskPayload]
    aggregate: PreparedAggregatePayload
    seed: int
    seeds: list[int]
    benchmark_contract: PaperModeContractPayload
    baseline_catalog: list[dict[str, object]]
    baseline_failures: list[dict[str, str]]
    prepared_data: PreparedDataPayload
    paper_mode_contract: PreparedPaperModeContractPayload


class SeedAggregatePayload(TypedDict):
    task_id: str
    model_id: str
    metric: str
    mean: float
    std: float
    ci_low: float
    ci_high: float
    n_seeds: int


class SeedAggregatedTaskPayload(TypedDict):
    status: str
    source_modality: str | None
    target_modality: str | None
    metrics: dict[str, float]
    metrics_by_model: dict[str, dict[str, float]]
    ranking: list[RankingPayload]
    failures: list[dict[str, str]]
    notes: list[str]


class PreparedPaperModePayload(TypedDict):
    scope: ScopePayload
    tasks: dict[str, SeedAggregatedTaskPayload]
    aggregate: PreparedAggregatePayload
    seed: int
    seeds: list[int]
    benchmark_contract: PaperModeContractPayload
    baseline_catalog: list[dict[str, object]]
    baseline_failures: list[dict[str, str]]
    prepared_data: PreparedDataPayload
    paper_mode_contract: PreparedPaperModeContractPayload
    seed_results: list[PreparedBaselineSuitePayload]
    seed_aggregate: list[SeedAggregatePayload]
    representative_seed_tasks: dict[str, PreparedTaskPayload]


BaselineRunner = Callable[[SupervisedWindowTask, int, int], np.ndarray]
TaskAvailability = Callable[[SupervisedWindowTask], bool]
CatalogStatus = Callable[[set[str], set[str]], str]


@dataclass(frozen=True)
class BaselineCatalogEntry:
    model_id: str
    display_name: str
    status: str | CatalogStatus
    notes: str
    upstream_reference: str | None = None
    exact_reproduction: bool = False
    uses_upstream_code: bool = False
    uses_upstream_weights: bool = False

    def catalog_row(self, task_ids: set[str], modalities: set[str]) -> dict[str, object]:
        status = self.status(task_ids, modalities) if callable(self.status) else self.status
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "status": status,
            "notes": self.notes,
            "upstream_reference": self.upstream_reference,
            "exact_reproduction": self.exact_reproduction,
            "uses_upstream_code": self.uses_upstream_code,
            "uses_upstream_weights": self.uses_upstream_weights,
        }


@dataclass(frozen=True)
class ExecutableBaselineRunner:
    model_id: str
    runner: BaselineRunner
    available_for_task: TaskAvailability = lambda task: True

    def supports(self, task: SupervisedWindowTask) -> bool:
        return self.available_for_task(task)

    def predict(self, task: SupervisedWindowTask, seed: int, train_steps: int) -> np.ndarray:
        return self.runner(task, seed, train_steps)


def run_synthetic_baseline_suite(seed: int = 0, train_steps: int = 10) -> BaselineSuitePayload:
    """Run tiny local baselines on paired synthetic windows.

    This is intentionally a plumbing benchmark. It validates that all local
    baselines see identical tensor shapes and metrics under one task surface.
    """

    data = _make_paired_windows(seed=seed)
    tasks = (
        _future_task(data),
        _masked_reconstruction_task(data, seed=seed),
        _cross_modal_task(data),
    )
    return run_supervised_window_tasks(
        tasks,
        seed=seed,
        train_steps=train_steps,
        scope_status="synthetic-only",
        scope_notes=(
            "Validates baseline/task plumbing only.",
            "Do not interpret these rankings as scientific evidence.",
            "Real benchmark claims require prepared public data, strict held-out splits, and bootstrap CIs.",
        ),
    )


def run_supervised_window_tasks(
    tasks: tuple[SupervisedWindowTask, ...],
    seed: int = 0,
    train_steps: int = 10,
    scope_status: str = "prepared-data",
    scope_notes: tuple[str, ...] = (),
) -> BaselineSuitePayload:
    task_payloads: dict[str, TaskPayload] = {}
    rank_accumulator: dict[str, list[int]] = {}
    all_failures: list[dict[str, str]] = []
    for task in tasks:
        task_result = _run_task_models(task, seed=seed, train_steps=train_steps)
        task_payloads[task.task_id] = task_result
        all_failures.extend(task_result["failures"])
        for row in task_result["ranking"]:
            rank_accumulator.setdefault(str(row["model_id"]), []).append(int(row["rank"]))

    aggregate_rank = sorted(
        (
            {
                "model_id": model_id,
                "mean_rank": float(np.mean(ranks)),
                "tasks_ranked": len(ranks),
            }
            for model_id, ranks in rank_accumulator.items()
        ),
        key=lambda row: (float(row["mean_rank"]), str(row["model_id"])),
    )
    return {
        "scope": {
            "status": scope_status,
            "notes": list(scope_notes),
        },
        "tasks": task_payloads,
        "aggregate": {
            "selection_metric": "mse",
            "higher_is_better": False,
            "aggregate_rank": aggregate_rank,
        },
        "seed": int(seed),
        "seeds": [int(seed)],
        "benchmark_contract": {
            "required_seeds": list(CANONICAL_REQUIRED_SEEDS),
            "require_ci": True,
            "notes": [
                "Paper mode requires a passed prepared eval audit, nonempty rankings, all required seeds, and CI summaries.",
                "Task 3 is expected to replace the single-seed payload with aggregated seed results.",
            ],
        },
        "baseline_catalog": _baseline_catalog(tasks),
        "baseline_failures": all_failures,
    }


def _make_paired_windows(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_samples = 30
    n_time = 8
    latent_dim = 4
    latent = rng.normal(size=(n_samples, n_time, latent_dim)).astype(np.float32)
    latent[:, 1:] += 0.55 * latent[:, :-1]
    eeg_weights = rng.normal(size=(latent_dim, 6)).astype(np.float32)
    fmri_weights = rng.normal(size=(latent_dim, 5)).astype(np.float32)
    eeg = latent @ eeg_weights + 0.04 * rng.normal(size=(n_samples, n_time, 6)).astype(np.float32)
    fmri = latent @ fmri_weights + 0.04 * rng.normal(size=(n_samples, n_time, 5)).astype(np.float32)
    return {"eeg": eeg.astype(np.float32), "fmri": fmri.astype(np.float32)}


def _split(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_idx = max(1, int(round(x.shape[0] * 0.7)))
    return x[:split_idx], y[:split_idx], x[split_idx:], y[split_idx:]


def _future_task(data: dict[str, np.ndarray]) -> SupervisedWindowTask:
    x_train, y_train, x_test, y_test = _split(data["eeg"][:, :-1], data["eeg"][:, 1:])
    return SupervisedWindowTask(
        task_id="future_state_forecasting",
        source_modality="eeg",
        target_modality="eeg",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        notes=("predict next EEG window from previous neural state",),
    )


def _masked_reconstruction_task(data: dict[str, np.ndarray], seed: int) -> SupervisedWindowTask:
    rng = np.random.default_rng(seed + 17)
    clean = data["eeg"]
    mask = rng.random(clean.shape) < 0.2
    masked = clean.copy()
    masked[mask] = 0.0
    x_train, y_train, x_test, y_test = _split(masked, clean)
    _, _, mask_test, _ = _split(mask.astype(bool), mask.astype(bool))
    return SupervisedWindowTask(
        task_id="masked_neural_reconstruction",
        source_modality="eeg",
        target_modality="eeg",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        metric_mask=mask_test,
        notes=("mask time/channel entries and reconstruct clean EEG",),
    )


def _cross_modal_task(data: dict[str, np.ndarray]) -> SupervisedWindowTask:
    x_train, y_train, x_test, y_test = _split(data["eeg"], data["fmri"])
    return SupervisedWindowTask(
        task_id="cross_modal_translation",
        source_modality="eeg",
        target_modality="fmri",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        notes=("paired synthetic EEG to fMRI-like latent proxy",),
    )


def _run_task_models(task: SupervisedWindowTask, seed: int, train_steps: int) -> TaskPayload:
    predictions: dict[str, np.ndarray] = {}
    failures: list[BaselineFailure] = []
    for spec in _runnable_baseline_specs(task):
        try:
            prediction = spec.predict(task, seed, train_steps)
            _validate_prediction(task, spec.model_id, prediction)
            predictions[spec.model_id] = prediction
        except Exception as exc:  # noqa: BLE001 - benchmark failures are payload data.
            failures.append(BaselineFailure(model_id=spec.model_id, task_id=task.task_id, reason=str(exc)))

    metrics_by_model: dict[str, dict[str, float]] = {}
    for model_id, prediction in predictions.items():
        try:
            model_metrics = _metrics(
                task.y_test,
                prediction,
                task.metric_mask,
                source_modality=task.source_modality,
                target_modality=task.target_modality,
                seed=seed,
            )
            _validate_metrics(model_id, model_metrics)
            metrics_by_model[model_id] = model_metrics
        except Exception as exc:  # noqa: BLE001 - benchmark failures are payload data.
            failures.append(BaselineFailure(model_id=model_id, task_id=task.task_id, reason=f"metric failure: {exc}"))
    ranking = [
        {
            "model_id": row.model_id,
            "metric": row.metric,
            "value": row.value,
            "rank": row.rank,
        }
        for row in rank_models(metrics_by_model, metric="mse", higher_is_better=False)
    ]
    return {
        "status": "completed",
        "source_modality": task.source_modality,
        "target_modality": task.target_modality,
        "metrics_by_model": metrics_by_model,
        "ranking": ranking,
        "failures": [failure.to_dict() for failure in failures],
        "notes": list(task.notes),
    }


def _predict_persistence(task: SupervisedWindowTask) -> np.ndarray:
    x_test = np.asarray(task.x_test, dtype=np.float32)
    if x_test.shape[1:] == task.y_test.shape[1:]:
        return x_test.copy()
    if x_test.ndim == 3 and task.y_test.ndim == 3 and x_test.shape[-1] == task.y_test.shape[-1]:
        last_observed = x_test[:, -1:, :]
        return np.broadcast_to(last_observed, task.y_test.shape).astype(np.float32).copy()
    return _predict_train_mean(task.y_train, task.y_test.shape)


def _predict_train_mean(y_train: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    y_train = np.asarray(y_train, dtype=np.float32)
    if y_train.ndim == len(target_shape) and y_train.shape[1:] == target_shape[1:]:
        mean = np.mean(y_train, axis=0, keepdims=True)
    else:
        reduce_axes = tuple(range(max(0, y_train.ndim - 1)))
        mean = np.mean(y_train, axis=reduce_axes).reshape((1,) * (len(target_shape) - 1) + (target_shape[-1],))
    return np.broadcast_to(mean, target_shape).astype(np.float32).copy()


def _predict_random_permutation(y_train: np.ndarray, target_shape: tuple[int, ...], seed: int) -> np.ndarray:
    y_flat = _flatten_time(np.asarray(y_train, dtype=np.float32))
    if y_flat.shape[0] == 0:
        raise ValueError("random_permutation requires at least one training target")
    rng = np.random.default_rng(seed)
    n_rows = int(np.prod(target_shape[:-1]))
    if y_flat.shape[0] >= n_rows:
        indices = rng.permutation(y_flat.shape[0])[:n_rows]
    else:
        indices = rng.choice(y_flat.shape[0], size=n_rows, replace=True)
    noise = rng.normal(scale=1e-6, size=(n_rows, y_flat.shape[-1])).astype(np.float32)
    return (y_flat[indices] + noise).reshape(target_shape).astype(np.float32)


def _fit_ridge(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    model = NumpyRidgeBaseline(alpha=1e-2)
    model.fit(_flatten_time(x_train), _flatten_time(y_train))
    pred = model.predict(_flatten_time(x_test))
    return pred.reshape(x_test.shape[0], x_test.shape[1], y_train.shape[-1])


def _fit_autoregressive_ridge(task: SupervisedWindowTask) -> np.ndarray:
    if task.x_train.ndim != 3 or task.y_train.ndim != 3 or task.x_test.ndim != 3 or task.y_test.ndim != 3:
        raise ValueError("autoregressive_ridge requires [sample, time, feature] windows")
    if task.x_train.shape[1] < 2 or task.x_test.shape[1] < 2 or task.y_train.shape[1] < 2:
        raise ValueError("autoregressive_ridge requires at least two timepoints")
    if task.x_train.shape[1] != task.y_train.shape[1] or task.x_test.shape[1] != task.y_test.shape[1]:
        raise ValueError("autoregressive_ridge requires aligned source and target sequence lengths")

    model = NumpyRidgeBaseline(alpha=1e-2)
    model.fit(_flatten_time(task.x_train[:, :-1, :]), _flatten_time(task.y_train[:, 1:, :]))
    prediction = _predict_train_mean(task.y_train, task.y_test.shape)
    pred_tail = model.predict(_flatten_time(task.x_test[:, :-1, :]))
    prediction[:, 1:, :] = pred_tail.reshape(task.x_test.shape[0], task.x_test.shape[1] - 1, task.y_train.shape[-1])
    return prediction.astype(np.float32)


def _fit_torch_sequence_model(
    factory: Callable[[], nn.Module],
    task: SupervisedWindowTask,
    seed: int,
    steps: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    model = factory()
    x_train = torch.as_tensor(task.x_train, dtype=torch.float32)
    y_train = torch.as_tensor(task.y_train, dtype=torch.float32)
    x_test = torch.as_tensor(task.x_test, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(x_test).detach().cpu().numpy()


def _fit_neurotwin(task: SupervisedWindowTask, seed: int, steps: int) -> np.ndarray:
    torch.manual_seed(seed)
    model = NeuralStateSpaceTranslator(
        input_dims={task.source_modality: task.x_train.shape[-1]},
        output_dims={task.target_modality: task.y_train.shape[-1]},
        config=NeuralStateSpaceTranslatorConfig(latent_dim=24, n_layers=1, subject_adapter_dim=8),
    )
    x_train = torch.as_tensor(task.x_train, dtype=torch.float32)
    y_train = torch.as_tensor(task.y_train, dtype=torch.float32)
    x_test = torch.as_tensor(task.x_test, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_task(
            {task.source_modality: x_train},
            target_modality=task.target_modality,
            task="forecast" if task.task_id == "future_state_forecasting" else "reconstruction",
        )
        loss = loss_fn(output["prediction"], y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        output = model.forward_task(
            {task.source_modality: x_test},
            target_modality=task.target_modality,
            task="forecast" if task.task_id == "future_state_forecasting" else "reconstruction",
        )
    return output["prediction"].detach().cpu().numpy()


def _fit_tribe_style(task: SupervisedWindowTask, seed: int, steps: int) -> np.ndarray:
    return _fit_torch_sequence_model(
        lambda: TribeStyleStimulusEncoder(task.x_train.shape[-1], task.y_train.shape[-1], hidden_dim=24),
        task,
        seed=seed,
        steps=steps,
    )


def _is_stimulus_fmri_task(task: SupervisedWindowTask) -> bool:
    return task.task_id == "stimulus_to_fmri_response" and task.source_modality == "stimulus" and task.target_modality == "fmri"


def _tribe_style_catalog_status(task_ids: set[str], modalities: set[str]) -> str:
    return "clean_room_approximation" if "stimulus_to_fmri_response" in task_ids and "fmri" in modalities else "unavailable"


def _brainvista_catalog_status(task_ids: set[str], modalities: set[str]) -> str:
    return "approximation" if "future_state_forecasting" in task_ids and "fmri" in modalities else "unavailable"


def _brain_of_catalog_status(task_ids: set[str], modalities: set[str]) -> str:
    return "approximation" if "masked_neural_reconstruction" in task_ids and len(modalities) >= 2 else "unavailable"


def _brainomni_catalog_status(task_ids: set[str], modalities: set[str]) -> str:
    return "approximation" if modalities & {"eeg", "meg"} else "unavailable"


BASELINE_CATALOG: tuple[BaselineCatalogEntry, ...] = (
    BaselineCatalogEntry(
        model_id="persistence",
        display_name="Persistence",
        status="local_baseline",
        notes="Last-observation or identity-style forecast with shape-safe fallback.",
    ),
    BaselineCatalogEntry(
        model_id="train_mean",
        display_name="Train Mean",
        status="local_baseline",
        notes="Broadcast train-target mean negative baseline.",
    ),
    BaselineCatalogEntry(
        model_id="random_permutation",
        display_name="Random Permutation",
        status="negative_control",
        notes="Seeded permutation of training targets with target-shaped output.",
    ),
    BaselineCatalogEntry(
        model_id="linear_ridge",
        display_name="Linear Ridge",
        status="local_baseline",
        notes="Closed-form sanity baseline on identical prepared windows.",
    ),
    BaselineCatalogEntry(
        model_id="autoregressive_ridge",
        display_name="Autoregressive Ridge",
        status="local_baseline",
        notes="Ridge from previous source timepoint to next target timepoint where sequence shapes allow.",
    ),
    BaselineCatalogEntry(
        model_id="mlp",
        display_name="MLP",
        status="local_baseline",
        notes="Per-timepoint neural-window baseline.",
    ),
    BaselineCatalogEntry(
        model_id="tcn",
        display_name="TCN",
        status="local_baseline",
        notes="Local temporal convolution baseline.",
    ),
    BaselineCatalogEntry(
        model_id="transformer",
        display_name="Transformer",
        status="local_baseline",
        notes="Small local Transformer with shared splits.",
    ),
    BaselineCatalogEntry(
        model_id="ssm_fallback",
        display_name="SSM Fallback",
        status="local_baseline",
        notes="GRU-based SSM fallback until Mamba is pinned.",
    ),
    BaselineCatalogEntry(
        model_id="neurotwin",
        display_name="NeuroTwin",
        status="local_baseline",
        notes="Current NeuroTwin implementation under the same task API.",
    ),
    BaselineCatalogEntry(
        model_id="tribe_style",
        display_name="TRIBE-Style",
        status=_tribe_style_catalog_status,
        notes="NeuroTwin-native stimulus-to-fMRI approximation; not an exact TRIBE v2 reproduction.",
        upstream_reference="TRIBE v2",
        exact_reproduction=False,
        uses_upstream_code=False,
        uses_upstream_weights=False,
    ),
    BaselineCatalogEntry(
        model_id="brainvista_style",
        display_name="BrainVista-Style",
        status=_brainvista_catalog_status,
        notes="Approximate autoregressive fMRI rollout lane; not an exact BrainVista reproduction.",
        upstream_reference="BrainVista",
    ),
    BaselineCatalogEntry(
        model_id="brain_of_style",
        display_name="Brain-OF-Style",
        status=_brain_of_catalog_status,
        notes="Approximate multimodal masked reconstruction lane; not an exact Brain-OF reproduction.",
        upstream_reference="Brain-OF",
    ),
    BaselineCatalogEntry(
        model_id="brainomni_style",
        display_name="BrainOmni-Style",
        status=_brainomni_catalog_status,
        notes="Approximate EEG/MEG tokenizer lane; not an exact BrainOmni reproduction.",
        upstream_reference="BrainOmni",
    ),
    BaselineCatalogEntry(
        model_id="braindecode_wrapper",
        display_name="Braindecode Wrapper",
        status="unavailable",
        notes="Optional EEG wrapper slot; exact use requires installed Braindecode and compatible task protocols.",
        upstream_reference="Braindecode",
    ),
    BaselineCatalogEntry(
        model_id="cebra_wrapper",
        display_name="CEBRA Wrapper",
        status="unavailable",
        notes="Optional neural-behavior embedding wrapper slot; exact use requires installed CEBRA and aligned behavior data.",
        upstream_reference="CEBRA",
    ),
)


EXECUTABLE_BASELINE_RUNNERS: tuple[ExecutableBaselineRunner, ...] = (
    ExecutableBaselineRunner("persistence", lambda task, seed, steps: _predict_persistence(task)),
    ExecutableBaselineRunner("train_mean", lambda task, seed, steps: _predict_train_mean(task.y_train, task.y_test.shape)),
    ExecutableBaselineRunner("random_permutation", lambda task, seed, steps: _predict_random_permutation(task.y_train, task.y_test.shape, seed=seed + 101)),
    ExecutableBaselineRunner("linear_ridge", lambda task, seed, steps: _fit_ridge(task.x_train, task.y_train, task.x_test)),
    ExecutableBaselineRunner("autoregressive_ridge", lambda task, seed, steps: _fit_autoregressive_ridge(task)),
    ExecutableBaselineRunner(
        "mlp",
        lambda task, seed, steps: _fit_torch_sequence_model(
            lambda: TorchMLPBaseline(task.x_train.shape[-1], task.y_train.shape[-1], hidden_dim=24),
            task,
            seed=seed + 1,
            steps=steps,
        ),
    ),
    ExecutableBaselineRunner(
        "tcn",
        lambda task, seed, steps: _fit_torch_sequence_model(
            lambda: TorchTCNBaseline(task.x_train.shape[-1], task.y_train.shape[-1], hidden_dim=24),
            task,
            seed=seed + 2,
            steps=steps,
        ),
    ),
    ExecutableBaselineRunner(
        "transformer",
        lambda task, seed, steps: _fit_torch_sequence_model(
            lambda: TinyTransformerBaseline(
                task.x_train.shape[-1],
                task.y_train.shape[-1],
                latent_dim=24,
                n_heads=4,
                n_layers=1,
            ),
            task,
            seed=seed + 3,
            steps=steps,
        ),
    ),
    ExecutableBaselineRunner(
        "ssm_fallback",
        lambda task, seed, steps: _fit_torch_sequence_model(
            lambda: TinySSMBaseline(task.x_train.shape[-1], task.y_train.shape[-1], latent_dim=24, n_layers=1),
            task,
            seed=seed + 4,
            steps=steps,
        ),
    ),
    ExecutableBaselineRunner("neurotwin", lambda task, seed, steps: _fit_neurotwin(task, seed=seed + 5, steps=steps)),
    ExecutableBaselineRunner(
        "tribe_style",
        lambda task, seed, steps: _fit_tribe_style(task, seed=seed + 6, steps=steps),
        available_for_task=_is_stimulus_fmri_task,
    ),
)


def _runnable_baseline_specs(task: SupervisedWindowTask) -> tuple[ExecutableBaselineRunner, ...]:
    return tuple(spec for spec in EXECUTABLE_BASELINE_RUNNERS if spec.supports(task))


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_mask: np.ndarray | None,
    source_modality: str,
    target_modality: str,
    seed: int = 0,
) -> dict[str, float]:
    if metric_mask is not None:
        y_true_metric = y_true[metric_mask]
        y_pred_metric = y_pred[metric_mask]
    else:
        y_true_metric = y_true
        y_pred_metric = y_pred
    values = {
        "mse": mse(y_true_metric, y_pred_metric),
        "mae": mae(y_true_metric, y_pred_metric),
        "pearsonr": pearsonr(y_true_metric, y_pred_metric),
        "spearmanr": spearmanr(y_true_metric, y_pred_metric),
        "r2": r2_score(y_true_metric, y_pred_metric),
    }
    squared_error = (np.asarray(y_true_metric, dtype=float).ravel() - np.asarray(y_pred_metric, dtype=float).ravel()) ** 2
    absolute_error = np.abs(np.asarray(y_true_metric, dtype=float).ravel() - np.asarray(y_pred_metric, dtype=float).ravel())
    values["mse_ci_low"], values["mse_ci_high"] = bootstrap_ci(squared_error, seed=seed, n_boot=200)
    values["mae_ci_low"], values["mae_ci_high"] = bootstrap_ci(absolute_error, seed=seed + 1, n_boot=200)
    if source_modality in {"eeg", "meg"} or target_modality in {"eeg", "meg"}:
        values["spectral_error"] = spectral_error(y_true, y_pred)
        values["bandpower_error"] = bandpower_error(y_true, y_pred)
    if target_modality == "fmri":
        values["regionwise_pearsonr"] = regionwise_pearsonr(y_true, y_pred)
    if metric_mask is not None:
        values["masked_mse"] = values["mse"]
    return values


def _flatten_time(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])


def _validate_prediction(task: SupervisedWindowTask, model_id: str, prediction: np.ndarray) -> None:
    prediction = np.asarray(prediction)
    if prediction.shape != task.y_test.shape:
        raise ValueError(f"{model_id} prediction shape {prediction.shape} does not match target {task.y_test.shape}")
    if not np.isfinite(prediction).all():
        raise ValueError(f"{model_id} prediction contains NaN or Inf")


def _validate_metrics(model_id: str, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        if not isinstance(value, (int, float, np.floating)) or not np.isfinite(float(value)):
            raise ValueError(f"{model_id} metric {key} is not finite: {value}")


def _baseline_catalog(tasks: tuple[SupervisedWindowTask, ...]) -> list[dict[str, object]]:
    task_ids = {task.task_id for task in tasks}
    modalities = {task.source_modality for task in tasks} | {task.target_modality for task in tasks}
    return [spec.catalog_row(task_ids, modalities) for spec in BASELINE_CATALOG]
