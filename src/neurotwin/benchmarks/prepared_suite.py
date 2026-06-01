from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neurotwin.benchmarks.baseline_suite import SupervisedWindowTask, run_supervised_window_tasks
from neurotwin.benchmarks.tasks import (
    TaskResult,
    run_dataset_site_generalization_task,
    run_subject_adaptation_task,
)
from neurotwin.data.event_io import event_manifest_summary, load_event_batches
from neurotwin.data.manifest_io import load_split_manifest
from neurotwin.data.schemas import NeuralEventBatch
from neurotwin.data.split_manifest import SplitManifest
from neurotwin.data.windows import WindowSpec, batch_to_windows
from neurotwin.eval.audit import audit_prepared_eval_inputs
from neurotwin.eval.metrics import bootstrap_ci, rank_models
from neurotwin.eval.paper_gate import (
    CANONICAL_REQUIRED_SEEDS,
    PaperModeGateReport,
    format_paper_mode_gate,
    validate_paper_mode_payload,
)
from neurotwin.repro import write_json


@dataclass(frozen=True)
class PreparedSuiteConfig:
    event_manifest: str | Path
    split_manifest: str | Path
    window_length: int = 8
    stride: int = 8
    seed: int = 0
    train_steps: int = 5
    require_ci: bool = True


def run_prepared_baseline_suite(
    config: PreparedSuiteConfig,
    out_dir: str | Path | None = None,
) -> dict[str, object]:
    batches = load_event_batches(config.event_manifest)
    split = load_split_manifest(config.split_manifest)
    tasks, skipped = build_prepared_window_tasks(
        batches,
        split,
        window_length=config.window_length,
        stride=config.stride,
        seed=config.seed,
    )
    if tasks:
        payload = run_supervised_window_tasks(
            tasks,
            seed=config.seed,
            train_steps=config.train_steps,
            scope_status=_scope_status(batches),
            scope_notes=(
                "Uses prepared event batches and a recording-level split manifest.",
                "Claims still require real public data, preregistered protocols, and confidence intervals.",
            ),
        )
    else:
        payload = {
            "scope": {
                "status": _scope_status(batches),
                "notes": ["No runnable supervised window tasks were available for this prepared manifest."],
            },
            "tasks": {},
            "aggregate": {"selection_metric": "mse", "higher_is_better": False, "aggregate_rank": []},
        }
    auxiliary_tasks, auxiliary_skipped = run_prepared_auxiliary_tasks(
        batches,
        split,
        window_length=config.window_length,
        stride=config.stride,
    )
    payload["tasks"].update(auxiliary_tasks)  # type: ignore[union-attr]
    skipped.extend(auxiliary_skipped)
    payload["prepared_data"] = {
        "event_manifest": str(config.event_manifest),
        "split_manifest": str(config.split_manifest),
        "event_summary": event_manifest_summary(config.event_manifest),
        "window_length": config.window_length,
        "stride": config.stride,
        "skipped_tasks": skipped,
    }
    payload["paper_mode_contract"] = {
        "required_seeds": list(CANONICAL_REQUIRED_SEEDS),
        "observed_seeds": payload.get("seeds", [config.seed]),
        "require_ci": bool(config.require_ci),
        "gate_status": "not_run",
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "prepared_baseline_suite.json", payload)
        write_json(out / "baseline_failures.json", payload.get("baseline_failures", []))
    return payload


def run_prepared_baseline_suite_multi_seed(
    config: PreparedSuiteConfig,
    seeds: tuple[int, ...] | list[int] = CANONICAL_REQUIRED_SEEDS,
    out_dir: str | Path | None = None,
) -> dict[str, object]:
    """Run prepared baselines for multiple seeds and write paper-mode artifacts."""

    seed_values = tuple(int(seed) for seed in seeds)
    seed_results = []
    for seed in seed_values:
        seed_payload = run_prepared_baseline_suite(
            PreparedSuiteConfig(
                event_manifest=config.event_manifest,
                split_manifest=config.split_manifest,
                window_length=config.window_length,
                stride=config.stride,
                seed=seed,
                train_steps=config.train_steps,
                require_ci=config.require_ci,
            ),
            out_dir=None,
        )
        _add_degenerate_task_metric_ci(seed_payload)
        seed_payload["seed"] = seed
        seed_payload["seeds"] = [seed]
        seed_results.append(seed_payload)

    payload = _merge_seed_payloads(config, seed_values, seed_results)
    audit = audit_prepared_eval_inputs(
        config.event_manifest,
        config.split_manifest,
        window_length=config.window_length,
        stride=config.stride,
        out_dir=out_dir,
        require_windows=True,
    )
    payload["eval_audit"] = audit.to_dict()
    gate = validate_paper_mode_payload(payload, audit_report=audit, require_ci=config.require_ci)
    payload["paper_mode_gate"] = gate.to_dict()
    payload["paper_mode_contract"] = {
        "required_seeds": list(gate.required_seeds),
        "observed_seeds": list(gate.observed_seeds),
        "require_ci": gate.require_ci,
        "gate_status": "passed" if gate.passed else "failed",
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "prepared_baseline_suite.json", payload)
        write_json(out / "seed_aggregate.json", payload.get("seed_aggregate", []))
        (out / "seed_aggregate.csv").write_text(
            _seed_aggregate_csv(payload.get("seed_aggregate", [])),
            encoding="utf-8",
        )
        write_json(out / "baseline_failures.json", payload.get("baseline_failures", []))
        write_json(out / "paper_mode_gate.json", gate.to_dict())
    return payload


def format_prepared_baseline_report(payload: dict[str, object]) -> str:
    prepared = payload.get("prepared_data", {})
    scope = payload.get("scope", {})
    aggregate = payload.get("aggregate", {})
    lines = [
        "# NeuroTwin Prepared Baseline Suite",
        "",
        f"scope={scope.get('status') if isinstance(scope, dict) else 'unknown'}",
    ]
    if isinstance(prepared, dict):
        lines.extend(
            [
                f"event_manifest={prepared.get('event_manifest')}",
                f"split_manifest={prepared.get('split_manifest')}",
                f"window_length={prepared.get('window_length')}",
                f"stride={prepared.get('stride')}",
                "",
            ]
        )
    if isinstance(aggregate, dict):
        lines.append("## aggregate_rank")
        for row in aggregate.get("aggregate_rank", []):
            if isinstance(row, dict):
                lines.append(
                    f"ranked_model={row.get('model_id')} mean_rank={row.get('mean_rank')} tasks={row.get('tasks_ranked')}"
                )
        lines.append("")
    seed_aggregate = payload.get("seed_aggregate", [])
    if isinstance(seed_aggregate, list) and seed_aggregate:
        lines.append("## seed_aggregate")
        for row in seed_aggregate[:20]:
            if isinstance(row, dict):
                lines.append(
                    "seed_metric="
                    f"{row.get('task_id')}:{row.get('model_id')}:{row.get('metric')} "
                    f"mean={row.get('mean')} std={row.get('std')} "
                    f"ci95=[{row.get('ci_low')},{row.get('ci_high')}] n={row.get('n_seeds')}"
                )
        if len(seed_aggregate) > 20:
            lines.append(f"seed_metric_rows_omitted={len(seed_aggregate) - 20}")
        lines.append("")
    catalog = payload.get("baseline_catalog", [])
    if isinstance(catalog, list):
        lines.append("## baseline_catalog")
        for row in catalog:
            if isinstance(row, dict):
                lines.append(f"{row.get('model_id')}: {row.get('status')} - {row.get('notes')}")
        lines.append("")
    tasks = payload.get("tasks", {})
    if isinstance(tasks, dict):
        for task_id, result in tasks.items():
            if not isinstance(result, dict):
                continue
            lines.append(f"## {task_id}")
            lines.append(f"status={result.get('status')}")
            metrics = result.get("metrics", {})
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    lines.append(f"{key}={value}")
            for row in result.get("ranking", []):
                if isinstance(row, dict):
                    model_id = row.get("model_id")
                    metric = row.get("metric")
                    suffix = ""
                    metrics_by_model = result.get("metrics_by_model", {})
                    if isinstance(metrics_by_model, dict) and model_id in metrics_by_model and metric == "mse":
                        model_metrics = metrics_by_model.get(model_id, {})
                        if isinstance(model_metrics, dict):
                            suffix = f" ci95=[{model_metrics.get('mse_ci_low')},{model_metrics.get('mse_ci_high')}]"
                    lines.append(f"{model_id}_rank={row.get('rank')} {metric}={row.get('value')}{suffix}")
            for row in result.get("failures", []):
                if isinstance(row, dict):
                    lines.append(f"failed_model={row.get('model_id')} reason={row.get('reason')}")
            lines.append("")
    failures = payload.get("baseline_failures", [])
    if isinstance(failures, list) and failures:
        lines.append("## baseline_failures")
        for row in failures:
            if isinstance(row, dict):
                lines.append(f"{row.get('task_id')}:{row.get('model_id')}: {row.get('reason')}")
        lines.append("")
    if isinstance(prepared, dict) and prepared.get("skipped_tasks"):
        lines.append("## skipped_tasks")
        for row in prepared.get("skipped_tasks", []):
            if isinstance(row, dict):
                lines.append(f"{row.get('task_id')}: {row.get('reason')}")
        lines.append("")
    gate = payload.get("paper_mode_gate")
    if isinstance(gate, dict):
        lines.append("## paper_mode_gate")
        lines.append(format_paper_mode_gate(PaperModeGateReport(**gate)))
        lines.append("")
    lines.append("Prepared-data rankings are benchmark plumbing unless run on real held-out public data with locked protocols.")
    return "\n".join(lines)


def run_prepared_auxiliary_tasks(
    batches: list[NeuralEventBatch],
    split: SplitManifest,
    window_length: int,
    stride: int,
) -> tuple[dict[str, dict[str, object]], list[dict[str, str]]]:
    windows_by_split = _windows_by_split(batches, split, WindowSpec(length=window_length, stride=stride))
    results: dict[str, dict[str, object]] = {}
    skipped: list[dict[str, str]] = []
    adaptation = _subject_adaptation_from_windows(windows_by_split)
    if adaptation is None:
        skipped.append({"task_id": "few_shot_subject_adaptation", "reason": "need at least one held-out subject with support/query windows"})
    else:
        results[adaptation.task_id] = _task_result_to_dict(adaptation)

    generalization = _dataset_site_generalization_from_windows(windows_by_split)
    if generalization is None:
        skipped.append({"task_id": "dataset_site_generalization", "reason": "need train/test windows from different datasets or sites"})
    else:
        results[generalization.task_id] = _task_result_to_dict(generalization)
    return results, skipped


def build_prepared_window_tasks(
    batches: list[NeuralEventBatch],
    split: SplitManifest,
    window_length: int,
    stride: int,
    seed: int = 0,
) -> tuple[tuple[SupervisedWindowTask, ...], list[dict[str, str]]]:
    spec = WindowSpec(length=window_length, stride=stride)
    skipped: list[dict[str, str]] = []
    windows_by_split = _windows_by_split(batches, split, spec)
    split_keys = _split_record_keys(split)
    for batch in batches:
        if _record_id(batch) not in split_keys:
            skipped.append({"task_id": "all", "reason": f"event not present in split manifest: {_record_id(batch)}"})

    tasks: list[SupervisedWindowTask] = []
    future = _future_task_from_windows(windows_by_split)
    if future is not None:
        tasks.append(future)
    else:
        skipped.append({"task_id": "future_state_forecasting", "reason": "need train and test windows for one modality"})

    masked = _masked_task_from_windows(windows_by_split, seed=seed)
    if masked is not None:
        tasks.append(masked)
    else:
        skipped.append({"task_id": "masked_neural_reconstruction", "reason": "need train and test windows for one modality"})

    cross_modal = _cross_modal_task_from_windows(windows_by_split)
    if cross_modal is not None:
        tasks.append(cross_modal)
    else:
        skipped.append({"task_id": "cross_modal_translation", "reason": "need paired train/test windows for two modalities"})

    stimulus_fmri = _stimulus_fmri_task_from_windows(windows_by_split)
    if stimulus_fmri is not None:
        tasks.append(stimulus_fmri)
    else:
        skipped.append({"task_id": "stimulus_to_fmri_response", "reason": "need fMRI train/test windows with aligned stimulus embeddings"})
    return tuple(tasks), skipped


def _subject_adaptation_from_windows(windows_by_split: dict[str, list[NeuralEventBatch]]) -> TaskResult | None:
    modality = _first_modality_with_splits(windows_by_split)
    if modality is None:
        return None
    by_subject: dict[str, list[np.ndarray]] = {}
    for window in windows_by_split["test"]:
        if window.modality == modality:
            by_subject.setdefault(window.subject_id, []).append(window.signal)
    for subject_id, signals in sorted(by_subject.items()):
        if len(signals) >= 2:
            metrics: dict[str, float] = {}
            notes = [f"prepared held-out subject={subject_id} modality={modality}"]
            result: TaskResult | None = None
            for support_size in (1, 5, 20):
                if len(signals) <= support_size:
                    continue
                support = np.asarray(signals[:support_size], dtype=np.float32)
                query = np.asarray(signals[support_size:], dtype=np.float32)
                if query.size == 0:
                    continue
                current = run_subject_adaptation_task(support, query)
                result = current
                for key, value in current.metrics.items():
                    metrics[f"k{support_size}_{key}"] = float(value)
                first = windows_by_split["test"][0]
                sampling_rate = first.metadata.get("sampling_rate") or first.metadata.get("tr")
                if sampling_rate:
                    try:
                        minutes = (support_size * support.shape[1]) / float(sampling_rate) / 60.0
                        metrics[f"k{support_size}_support_minutes"] = float(minutes)
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
            if result is None:
                continue
            return TaskResult(
                result.task_id,
                result.status,
                metrics,
                tuple(notes),
            )
    return None


def _dataset_site_generalization_from_windows(windows_by_split: dict[str, list[NeuralEventBatch]]) -> TaskResult | None:
    modality = _first_modality_with_splits(windows_by_split)
    if modality is None:
        return None
    source_groups = _group_windows(windows_by_split["train"], modality)
    target_groups = _group_windows(windows_by_split["test"], modality)
    for source_key, source_signals in sorted(source_groups.items()):
        for target_key, target_signals in sorted(target_groups.items()):
            if source_key == target_key:
                continue
            source = np.asarray(source_signals, dtype=np.float32)
            target = np.asarray(target_signals, dtype=np.float32)
            if source.size == 0 or target.size == 0:
                continue
            n_time = min(source.shape[1], target.shape[1])
            n_space = min(source.shape[2], target.shape[2])
            result = run_dataset_site_generalization_task(
                source[:, :n_time, :n_space],
                target[:, :n_time, :n_space],
                source_name="/".join(source_key),
                target_name="/".join(target_key),
            )
            return result
    return None


def _future_task_from_windows(windows_by_split: dict[str, list[NeuralEventBatch]]) -> SupervisedWindowTask | None:
    modality = _first_modality_with_splits(windows_by_split)
    if modality is None:
        return None
    train = [window.signal for window in windows_by_split["train"] if window.modality == modality]
    val = [window.signal for window in windows_by_split["val"] if window.modality == modality]
    test = [window.signal for window in windows_by_split["test"] if window.modality == modality]
    x_train, y_train = _future_xy(train)
    x_val, y_val = _future_xy(val)
    x_test, y_test = _future_xy(test)
    if x_train is None or x_test is None or y_train is None or y_test is None:
        return None
    return SupervisedWindowTask(
        task_id="future_state_forecasting",
        source_modality=modality,
        target_modality=modality,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_val=x_val,
        y_val=y_val,
        notes=(f"prepared {modality} next-state windows",),
    )


def _masked_task_from_windows(
    windows_by_split: dict[str, list[NeuralEventBatch]],
    seed: int,
) -> SupervisedWindowTask | None:
    modality = _first_modality_with_splits(windows_by_split)
    if modality is None:
        return None
    train = np.asarray([window.signal for window in windows_by_split["train"] if window.modality == modality], dtype=np.float32)
    val = np.asarray([window.signal for window in windows_by_split["val"] if window.modality == modality], dtype=np.float32)
    test = np.asarray([window.signal for window in windows_by_split["test"] if window.modality == modality], dtype=np.float32)
    if train.size == 0 or test.size == 0:
        return None
    rng = np.random.default_rng(seed + 31)
    train_mask = rng.random(train.shape) < 0.2
    val_mask = rng.random(val.shape) < 0.2 if val.size else None
    test_mask = rng.random(test.shape) < 0.2
    x_train = train.copy()
    x_val = val.copy() if val.size else None
    x_test = test.copy()
    x_train[train_mask] = 0.0
    if x_val is not None and val_mask is not None:
        x_val[val_mask] = 0.0
    x_test[test_mask] = 0.0
    return SupervisedWindowTask(
        task_id="masked_neural_reconstruction",
        source_modality=modality,
        target_modality=modality,
        x_train=x_train,
        y_train=train,
        x_test=x_test,
        y_test=test,
        metric_mask=test_mask,
        x_val=x_val,
        y_val=val if val.size else None,
        val_metric_mask=val_mask,
        notes=(f"prepared {modality} masked reconstruction",),
    )


def _cross_modal_task_from_windows(windows_by_split: dict[str, list[NeuralEventBatch]]) -> SupervisedWindowTask | None:
    modalities = sorted({window.modality for split_windows in windows_by_split.values() for window in split_windows})
    if len(modalities) < 2:
        return None
    source = "eeg" if "eeg" in modalities else modalities[0]
    target_candidates = [modality for modality in modalities if modality != source]
    target = "fmri" if "fmri" in target_candidates else target_candidates[0]
    train_pairs = _paired_windows(windows_by_split["train"], source, target)
    val_pairs = _paired_windows(windows_by_split["val"], source, target)
    test_pairs = _paired_windows(windows_by_split["test"], source, target)
    if not train_pairs or not test_pairs:
        return None
    x_train, y_train = _stack_pairs(train_pairs)
    x_val, y_val = _stack_pairs(val_pairs) if val_pairs else (None, None)
    x_test, y_test = _stack_pairs(test_pairs)
    return SupervisedWindowTask(
        task_id="cross_modal_translation",
        source_modality=source,
        target_modality=target,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_val=x_val,
        y_val=y_val,
        notes=(f"prepared paired {source}->{target} windows",),
    )


def _stimulus_fmri_task_from_windows(windows_by_split: dict[str, list[NeuralEventBatch]]) -> SupervisedWindowTask | None:
    train = _stimulus_fmri_xy(windows_by_split["train"])
    val = _stimulus_fmri_xy(windows_by_split["val"])
    test = _stimulus_fmri_xy(windows_by_split["test"])
    if train is None or test is None:
        return None
    x_train, y_train = train
    x_test, y_test = test
    x_val, y_val = val if val is not None else (None, None)
    return SupervisedWindowTask(
        task_id="stimulus_to_fmri_response",
        source_modality="stimulus",
        target_modality="fmri",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_val=x_val,
        y_val=y_val,
        notes=(
            "prepared stimulus->fMRI response windows",
            "tribe_style is a clean_room_approximation, not an exact TRIBE v2 reproduction",
        ),
    )


def _first_modality_with_splits(windows_by_split: dict[str, list[NeuralEventBatch]]) -> str | None:
    train_modalities = {window.modality for window in windows_by_split["train"]}
    test_modalities = {window.modality for window in windows_by_split["test"]}
    for preferred in ("eeg", "fmri", "meg", "spikes"):
        if preferred in train_modalities and preferred in test_modalities:
            return preferred
    shared = sorted(train_modalities & test_modalities)
    return shared[0] if shared else None


def _future_xy(signals: list[np.ndarray]) -> tuple[np.ndarray | None, np.ndarray | None]:
    usable = [np.asarray(signal, dtype=np.float32) for signal in signals if signal.shape[0] >= 2]
    if not usable:
        return None, None
    return np.asarray([signal[:-1] for signal in usable], dtype=np.float32), np.asarray([signal[1:] for signal in usable], dtype=np.float32)


def _paired_windows(
    windows: list[NeuralEventBatch],
    source: str,
    target: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str, str, str, int], dict[str, NeuralEventBatch]] = {}
    for window in windows:
        key = (
            window.dataset,
            window.subject_id,
            window.session_id,
            window.site_id,
            int(window.metadata.get("window_start_index", 0)),
        )
        grouped.setdefault(key, {})[window.modality] = window
    pairs = []
    for group in grouped.values():
        if source in group and target in group:
            source_signal = group[source].signal
            target_signal = group[target].signal
            n_time = min(source_signal.shape[0], target_signal.shape[0])
            pairs.append((source_signal[:n_time], target_signal[:n_time]))
    return pairs


def _stimulus_fmri_xy(windows: list[NeuralEventBatch]) -> tuple[np.ndarray, np.ndarray] | None:
    pairs = [
        (np.asarray(window.stimulus_embedding, dtype=np.float32), np.asarray(window.signal, dtype=np.float32))
        for window in windows
        if window.modality == "fmri"
        and window.stimulus_embedding is not None
        and window.stimulus_embedding.shape[0] == window.signal.shape[0]
    ]
    if not pairs:
        return None
    return (
        np.asarray([stimulus for stimulus, _ in pairs], dtype=np.float32),
        np.asarray([signal for _, signal in pairs], dtype=np.float32),
    )


def _windows_by_split(
    batches: list[NeuralEventBatch],
    split: SplitManifest,
    spec: WindowSpec,
) -> dict[str, list[NeuralEventBatch]]:
    split_keys = _split_record_keys(split)
    windows_by_split: dict[str, list[NeuralEventBatch]] = {"train": [], "val": [], "test": []}
    for batch in batches:
        split_name = split_keys.get(_record_id(batch))
        if split_name is None:
            continue
        windows_by_split[split_name].extend(batch_to_windows(batch, spec))
    return windows_by_split


def _group_windows(windows: list[NeuralEventBatch], modality: str) -> dict[tuple[str, str], list[np.ndarray]]:
    groups: dict[tuple[str, str], list[np.ndarray]] = {}
    for window in windows:
        if window.modality != modality:
            continue
        groups.setdefault((window.dataset, window.site_id), []).append(window.signal)
    return groups


def _stack_pairs(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([source for source, _ in pairs], dtype=np.float32),
        np.asarray([target for _, target in pairs], dtype=np.float32),
    )


def _record_id(batch: NeuralEventBatch) -> str:
    return str(batch.metadata.get("record_id") or batch.metadata.get("source_record_id"))


def _split_record_keys(split: SplitManifest) -> dict[str, str]:
    keys = {}
    for split_name in ("train", "val", "test"):
        for record in getattr(split, split_name):
            keys[record.record_id] = split_name
    return keys


def _scope_status(batches: list[NeuralEventBatch]) -> str:
    return "prepared-synthetic" if all(batch.metadata.get("synthetic") for batch in batches) else "prepared-data"


def _task_result_to_dict(result: TaskResult) -> dict[str, object]:
    return {
        "status": result.status,
        "metrics": result.metrics,
        "notes": result.notes,
    }


def _merge_seed_payloads(
    config: PreparedSuiteConfig,
    seeds: tuple[int, ...],
    seed_results: list[dict[str, object]],
) -> dict[str, object]:
    first = seed_results[0] if seed_results else {}
    aggregate_rank = _aggregate_seed_ranks(seed_results)
    seed_aggregate = _aggregate_seed_metrics(seed_results)
    tasks = _aggregate_seed_tasks(first, seed_aggregate)
    failures: list[dict[str, str]] = []
    for result in seed_results:
        raw_failures = result.get("baseline_failures", [])
        if isinstance(raw_failures, list):
            for failure in raw_failures:
                if isinstance(failure, dict):
                    failures.append({str(key): str(value) for key, value in failure.items()})
    return {
        "scope": first.get("scope", {"status": "prepared-data", "notes": []}),
        "tasks": tasks,
        "aggregate": {
            "selection_metric": "mse",
            "higher_is_better": False,
            "aggregate_rank": aggregate_rank,
        },
        "seed": int(seeds[0]) if seeds else int(config.seed),
        "seeds": [int(seed) for seed in seeds],
        "seed_results": seed_results,
        "seed_aggregate": seed_aggregate,
        "representative_seed_tasks": first.get("tasks", {}),
        "prepared_data": first.get(
            "prepared_data",
            {
                "event_manifest": str(config.event_manifest),
                "split_manifest": str(config.split_manifest),
                "event_summary": event_manifest_summary(config.event_manifest),
                "window_length": config.window_length,
                "stride": config.stride,
                "skipped_tasks": [],
            },
        ),
        "paper_mode_contract": {
            "required_seeds": list(CANONICAL_REQUIRED_SEEDS),
            "observed_seeds": [int(seed) for seed in seeds],
            "require_ci": bool(config.require_ci),
            "gate_status": "not_run",
        },
        "baseline_catalog": first.get("baseline_catalog", []),
        "baseline_failures": failures,
    }


def _aggregate_seed_tasks(
    representative_payload: dict[str, object],
    seed_aggregate: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    representative_tasks = representative_payload.get("tasks", {})
    if not isinstance(representative_tasks, dict):
        representative_tasks = {}

    rows_by_task: dict[str, list[dict[str, object]]] = {}
    for row in seed_aggregate:
        task_id = row.get("task_id")
        if task_id is not None:
            rows_by_task.setdefault(str(task_id), []).append(row)

    tasks: dict[str, dict[str, object]] = {}
    for task_id, rows in sorted(rows_by_task.items()):
        representative = representative_tasks.get(task_id, {})
        if not isinstance(representative, dict):
            representative = {}
        metrics: dict[str, float] = {}
        metrics_by_model: dict[str, dict[str, float]] = {}
        for row in rows:
            model_id = row.get("model_id")
            metric = row.get("metric")
            if model_id is None or metric is None:
                continue
            metric_name = str(metric)
            mean = _finite_row_float(row, "mean")
            ci_low = _finite_row_float(row, "ci_low")
            ci_high = _finite_row_float(row, "ci_high")
            if mean is None or ci_low is None or ci_high is None:
                continue
            target = metrics if str(model_id) == "task_metric" else metrics_by_model.setdefault(str(model_id), {})
            target[metric_name] = mean
            target[f"{metric_name}_ci_low"] = ci_low
            target[f"{metric_name}_ci_high"] = ci_high

        ranking = [
            {"model_id": row.model_id, "metric": row.metric, "value": row.value, "rank": row.rank}
            for row in rank_models(metrics_by_model, metric="mse", higher_is_better=False)
        ] if metrics_by_model and all("mse" in model_metrics for model_metrics in metrics_by_model.values()) else []
        tasks[task_id] = {
            "status": "seed_aggregated",
            "source_modality": representative.get("source_modality"),
            "target_modality": representative.get("target_modality"),
            "metrics": metrics,
            "metrics_by_model": metrics_by_model,
            "ranking": ranking,
            "failures": [],
            "notes": [
                "aggregate across seed_results; per-seed concrete results are stored in seed_results",
                *_representative_notes(representative),
            ],
        }
    return tasks


def _representative_notes(task_payload: dict[str, object]) -> list[str]:
    notes = task_payload.get("notes", [])
    return [str(note) for note in notes] if isinstance(notes, list) else []


def _finite_row_float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool) and np.isfinite(float(value)):
        return float(value)
    return None


def _aggregate_seed_ranks(seed_results: list[dict[str, object]]) -> list[dict[str, object]]:
    concrete_ranks = _collect_concrete_seed_ranks(seed_results)
    ranks_by_model = concrete_ranks or _collect_aggregate_seed_ranks(seed_results)
    return sorted(
        (
            {
                "model_id": model_id,
                "mean_rank": float(np.mean(ranks)),
                "std_rank": float(np.std(ranks)),
                "tasks_ranked": len(ranks),
                "n_seeds": len(seed_keys),
            }
            for model_id, values in ranks_by_model.items()
            for ranks, seed_keys in [([rank for _, rank in values], {seed_key for seed_key, _ in values})]
            if ranks
        ),
        key=lambda row: (float(row["mean_rank"]), str(row["model_id"])),
    )


def _collect_concrete_seed_ranks(seed_results: list[dict[str, object]]) -> dict[str, list[tuple[str, float]]]:
    ranks_by_model: dict[str, list[tuple[str, float]]] = {}
    for result_idx, result in enumerate(seed_results):
        seed_key = _seed_key(result, result_idx)
        tasks = result.get("tasks", {})
        if not isinstance(tasks, dict):
            continue
        for task_payload in tasks.values():
            if not isinstance(task_payload, dict):
                continue
            ranking = task_payload.get("ranking", [])
            if not isinstance(ranking, list):
                continue
            for row in ranking:
                if not isinstance(row, dict):
                    continue
                model_id = row.get("model_id")
                if model_id is None:
                    continue
                try:
                    rank = float(row.get("rank", row.get("mean_rank")))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(rank):
                    ranks_by_model.setdefault(str(model_id), []).append((seed_key, rank))
    return ranks_by_model


def _collect_aggregate_seed_ranks(seed_results: list[dict[str, object]]) -> dict[str, list[tuple[str, float]]]:
    ranks_by_model: dict[str, list[tuple[str, float]]] = {}
    for result_idx, result in enumerate(seed_results):
        seed_key = _seed_key(result, result_idx)
        aggregate = result.get("aggregate", {})
        if not isinstance(aggregate, dict):
            continue
        for row in aggregate.get("aggregate_rank", []):
            if not isinstance(row, dict):
                continue
            model_id = row.get("model_id")
            if model_id is None:
                continue
            try:
                rank = float(row.get("mean_rank", row.get("rank")))
            except (TypeError, ValueError):
                continue
            if np.isfinite(rank):
                ranks_by_model.setdefault(str(model_id), []).append((seed_key, rank))
    return ranks_by_model


def _seed_key(result: dict[str, object], fallback_index: int) -> str:
    value = result.get("seed", fallback_index)
    return str(value)


def _aggregate_seed_metrics(seed_results: list[dict[str, object]]) -> list[dict[str, object]]:
    values: dict[tuple[str, str, str], list[float]] = {}
    for result in seed_results:
        tasks = result.get("tasks", {})
        if not isinstance(tasks, dict):
            continue
        for task_id, task_payload in tasks.items():
            if not isinstance(task_payload, dict):
                continue
            metrics_by_model = task_payload.get("metrics_by_model", {})
            if isinstance(metrics_by_model, dict):
                for model_id, metrics in metrics_by_model.items():
                    if isinstance(metrics, dict):
                        _collect_metric_values(values, str(task_id), str(model_id), metrics)
            task_metrics = task_payload.get("metrics", {})
            if isinstance(task_metrics, dict):
                _collect_metric_values(values, str(task_id), "task_metric", task_metrics)
    rows = []
    for (task_id, model_id, metric), metric_values in sorted(values.items()):
        arr = np.asarray(metric_values, dtype=float)
        if arr.size == 0 or not np.isfinite(arr).all():
            continue
        if arr.size == 1:
            ci_low = ci_high = float(arr[0])
        else:
            ci_low, ci_high = bootstrap_ci(arr, seed=_stable_metric_seed(task_id, model_id, metric), n_boot=200)
        rows.append(
            {
                "task_id": task_id,
                "model_id": model_id,
                "metric": metric,
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n_seeds": int(arr.size),
            }
        )
    return rows


def _collect_metric_values(
    values: dict[tuple[str, str, str], list[float]],
    task_id: str,
    model_id: str,
    metrics: dict[str, Any],
) -> None:
    for metric, value in metrics.items():
        if metric.endswith(("_ci_low", "_ci_high")):
            continue
        if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool) and np.isfinite(float(value)):
            values.setdefault((task_id, model_id, metric), []).append(float(value))


def _add_degenerate_task_metric_ci(payload: dict[str, object]) -> None:
    tasks = payload.get("tasks", {})
    if not isinstance(tasks, dict):
        return
    for task_payload in tasks.values():
        if not isinstance(task_payload, dict):
            continue
        metrics = task_payload.get("metrics")
        if isinstance(metrics, dict):
            _add_degenerate_ci(metrics)


def _add_degenerate_ci(metrics: dict[str, Any]) -> None:
    for metric, value in list(metrics.items()):
        if metric.endswith(("_ci_low", "_ci_high")):
            continue
        if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool) and np.isfinite(float(value)):
            metrics.setdefault(f"{metric}_ci_low", float(value))
            metrics.setdefault(f"{metric}_ci_high", float(value))


def _seed_aggregate_csv(rows: object) -> str:
    header = ("task_id", "model_id", "metric", "mean", "std", "ci_low", "ci_high", "n_seeds")
    lines = [",".join(header)]
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                lines.append(",".join(_csv_cell(row.get(key, "")) for key in header))
    return "\n".join(lines) + "\n"


def _csv_cell(value: object) -> str:
    text = str(value)
    if any(char in text for char in (",", "\"", "\n")):
        return "\"" + text.replace("\"", "\"\"") + "\""
    return text


def _stable_metric_seed(task_id: str, model_id: str, metric: str) -> int:
    return sum(ord(char) for char in f"{task_id}:{model_id}:{metric}") % 1_000_003
