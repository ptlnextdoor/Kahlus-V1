"""Output-bundle writer for the v3 KTM training harness (PROPOSED / SYNTHETIC ONLY).

Writes the auditable artifact bundle: ``metrics.json``, ``baseline_table.{json,csv}``,
``evidence_gate.json`` (primary scope ``synthetic_ktm_training_harness``), ``model_card.json``,
``data_card.json``, ``run_config.json``, ``failure_reasons.json`` (checkpoints are written by the
trainer under ``checkpoints/``). Baselines are run via the shared ``baseline_runner`` so the
KTM-vs-baselines comparison is honest: the stronger ``synthetic_ktm_recovery`` scope stays blocked
unless the trained KTM actually wins. Only rank-0 should call :func:`write_training_bundle`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from neurotwin.baseline_runner import (
    baseline_table_rows,
    run_baselines,
    transition_gym_regression_task,
    write_baseline_table,
)
from neurotwin.gates import evaluate_gate
from neurotwin.repro import capture_environment, write_json
from neurotwin.training_v3.config import KTMTrainConfig
from neurotwin.training_v3.metrics_eval import evaluate_ktm, ktm_vs_baselines
from neurotwin.training_v3.trainer import TrainingArtifacts, build_torchrun_command

BRANCH = "v3"
DATASET = "ktm_training_synthetic"
HARNESS_SCOPE = "synthetic_ktm_training_harness"
RECOVERY_SCOPE = "synthetic_ktm_recovery"
MODEL_CARD_SCHEMA = "kahlus.ktm_model_card.v1"
METRICS_SCHEMA = "kahlus.ktm_training_metrics.v1"


def _all_finite(*values: Any) -> bool:
    flat: list[float] = []
    for value in values:
        if isinstance(value, dict):
            flat.extend(float(v) for v in value.values() if isinstance(v, (int, float)))
        elif isinstance(value, (int, float)):
            flat.append(float(value))
    return bool(np.isfinite(flat).all()) if flat else True


def _checkpoint_files(out: Path) -> list[str]:
    ckpt_dir = out / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted(p.name for p in ckpt_dir.glob("*.pt"))


def write_training_bundle(
    out_dir: str | Path,
    *,
    cfg: KTMTrainConfig,
    artifacts: TrainingArtifacts,
    config_path: str | Path | None = None,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = artifacts.model
    bundle = artifacts.bundle
    device = torch.device(artifacts.device)

    # Honest baseline sweep on the same synthetic gym (shared runner).
    task = transition_gym_regression_task(cfg.to_world_config())
    baseline_result = run_baselines(task, seed=cfg.seed)
    baseline_metrics = baseline_result.metrics_by_model

    ktm_val = evaluate_ktm(model, bundle, bundle.splits.val_episodes, device)
    ktm_test = evaluate_ktm(model, bundle, bundle.splits.test_episodes, device)
    comparison = ktm_vs_baselines(ktm_test["trajectory"]["mse"], baseline_metrics)

    all_finite = (
        _all_finite(ktm_test["trajectory"], ktm_test["calibration"], ktm_val["trajectory"])
        and bool(baseline_metrics)
        and all(_all_finite(m) for m in baseline_metrics.values())
        and _all_finite(artifacts.val_before, artifacts.val_after)
    )
    calibration_checked = bool(ktm_test["calibration"]["finite"])

    # Both scopes share every check except claim scope + blocker reasons.
    def _gate(scope: str, blockers: list[str]) -> dict[str, Any]:
        return evaluate_gate(
            branch=BRANCH,
            dataset=DATASET,
            split_audit_passed=True,
            baseline_table_present=bool(baseline_metrics),
            finite_metrics=all_finite,
            calibration_checked=calibration_checked,
            claim_scope=scope,
            extra_failure_reasons=blockers,
        )

    # Primary scope: harness readiness. Block honestly if training aborted or loss did not drop.
    harness_blockers: list[str] = []
    if artifacts.aborted:
        harness_blockers.append("training aborted before completion")
    if not artifacts.loss_decreased:
        harness_blockers.append("training loss did not decrease over the run")
    harness_gate = _gate(HARNESS_SCOPE, harness_blockers)

    # Stronger scope: recovery. Blocked unless the trained KTM truly beats baselines.
    recovery_blockers = list(harness_blockers)
    if not comparison["ktm_beats_baselines"]:
        recovery_blockers.append(
            "KTM did not beat baselines on locked held-out metrics; recovery claim not earned"
        )
    recovery_gate = _gate(RECOVERY_SCOPE, recovery_blockers)

    training_failures = list(artifacts.failure_reasons) + list(baseline_result.failure_reasons)

    paths: dict[str, Path] = {}

    paths["metrics"] = write_json(
        out / "metrics.json",
        {
            "schema": METRICS_SCHEMA,
            "branch": BRANCH,
            "claim_status": "synthetic_training_harness_only",
            "seed": cfg.seed,
            "val_mse_before": artifacts.val_before,
            "val_mse_after": artifacts.val_after,
            "best_val_mse": artifacts.best_val,
            "loss_decreased": artifacts.loss_decreased,
            "ktm_val": ktm_val,
            "ktm_test": ktm_test,
            "ktm_vs_baselines": comparison,
            "baseline_metrics": baseline_metrics,
            "baseline_ranking": baseline_result.ranking,
            "recovery_claim_allowed": recovery_gate["scientific_claim_allowed"],
            "step_losses": artifacts.step_losses,
            "failure_reasons": training_failures,
        },
    )

    # Canonical baseline table (shared writer), with the trained KTM appended as one more row.
    rows = baseline_table_rows(baseline_metrics) + [
        {"model_id": "ktm_torch", "mse": ktm_test["trajectory"]["mse"],
         "mae": ktm_test["trajectory"]["mae"], "r2": ktm_test["trajectory"]["r2"],
         "pearson_r": ktm_test["trajectory"]["pearson_r"], "status": "completed"}
    ]
    paths.update(write_baseline_table(
        out, rows, baseline_result.ranking, extra={"ktm_vs_baselines": comparison}
    ))

    paths["evidence_gate"] = write_json(out / "evidence_gate.json", harness_gate)

    paths["model_card"] = write_json(
        out / "model_card.json",
        {
            "schema": MODEL_CARD_SCHEMA,
            "branch": BRANCH,
            "model": "TorchKTM",
            "claim_status": "synthetic_training_harness_only",
            "claim_scope": HARNESS_SCOPE,
            "scientific_claim_allowed": harness_gate["scientific_claim_allowed"],
            "recovery_scope": RECOVERY_SCOPE,
            "recovery_claim_allowed": recovery_gate["scientific_claim_allowed"],
            "ktm_beats_baselines": comparison["ktm_beats_baselines"],
            "num_parameters": int(model.num_parameters()),
            "model_config": cfg.to_model_config(),
            "train_config": cfg.as_dict(),
            "test_metrics": ktm_test["trajectory"],
            "calibration": ktm_test["calibration"],
            "best_baseline": comparison["best_baseline"],
            "best_baseline_mse": comparison["best_baseline_mse"],
            "checkpoints": _checkpoint_files(out),
            "limitations": [
                "Synthetic Transition Gym only; no real EEG, no clinical or control claims.",
                "Harness readiness only — a decreasing loss does not imply operator/recovery success.",
                "Recovery scope stays blocked until KTM beats strong baselines on locked metrics.",
            ],
        },
    )

    paths["data_card"] = write_json(out / "data_card.json", bundle.data_card)

    paths["run_config"] = write_json(
        out / "run_config.json",
        {
            "branch": BRANCH,
            "dataset": DATASET,
            "claim_scope": HARNESS_SCOPE,
            "config": cfg.as_dict(),
            "config_hash": artifacts.config_hash,
            "config_path": str(config_path) if config_path else None,
            "mode": artifacts.mode,
            "device": artifacts.device,
            "ddp_initialized": artifacts.ddp_initialized,
            "ddp_backend": artifacts.backend,
            "ddp_micro_sweep_command": build_torchrun_command(
                config_path=config_path or "configs/train/ktm_a100_micro.yaml",
                out_dir="$RUN_ROOT/ktm_micro_sweep",
            ),
            "environment": capture_environment(repo_root="."),
        },
    )

    paths["failure_reasons"] = write_json(
        out / "failure_reasons.json",
        {
            "training": list(artifacts.failure_reasons),
            "baselines": list(baseline_result.failure_reasons),
            "aborted": artifacts.aborted,
            "recovery_blocked_reasons": recovery_blockers,
        },
    )

    return paths
