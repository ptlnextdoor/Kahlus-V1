"""Multi-seed + generator-family red-team runner for the KTM recovery candidate (SYNTHETIC ONLY).

Drives the adversarial falsifier end to end on CPU / single GPU (never A100):

* trains the trainable ``TorchKTM`` and the *symmetric* best-val baselines once per seed on the
  original (``linear``) generator and records each fair comparison as a :class:`SeedOutcome`;
* re-runs the comparison on alternate generator families (``nonlinear`` / ``quadratic``) to test
  whether the win survives when the generator is bent away from the KTM's inductive bias;
* feeds both into :func:`recovery_redteam_gate`, which keeps ``synthetic_ktm_recovery`` blocked
  unless every check passes.

No A100, no real data, no scientific-superiority claim — a surviving candidate is still synthetic.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from neurotwin.baseline_runner import run_baselines, transition_gym_regression_task
from neurotwin.repro import write_json
from neurotwin.runtime.distributed import DistributedInfo
from neurotwin.training_v3.config import KTMTrainConfig
from neurotwin.training_v3.metrics_eval import evaluate_ktm, ktm_vs_baselines
from neurotwin.training_v3.redteam import (
    MIN_SEEDS,
    GeneratorFamilyOutcome,
    SeedOutcome,
    recovery_redteam_gate,
)
from neurotwin.training_v3.trainer import train_ktm

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_FAMILIES: tuple[str, ...] = ("linear", "nonlinear", "quadratic")
REPORT_SCHEMA = "kahlus.ktm_recovery_redteam_report.v1"

_INFO = DistributedInfo(rank=0, local_rank=0, world_size=1)


def _profile_diagnostics(model: Any, bundle: Any, device: torch.device) -> dict[str, Any]:
    """Per-perturbation / per-horizon error + response-profile loss for the trained KTM."""

    idx = np.asarray(bundle.splits.test_episodes, dtype=int)
    history = torch.from_numpy(np.asarray(bundle.history_eeg, dtype=np.float32)[idx]).to(device)
    model.eval()
    with torch.no_grad():
        prof = model.predict_response_profile(history).detach().cpu().numpy().astype(np.float64)
    target = np.asarray(bundle.response_eeg, dtype=np.float64)[idx]  # (B, K, H, C)
    se = (prof - target) ** 2
    return {
        "per_perturbation_mse": se.mean(axis=(0, 2, 3)).tolist(),
        "per_horizon_mse": se.mean(axis=(0, 1, 3)).tolist(),
        "response_profile_mse": float(((prof.mean(axis=2) - target.mean(axis=2)) ** 2).mean()),
        "heldout_composition_note": (
            "TorchKTM predicts single-perturbation response profiles; it has no compose operator "
            "and the bundle has no composed-response ground truth, so held-out composition error is "
            "not scored for the model (composition generalization is structurally absent)."
        ),
        "heldout_compositions": len(bundle.splits.heldout_compositions),
        "finite": bool(np.isfinite(se).all()),
    }


def _one_comparison(cfg: KTMTrainConfig) -> dict[str, Any]:
    """Train the KTM + symmetric best-val baselines once; return the locked fair comparison."""

    cfg = cfg.validate()
    artifacts = train_ktm(cfg, out_dir=None, dist_info=_INFO)
    device = torch.device(artifacts.device)
    baseline_steps = cfg.baseline_train_steps or cfg.steps
    task = transition_gym_regression_task(cfg.to_world_config())
    baseline_result = run_baselines(
        task, seed=cfg.seed, train_steps=baseline_steps, select_best_val=True
    )
    ktm_test = evaluate_ktm(
        artifacts.model, artifacts.bundle, artifacts.bundle.splits.test_episodes, device
    )
    comparison = ktm_vs_baselines(
        ktm_test["trajectory"]["mse"],
        baseline_result.metrics_by_model,
        ktm_train_steps=cfg.steps,
        baseline_train_steps=baseline_steps,
        ktm_world_size=1,
        ktm_global_batch_size=cfg.batch_size,
        margin=cfg.recovery_margin,
    )
    return {
        "comparison": comparison,
        "ktm_checkpoint_policy": "best_val" if artifacts.selected_checkpoint == "best_val" else "final_step",
        "baseline_checkpoint_policy": "best_val" if baseline_result.selection_policy == "symmetric_best_val" else "final_step",
        "diagnostics": _profile_diagnostics(artifacts.model, artifacts.bundle, device),
        "ktm_test": ktm_test["trajectory"],
    }


def _seed_outcome(seed: int, result: dict[str, Any]) -> SeedOutcome:
    c = result["comparison"]
    return SeedOutcome(
        seed=int(seed),
        ktm_mse=float(c["ktm_mse"]),
        best_baseline=c["best_baseline"] or "none",
        best_baseline_mse=float(c["best_baseline_mse"] if c["best_baseline_mse"] is not None else float("inf")),
        relative_improvement=float(c["relative_improvement"] or 0.0),
        comparison_locked=bool(c["comparison_locked"]),
        ktm_checkpoint_policy=result["ktm_checkpoint_policy"],
        baseline_checkpoint_policy=result["baseline_checkpoint_policy"],
    )


def architecture_affinity_summary(
    families: Sequence[GeneratorFamilyOutcome], margin: float
) -> dict[str, Any]:
    """Does the win look aligned to the generator? Wins on linear but fails on >=1 alt family."""

    by_family = {f.family: float(f.relative_improvement) for f in families}
    linear = by_family.get("linear")
    alternates = {k: v for k, v in by_family.items() if k != "linear"}
    linear_wins = linear is not None and linear >= margin
    alt_failures = [k for k, v in alternates.items() if v < margin]
    appears_aligned = bool(linear_wins and alt_failures)
    return {
        "by_family_relative_improvement": by_family,
        "linear_wins": bool(linear_wins),
        "alternate_family_failures": sorted(alt_failures),
        "appears_generator_aligned": appears_aligned,
        "interpretation": (
            "KTM wins on the linear generator but fails on at least one alternate family — the win "
            "may be alignment between the model's operator path and the generator's structure, not "
            "general recovery."
            if appears_aligned
            else "No linear-only affinity detected across the tested generator families."
        ),
    }


def run_redteam(
    base_cfg: KTMTrainConfig,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    families: Sequence[str] = DEFAULT_FAMILIES,
    margin: float | None = None,
    min_seeds: int = MIN_SEEDS,
) -> dict[str, Any]:
    """Run the multi-seed + generator-family battery and return the full red-team report dict."""

    margin = float(base_cfg.recovery_margin if margin is None else margin)

    seed_outcomes: list[SeedOutcome] = []
    seed_details: list[dict[str, Any]] = []
    base_diagnostics: dict[str, Any] | None = None
    for s in seeds:
        result = _one_comparison(replace(base_cfg, seed=int(s), dynamics_family="linear"))
        seed_outcomes.append(_seed_outcome(int(s), result))
        seed_details.append({"seed": int(s), "ktm_test": result["ktm_test"]})
        if base_diagnostics is None:
            base_diagnostics = result["diagnostics"]

    family_outcomes: list[GeneratorFamilyOutcome] = []
    for fam in families:
        result = _one_comparison(replace(base_cfg, seed=int(base_cfg.seed), dynamics_family=str(fam)))
        c = result["comparison"]
        family_outcomes.append(
            GeneratorFamilyOutcome(
                family=str(fam),
                ktm_mse=float(c["ktm_mse"]),
                best_baseline=c["best_baseline"] or "none",
                best_baseline_mse=float(c["best_baseline_mse"] if c["best_baseline_mse"] is not None else float("inf")),
                relative_improvement=float(c["relative_improvement"] or 0.0),
                margin=margin,
            )
        )

    dossier = recovery_redteam_gate(
        seeds=seed_outcomes, families=family_outcomes, margin=margin, min_seeds=int(min_seeds)
    )
    return {
        "schema": REPORT_SCHEMA,
        "claim_status": "synthetic_recovery_candidate_under_review",
        "branch": "v3",
        "recovery_allowed": dossier["recovery_allowed"],
        "blocker_reasons": dossier["blocker_reasons"],
        "redteam_gate": dossier,
        "architecture_affinity": architecture_affinity_summary(family_outcomes, margin),
        "diagnostics": base_diagnostics or {},
        "seed_details": seed_details,
        "config": base_cfg.as_dict(),
    }


def format_redteam_md(report: dict[str, Any]) -> str:
    g = report["redteam_gate"]
    summ = g["seed_summary"]
    aff = report["architecture_affinity"]
    lines = [
        "# KTM Recovery Candidate — Red-Team Falsifier Report (SYNTHETIC ONLY)",
        "",
        f"- schema: {report['schema']}",
        f"- recovery_allowed: **{report['recovery_allowed']}**",
        f"- blocker_reasons: {report['blocker_reasons'] or 'none'}",
        f"- margin: {g['margin']}  min_seeds: {g['min_seeds']}",
        "",
        "## Multi-seed (original linear generator)",
        "",
        f"- n_seeds: {summ['n_seeds']}  mean: {summ['mean']:.4g}  std: {summ['std']:.4g}  "
        f"min: {summ['min']:.4g}  max: {summ['max']:.4g}",
        f"- lower_bound_95: {summ['lower_bound_95']:.4g}  n_positive: {summ['n_positive']}/{summ['n_seeds']}",
        "",
        "| seed | rel_improvement | best_baseline | ktm_mse | locked | ktm_ckpt | baseline_ckpt |",
        "| ---: | ---: | --- | ---: | --- | --- | --- |",
    ]
    for s in g["per_seed"]:
        lines.append(
            f"| {s['seed']} | {s['relative_improvement']:.4g} | {s['best_baseline']} | "
            f"{s['ktm_mse']:.4g} | {s['comparison_locked']} | {s['ktm_checkpoint_policy']} | "
            f"{s['baseline_checkpoint_policy']} |"
        )
    lines += [
        "",
        "## Generator-family generalization",
        "",
        f"- appears_generator_aligned: **{aff['appears_generator_aligned']}**",
        f"- {aff['interpretation']}",
        "",
        "| family | rel_improvement | best_baseline | recovery_allowed_for_family |",
        "| --- | ---: | --- | --- |",
    ]
    for f in g["generator_families"]:
        lines.append(
            f"| {f['family']} | {f['relative_improvement']:.4g} | {f['best_baseline']} | "
            f"{f['recovery_allowed_for_family']} |"
        )
    diag = report.get("diagnostics", {})
    lines += [
        "",
        "## Diagnostics (base seed, linear)",
        "",
        f"- response_profile_mse: {diag.get('response_profile_mse')}",
        f"- per_perturbation_mse: {diag.get('per_perturbation_mse')}",
        f"- per_horizon_mse: {diag.get('per_horizon_mse')}",
        f"- {diag.get('heldout_composition_note', '')}",
        "",
        "## Bottom line",
        "",
        "Synthetic recovery candidate. A surviving candidate is still synthetic-only evidence — no "
        "real-EEG, clinical, control, consciousness, or model-superiority claim is implied.",
        "",
    ]
    return "\n".join(lines)


def write_redteam_report(out_dir: str | Path, report: dict[str, Any]) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = write_json(out / "ktm_redteam_report.json", report)
    md_path = out / "ktm_redteam_report.md"
    md_path.write_text(format_redteam_md(report), encoding="utf-8")
    return {"ktm_redteam_report_json": json_path, "ktm_redteam_report_md": md_path}
