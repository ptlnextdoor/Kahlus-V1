"""KTM-vs-SSM failure analysis (Sprint 3B) — PROPOSED / SYNTHETIC ONLY, CPU / single-GPU.

Diagnoses *why* the trained ``TorchKTM`` loses to the strongest matched baseline (``ssm_fallback``)
on the synthetic Transition Gym, without launching A100, scaling, or claiming recovery. It:

* breaks the held-out response error down by **horizon / perturbation / channel** for both models,
* runs a KTM-vs-SSM head-to-head **autopsy** on the *same* held-out episodes (error ratios, where
  SSM wins, where KTM wins),
* reports the **loss-component** breakdown (trajectory / profile / NLL) + calibration,
* optionally sweeps small **loss / architecture ablations** (opt-in; off by default),
* derives a **data-driven failure hypothesis** from the measured slice ratios.

No real data, no clinical / consciousness / Orch-OR claim, no scientific-superiority claim. This
tool only *reads* a trained model — it never relaxes a gate. The ``synthetic_ktm_recovery`` scope
stays blocked: recovery is earned only by the red-team battery (``redteam_runner``), never by a
single locked win here.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from neurotwin.baseline_runner import (
    regression_metrics,
    selected_model_predictions,
    transition_gym_regression_task,
)
from neurotwin.repro import write_json
from neurotwin.runtime.distributed import DistributedInfo
from neurotwin.training_v3.config import KTMTrainConfig
from neurotwin.training_v3.metrics_eval import evaluate_ktm, fair_ktm_vs_baselines
from neurotwin.training_v3.objective import ktm_loss
from neurotwin.training_v3.trainer import TrainingArtifacts, train_ktm

SCHEMA = "kahlus.ktm_failure_analysis.v1"
REPORT_SCHEMA = "kahlus.ktm_failure_analysis_report.v1"
SSM_MODEL_ID = "ssm_fallback"
# A single locked win never earns recovery — only the red-team battery does. Hard-coded so a lucky
# ablation cannot leak a recovery claim out of this read-only diagnostic.
RECOVERY_NOTE = (
    "recovery requires the red-team battery (>=5 seeds with positive lower bound + generator-family "
    "generalization via redteam_runner); a single locked win here does NOT earn synthetic_ktm_recovery"
)

_INFO = DistributedInfo(rank=0, local_rank=0, world_size=1)


def build_ablations(base_cfg: KTMTrainConfig) -> list[tuple[str, dict[str, Any]]]:
    """Named ablation overrides relative to ``base_cfg`` (single source of truth for the matrix).

    Dimension sweeps scale to the base config so the same matrix works at smoke and A100 scale; the
    mirrored ``configs/train/ktm_ablation_*.yaml`` files carry concrete smoke-scale numbers.
    """

    md, ed = int(base_cfg.memory_dim), int(base_cfg.embed_dim)
    w_profile = float(base_cfg.w_profile) or 0.5
    w_nll = float(base_cfg.w_nll) or 0.1
    return [
        ("trajectory_only", {"w_profile": 0.0, "w_nll": 0.0}),
        ("traj_profile", {"w_profile": w_profile, "w_nll": 0.0}),
        ("traj_nll", {"w_profile": 0.0, "w_nll": w_nll}),
        ("full_objective", {}),  # the base objective, unchanged — the reference row
        ("uncertainty_off", {"w_nll": 0.0}),
        ("uncertainty_on", {"w_nll": w_nll}),
        ("memory_dim_small", {"memory_dim": max(4, md // 2)}),
        ("memory_dim_large", {"memory_dim": md * 2}),
        ("embed_dim_small", {"embed_dim": max(4, ed // 2)}),
        ("embed_dim_large", {"embed_dim": ed * 2}),
    ]


# Derived from the single source of truth above so the two cannot drift.
ABLATION_LABELS: tuple[str, ...] = tuple(label for label, _ in build_ablations(KTMTrainConfig()))


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("inf")


def _ratios(nums: Sequence[float], dens: Sequence[float]) -> list[float]:
    return [_ratio(float(n), float(d)) for n, d in zip(nums, dens)]


def _slice_mses(se: np.ndarray) -> dict[str, list[float]]:
    """Per-perturbation / per-horizon / per-channel MSE from a ``(B, K, H, C)`` squared-error tensor."""

    return {
        "per_perturbation_mse": se.mean(axis=(0, 2, 3)).tolist(),  # (K,)
        "per_horizon_mse": se.mean(axis=(0, 1, 3)).tolist(),       # (H,)
        "per_channel_mse": se.mean(axis=(0, 1, 2)).tolist(),       # (C,)
    }


def _test_split(artifacts: TrainingArtifacts) -> tuple[Any, np.ndarray, torch.device, torch.Tensor]:
    """``(bundle, idx, device, history_tensor)`` for the held-out test episodes.

    Single source for the test-episode indexing + float32-cast + device-move contract shared by the
    autopsy and the loss-component breakdown.
    """

    bundle = artifacts.bundle
    idx = np.asarray(list(bundle.splits.test_episodes), dtype=int)
    device = torch.device(artifacts.device)
    history = torch.from_numpy(np.asarray(bundle.history_eeg, dtype=np.float32)[idx]).to(device)
    return bundle, idx, device, history


def _ktm_profile_on_test(artifacts: TrainingArtifacts) -> tuple[np.ndarray, np.ndarray]:
    """KTM response profile ``(B, K, H, C)`` and the matching target on the held-out test split."""

    bundle, idx, _device, history = _test_split(artifacts)
    artifacts.model.eval()
    with torch.no_grad():
        prof = artifacts.model.predict_response_profile(history).detach().cpu().numpy().astype(np.float64)
    target = np.asarray(bundle.response_eeg, dtype=np.float64)[idx]  # (B, K, H, C)
    return prof, target


def _ssm_profile_on_test(cfg: KTMTrainConfig, shape: tuple[int, int, int, int]) -> tuple[np.ndarray, str]:
    """Strongest-baseline (best-val SSM) prediction reshaped to ``(B, K, H, C)`` on the same split.

    The task is rebuilt from the same world config + seed, so its leakage-checked episode splits and
    ordering match the KTM bundle's; the flattened ``(K*H*C)`` prediction reshapes back element-for-
    element to the KTM target. Imports the shared selected-baseline helper so baseline code is reused,
    never modified or weakened.
    """

    b, k, h, c = shape
    task = transition_gym_regression_task(cfg.to_world_config())
    steps = int(cfg.baseline_train_steps or cfg.steps)
    res = selected_model_predictions(SSM_MODEL_ID, task, steps, int(cfg.seed))
    pred = np.asarray(res["pred_best_val"], dtype=np.float64)  # (n_test, K*H*C)
    if pred.shape != (b, k * h * c):
        raise ValueError(f"ssm prediction shape {pred.shape} != expected {(b, k * h * c)}")
    return pred.reshape(b, k, h, c), str(res["checkpoint_policy"])


def _where(ratios: Sequence[float], *, ssm_wins: bool) -> list[int]:
    """Indices where SSM beats KTM (ratio > 1) or KTM beats SSM (ratio < 1)."""

    if ssm_wins:
        return [i for i, r in enumerate(ratios) if r > 1.0]
    return [i for i, r in enumerate(ratios) if r < 1.0]


def _extreme(ratios: Sequence[float], *, worst: bool) -> dict[str, float] | None:
    if not ratios:
        return None
    arr = np.asarray(ratios, dtype=np.float64)
    idx = int(np.argmax(arr)) if worst else int(np.argmin(arr))
    return {"index": idx, "ratio_ktm_over_ssm": float(arr[idx])}


def ktm_vs_ssm_autopsy(artifacts: TrainingArtifacts, cfg: KTMTrainConfig) -> dict[str, Any]:
    """KTM-vs-SSM head-to-head on the same held-out episodes: overall + per-slice error ratios."""

    ktm_prof, target = _ktm_profile_on_test(artifacts)
    b, k, h, c = target.shape
    ssm_prof, ssm_policy = _ssm_profile_on_test(cfg, (b, k, h, c))

    se_ktm = (ktm_prof - target) ** 2
    se_ssm = (ssm_prof - target) ** 2
    flat_t = target.reshape(b, -1)
    ktm_overall = regression_metrics(flat_t, ktm_prof.reshape(b, -1))
    ssm_overall = regression_metrics(flat_t, ssm_prof.reshape(b, -1))
    overall_ratio = _ratio(ktm_overall["mse"], ssm_overall["mse"])

    ktm_slices = _slice_mses(se_ktm)
    ssm_slices = _slice_mses(se_ssm)
    per_k_ratio = _ratios(ktm_slices["per_perturbation_mse"], ssm_slices["per_perturbation_mse"])
    per_h_ratio = _ratios(ktm_slices["per_horizon_mse"], ssm_slices["per_horizon_mse"])
    per_c_ratio = _ratios(ktm_slices["per_channel_mse"], ssm_slices["per_channel_mse"])

    # Horizon-averaged response-profile distance (profile magnitude error).
    rp_ktm = float(((ktm_prof.mean(axis=2) - target.mean(axis=2)) ** 2).mean())
    rp_ssm = float(((ssm_prof.mean(axis=2) - target.mean(axis=2)) ** 2).mean())

    finite = bool(np.isfinite(se_ktm).all() and np.isfinite(se_ssm).all())
    return {
        "n_test_episodes": int(b),
        "dims": {"K": int(k), "H": int(h), "C": int(c)},
        "ssm_model_id": SSM_MODEL_ID,
        "ssm_checkpoint_policy": ssm_policy,
        "finite": finite,
        "overall": {
            "ktm": ktm_overall,
            "ssm": ssm_overall,
            "ratio_ktm_over_ssm": overall_ratio,
            "ssm_beats_ktm": bool(overall_ratio > 1.0),
        },
        "per_perturbation": {
            "ktm_mse": ktm_slices["per_perturbation_mse"],
            "ssm_mse": ssm_slices["per_perturbation_mse"],
            "ratio_ktm_over_ssm": per_k_ratio,
        },
        "per_horizon": {
            "ktm_mse": ktm_slices["per_horizon_mse"],
            "ssm_mse": ssm_slices["per_horizon_mse"],
            "ratio_ktm_over_ssm": per_h_ratio,
        },
        "per_channel": {
            "ktm_mse": ktm_slices["per_channel_mse"],
            "ssm_mse": ssm_slices["per_channel_mse"],
            "ratio_ktm_over_ssm": per_c_ratio,
        },
        "response_profile_distance": {"ktm": rp_ktm, "ssm": rp_ssm},
        "where_ssm_beats_ktm": {
            "perturbations": _where(per_k_ratio, ssm_wins=True),
            "horizons": _where(per_h_ratio, ssm_wins=True),
            "channels": _where(per_c_ratio, ssm_wins=True),
            "worst_perturbation": _extreme(per_k_ratio, worst=True),
        },
        "where_ktm_beats_ssm": {
            "perturbations": _where(per_k_ratio, ssm_wins=False),
            "horizons": _where(per_h_ratio, ssm_wins=False),
            "channels": _where(per_c_ratio, ssm_wins=False),
            "best_perturbation": _extreme(per_k_ratio, worst=False),
        },
    }


def loss_component_breakdown(artifacts: TrainingArtifacts, cfg: KTMTrainConfig) -> dict[str, Any]:
    """Average trajectory / profile / NLL loss terms over the test split (+ calibration summary).

    Evaluates the real model-forward path once per perturbation (faithful ``log_var`` from the
    uncertainty head) and averages — no dataset coupling, no training step.
    """

    bundle, idx, device, history = _test_split(artifacts)
    profile_target = torch.from_numpy(np.asarray(bundle.response_eeg, dtype=np.float32)[idx]).to(device)
    model = artifacts.model
    model.eval()
    n_k = int(profile_target.shape[1])

    components: list[dict[str, float]] = []
    with torch.no_grad():
        for k_idx in range(n_k):
            kvec = torch.full((history.shape[0],), k_idx, dtype=torch.long, device=device)
            pred, log_var, prof_pred = model(history, kvec, return_profile=True)
            _loss, comp = ktm_loss(
                pred, log_var, profile_target[:, k_idx], cfg,
                profile_pred=prof_pred, profile_target=profile_target,
            )
            components.append(comp)

    avg = {key: float(np.mean([c[key] for c in components])) for key in ("trajectory", "profile", "nll", "total")}
    avg["calibration"] = evaluate_ktm(model, bundle, bundle.splits.test_episodes, device)["calibration"]
    avg["finite"] = bool(all(np.isfinite(avg[key]) for key in ("trajectory", "profile", "nll", "total")))
    return avg


def run_ablations(
    base_cfg: KTMTrainConfig,
    *,
    ablations: Sequence[tuple[str, dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Train each ablation (cpu_smoke-scale) and record test MSE, loss terms, and the locked comparison.

    ``recovery_allowed`` is hard ``False`` on every row: recovery is earned only by the red-team
    battery, never by a single locked win. ``beats_best_baseline_locked`` reports the honest locked
    single-seed comparison without conflating it with recovery.
    """

    matrix = list(ablations) if ablations is not None else build_ablations(base_cfg)
    rows: list[dict[str, Any]] = []
    for label, overrides in matrix:
        cfg = replace(base_cfg, **overrides).validate()
        artifacts = train_ktm(cfg, out_dir=None, dist_info=_INFO)
        device = torch.device(artifacts.device)
        _baseline, ktm_test, comparison, _steps = fair_ktm_vs_baselines(
            artifacts.model, artifacts.bundle, cfg, device=device, world_size=1
        )
        loss_comp = loss_component_breakdown(artifacts, cfg)
        rows.append({
            "label": label,
            "overrides": dict(overrides),
            "ktm_test_mse": float(ktm_test["trajectory"]["mse"]),
            "best_baseline": comparison["best_baseline"],
            "best_baseline_mse": comparison["best_baseline_mse"],
            "relative_improvement": comparison["relative_improvement"],
            "comparison_locked": bool(comparison["comparison_locked"]),
            "beats_best_baseline_locked": bool(comparison["ktm_beats_baselines"]),
            "recovery_allowed": False,
            "loss_components": {key: loss_comp[key] for key in ("trajectory", "profile", "nll", "total")},
        })
    return rows


def best_failure_hypothesis(autopsy: dict[str, Any]) -> str:
    """Pick the dominant failure mode from the measured slice ratios (not asserted up front)."""

    overall_ratio = float(autopsy["overall"]["ratio_ktm_over_ssm"])
    h_ratios = [float(r) for r in autopsy["per_horizon"]["ratio_ktm_over_ssm"]]
    k_ratios = [float(r) for r in autopsy["per_perturbation"]["ratio_ktm_over_ssm"]]

    if overall_ratio <= 1.0:
        return (
            f"KTM matches/beats SSM overall on this single run (ktm/ssm MSE ratio={overall_ratio:.3g}); "
            "no dominant wound localized here — confirm stability with a multi-seed red-team before "
            "any architecture change."
        )
    rising = bool(h_ratios and h_ratios[-1] > h_ratios[0] * 1.25)
    k_spread = (max(k_ratios) / min(k_ratios)) if (k_ratios and min(k_ratios) > 0) else float("inf")
    if rising:
        return (
            f"Long-horizon degradation: KTM error grows with horizon vs SSM (ratio {h_ratios[0]:.3g} -> "
            f"{h_ratios[-1]:.3g}). The profile decoder / rollout weakens at longer horizons — focus on "
            "the decoder horizon conditioning before scaling."
        )
    if k_spread > 1.5:
        worst = autopsy["where_ssm_beats_ktm"].get("worst_perturbation")
        worst_str = f" (worst perturbation index {worst['index']}, ratio {worst['ratio_ktm_over_ssm']:.3g})" if worst else ""
        return (
            f"Perturbation-specific misalignment: KTM loses unevenly across perturbations "
            f"(ratio spread {k_spread:.3g}){worst_str}. The operator path may be aligned to only some "
            "perturbations — inspect the operator/profile-decoder split per perturbation."
        )
    return (
        f"Global underfit: KTM is uniformly ~{overall_ratio:.3g}x worse than SSM across horizon, "
        "perturbation, and channel slices. The structured KTM underfits the easy linear generator — "
        "raise capacity (embed/memory dim) or optimizer steps before changing architecture."
    )


def run_failure_analysis(
    base_cfg: KTMTrainConfig,
    *,
    run_ablations_flag: bool = False,
) -> dict[str, Any]:
    """Train the base KTM once, run the autopsy + loss breakdown (+ optional ablations), report.

    ``run_ablations_flag`` defaults to False so the normal CPU command runs the fast base autopsy
    only and never surprise-runs the full sweep.
    """

    cfg = base_cfg.validate()
    artifacts = train_ktm(cfg, out_dir=None, dist_info=_INFO)
    device = torch.device(artifacts.device)
    _baseline, _ktm_test, comparison, _steps = fair_ktm_vs_baselines(
        artifacts.model, artifacts.bundle, cfg, device=device, world_size=1
    )
    autopsy = ktm_vs_ssm_autopsy(artifacts, cfg)
    loss_comp = loss_component_breakdown(artifacts, cfg)
    ablations = run_ablations(cfg) if run_ablations_flag else []
    hypothesis = best_failure_hypothesis(autopsy)

    return {
        "schema": REPORT_SCHEMA,
        "result_schema": SCHEMA,
        "claim_status": "synthetic_model_failure_analysis",
        "branch": "v3",
        "synthetic_only": True,
        "claim_scope": "synthetic_ktm_training_harness",
        "recovery_claim_allowed": False,
        "best_baseline": comparison["best_baseline"],
        "comparison": comparison,
        "autopsy": autopsy,
        "loss_components": loss_comp,
        "ablations_ran": bool(run_ablations_flag),
        "ablations": ablations,
        "best_failure_hypothesis": hypothesis,
        "config": cfg.as_dict(),
        "notes": [
            "PROPOSED / SYNTHETIC ONLY — synthetic Transition Gym; no real EEG, clinical, control, "
            "consciousness, Orch-OR, or model-superiority claim.",
            RECOVERY_NOTE,
        ],
    }


def _fmt(x: Any) -> str:
    try:
        return f"{float(x):.4g}"
    except (TypeError, ValueError):
        return str(x)


def format_failure_md(report: dict[str, Any]) -> str:
    a = report["autopsy"]
    ov = a["overall"]
    lc = report["loss_components"]
    lines = [
        "# KTM Failure Analysis vs SSM (SYNTHETIC ONLY)",
        "",
        f"- schema: {report['schema']}",
        f"- claim_scope: {report['claim_scope']}",
        f"- recovery_claim_allowed: **{report['recovery_claim_allowed']}**",
        f"- best_baseline: {report['best_baseline']}",
        f"- ssm_beats_ktm: **{ov['ssm_beats_ktm']}**  (ktm/ssm MSE ratio: {_fmt(ov['ratio_ktm_over_ssm'])})",
        "",
        "## Overall (held-out test)",
        "",
        "| model | mse | mae | r2 | pearson_r |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| KTM | {_fmt(ov['ktm']['mse'])} | {_fmt(ov['ktm']['mae'])} | {_fmt(ov['ktm']['r2'])} | {_fmt(ov['ktm']['pearson_r'])} |",
        f"| SSM | {_fmt(ov['ssm']['mse'])} | {_fmt(ov['ssm']['mae'])} | {_fmt(ov['ssm']['r2'])} | {_fmt(ov['ssm']['pearson_r'])} |",
        "",
        "## Error by horizon (ktm / ssm / ratio)",
        "",
        "| horizon | ktm_mse | ssm_mse | ratio_ktm_over_ssm |",
        "| ---: | ---: | ---: | ---: |",
    ]
    ph = a["per_horizon"]
    for i, (kk, ss, rr) in enumerate(zip(ph["ktm_mse"], ph["ssm_mse"], ph["ratio_ktm_over_ssm"])):
        lines.append(f"| {i} | {_fmt(kk)} | {_fmt(ss)} | {_fmt(rr)} |")
    lines += [
        "",
        "## Error by perturbation (ktm / ssm / ratio)",
        "",
        "| perturbation | ktm_mse | ssm_mse | ratio_ktm_over_ssm |",
        "| ---: | ---: | ---: | ---: |",
    ]
    pk = a["per_perturbation"]
    for i, (kk, ss, rr) in enumerate(zip(pk["ktm_mse"], pk["ssm_mse"], pk["ratio_ktm_over_ssm"])):
        lines.append(f"| {i} | {_fmt(kk)} | {_fmt(ss)} | {_fmt(rr)} |")
    lines += [
        "",
        "## Loss components (test, averaged over perturbations)",
        "",
        f"- trajectory: {_fmt(lc['trajectory'])}  profile: {_fmt(lc['profile'])}  "
        f"nll: {_fmt(lc['nll'])}  total: {_fmt(lc['total'])}",
        f"- calibration 1sigma coverage: {_fmt(lc['calibration']['empirical_coverage_1sigma'])} "
        f"(nominal {_fmt(lc['calibration']['nominal_1sigma'])})",
        "",
    ]
    if report["ablations_ran"]:
        lines += [
            "## Ablations (cpu_smoke)",
            "",
            "| label | ktm_test_mse | best_baseline | rel_improvement | beats_baseline_locked | recovery |",
            "| --- | ---: | --- | ---: | --- | --- |",
        ]
        for row in report["ablations"]:
            lines.append(
                f"| {row['label']} | {_fmt(row['ktm_test_mse'])} | {row['best_baseline']} | "
                f"{_fmt(row['relative_improvement'])} | {row['beats_best_baseline_locked']} | "
                f"{row['recovery_allowed']} |"
            )
        lines.append("")
    lines += [
        "## Best failure hypothesis",
        "",
        report["best_failure_hypothesis"],
        "",
        "## Bottom line",
        "",
        "Synthetic failure analysis only. The recovery scope stays blocked; " + RECOVERY_NOTE + ".",
        "",
    ]
    return "\n".join(lines)


def _error_by_slice_csv(report: dict[str, Any]) -> str:
    a = report["autopsy"]
    rows = ["dimension,index,ktm_mse,ssm_mse,ratio_ktm_over_ssm"]
    for dim, key in (("horizon", "per_horizon"), ("perturbation", "per_perturbation"), ("channel", "per_channel")):
        block = a[key]
        for i, (kk, ss, rr) in enumerate(zip(block["ktm_mse"], block["ssm_mse"], block["ratio_ktm_over_ssm"])):
            rows.append(f"{dim},{i},{kk!r},{ss!r},{rr!r}")
    return "\n".join(rows) + "\n"


def write_failure_analysis(out_dir: str | Path, report: dict[str, Any]) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = write_json(out / "ktm_failure_analysis.json", report)
    md_path = out / "ktm_failure_analysis.md"
    md_path.write_text(format_failure_md(report), encoding="utf-8")
    csv_path = out / "ktm_error_by_slice.csv"
    csv_path.write_text(_error_by_slice_csv(report), encoding="utf-8")
    return {
        "ktm_failure_analysis_json": json_path,
        "ktm_failure_analysis_md": md_path,
        "ktm_error_by_slice_csv": csv_path,
    }
