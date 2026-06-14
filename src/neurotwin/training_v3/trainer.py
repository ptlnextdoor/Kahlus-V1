"""Standalone v3 KTM training loop (PROPOSED / SYNTHETIC ONLY).

Reuses the repo's DDP helpers (``runtime/distributed``) and repro utilities — it does NOT import
the frozen v1 ``training/prepared_*`` modules, so v1 stays isolated. Supports cpu_smoke /
single_gpu / ddp modes, gradient accumulation, autocast precision, grad clipping, a finite/NaN
micro-batch skip, a loss-explosion abort, best-val checkpointing, and resume. No A100 is launched
here; ``build_torchrun_command`` only *builds* the future micro-sweep command.
"""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Iterator

import torch

from neurotwin.repro import append_jsonl, set_global_seed, stable_hash, write_json
from neurotwin.runtime.distributed import (
    DistributedInfo,
    barrier_if_distributed,
    cleanup_process_group,
    get_distributed_info,
    maybe_init_process_group,
    wrap_ddp_if_initialized,
)
from neurotwin.models.ktm import TorchKTM
from neurotwin.training_v3.checkpoint import (
    apply_resume,
    load_resume,
    resume_start_step,
    save_ktm_checkpoint,
)
from neurotwin.training_v3.config import KTMTrainConfig
from neurotwin.training_v3.dataset import make_dataloaders
from neurotwin.training_v3.metrics_eval import evaluate_ktm
from neurotwin.training_v3.objective import LossExplosionGuard, is_finite_loss, ktm_loss
from neurotwin.transition_gym import build_transition_gym

DEFAULT_TRAIN_SCRIPT = "scripts/run_ktm_train.py"


@dataclass(frozen=True)
class TrainingArtifacts:
    """Explicit result of :func:`train_ktm`, consumed by the bundle writer / script / tests."""

    model: TorchKTM
    bundle: Any  # TransitionGymBundle
    device: str
    mode: str
    dist_info: DistributedInfo
    ddp_initialized: bool
    backend: str | None
    val_before: float
    val_after: float
    best_val: float
    loss_decreased: bool
    aborted: bool
    config_hash: str
    step_losses: list[dict[str, float]] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)
    best_checkpoint: Path | None = None
    last_checkpoint: Path | None = None
    selected_checkpoint: str = "last"
    selected_checkpoint_step: int = 0


def resolve_device(mode: str, dist_info: DistributedInfo) -> torch.device:
    if mode == "cpu_smoke":
        return torch.device("cpu")
    if mode == "single_gpu":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ddp
    if torch.cuda.is_available():
        return torch.device(f"cuda:{dist_info.local_rank}")
    return torch.device("cpu")


def build_torchrun_command(
    *,
    config_path: str | Path,
    out_dir: str | Path,
    nproc: int = 8,
    script: str = DEFAULT_TRAIN_SCRIPT,
) -> list[str]:
    """Build (do NOT run) the future N-GPU torchrun command for the micro-sweep."""

    return [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={int(nproc)}",
        script,
        "--config",
        str(config_path),
        "--out-dir",
        str(out_dir),
        "--mode",
        "ddp",
    ]


def _cycle(loader: Any) -> Iterator[Any]:
    while True:
        for batch in loader:
            yield batch


def _autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision in {"bf16", "fp16"}:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return lambda: torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext


def _val_mse(model: TorchKTM, bundle: Any, episodes: Any, device: torch.device) -> float:
    return float(evaluate_ktm(model, bundle, episodes, device)["trajectory"]["mse"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(out: Path | None, dist_info: DistributedInfo, **event: Any) -> None:
    """Append one event to ``progress.jsonl`` (rank-0 only). Survives a later crash."""

    if out is None or not dist_info.is_rank_zero:
        return
    event.setdefault("ts", _now())
    append_jsonl(out / "progress.jsonl", event)


def _write_status(out: Path | None, dist_info: DistributedInfo, **fields: Any) -> None:
    """Overwrite ``run_status.json`` (rank-0 only) — a one-glance snapshot of where the run is."""

    if out is None or not dist_info.is_rank_zero:
        return
    fields.setdefault("updated_at", _now())
    write_json(out / "run_status.json", fields)


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_ktm(
    cfg: KTMTrainConfig,
    *,
    out_dir: str | Path | None = None,
    dist_info: DistributedInfo | None = None,
) -> TrainingArtifacts:
    """Train a TorchKTM on a synthetic Transition Gym; return artifacts (no claim)."""

    cfg = cfg.validate()
    dist_info = dist_info or get_distributed_info()
    out = Path(out_dir) if out_dir is not None else None
    set_global_seed(cfg.seed)
    _write_status(out, dist_info, status="running", phase="setup",
                  completed_steps=0, total_steps=cfg.steps, mode=cfg.mode, started_at=_now())
    _progress(out, dist_info, event="run_started", mode=cfg.mode, total_steps=cfg.steps, seed=cfg.seed)

    ddp_initialized, backend = maybe_init_process_group(dist_info)
    completed_steps = 0
    phase = "setup"
    try:
        device = resolve_device(cfg.mode, dist_info)

        phase = "data"
        _write_status(out, dist_info, status="running", phase=phase,
                      completed_steps=0, total_steps=cfg.steps)
        bundle = build_transition_gym(cfg.to_world_config())
        bundle.splits.assert_no_episode_leakage()
        bundle.splits.assert_no_composition_leakage()
        train_loader, _val_loader = make_dataloaders(
            bundle, batch_size=cfg.batch_size, seed=cfg.seed, dist_info=dist_info
        )

        phase = "model"
        _write_status(out, dist_info, status="running", phase=phase,
                      completed_steps=0, total_steps=cfg.steps)
        model = TorchKTM(cfg.to_model_config()).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        config_hash = stable_hash(cfg.as_dict())

        start_step = 0
        best_val = math.inf
        if cfg.resume_path:
            checkpoint = load_resume(cfg.resume_path, device)
            apply_resume(checkpoint, model=model, optimizer=optimizer)
            start_step = resume_start_step(checkpoint)
            best_val = float(checkpoint.get("best_val", math.inf)) if checkpoint else math.inf

        ddp_model = wrap_ddp_if_initialized(model, dist_info.local_rank)
        guard = LossExplosionGuard(cfg.loss_explosion_factor)
        autocast_ctx = _autocast(device, cfg.precision)

        ckpt_dir = out / "checkpoints" if out is not None else None
        failure_reasons: list[str] = []
        step_losses: list[dict[str, float]] = []
        best_ckpt_path: Path | None = None
        last_ckpt_path: Path | None = None
        best_state_dict: dict[str, torch.Tensor] | None = None
        best_step = int(start_step)
        selected_checkpoint = "last"
        selected_checkpoint_step = int(start_step)
        aborted = False
        completed_steps = start_step

        phase = "train"
        val_before = _val_mse(model, bundle, bundle.splits.val_episodes, device)
        if not math.isfinite(best_val) or val_before < best_val:
            best_val = float(val_before)
            best_step = int(start_step)
            best_state_dict = _clone_state_dict(model)
            if ckpt_dir is not None and dist_info.is_rank_zero:
                best_ckpt_path = save_ktm_checkpoint(
                    ckpt_dir / "best.pt", step=start_step, model=model, optimizer=optimizer,
                    best_val=best_val, rng_state={"torch": torch.get_rng_state()},
                    config_hash=config_hash,
                )
        _write_status(out, dist_info, status="running", phase=phase, device=str(device),
                      completed_steps=start_step, total_steps=cfg.steps, val_before=val_before)
        _progress(out, dist_info, event="val_before", val_mse=val_before, device=str(device))
        train_iter = _cycle(train_loader)
        accum = max(1, cfg.gradient_accumulation_steps)

        for step in range(start_step, cfg.steps):
            ddp_model.train()
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            skipped = False
            for _micro in range(accum):
                batch = next(train_iter)
                if len(batch) == 4:
                    history, k, target, profile_target = batch
                else:
                    history, k, target = batch
                    profile_target = None
                history = history.to(device)
                k = k.to(device)
                target = target.to(device)
                if profile_target is not None:
                    profile_target = profile_target.to(device)
                with autocast_ctx():
                    if profile_target is not None:
                        pred, log_var, profile_pred = ddp_model(history, k, return_profile=True)
                        loss, _components = ktm_loss(
                            pred, log_var, target, cfg,
                            profile_pred=profile_pred, profile_target=profile_target,
                        )
                    else:
                        pred, log_var = ddp_model(history, k)
                        loss, _components = ktm_loss(pred, log_var, target, cfg)
                loss_value = float(loss.detach())
                if not is_finite_loss(loss_value):
                    failure_reasons.append(f"non-finite loss at step {step}; micro-batch skipped")
                    _progress(out, dist_info, event="nonfinite_skip", step=step)
                    optimizer.zero_grad(set_to_none=True)
                    skipped = True
                    break
                (loss / accum).backward()
                accum_loss += loss_value / accum
            if skipped:
                continue

            if guard.update(accum_loss):
                failure_reasons.append(
                    f"loss explosion at step {step} (loss={accum_loss:.4g}); training aborted"
                )
                _progress(out, dist_info, event="loss_explosion", step=step, loss=accum_loss)
                optimizer.zero_grad(set_to_none=True)
                aborted = True
                break

            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            step_losses.append({"step": float(step), "loss": float(accum_loss)})
            completed_steps = step + 1
            # Per-step progress persists incrementally, so a mid-run crash keeps the loss trace.
            _progress(out, dist_info, event="step", step=step, loss=accum_loss)

            final_step = step + 1 == cfg.steps
            if (step + 1) % cfg.eval_every_steps == 0 or final_step:
                val_mse = _val_mse(model, bundle, bundle.splits.val_episodes, device)
                _progress(out, dist_info, event="eval", step=step, val_mse=val_mse,
                          completed_steps=completed_steps)
                _write_status(out, dist_info, status="running", phase="train",
                              completed_steps=completed_steps, total_steps=cfg.steps,
                              val_before=val_before, best_val=min(best_val, val_mse))
                if val_mse < best_val:
                    best_val = val_mse
                    best_step = step + 1
                    best_state_dict = _clone_state_dict(model)
                    if ckpt_dir is not None and dist_info.is_rank_zero:
                        best_ckpt_path = save_ktm_checkpoint(
                            ckpt_dir / "best.pt", step=step + 1, model=model, optimizer=optimizer,
                            best_val=best_val, rng_state={"torch": torch.get_rng_state()},
                            config_hash=config_hash,
                        )
                        _progress(out, dist_info, event="checkpoint", kind="best",
                                  step=step + 1, best_val=best_val)
            if ckpt_dir is not None and dist_info.is_rank_zero and (
                (step + 1) % cfg.checkpoint_every_steps == 0 or final_step
            ):
                last_ckpt_path = save_ktm_checkpoint(
                    ckpt_dir / "last.pt", step=step + 1, model=model, optimizer=optimizer,
                    best_val=best_val, rng_state={"torch": torch.get_rng_state()},
                    config_hash=config_hash,
                )
                _progress(out, dist_info, event="checkpoint", kind="last", step=step + 1)

        if best_state_dict is not None:
            model.load_state_dict(deepcopy(best_state_dict))
            selected_checkpoint = "best_val"
            selected_checkpoint_step = int(best_step)
            _progress(out, dist_info, event="selected_checkpoint", kind=selected_checkpoint,
                      step=selected_checkpoint_step, best_val=float(best_val))
        val_after = _val_mse(model, bundle, bundle.splits.val_episodes, device)
        _progress(out, dist_info, event="val_after", val_mse=val_after, aborted=aborted)
        _write_status(out, dist_info, status="training_complete", phase="train",
                      completed_steps=completed_steps, total_steps=cfg.steps,
                      val_before=val_before, val_after=val_after, best_val=float(best_val),
                      selected_checkpoint=selected_checkpoint,
                      selected_checkpoint_step=selected_checkpoint_step,
                      aborted=aborted)
    except Exception as exc:  # noqa: BLE001 - persist where/why training died, then re-raise.
        _progress(out, dist_info, event="training_error", phase=phase,
                  error_type=type(exc).__name__, error=str(exc))
        _write_status(out, dist_info, status="failed", phase=phase,
                      completed_steps=completed_steps, total_steps=cfg.steps,
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        barrier_if_distributed()
        cleanup_process_group()

    return TrainingArtifacts(
        model=model,
        bundle=bundle,
        device=str(device),
        mode=cfg.mode,
        dist_info=dist_info,
        ddp_initialized=bool(ddp_initialized),
        backend=backend,
        val_before=val_before,
        val_after=val_after,
        best_val=float(best_val),
        loss_decreased=bool(val_after < val_before),
        aborted=aborted,
        config_hash=config_hash,
        step_losses=step_losses,
        failure_reasons=failure_reasons,
        best_checkpoint=best_ckpt_path,
        last_checkpoint=last_ckpt_path,
        selected_checkpoint=selected_checkpoint,
        selected_checkpoint_step=selected_checkpoint_step,
    )
