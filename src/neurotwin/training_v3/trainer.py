"""Standalone v3 KTM training loop (PROPOSED / SYNTHETIC ONLY).

Reuses the repo's DDP helpers (``runtime/distributed``) and repro utilities — it does NOT import
the frozen v1 ``training/prepared_*`` modules, so v1 stays isolated. Supports cpu_smoke /
single_gpu / ddp modes, gradient accumulation, autocast precision, grad clipping, a finite/NaN
micro-batch skip, a loss-explosion abort, best-val checkpointing, and resume. No A100 is launched
here; ``build_torchrun_command`` only *builds* the future micro-sweep command.
"""

from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
from typing import Any, Iterator

import torch

from neurotwin.repro import set_global_seed, stable_hash
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


def train_ktm(
    cfg: KTMTrainConfig,
    *,
    out_dir: str | Path | None = None,
    dist_info: DistributedInfo | None = None,
    debug_force_nan_steps: frozenset[int] = frozenset(),
    debug_force_explode: bool = False,
) -> dict[str, Any]:
    """Train a TorchKTM on a synthetic Transition Gym; return an artifacts dict (no claim)."""

    cfg = cfg.validate()
    dist_info = dist_info or get_distributed_info()
    set_global_seed(cfg.seed)
    ddp_initialized, backend = maybe_init_process_group(dist_info)
    device = resolve_device(cfg.mode, dist_info)

    bundle = build_transition_gym(cfg.to_world_config())
    bundle.splits.assert_no_episode_leakage()
    bundle.splits.assert_no_composition_leakage()
    train_loader, _val_loader = make_dataloaders(
        bundle, batch_size=cfg.batch_size, seed=cfg.seed, dist_info=dist_info
    )

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

    ckpt_dir = Path(out_dir) / "checkpoints" if out_dir is not None else None
    failure_reasons: list[str] = []
    step_losses: list[dict[str, float]] = []
    best_ckpt_path: Path | None = None
    last_ckpt_path: Path | None = None
    aborted = False

    val_before = _val_mse(model, bundle, bundle.splits.val_episodes, device)
    train_iter = _cycle(train_loader)
    accum = max(1, cfg.gradient_accumulation_steps)

    for step in range(start_step, cfg.steps):
        ddp_model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        skipped = False
        for _micro in range(accum):
            history, k, target = next(train_iter)
            history = history.to(device)
            k = k.to(device)
            target = target.to(device)
            with autocast_ctx():
                pred, log_var = ddp_model(history, k)
                loss, _components = ktm_loss(pred, log_var, target, cfg)
            if step in debug_force_nan_steps:
                loss = loss * torch.tensor(float("nan"), device=device)
            if debug_force_explode and step >= guard.warmup:
                loss = loss * 1e6
            loss_value = float(loss.detach())
            if not is_finite_loss(loss_value):
                failure_reasons.append(f"non-finite loss at step {step}; micro-batch skipped")
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
            optimizer.zero_grad(set_to_none=True)
            aborted = True
            break

        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        step_losses.append({"step": float(step), "loss": float(accum_loss)})

        final_step = step + 1 == cfg.steps
        if (step + 1) % cfg.eval_every_steps == 0 or final_step:
            val_mse = _val_mse(model, bundle, bundle.splits.val_episodes, device)
            if val_mse < best_val:
                best_val = val_mse
                if ckpt_dir is not None and dist_info.is_rank_zero:
                    best_ckpt_path = save_ktm_checkpoint(
                        ckpt_dir / "best.pt", step=step + 1, model=model, optimizer=optimizer,
                        best_val=best_val, rng_state={"torch": torch.get_rng_state()},
                        config_hash=config_hash,
                    )
        if ckpt_dir is not None and dist_info.is_rank_zero and (
            (step + 1) % cfg.checkpoint_every_steps == 0 or final_step
        ):
            last_ckpt_path = save_ktm_checkpoint(
                ckpt_dir / "last.pt", step=step + 1, model=model, optimizer=optimizer,
                best_val=best_val, rng_state={"torch": torch.get_rng_state()},
                config_hash=config_hash,
            )

    val_after = _val_mse(model, bundle, bundle.splits.val_episodes, device)
    barrier_if_distributed()
    cleanup_process_group()

    return {
        "model": model,
        "bundle": bundle,
        "device": str(device),
        "mode": cfg.mode,
        "dist_info": dist_info,
        "ddp_initialized": bool(ddp_initialized),
        "backend": backend,
        "val_before": val_before,
        "val_after": val_after,
        "best_val": float(best_val),
        "loss_decreased": bool(val_after < val_before),
        "step_losses": step_losses,
        "failure_reasons": failure_reasons,
        "aborted": aborted,
        "config_hash": config_hash,
        "best_checkpoint": best_ckpt_path,
        "last_checkpoint": last_ckpt_path,
    }
