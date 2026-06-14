"""Checkpoint save / resume for the v3 KTM training harness.

PROPOSED / SYNTHETIC ONLY. Generic ``torch.save`` / ``torch.load`` of a small dict (no v1
coupling). Loading prefers ``weights_only=True`` (safe unpickling) with a fallback for older
PyTorch that lacks the kwarg. Only rank-0 should call :func:`save_ktm_checkpoint`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neurotwin.runtime.distributed import unwrap_model


def save_ktm_checkpoint(
    path: str | Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val: float,
    rng_state: dict[str, Any] | None = None,
    config_hash: str | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val": float(best_val),
        "rng_state": rng_state or {},
        "config_hash": config_hash,
    }
    torch.save(payload, out)
    return out


def load_resume(path: str | Path, device: torch.device | str) -> dict[str, Any] | None:
    """Load a checkpoint dict, or ``None`` if the file is absent."""

    src = Path(path)
    if not src.exists():
        return None
    try:
        return torch.load(src, map_location=device, weights_only=True)
    except TypeError:
        # Installed PyTorch predates the weights_only kwarg.
        return torch.load(src, map_location=device)


def resume_start_step(checkpoint: dict[str, Any] | None) -> int:
    if not checkpoint:
        return 0
    return int(checkpoint.get("step", 0))


def apply_resume(
    checkpoint: dict[str, Any] | None,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    if not checkpoint:
        return
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
