from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys

import torch


@dataclass(frozen=True)
class DistributedInfo:
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0


def get_distributed_info() -> DistributedInfo:
    return DistributedInfo(
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )


def get_rank_metrics_path(run_dir: str | Path, info: DistributedInfo | None = None) -> Path:
    info = info or get_distributed_info()
    run_path = Path(run_dir)
    if info.is_distributed:
        return run_path / f"metrics.rank{info.rank}.jsonl"
    return run_path / "metrics.jsonl"


def maybe_init_process_group(info: DistributedInfo | None = None) -> tuple[bool, str | None]:
    info = info or get_distributed_info()
    if not info.is_distributed:
        return False, None
    if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
        return False, None
    if not torch.distributed.is_available():
        return False, None
    if torch.distributed.is_initialized():
        return True, torch.distributed.get_backend()
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(info.local_rank)
    torch.distributed.init_process_group(backend=backend, rank=info.rank, world_size=info.world_size)
    return True, backend


def barrier_if_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def distributed_any(value: bool, *, device: torch.device | None = None) -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return bool(value)
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    flag = torch.tensor(1 if value else 0, dtype=torch.int32, device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(flag.item())


def cleanup_process_group() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask completed training artifacts.
            print(f"warning=distributed_cleanup_failed detail={exc}", file=sys.stderr)


def wrap_ddp_if_initialized(model: torch.nn.Module, local_rank: int = 0) -> torch.nn.Module:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return model
    if torch.cuda.is_available():
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    return torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)
