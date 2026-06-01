#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


def _nccl_version() -> str | None:
    try:
        version = torch.cuda.nccl.version()  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError):
        return None
    return ".".join(str(part) for part in version) if isinstance(version, tuple) else str(version)


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"{name} must be an integer, got: {raw}") from None
    if value < 1:
        raise SystemExit(f"{name} must be positive, got: {raw}")
    return value


def _payload(expected_gpus: int) -> dict[str, Any]:
    visible_count = torch.cuda.device_count()
    names: list[str] = []
    for index in range(visible_count):
        try:
            names.append(torch.cuda.get_device_name(index))
        except (AssertionError, RuntimeError):
            names.append("")
    return {
        "passed": bool(torch.cuda.is_available()) and visible_count == expected_gpus,
        "docker_image": os.environ.get("DOCKER_IMAGE"),
        "host_gpu_ids": os.environ.get("HOST_GPU_IDS"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
        "RANK": os.environ.get("RANK"),
        "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
        "NPROC_PER_NODE": os.environ.get("NPROC_PER_NODE"),
        "expected_gpu_count": expected_gpus,
        "visible_gpu_count": visible_count,
        "visible_gpu_names": names,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "nccl_version": _nccl_version(),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: scripts/docker_gpu_preflight.py <output-json>", file=sys.stderr)
        return 2
    out = Path(argv[1])
    expected_gpus = _positive_int("GPU_COUNT", 6)
    payload = _payload(expected_gpus)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["cuda_available"]:
        print("CUDA is not available inside Docker.", file=sys.stderr)
        return 10
    if int(payload["visible_gpu_count"]) != expected_gpus:
        print(
            f"Expected exactly {expected_gpus} visible CUDA devices, saw {payload['visible_gpu_count']}.",
            file=sys.stderr,
        )
        return 11
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
