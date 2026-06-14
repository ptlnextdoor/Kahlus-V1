#!/usr/bin/env python3
"""Local trainer for the Kahlus v3 KTM on the synthetic Transition Gym.

PROPOSED / SYNTHETIC ONLY. Trains the trainable ``TorchKTM`` (cpu_smoke / single_gpu / ddp),
writes the auditable output bundle, and prints a summary. Earns only the narrow
``synthetic_ktm_training_harness`` scope; the stronger ``synthetic_ktm_recovery`` stays blocked
unless the trained KTM actually beats baselines. No A100 launched here; no real data; no claim of
model success. The 8xA100 micro-sweep command is built (printed) but never executed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.repro import capture_environment, write_json  # noqa: E402
from neurotwin.runtime.distributed import get_distributed_info  # noqa: E402
from neurotwin.training_v3 import (  # noqa: E402
    KTMTrainConfig,
    build_torchrun_command,
    train_ktm,
    write_training_bundle,
)


def _build_config(args: argparse.Namespace) -> KTMTrainConfig:
    from neurotwin.config import load_config

    payload: dict = dict(load_config(args.config)) if args.config else {}
    payload["mode"] = args.mode
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.steps is not None:
        payload["steps"] = args.steps
    if args.resume:
        payload["resume_path"] = args.resume
    return KTMTrainConfig.from_mapping(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", default="cpu_smoke", choices=["cpu_smoke", "single_gpu", "ddp"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = _build_config(args)
    out = Path(args.out_dir)
    dist_info = get_distributed_info()

    artifacts = train_ktm(cfg, out_dir=out, dist_info=dist_info)

    if dist_info.is_rank_zero:
        paths = write_training_bundle(out, cfg=cfg, artifacts=artifacts, config_path=args.config)
        # Standalone environment.json for the handoff/evidence contract (torch/cuda/nccl, visible
        # GPU count+names, CUDA_VISIBLE_DEVICES, WORLD_SIZE, docker image, git commit).
        paths["environment"] = write_json(out / "environment.json", capture_environment(argv=sys.argv))
        gate_path = paths["evidence_gate"]
        from neurotwin.gates import read_evidence_gate

        gate = read_evidence_gate(gate_path)
        print(f"branch=v3 model=TorchKTM mode={cfg.mode} device={artifacts.device} out_dir={out.resolve()}")
        print(f"val_mse_before={artifacts.val_before:.6g} val_mse_after={artifacts.val_after:.6g} "
              f"best_val_mse={artifacts.best_val:.6g} loss_decreased={artifacts.loss_decreased}")
        print(f"claim_scope={gate['claim_scope']} scientific_claim_allowed={gate['scientific_claim_allowed']}")
        print(f"aborted={artifacts.aborted} failure_reasons={artifacts.failure_reasons}")
        print(f"checkpoints best={artifacts.best_checkpoint} last={artifacts.last_checkpoint}")
        print("bundle=" + ", ".join(f"{key}={value}" for key, value in paths.items()))
        print("future_micro_sweep_cmd=" + " ".join(build_torchrun_command(
            config_path="configs/train/ktm_a100_micro.yaml", out_dir="$RUN_ROOT/ktm_micro_sweep")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
