from __future__ import annotations

from pathlib import Path

import torch

from neurotwin.runtime.distributed import DistributedInfo
from neurotwin.training.prepared_checkpoints import load_resume_checkpoint, resume_start_step
from neurotwin.training.prepared_types import PreparedRuntimeContext, PreparedTrainingRunPaths


def build_prepared_runtime_context(dist_info: DistributedInfo, paths: PreparedTrainingRunPaths) -> PreparedRuntimeContext:
    device = torch.device(f"cuda:{dist_info.local_rank}" if torch.cuda.is_available() else "cpu")
    resume_checkpoint = load_resume_checkpoint(paths.resume_path, device)
    return PreparedRuntimeContext(
        device=device,
        dist_info=dist_info,
        resume_checkpoint=resume_checkpoint,
        paths=paths,
        start_step=resume_start_step(resume_checkpoint),
        checkpoint_dir=Path(paths.checkpoint_path).parent if paths.checkpoint_path is not None else None,
    )
