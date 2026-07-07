"""Kahlus v1 EEG future-window baseline lane.

This package is additive: it reuses the existing split/audit, baseline, scoring, and gate
infrastructure while keeping the frozen v1 training/eval paths untouched.
"""

from __future__ import annotations

from neurotwin.eeg_v1.dataset import EEGV1Dataset, build_future_forecasting_task, make_synthetic_eeg_v1_dataset
from neurotwin.eeg_v1.adaptation import (
    FewShotAdaptationTask,
    SubjectAdaptationSplit,
    audit_adaptation_checksum_manifest,
    build_fewshot_adaptation_task,
    run_fewshot_adaptation,
    write_fewshot_adaptation_artifacts,
)
from neurotwin.eeg_v1.gates import (
    EEG_V1_ADAPTATION_CLAIM_SCOPE,
    EEG_V1_CLAIM_SCOPE,
    build_eeg_v1_adaptation_gate,
    build_eeg_v1_gate,
)
from neurotwin.eeg_v1.hbn import load_hbn_eeg_local_dataset
from neurotwin.eeg_v1.metrics import smoothness_loss
from neurotwin.eeg_v1.reporting import (
    audit_eeg_v1_checksum_manifest,
    audit_eeg_v1_split,
    run_eeg_v1_autocorrelation_diagnostics,
    run_eeg_v1_baselines,
    write_eeg_v1_artifacts,
)

__all__ = [
    "EEGV1Dataset",
    "FewShotAdaptationTask",
    "SubjectAdaptationSplit",
    "EEG_V1_ADAPTATION_CLAIM_SCOPE",
    "EEG_V1_CLAIM_SCOPE",
    "audit_eeg_v1_checksum_manifest",
    "audit_eeg_v1_split",
    "audit_adaptation_checksum_manifest",
    "build_eeg_v1_adaptation_gate",
    "build_eeg_v1_gate",
    "build_fewshot_adaptation_task",
    "build_future_forecasting_task",
    "load_hbn_eeg_local_dataset",
    "make_synthetic_eeg_v1_dataset",
    "run_fewshot_adaptation",
    "run_eeg_v1_autocorrelation_diagnostics",
    "run_eeg_v1_baselines",
    "smoothness_loss",
    "write_eeg_v1_artifacts",
    "write_fewshot_adaptation_artifacts",
]
