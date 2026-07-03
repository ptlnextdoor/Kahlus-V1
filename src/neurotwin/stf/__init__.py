"""Kahlus-STF passive state-transition forecasting benchmark lane.

STF is a research benchmark contract for longitudinal epilepsy/sleep monitoring.
It is not a diagnostic, treatment, stimulation, or wearable-device claim.
"""

from __future__ import annotations

from neurotwin.stf.benchmark import (
    BLOCKED_STF_CLAIM_TERMS,
    REQUIRED_STF_BASELINES_BY_TASK,
    REQUIRED_STF_NEGATIVE_CONTROLS,
    REQUIRED_STF_SPLITS,
    REQUIRED_STF_TASKS,
    STF_CLAIM_SCOPE,
    audit_stf_benchmark_contract,
    build_stf_gate,
    stf_benchmark_contract,
)
from neurotwin.stf.chb_mit import (
    parse_chb_mit_summary_dir,
    parse_chb_mit_summary_text,
    run_chb_mit_public_smoke,
    write_chb_mit_public_smoke,
)
from neurotwin.stf.public_data import (
    CHB_MIT_DATASET_ID,
    CHB_MIT_FILE_BASE_URL,
    CHB_MIT_PHYSIONET_URL,
    audit_chb_mit_root,
    fetch_chb_mit_smoke_subset,
    stf_public_dataset_registry,
    write_chb_mit_root_audit,
)
from neurotwin.stf.smoke import run_stf_synthetic_smoke, write_stf_smoke_artifacts

__all__ = [
    "BLOCKED_STF_CLAIM_TERMS",
    "CHB_MIT_DATASET_ID",
    "CHB_MIT_FILE_BASE_URL",
    "CHB_MIT_PHYSIONET_URL",
    "REQUIRED_STF_BASELINES_BY_TASK",
    "REQUIRED_STF_NEGATIVE_CONTROLS",
    "REQUIRED_STF_SPLITS",
    "REQUIRED_STF_TASKS",
    "STF_CLAIM_SCOPE",
    "audit_chb_mit_root",
    "audit_stf_benchmark_contract",
    "build_stf_gate",
    "fetch_chb_mit_smoke_subset",
    "parse_chb_mit_summary_dir",
    "parse_chb_mit_summary_text",
    "run_chb_mit_public_smoke",
    "run_stf_synthetic_smoke",
    "stf_public_dataset_registry",
    "stf_benchmark_contract",
    "write_chb_mit_root_audit",
    "write_chb_mit_public_smoke",
    "write_stf_smoke_artifacts",
]
