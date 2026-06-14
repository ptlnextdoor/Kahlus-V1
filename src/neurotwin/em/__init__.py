"""Kahlus-EM Stage 0 scaffold (PROPOSED / SYNTHETIC, NO-HUMAN).

Safe scaffolding only: EEG idle/phantom recording metadata schemas, room/device environment
logging, offline geomagnetic context loading, PSD/channel artifact feature extraction,
descriptive EM-response metrics, an artifact audit, and an EM-branch evidence gate.

Hard safety boundary: NO stimulation, NO high voltage, NO plasma/coils/gas, NO God Helmet,
NO human protocol execution, NO clinical/diagnostic claim. Stage 0 only asks whether
environment/device changes affect EEG hardware even without a brain, using synthetic phantom
data. See ``docs/roadmap/kahlus_implementation_status.md``.
"""

from __future__ import annotations

from neurotwin.em.artifact_audit import (
    format_artifact_report_md,
    run_artifact_audit,
    synthesize_idle_recording,
)
from neurotwin.em.artifact_severity import (
    EEG_BANDS,
    artifact_severity_summary,
    band_contamination_score,
    channel_contamination_score,
    contamination_map,
    overall_artifact_severity,
    severity_verdict,
)
from neurotwin.em.eeg_artifact_features import channel_artifact_features, compute_psd
from neurotwin.em.em_context_schema import (
    EMContext,
    IdleRecordingMetadata,
    PhantomRecordingSchema,
    RoomEnvironmentLog,
)
from neurotwin.em.em_gates import build_em_artifact_audit_gate
from neurotwin.em.em_response_metrics import ARTIFACT_MODEL, summarize_em_response
from neurotwin.em.geomagnetic_fetcher import fetch_geomagnetic
from neurotwin.em.room_emf_logger import RoomEMFLogger
from neurotwin.em.stage0_report import (
    build_stage0_report,
    format_stage0_report_md,
    recommendation,
    write_stage0_report,
)

__all__ = [
    "EMContext",
    "IdleRecordingMetadata",
    "PhantomRecordingSchema",
    "RoomEnvironmentLog",
    "RoomEMFLogger",
    "ARTIFACT_MODEL",
    "compute_psd",
    "channel_artifact_features",
    "summarize_em_response",
    "synthesize_idle_recording",
    "run_artifact_audit",
    "format_artifact_report_md",
    "build_em_artifact_audit_gate",
    "fetch_geomagnetic",
    "EEG_BANDS",
    "contamination_map",
    "channel_contamination_score",
    "band_contamination_score",
    "overall_artifact_severity",
    "severity_verdict",
    "artifact_severity_summary",
    "build_stage0_report",
    "format_stage0_report_md",
    "recommendation",
    "write_stage0_report",
]
