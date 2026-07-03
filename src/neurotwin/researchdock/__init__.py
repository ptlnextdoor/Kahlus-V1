"""ResearchDock synthetic response-profile scaffold.

ResearchDock is a prototype data-collection branch for synchronized task,
behavior, pupil/gaze, physiology, and self-report response profiles. This package
contains no clinical claim logic, no stimulation, and no real participant data.
"""

from __future__ import annotations

from neurotwin.researchdock.data_card import build_researchdock_data_card
from neurotwin.researchdock.export import export_researchdock_sessions
from neurotwin.researchdock.gates import RESEARCHDOCK_ALLOWED_CLAIM_SCOPE, build_researchdock_gate
from neurotwin.researchdock.metrics import (
    audit_response_profile_readiness,
    compute_researchdock_metrics,
    response_profile_vector,
)
from neurotwin.researchdock.observation_model import (
    ResearchDockObservationModelConfig,
    ResearchDockObservationTask,
    build_researchdock_observation_task,
    run_researchdock_observation_benchmark,
    write_researchdock_observation_artifacts,
)
from neurotwin.researchdock.pilot_preflight import (
    build_researchdock_pilot_manifest,
    run_researchdock_pilot_preflight,
    write_researchdock_pilot_preflight_artifacts,
)
from neurotwin.researchdock.protocol import (
    ResearchDockProtocolBlock,
    ResearchDockSessionProtocol,
    build_rd1_session_protocol,
    protocol_to_dict,
    researchdock_interface_contract,
)
from neurotwin.researchdock.public_datasets import (
    ResearchDockPublicDatasetReview,
    researchdock_public_dataset_reviews,
    write_researchdock_public_dataset_review,
)
from neurotwin.researchdock.quality import assess_trial_quality, summarize_quality_flags
from neurotwin.researchdock.schemas import (
    ResearchDockSensorPacket,
    ResearchDockSelfReport,
    ResearchDockSession,
    ResearchDockTaskEvent,
    ResearchDockTrial,
)
from neurotwin.researchdock.synthetic import make_synthetic_researchdock_sessions
from neurotwin.researchdock.tasks import ResearchDockTaskTemplate, researchdock_task_templates

__all__ = [
    "RESEARCHDOCK_ALLOWED_CLAIM_SCOPE",
    "ResearchDockObservationModelConfig",
    "ResearchDockObservationTask",
    "ResearchDockProtocolBlock",
    "ResearchDockPublicDatasetReview",
    "ResearchDockSensorPacket",
    "ResearchDockSelfReport",
    "ResearchDockSession",
    "ResearchDockSessionProtocol",
    "ResearchDockTaskEvent",
    "ResearchDockTaskTemplate",
    "ResearchDockTrial",
    "assess_trial_quality",
    "audit_response_profile_readiness",
    "build_researchdock_data_card",
    "build_researchdock_gate",
    "build_researchdock_observation_task",
    "build_researchdock_pilot_manifest",
    "build_rd1_session_protocol",
    "compute_researchdock_metrics",
    "export_researchdock_sessions",
    "make_synthetic_researchdock_sessions",
    "protocol_to_dict",
    "researchdock_public_dataset_reviews",
    "researchdock_task_templates",
    "researchdock_interface_contract",
    "response_profile_vector",
    "run_researchdock_observation_benchmark",
    "run_researchdock_pilot_preflight",
    "summarize_quality_flags",
    "write_researchdock_public_dataset_review",
    "write_researchdock_pilot_preflight_artifacts",
    "write_researchdock_observation_artifacts",
]
