"""NV-1 neurovisual symptom ontology and dataset registry side branch."""

from __future__ import annotations

from neurotwin.neurovisual.condition_map import build_condition_comparison_matrix
from neurotwin.neurovisual.dataset_registry import (
    REGISTRY_REQUIRED_FIELDS,
    SCORE_DEFINITIONS,
    build_metadata_only_adapter_plan,
    build_metadata_query_plan,
    build_registry_verification_summary,
    build_seed_dataset_registry,
    build_split_audit_plan,
    validate_registry_entry,
)
from neurotwin.neurovisual.gates import (
    NEUROVISUAL_ALLOWED_CLAIM_SCOPES,
    NEUROVISUAL_BLOCKED_CLAIMS,
    evaluate_neurovisual_claim_gate,
)
from neurotwin.neurovisual.intake import NeurovisualIntakeProfile, build_episode_intake_profile
from neurotwin.neurovisual.manifest import (
    LOCAL_MANIFEST_REQUIRED_FIELDS,
    build_local_manifest_schema,
    validate_local_manifest_records,
)
from neurotwin.neurovisual.ontology import (
    NOT_DIAGNOSIS_NOTICE,
    REQUIRED_ONTOLOGY_FIELDS,
    NeurovisualEpisodeProfile,
    default_episode_profile,
    episode_profile_to_dict,
)
from neurotwin.neurovisual.split_audit import (
    ALLOWED_SPLIT_NAMES,
    SPLIT_AUDIT_REQUIRED_FIELDS,
    validate_local_split_records,
)

__all__ = [
    "ALLOWED_SPLIT_NAMES",
    "LOCAL_MANIFEST_REQUIRED_FIELDS",
    "NEUROVISUAL_ALLOWED_CLAIM_SCOPES",
    "NEUROVISUAL_BLOCKED_CLAIMS",
    "NOT_DIAGNOSIS_NOTICE",
    "REGISTRY_REQUIRED_FIELDS",
    "REQUIRED_ONTOLOGY_FIELDS",
    "SCORE_DEFINITIONS",
    "SPLIT_AUDIT_REQUIRED_FIELDS",
    "NeurovisualEpisodeProfile",
    "NeurovisualIntakeProfile",
    "build_condition_comparison_matrix",
    "build_episode_intake_profile",
    "build_local_manifest_schema",
    "build_metadata_only_adapter_plan",
    "build_metadata_query_plan",
    "build_registry_verification_summary",
    "build_seed_dataset_registry",
    "build_split_audit_plan",
    "default_episode_profile",
    "episode_profile_to_dict",
    "evaluate_neurovisual_claim_gate",
    "validate_local_manifest_records",
    "validate_local_split_records",
    "validate_registry_entry",
]
