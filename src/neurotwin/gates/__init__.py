"""Unified, branch-aware evidence gate for the Kahlus v1/v2/v3/EM lanes.

This package is intentionally separate from ``neurotwin.reports.evidence_gate`` (the
load-bearing v1 prepared-run gate). The unified gate is the cross-branch schema described
in the unified dossier; it is consumed by the new synthetic v2/v3 scaffolds and the EM
Stage 0 artifact-audit scaffold. It makes no scientific claim by default.
"""

from __future__ import annotations

from neurotwin.gates.unified_gate import (
    ALLOWED_BRANCHES,
    NARROW_CLAIM_SCOPES,
    GATE_SCHEMA,
    evaluate_gate,
    read_evidence_gate,
    write_evidence_gate,
)

__all__ = [
    "ALLOWED_BRANCHES",
    "NARROW_CLAIM_SCOPES",
    "GATE_SCHEMA",
    "evaluate_gate",
    "read_evidence_gate",
    "write_evidence_gate",
]
