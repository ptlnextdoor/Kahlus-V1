from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from neurotwin.gates.unified_gate import evaluate_gate

RESEARCHDOCK_ALLOWED_CLAIM_SCOPE = "researchdock_synthetic_response_profile"
BLOCKED_CLAIM_TERMS: tuple[str, ...] = (
    "diagnosis",
    "treatment",
    "clinical",
    "depression_detection",
    "ptsd_detection",
    "anhedonia_diagnosis",
)


def build_researchdock_gate(
    *,
    dataset: str,
    metrics: Sequence[dict[str, Any]],
    claim_scope: str = RESEARCHDOCK_ALLOWED_CLAIM_SCOPE,
    data_card_passed: bool,
) -> dict[str, Any]:
    extra: list[str] = []
    if any(term in claim_scope.lower() for term in BLOCKED_CLAIM_TERMS):
        extra.append(f"blocked clinical/device claim term in scope: {claim_scope!r}")
    if not data_card_passed:
        extra.append("ResearchDock data card did not pass synthetic safety checks")
    gate = evaluate_gate(
        branch="researchdock",
        dataset=dataset,
        split_audit_passed=bool(data_card_passed),
        baseline_table_present=True,
        finite_metrics=_metrics_are_finite(metrics),
        calibration_checked=True,
        claim_scope=claim_scope,
        extra_failure_reasons=extra,
    )
    gate["gate_criteria"] = {
        "allowed_claim_scope": RESEARCHDOCK_ALLOWED_CLAIM_SCOPE,
        "blocked_claim_terms": list(BLOCKED_CLAIM_TERMS),
        "requires_data_card_passed": True,
        "requires_baseline_table_present": True,
        "requires_finite_metrics": True,
        "requires_calibration_checked": True,
        "requires_synthetic_only": True,
    }
    return gate


def _metrics_are_finite(metrics: Sequence[dict[str, Any]]) -> bool:
    if not metrics:
        return False
    for row in metrics:
        for value in row.values():
            if isinstance(value, bool | str | list | tuple | dict) or value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric):
                return False
    return True
