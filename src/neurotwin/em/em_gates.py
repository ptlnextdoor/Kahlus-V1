"""EM-branch evidence gate wrapper.

Wraps the unified gate with ``branch="em"`` and the narrow ``em_artifact_audit_no_human``
claim scope. Stage 0 computes no calibration and makes no scientific claim, so the gate is
designed to block any claim while still recording a clean, auditable status.
"""

from __future__ import annotations

from typing import Any

from neurotwin.gates import evaluate_gate


def build_em_artifact_audit_gate(
    report: dict[str, Any],
    *,
    dataset: str = "em_stage0_artifact_audit",
) -> dict[str, Any]:
    response = report.get("response", {}) if isinstance(report, dict) else {}
    finite = bool(response.get("finite", False))
    has_comparison = bool(response.get("feature_deltas"))
    return evaluate_gate(
        branch="em",
        dataset=dataset,
        split_audit_passed=True,  # baseline vs condition are explicitly separate, controlled inputs
        baseline_table_present=has_comparison,  # the baseline-condition comparison stands in for the table
        finite_metrics=finite,
        calibration_checked=False,  # Stage 0 computes no calibration
        claim_scope="em_artifact_audit_no_human",
        extra_failure_reasons=[
            "Kahlus-EM Stage 0 is a descriptive no-human artifact audit; no scientific claim is made",
        ],
    )
