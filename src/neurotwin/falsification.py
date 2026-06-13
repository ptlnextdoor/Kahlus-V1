"""Shared falsification-benchmark harness for the synthetic v-lane benchmarks.

A neutral core both the v2 (dual-field) and v3 (Transition Gym) benchmarks depend on — and
that future lanes (EM, ...) can reuse — so the Outcome type, finite check, evidence-gate
assembly, and report/write boilerplate live in one place instead of being hand-copied per lane.

This core is lane-agnostic: it imports no lane (v2/v3/EM) module, so depending on it never
couples lanes to each other. PROPOSED / SYNTHETIC ONLY — it builds claim-blocking gates, it
does not produce results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from neurotwin.gates import evaluate_gate, write_evidence_gate
from neurotwin.repro import write_json


@dataclass(frozen=True)
class Outcome:
    """One falsification diagnostic result: a name, pass/fail, numeric detail, and reason."""

    name: str
    passed: bool
    detail: dict[str, Any]
    reason: str = ""


def _iter_numbers(value: Any) -> Iterator[float]:
    """Yield every scalar number nested anywhere inside a detail value."""

    if isinstance(value, dict):
        for inner in value.values():
            yield from _iter_numbers(inner)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            yield from _iter_numbers(inner)
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        yield float(value)


def outcomes_finite(outcomes: Sequence[Outcome]) -> bool:
    """True iff every numeric value in every outcome detail (incl. nested) is finite."""

    values = [v for outcome in outcomes for v in _iter_numbers(outcome.detail)]
    return bool(np.isfinite(values).all()) if values else True


def outcome_dicts(outcomes: Sequence[Outcome]) -> list[dict[str, Any]]:
    return [{"name": o.name, "passed": o.passed, "detail": o.detail, "reason": o.reason} for o in outcomes]


def assemble_gate(
    *,
    branch: str,
    dataset: str,
    claim_scope: str,
    outcomes: Sequence[Outcome],
    required: Sequence[str],
    split_audit_passed: bool = True,
    baseline_table_present: bool = True,
    extra_finite: bool = True,
) -> dict[str, Any]:
    """Build the unified evidence gate for a falsification run.

    The narrow claim is calibrated/validated by the ``required`` diagnostics passing; any
    required failure is surfaced as a gate failure reason and blocks the claim. ``extra_finite``
    folds in finiteness of inputs outside the outcomes (e.g. a baseline leaderboard).
    """

    by_name = {o.name: o for o in outcomes}
    required_pass = all(by_name[name].passed for name in required)
    reasons = [f"{by_name[name].name}: {by_name[name].reason}"
               for name in required if not by_name[name].passed and by_name[name].reason]
    finite = outcomes_finite(outcomes) and bool(extra_finite)
    return evaluate_gate(
        branch=branch,
        dataset=dataset,
        split_audit_passed=split_audit_passed,
        baseline_table_present=baseline_table_present,
        finite_metrics=finite,
        calibration_checked=required_pass,
        claim_scope=claim_scope,
        extra_failure_reasons=reasons,
    )


def build_report(
    *,
    schema: str,
    branch: str,
    claim_scope: str,
    seed: int,
    config: dict[str, Any],
    outcomes: Sequence[Outcome],
    gate: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the standard falsification report; ``extra`` adds lane-specific keys."""

    report = {
        "schema": schema,
        "branch": branch,
        "claim_status": "synthetic_scaffold_only",
        "claim_scope": claim_scope,
        "seed": seed,
        "config": config,
        "diagnostics": outcome_dicts(outcomes),
        "falsification_passed": bool(gate["scientific_claim_allowed"]),
        "scientific_claim_allowed": bool(gate["scientific_claim_allowed"]),
        "failure_reasons": list(gate["failure_reasons"]),
        "evidence_gate": gate,
    }
    if extra:
        report.update(extra)
    return report


def write_report(out_dir: str | Path, *, report: dict[str, Any], gate: dict[str, Any], prefix: str) -> dict[str, Path]:
    """Write ``{prefix}_benchmark_report.json`` + ``{prefix}_evidence_gate.json``."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = write_json(out / f"{prefix}_benchmark_report.json", report)
    gate_path = write_evidence_gate(out / f"{prefix}_evidence_gate.json", gate)
    return {"report": report_path, "evidence_gate": gate_path}
