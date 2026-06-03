from __future__ import annotations

from pathlib import Path
from typing import Any

from neurotwin.eval.paper_gate import paper_mode_gate_allows_claim
from neurotwin.reports.artifact_bundle import (
    ModelCardSourceArtifacts,
    append_artifact_errors,
    diagnostic_status,
    format_aggregate_rank,
    join_list,
    load_model_card_source_artifacts,
    write_paper_artifact_aliases,
)


def generate_model_card_report(run_dir: str | Path, out: str | Path | None = None) -> str:
    run_path = Path(run_dir)
    source = load_model_card_source_artifacts(run_path)
    aliases = write_paper_artifact_aliases(source)
    card_path = Path(out) if out is not None else source.run_dir / "EEG_MODEL_CARD.md"
    lines = model_card_lines(source, aliases=aliases)
    if out is not None:
        card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join([*lines, "", f"model_card={card_path}"])


def model_card_lines(source: ModelCardSourceArtifacts, *, aliases: list[str]) -> list[str]:
    summary_payload = source.summary if isinstance(source.summary, dict) else {}
    prepared_payload = source.prepared if isinstance(source.prepared, dict) else {}
    eval_payload = source.eval_audit if isinstance(source.eval_audit, dict) else {}
    gate_payload = source.claim_gate if isinstance(source.claim_gate, dict) else {}
    evidence_gate_payload = source.evidence_gate if isinstance(source.evidence_gate, dict) else {}
    prepared_data = prepared_payload.get("prepared_data", {}) if isinstance(prepared_payload.get("prepared_data"), dict) else {}
    event_summary = prepared_data.get("event_summary", {}) if isinstance(prepared_data.get("event_summary"), dict) else {}
    scope = prepared_payload.get("scope", {}) if isinstance(prepared_payload.get("scope"), dict) else {}
    stimulus_evidence = _first_dict(summary_payload.get("stimulus_evidence"), prepared_data.get("stimulus_evidence"))
    quarantined_tasks = summary_payload.get("quarantined_tasks", [])
    summary_claim = bool(summary_payload.get("scientific_claim_allowed"))
    gate_allowed = paper_mode_gate_allows_claim(gate_payload)
    lines = [
        "# EEG Model Card",
        "",
        "## Intended Claim",
        "",
        "NeuroTwin evaluates neural translation under executable leakage controls. This card does not support clinical, diagnostic, SOTA, or first-foundation-model claims.",
        "",
        "## Run Scope",
        "",
        f"- run_dir: {source.run_dir}",
        f"- status: {summary_payload.get('status', 'unknown')}",
        f"- scope: {scope.get('status', 'unknown')}",
        f"- synthetic_only: {summary_payload.get('synthetic_only', event_summary.get('synthetic_only', 'unknown'))}",
        f"- scientific_claim_allowed: {summary_claim}",
        f"- paper_mode_gate_allows_claim: {gate_allowed}",
        "",
        "## Data And Protocol",
        "",
        f"- event_manifest: {prepared_data.get('event_manifest', 'unknown')}",
        f"- split_manifest: {prepared_data.get('split_manifest', 'unknown')}",
        f"- modalities: {join_list(event_summary.get('modalities'))}",
        f"- datasets: {join_list(event_summary.get('datasets'))}",
        f"- subjects: {event_summary.get('subjects', 'unknown')}",
        f"- window_length: {prepared_data.get('window_length', 'unknown')}",
        f"- stride: {prepared_data.get('stride', 'unknown')}",
        f"- stimulus_evidence: {_format_stimulus_evidence(stimulus_evidence)}",
        "",
        "## Leakage And Claim Gates",
        "",
        f"- eval_audit_passed: {eval_payload.get('passed', 'missing')}",
        f"- paper_mode_gate_passed: {gate_payload.get('passed', 'missing')}",
        f"- evidence_gate_passed: {evidence_gate_payload.get('passed', 'missing')}",
        f"- checked: {join_list(eval_payload.get('checked'))}",
        f"- gate_violations: {join_list(gate_payload.get('violations'))}",
        f"- evidence_gate_failures: {join_list(evidence_gate_payload.get('failures'))}",
        "",
        "## Baselines And Metrics",
        "",
        f"- seeds: {join_list(prepared_payload.get('seeds'))}",
        f"- aggregate_rank: {format_aggregate_rank(prepared_payload)}",
        f"- baseline_failures: {len(prepared_payload.get('baseline_failures', [])) if isinstance(prepared_payload.get('baseline_failures'), list) else 'unknown'}",
        f"- quarantined_tasks: {_format_quarantined_tasks(quarantined_tasks)}",
        "",
        "## Paper Diagnostics",
        "",
        f"- leakage_demo: {diagnostic_status(source.leakage_demo)}",
        f"- identity_probe: {diagnostic_status(source.identity_probe)}",
        f"- identity_confounding_risk: {source.identity_probe.get('identity_confounding_risk', 'missing') if isinstance(source.identity_probe, dict) else 'missing'}",
        "",
        "## Artifacts",
        "",
    ]
    if aliases:
        lines.extend(f"- {alias}" for alias in aliases)
    else:
        lines.append("- no paper artifact aliases were written")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Segment/window split results are negative controls and are never claim eligible.",
            "- Model/scientific claim allowance is controlled by summary.json and requires real prepared data, required seeds, confidence intervals, and a passed colocated claim gate.",
            "- Stimulus-to-fMRI claims require real precomputed stimulus features; hash-derived stimulus plumbing is not claim eligible.",
            "- TRIBE-style and Brain-OF-style lanes are local approximations unless exact upstream code or weights are explicitly declared.",
        ]
    )
    append_artifact_errors(lines, source.metrics, source.prepared, source.eval_audit, source.claim_gate)
    return lines


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _format_stimulus_evidence(value: dict[str, Any]) -> str:
    if not value:
        return "missing"
    return (
        f"{value.get('status', 'unknown')} "
        f"claim_eligible={value.get('claim_eligible', False)} "
        f"require_real_stimulus={value.get('require_real_stimulus', False)}"
    )


def _format_quarantined_tasks(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    formatted = []
    for row in value:
        if isinstance(row, dict):
            formatted.append(f"{row.get('task_id')}:{row.get('reason')}")
    return "; ".join(formatted) if formatted else "none"
