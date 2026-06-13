"""Kahlus-EM Stage 0 artifact report generator (no-human, synthetic).

Turns a Stage 0 artifact-audit output (from :func:`neurotwin.em.run_artifact_audit`) plus the
raw phantom signals into a useful artifact-reporting bundle: a channel x band contamination
map, an artifact severity score (pass/warn/fail), a pass-capable EM evidence gate routed
through the shared falsification core, and a Markdown ``stage0_artifact_report.md``.

Gate semantics (extremely narrow): ``scientific_claim_allowed=true`` is set ONLY when the
no-human Stage 0 artifact audit and report generation pass all required checks, with the exact
scope ``em_stage0_artifact_audit``. It means ONLY: "Artifact audit/report generation passed for
no-human Stage 0 hardware/environment characterization." It NEVER licenses a predictive human
EEG, causal EM, brain-modulation, consciousness, clinical, God Helmet, or stimulation claim;
every broader scope stays blocked by the unified gate allowlist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from neurotwin.em.artifact_severity import artifact_severity_summary, contamination_map
from neurotwin.em.em_response_metrics import ARTIFACT_MODEL
from neurotwin.falsification import Outcome, assemble_gate, build_report, write_report

REPORT_SCHEMA = "kahlus.em_stage0_artifact_report.v1"
CLAIM_SCOPE = "em_stage0_artifact_audit"
DATASET = "em_stage0_artifact_audit"
REPORT_PREFIX = "em_stage0"
REPORT_MD_NAME = "stage0_artifact_report.md"

_REQUIRED = [
    "severity_finite",
    "contamination_map_well_formed",
    "audit_finite",
    "no_human_attribution",
]


def _stage0_outcomes(
    *,
    audit_report: Mapping[str, Any],
    cmap: Mapping[str, Any],
    severity: Mapping[str, Any],
) -> list[Outcome]:
    response = audit_report.get("response", {}) if isinstance(audit_report, dict) else {}
    audit_finite = bool(response.get("finite", False))
    n_channels = int(cmap.get("n_channels", 0))
    n_bands = len(cmap.get("bands", []))
    well_formed = bool(n_channels >= 1 and n_bands >= 1 and cmap.get("delta_vs_baseline"))
    baseline_human = bool(audit_report.get("baseline_context", {}).get("involves_human", False))
    condition_human = bool(audit_report.get("condition_context", {}).get("involves_human", False))
    no_human = not (baseline_human or condition_human)
    return [
        Outcome(
            "severity_finite",
            bool(severity.get("finite", False)),
            {
                "overall_artifact_severity": float(severity.get("overall_artifact_severity", 0.0)),
                "channel_contamination_score_mean": float(
                    severity.get("channel_contamination_score_mean", 0.0)
                ),
            },
            "" if severity.get("finite", False) else "non-finite severity score",
        ),
        Outcome(
            "contamination_map_well_formed",
            well_formed,
            {"n_channels": n_channels, "n_bands": n_bands},
            "" if well_formed else "contamination map missing channels/bands/deltas",
        ),
        Outcome(
            "audit_finite",
            audit_finite,
            {"environment_effect_detected": bool(response.get("environment_effect_detected", False))},
            "" if audit_finite else "non-finite audit response",
        ),
        Outcome(
            "no_human_attribution",
            no_human,
            {"baseline_involves_human": baseline_human, "condition_involves_human": condition_human},
            "" if no_human else "human involvement is forbidden in Stage 0",
        ),
    ]


def build_stage0_report(
    *,
    audit_report: Mapping[str, Any],
    conditions: Mapping[str, Any],
    fs_hz: float,
    line_freq_hz: float,
    seed: int,
    config: Mapping[str, Any] | None = None,
    environment_log_summary: Mapping[str, Any] | None = None,
    baseline_label: str = "baseline",
    condition_label: str = "perturbed_environment",
) -> dict[str, Any]:
    """Assemble the Stage 0 report bundle (severity, contamination map, gate, report dict)."""

    cmap = contamination_map(
        conditions, fs_hz=fs_hz, line_freq_hz=line_freq_hz, baseline_label=baseline_label
    )
    severity = artifact_severity_summary(
        conditions[baseline_label],
        conditions[condition_label],
        fs_hz=fs_hz,
        line_freq_hz=line_freq_hz,
        cmap=cmap,
        condition_label=condition_label,
    )
    outcomes = _stage0_outcomes(audit_report=audit_report, cmap=cmap, severity=severity)
    gate = assemble_gate(
        branch="em",
        dataset=DATASET,
        claim_scope=CLAIM_SCOPE,
        outcomes=outcomes,
        required=_REQUIRED,
        split_audit_passed=True,
        baseline_table_present=bool(cmap.get("delta_vs_baseline")),
    )
    env_summary = dict(environment_log_summary) if environment_log_summary else {
        "status": "no_environment_log_provided",
        "n_entries": 0,
    }
    report = build_report(
        schema=REPORT_SCHEMA,
        branch="em",
        claim_scope=CLAIM_SCOPE,
        seed=seed,
        config=dict(config) if config else {},
        outcomes=outcomes,
        gate=gate,
        extra={
            "claim_status": "descriptive_no_human_artifact_audit",
            "artifact_model": ARTIFACT_MODEL,
            "audit": dict(audit_report),
            "contamination_map": cmap,
            "severity": severity,
            "environment_log_summary": env_summary,
            "condition_label": condition_label,
        },
    )
    return {
        "report": report,
        "gate": gate,
        "contamination_map": cmap,
        "severity": severity,
        "environment_log_summary": env_summary,
        "outcomes": outcomes,
        "condition_label": condition_label,
    }


def recommendation(bundle: Mapping[str, Any]) -> str:
    """A plain pass/fail recommendation combining gate validity and severity verdict."""

    gate = bundle["gate"]
    severity = bundle["severity"]
    if not gate.get("scientific_claim_allowed", False):
        return "AUDIT INCOMPLETE — gate did not pass; do not rely on this Stage 0 report."
    verdict = severity.get("verdict", "unknown")
    if verdict == "fail":
        return "AUDIT PASSED, HARDWARE CONTAMINATED — high artifact severity; clean environment/device before any downstream use."
    if verdict == "warn":
        return "AUDIT PASSED, ELEVATED ARTIFACT — review environment/device noise before downstream use."
    return "AUDIT PASSED, LOW ARTIFACT — Stage 0 hardware/environment characterization is clean."


def format_stage0_report_md(bundle: Mapping[str, Any]) -> str:
    """Render the Stage 0 artifact report as Markdown with all required sections."""

    report = bundle["report"]
    gate = bundle["gate"]
    severity = bundle["severity"]
    cmap = bundle["contamination_map"]
    audit = report.get("audit", {})
    env = bundle["environment_log_summary"]
    condition_label = bundle["condition_label"]
    baseline_label = cmap["baseline_label"]

    lines: list[str] = [
        "# Kahlus-EM Stage 0 Artifact Report (no-human, synthetic)",
        "",
        "## Run metadata",
        "",
        f"- schema: {report.get('schema')}",
        f"- branch: {report.get('branch')}",
        f"- claim_scope: {report.get('claim_scope')}",
        f"- claim_status: {report.get('claim_status')}",
        f"- seed: {report.get('seed')}",
        f"- fs_hz: {audit.get('fs_hz')}  line_freq_hz: {audit.get('line_freq_hz')}",
        f"- artifact_model: {report.get('artifact_model')}",
        "",
        "## Device / environment log summary",
        "",
    ]
    if env:
        lines.extend(f"- {key}: {value}" for key, value in env.items())
    else:
        lines.append("- none")

    base_means = audit.get("baseline_features_mean", {})
    cond_means = audit.get("condition_features_mean", {})
    lines.extend(
        [
            "",
            "## PSD / channel artifact summary (mean over channels)",
            "",
            "| feature | baseline | condition | abs delta |",
            "| --- | --- | --- | --- |",
        ]
    )
    for key in sorted(set(base_means) | set(cond_means)):
        b = float(base_means.get(key, 0.0))
        c = float(cond_means.get(key, 0.0))
        lines.append(f"| {key} | {b:.6g} | {c:.6g} | {abs(c - b):.6g} |")

    base_band = cmap["conditions"][baseline_label]["band_power_mean"]
    cond_band = cmap["conditions"][condition_label]["band_power_mean"]
    delta_band = cmap["delta_vs_baseline"][condition_label]["band_power_mean_abs_delta"]
    band_sev = severity["band_contamination_score"]
    lines.extend(
        [
            "",
            f"## Contamination map (channel x band; means over {cmap['n_channels']} channels)",
            "",
            f"- attribution: {cmap['attribution']}",
            f"- conditions: {baseline_label} vs {condition_label}",
            "",
            "| band | baseline_mean | condition_mean | abs_delta | band_severity |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for band in cmap["bands"]:
        lines.append(
            f"| {band} | {base_band[band]:.6g} | {cond_band[band]:.6g} | "
            f"{delta_band[band]:.6g} | {band_sev[band]:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Severity score",
            "",
            f"- channel_contamination_score_mean: {severity['channel_contamination_score_mean']:.4f}",
            f"- overall_artifact_severity: {severity['overall_artifact_severity']:.4f}",
            f"- verdict: {severity['verdict']}",
            f"- thresholds: warn>={severity['thresholds']['warn']}  fail>={severity['thresholds']['fail']}",
            "",
            "## Gate result",
            "",
            f"- scientific_claim_allowed: {gate.get('scientific_claim_allowed')}",
            f"- claim_scope: {gate.get('claim_scope')}",
            f"- finite_metrics: {gate.get('finite_metrics')}",
            f"- calibration_checked: {gate.get('calibration_checked')}",
            f"- failure_reasons: {gate.get('failure_reasons') or 'none'}",
            "",
            "## Pass / fail recommendation",
            "",
            f"- {recommendation(bundle)}",
            "",
            "## Safety disclaimer",
            "",
            "- No human subject. No human protocol. No stimulation. No high voltage. No DBD plasma.",
            "- No helium or argon hardware. No coils. No God Helmet.",
            "- No clinical claim. No consciousness claim. No causal EM claim. No predictive human EEG claim.",
            "- Synthetic phantom / no-human data only; offline, no network access.",
            "- `scientific_claim_allowed=true` means ONLY that the no-human Stage 0 artifact audit and "
            "report generation passed — nothing about brains, humans, modulation, or consciousness.",
            "",
        ]
    )
    return "\n".join(lines)


def write_stage0_report(
    out_dir: str | Path,
    bundle: Mapping[str, Any],
    *,
    prefix: str = REPORT_PREFIX,
) -> dict[str, Path]:
    """Write the Markdown report + shared-core report/gate JSONs; return their paths."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = write_report(out, report=bundle["report"], gate=bundle["gate"], prefix=prefix)
    md_path = out / REPORT_MD_NAME
    md_path.write_text(format_stage0_report_md(bundle), encoding="utf-8")
    paths["stage0_report_md"] = md_path
    return paths
