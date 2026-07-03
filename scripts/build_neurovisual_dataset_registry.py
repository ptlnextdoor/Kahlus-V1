#!/usr/bin/env python3
"""Build the local NV-1 dataset discovery registry from verified seed metadata.

This is metadata-only. It does not download datasets, run A100, or create adapters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.neurovisual import (  # noqa: E402
    SCORE_DEFINITIONS,
    build_local_manifest_schema,
    build_metadata_only_adapter_plan,
    build_metadata_query_plan,
    build_registry_verification_summary,
    build_seed_dataset_registry,
    build_split_audit_plan,
    evaluate_neurovisual_claim_gate,
    validate_local_split_records,
)
from neurotwin.repro import hash_file, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = build_seed_dataset_registry()
    summary = build_registry_verification_summary(registry)
    adapter_plan = build_metadata_only_adapter_plan(registry)
    query_plan = build_metadata_query_plan(registry)
    local_manifest_schema = build_local_manifest_schema(registry)
    split_audit_plan = build_split_audit_plan(registry)
    synthetic_split_manifest = _build_synthetic_split_manifest()
    synthetic_split_audit = validate_local_split_records(synthetic_split_manifest, registry=registry)
    report_text = _format_registry_report(registry, summary, adapter_plan)
    claim_gate = evaluate_neurovisual_claim_gate(
        claim_scope="dataset_registry_ready",
        payloads=[
            registry,
            summary,
            adapter_plan,
            query_plan,
            local_manifest_schema,
            split_audit_plan,
            synthetic_split_audit,
            report_text,
        ],
    )
    json_path = write_json(out_dir / "neurovisual_dataset_registry.json", registry)
    summary_path = write_json(out_dir / "neurovisual_registry_verification_summary.json", summary)
    adapter_plan_path = write_json(out_dir / "neurovisual_adapter_plan.json", adapter_plan)
    query_plan_path = write_json(out_dir / "neurovisual_metadata_query_plan.json", query_plan)
    local_manifest_schema_path = write_json(
        out_dir / "neurovisual_local_manifest_schema.json",
        local_manifest_schema,
    )
    split_audit_plan_path = write_json(out_dir / "neurovisual_split_audit_plan.json", split_audit_plan)
    synthetic_split_manifest_path = write_json(
        out_dir / "neurovisual_synthetic_split_manifest.json",
        synthetic_split_manifest,
    )
    synthetic_split_audit_path = write_json(
        out_dir / "neurovisual_synthetic_split_audit.json",
        synthetic_split_audit,
    )
    claim_gate_path = write_json(out_dir / "neurovisual_registry_claim_gate.json", claim_gate)
    report_path = out_dir / "neurovisual_dataset_registry.md"
    report_path.write_text(report_text, encoding="utf-8")
    evidence_manifest = _build_evidence_manifest(
        [
            json_path,
            summary_path,
            adapter_plan_path,
            query_plan_path,
            local_manifest_schema_path,
            split_audit_plan_path,
            synthetic_split_manifest_path,
            synthetic_split_audit_path,
            claim_gate_path,
            report_path,
        ]
    )
    evidence_manifest_path = write_json(out_dir / "neurovisual_registry_evidence_manifest.json", evidence_manifest)

    confirmed = [entry for entry in registry["entries"] if entry["verification_status"] == "confirmed"]
    unverified = [entry for entry in registry["entries"] if entry["verification_status"] == "unverified"]
    rejected = [entry for entry in registry["entries"] if entry["verification_status"] == "rejected"]
    print(f"branch=nv1 out_dir={out_dir.resolve()}")
    print(f"registry={json_path}")
    print(f"verification_summary={summary_path}")
    print(f"adapter_plan={adapter_plan_path}")
    print(f"metadata_query_plan={query_plan_path}")
    print(f"local_manifest_schema={local_manifest_schema_path}")
    print(f"split_audit_plan={split_audit_plan_path}")
    print(f"synthetic_split_manifest={synthetic_split_manifest_path}")
    print(f"synthetic_split_audit={synthetic_split_audit_path}")
    print(f"claim_gate={claim_gate_path}")
    print(f"evidence_manifest={evidence_manifest_path}")
    print(f"report={report_path}")
    print(f"confirmed={len(confirmed)} unverified={len(unverified)} rejected={len(rejected)}")
    print("bulk_dataset_download=false")
    print("a100_jobs_launched=false")
    if not claim_gate["passed"]:
        print(f"claim_gate_passed=false failure_reasons={claim_gate['failure_reasons']}")
        return 1
    return 0


def _build_evidence_manifest(artifact_paths: list[Path]) -> dict[str, object]:
    return {
        "schema": "kahlus.nv1.registry_evidence_manifest.v1",
        "scope": "nv1_dataset_registry_metadata_only",
        "execution": {
            "bulk_dataset_download": False,
            "a100_jobs_launched": False,
            "cluster_jobs_launched": False,
            "metadata_only": True,
        },
        "artifacts": [
            {
                "path": path.name,
                "sha256": hash_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
    }


def _build_synthetic_split_manifest() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split_name in ("train", "validation", "test"):
        rows.append(
            {
                "record_id": f"synthetic-{split_name}-record-001",
                "subject_id": f"synthetic-{split_name}-subject-001",
                "session_id": f"synthetic-{split_name}-session-001",
                "dataset_name": "CHB-MIT Scalp EEG Database",
                "signal_path": f"USER_PROVIDED_SIGNAL_PATH_{split_name.upper()}",
                "sampling_rate": 256,
                "channel_names": ["FP1-F7", "F7-T7"],
                "event_annotations_path": f"USER_PROVIDED_EVENT_ANNOTATIONS_PATH_{split_name.upper()}",
                "task_label": "retrospective_event_window_research",
                "license_or_access_confirmation": "user_confirmed_access_terms",
                "split_name": split_name,
            }
        )
    return rows


def _format_registry_report(
    registry: dict[str, object],
    summary: dict[str, object],
    adapter_plan: dict[str, object],
) -> str:
    entries = registry["entries"]
    assert isinstance(entries, list)
    lines = [
        "# NV-1 Neurovisual Dataset Discovery Registry",
        "",
        "NV-1 is a side branch for metadata discovery and structured symptom mapping. It is not a diagnosis tool.",
        "",
        "## Execution Boundary",
        "",
        "- no bulk dataset download",
        "- no A100 or cluster job",
        "- no private patient data",
        "- no adapter is written until metadata and access terms are verified",
        "",
        "## Score Definitions",
        "",
        *[f"- {score}: {definition}" for score, definition in sorted(SCORE_DEFINITIONS.items())],
        "",
        "## Registry Entries",
        "",
        "| adapter_priority | dataset | status | score | source | notes |",
        "| ---: | --- | --- | ---: | --- | --- |",
    ]
    for entry in sorted(entries, key=lambda item: item["adapter_priority"]):
        lines.append(
            "| "
            f"{entry['adapter_priority']} | "
            f"{entry['dataset_name']} | "
            f"{entry['verification_status']} | "
            f"{entry['neurovisual_relevance_score']} | "
            f"{entry['source_url_or_identifier']} | "
            f"{entry['notes']} |"
        )
    lines.extend(
        [
            "",
            "## OpenNeuro Verification Results",
            "",
            "No OpenNeuro accession is confirmed in this sprint. Prior IDs remain UNVERIFIED until catalog metadata confirms accession, subject count, task, modalities, and license.",
            "",
            "## Verification Summary Artifact",
            "",
            "`neurovisual_registry_verification_summary.json` records counts, dataset sources searched, OpenNeuro verification results, rejected datasets, score-3 assignments, and top adapter priorities.",
            "",
            "## Metadata-Only Adapter Plan",
            "",
            "`neurovisual_adapter_plan.json` records planned local-manifest fields, split/audit strategy, and leakage risks for the top confirmed adapter priorities. It does not implement adapters, download data, or run models.",
            "",
            "## Metadata Query Plan",
            "",
            "`neurovisual_metadata_query_plan.json` records planned OpenNeuro, NEMAR, MOABB, EEGDash, and public multimodal EEG catalog searches. The plan does not execute queries or confirm accessions.",
            "",
            "## Local Manifest Schema",
            "",
            "`neurovisual_local_manifest_schema.json` records required local-manifest fields, field type expectations, confirmed allowed dataset names, and blocked unverified or rejected dataset names.",
            "",
            "## Split Audit Plan",
            "",
            "`neurovisual_split_audit_plan.json` records required split keys, leakage checks, and baseline-before-model gates for confirmed adapter priorities. It does not execute splits or baselines.",
            "",
            "## Synthetic Split Fixture",
            "",
            "`neurovisual_synthetic_split_manifest.json` and `neurovisual_synthetic_split_audit.json` provide a stable local example for the split-audit CLI. The fixture uses symbolic `USER_PROVIDED_*` path labels only; it does not check raw files, download data, execute adapters, run baselines/models, or launch A100.",
            "",
            "## Claim Gate Artifact",
            "",
            "`neurovisual_registry_claim_gate.json` audits the registry, verification summary, adapter plan, and generated report under the `dataset_registry_ready` scope before the builder exits successfully.",
            "",
            "## Evidence Manifest",
            "",
            "`neurovisual_registry_evidence_manifest.json` records SHA-256 checksums and byte sizes for the registry, verification summary, adapter plan, metadata query plan, local manifest schema, split audit plan, synthetic split fixture, claim gate, and Markdown report.",
            "",
            "## Deferred Adapter Plan",
            "",
            "1. Validate metadata and access terms.",
            "2. Add local manifest adapter only after metadata is confirmed.",
            "3. Run split/leakage audit before any baseline ladder.",
            "4. Keep all claims research-only and non-diagnostic.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
