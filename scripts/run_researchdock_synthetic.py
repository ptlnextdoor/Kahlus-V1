#!/usr/bin/env python3
"""Run the ResearchDock synthetic response-profile scaffold.

No A100, no hardware access, no webcam access, no real participant data, and no
clinical claims. This is a local RD-0 artifact writer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.repro import write_json  # noqa: E402
from neurotwin.researchdock import (  # noqa: E402
    RESEARCHDOCK_ALLOWED_CLAIM_SCOPE,
    build_rd1_session_protocol,
    build_researchdock_observation_task,
    build_researchdock_data_card,
    build_researchdock_gate,
    build_researchdock_pilot_manifest,
    compute_researchdock_metrics,
    export_researchdock_sessions,
    audit_response_profile_readiness,
    make_synthetic_researchdock_sessions,
    protocol_to_dict,
    researchdock_interface_contract,
    run_researchdock_observation_benchmark,
    write_researchdock_pilot_preflight_artifacts,
    write_researchdock_public_dataset_review,
    write_researchdock_observation_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write-session-export", action="store_true")
    parser.add_argument("--run-observation-model", action="store_true")
    parser.add_argument("--write-public-dataset-review", action="store_true")
    parser.add_argument("--write-pilot-preflight", action="store_true")
    parser.add_argument("--write-profile-readiness", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = make_synthetic_researchdock_sessions(seed=args.seed)
    metrics = compute_researchdock_metrics(sessions)
    data_card = build_researchdock_data_card(sessions)
    data_card_passed = _data_card_passed(data_card)
    gate = build_researchdock_gate(
        dataset=str(data_card["dataset_id"]),
        metrics=metrics,
        claim_scope=RESEARCHDOCK_ALLOWED_CLAIM_SCOPE,
        data_card_passed=data_card_passed,
    )

    metrics_path = write_json(
        out_dir / "researchdock_metrics.json",
        {
            "dataset_id": data_card["dataset_id"],
            "seed": int(args.seed),
            "session_metrics": metrics,
        },
    )
    data_card_path = write_json(out_dir / "researchdock_data_card.json", data_card)
    gate_path = write_json(out_dir / "researchdock_evidence_gate.json", gate)
    failure_reasons_path = write_json(
        out_dir / "researchdock_failure_reasons.json",
        _failure_reasons_payload(gate, data_card),
    )
    protocol_path = None
    interface_path = None
    export_dir = None
    if args.write_session_export:
        protocol = build_rd1_session_protocol(seed=args.seed)
        protocol_path = write_json(out_dir / "researchdock_rd1_protocol.json", protocol_to_dict(protocol))
        interface_path = write_json(out_dir / "researchdock_interface_contract.json", researchdock_interface_contract())
        export_dir = out_dir / "session_export"
        export_researchdock_sessions(sessions, export_dir)
    observation_paths = None
    observation_result = None
    if args.run_observation_model:
        task = build_researchdock_observation_task(sessions, seed=args.seed)
        observation_result = run_researchdock_observation_benchmark(task)
        observation_paths = write_researchdock_observation_artifacts(out_dir, task=task, result=observation_result)
    public_dataset_paths = None
    if args.write_public_dataset_review:
        public_dataset_paths = write_researchdock_public_dataset_review(out_dir)
    pilot_preflight_paths = None
    if args.write_pilot_preflight:
        protocol = build_rd1_session_protocol(seed=args.seed)
        pilot_manifest = build_researchdock_pilot_manifest(sessions=sessions, protocol=protocol)
        pilot_preflight_paths = write_researchdock_pilot_preflight_artifacts(out_dir, pilot_manifest)
    profile_readiness_paths = None
    if args.write_profile_readiness:
        readiness = audit_response_profile_readiness(metrics)
        profile_readiness_paths = _write_profile_readiness_artifacts(out_dir, readiness)
    report_path = out_dir / "researchdock_report.md"
    report_path.write_text(
        _format_report(
            metrics,
            data_card,
            gate,
            failure_reasons=_failure_reasons_payload(gate, data_card),
            observation_result=observation_result,
            profile_readiness=readiness if args.write_profile_readiness else None,
            write_session_export=args.write_session_export,
            run_observation_model=args.run_observation_model,
            write_public_dataset_review=args.write_public_dataset_review,
            write_pilot_preflight=args.write_pilot_preflight,
            write_profile_readiness=args.write_profile_readiness,
        ),
        encoding="utf-8",
    )

    print(f"branch=researchdock dataset={data_card['dataset_id']} out_dir={out_dir.resolve()}")
    print(f"claim_scope={RESEARCHDOCK_ALLOWED_CLAIM_SCOPE}")
    print(f"sessions={len(sessions)} trials={data_card['n_trials']}")
    print(f"scientific_claim_allowed={gate['scientific_claim_allowed']}")
    print(f"metrics={metrics_path}")
    print(f"data_card={data_card_path}")
    print(f"evidence_gate={gate_path}")
    print(f"failure_reasons={failure_reasons_path}")
    if protocol_path is not None:
        print(f"rd1_protocol={protocol_path}")
    if interface_path is not None:
        print(f"interface_contract={interface_path}")
    if export_dir is not None:
        print(f"session_export={export_dir}")
    if observation_paths is not None:
        print(f"observation_model_metrics={observation_paths['metrics']}")
        print(f"observation_model_baselines={observation_paths['baseline_table_csv']}")
        print(f"observation_split_audit={observation_paths['split_audit']}")
        print(f"observation_model_report={observation_paths['report']}")
    if public_dataset_paths is not None:
        print(f"public_dataset_review={public_dataset_paths['json']}")
        print(f"public_dataset_report={public_dataset_paths['report']}")
    if pilot_preflight_paths is not None:
        print(f"pilot_manifest={pilot_preflight_paths['manifest']}")
        print(f"pilot_preflight_gate={pilot_preflight_paths['gate']}")
        print(f"pilot_preflight_report={pilot_preflight_paths['report']}")
    if profile_readiness_paths is not None:
        print(f"profile_readiness={profile_readiness_paths['json']}")
        print(f"profile_readiness_report={profile_readiness_paths['report']}")
    print(f"report={report_path}")
    return 0


def _data_card_passed(data_card: dict[str, object]) -> bool:
    return bool(
        data_card.get("dataset_id")
        and not data_card.get("contains_pii")
        and not data_card.get("contains_real_participant_data")
        and not data_card.get("contains_clinical_labels")
        and not data_card.get("contains_stimulation")
    )


def _failure_reasons_payload(gate: dict[str, object], data_card: dict[str, object]) -> dict[str, object]:
    return {
        "gate_failures": list(gate.get("failure_reasons", ())),
        "data_card_failures": _data_card_failure_reasons(data_card),
        "blocked_claim_terms": list(gate.get("gate_criteria", {}).get("blocked_claim_terms", ())),
    }


def _data_card_failure_reasons(data_card: dict[str, object]) -> list[str]:
    failures: list[str] = []
    if not data_card.get("dataset_id"):
        failures.append("dataset_id missing")
    for field, reason in (
        ("contains_pii", "data card contains PII"),
        ("contains_real_participant_data", "data card contains real participant data"),
        ("contains_clinical_labels", "data card contains clinical labels"),
        ("contains_stimulation", "data card contains stimulation"),
    ):
        if data_card.get(field):
            failures.append(reason)
    return failures


def _report_list(values: object) -> str:
    items = [str(value) for value in values] if isinstance(values, list | tuple) else []
    return ", ".join(items) if items else "none"


def _format_report(
    metrics: list[dict[str, object]],
    data_card: dict[str, object],
    gate: dict[str, object],
    *,
    failure_reasons: dict[str, object],
    observation_result: dict[str, object] | None,
    profile_readiness: dict[str, object] | None,
    write_session_export: bool,
    run_observation_model: bool,
    write_public_dataset_review: bool,
    write_pilot_preflight: bool,
    write_profile_readiness: bool,
) -> str:
    lines = [
        "# ResearchDock Synthetic Response-Profile Report",
        "",
        f"- dataset: {data_card['dataset_id']}",
        f"- sessions: {data_card['n_sessions']}",
        f"- trials: {data_card['n_trials']}",
        f"- claim_scope: {gate['claim_scope']}",
        f"- scientific_claim_allowed: {gate['scientific_claim_allowed']}",
        "",
        "## Synthetic Metrics",
        "",
        "| session_id | profile | reward_delta | rt_change | pupil_amp | accuracy |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(
            "| {session_id} | {profile} | {reward:.6g} | {rt:.6g} | {pupil:.6g} | {acc:.6g} |".format(
                session_id=row["session_id"],
                profile=row["profile"],
                reward=float(row["reward_response_delta"]),
                rt=float(row["reaction_time_change"]),
                pupil=float(row["pupil_response_amplitude"]),
                acc=float(row["task_accuracy_mean"]),
            )
        )
    quality_flags = data_card.get("quality_flags", ())
    lines.extend(
        [
            "",
            "## Data Card Summary",
            "",
            f"- contains_pii: {data_card['contains_pii']}",
            f"- contains_real_participant_data: {data_card['contains_real_participant_data']}",
            f"- contains_clinical_labels: {data_card['contains_clinical_labels']}",
            f"- contains_stimulation: {data_card['contains_stimulation']}",
            f"- modalities: {', '.join(str(value) for value in data_card.get('modalities', ()))}",
            f"- profiles: {', '.join(str(value) for value in data_card.get('profiles', ()))}",
            f"- quality_flags: {_report_list(quality_flags)}",
            "",
            "## Quality Flag Counts",
            "",
            *_quality_flag_count_rows(metrics),
            "",
            "## Evidence Artifact Index",
            "",
            *_evidence_artifact_index(
                write_session_export=write_session_export,
                run_observation_model=run_observation_model,
                write_public_dataset_review=write_public_dataset_review,
                write_pilot_preflight=write_pilot_preflight,
                write_profile_readiness=write_profile_readiness,
            ),
            "",
            "## Failure Reasons Summary",
            "",
            f"- gate_failure_count: {len(failure_reasons['gate_failures'])}",
            f"- data_card_failure_count: {len(failure_reasons['data_card_failures'])}",
            f"- blocked_claim_terms_count: {len(failure_reasons['blocked_claim_terms'])}",
            "",
            "## Failure Reasons Details",
            "",
            f"- gate_failures: {_report_list(failure_reasons['gate_failures'])}",
            f"- data_card_failures: {_report_list(failure_reasons['data_card_failures'])}",
            f"- blocked_claim_terms: {_report_list(failure_reasons['blocked_claim_terms'])}",
            "",
            "## Clinical Claim Boundary",
            "",
            "- ResearchDock RD-0 measures synthetic response profiles only.",
            "- It does not diagnose depression, PTSD, anhedonia, epilepsy, or any clinical condition.",
            "- It does not treat, stimulate, or recommend intervention.",
        ]
    )
    criteria = gate.get("gate_criteria")
    if isinstance(criteria, dict):
        lines.extend(
            [
                "",
                "## Evidence Gate Criteria",
                "",
                f"- allowed_claim_scope: {criteria['allowed_claim_scope']}",
                f"- blocked_claim_terms: {', '.join(criteria['blocked_claim_terms'])}",
                f"- requires_data_card_passed: {criteria['requires_data_card_passed']}",
                f"- requires_baseline_table_present: {criteria['requires_baseline_table_present']}",
                f"- requires_finite_metrics: {criteria['requires_finite_metrics']}",
                f"- requires_calibration_checked: {criteria['requires_calibration_checked']}",
                f"- requires_synthetic_only: {criteria['requires_synthetic_only']}",
            ]
        )
    if run_observation_model and observation_result is not None:
        lines.extend(
            [
                "",
                "## RD-2 Baseline Ladder Summary",
                "",
                *_observation_baseline_ladder_rows(observation_result),
                f"- best_baseline: {observation_result.get('best_baseline')}",
                f"- best_baseline_mse: {_report_float(observation_result.get('best_baseline_mse'))}",
                f"- observation_operator_beats_best_baseline: {observation_result.get('observation_operator_beats_best_baseline')}",
                "",
                "## RD-2 Split Audit Summary",
                "",
                *_observation_split_summary_rows(observation_result),
                "",
                "## RD-2 Missing-Modality Summary",
                "",
                *_observation_missing_modality_rows(observation_result),
            ]
        )
    if write_session_export:
        lines.extend(
            [
                "",
                "## RD-1 Local Prototype",
                "",
                "- Wrote deterministic task protocol, interface contract, and CSV session export.",
                "- The interface contract is design-only and does not open a camera or PPG device.",
            ]
        )
    if run_observation_model:
        lines.extend(
            [
                "",
                "## RD-2 Observation Model",
                "",
                "- Wrote synthetic multimodal observation-model metrics, baseline table, and report.",
                "- Observation artifacts: researchdock_observation_metrics.json, researchdock_observation_baselines.csv, researchdock_observation_split_audit.json, researchdock_observation_report.md.",
                "- Mean and ridge baselines are evaluated before the ResearchDock observation operator.",
                "- This is synthetic pretraining only and does not support clinical claims.",
            ]
        )
    if write_public_dataset_review:
        lines.extend(
            [
                "",
                "## RD-3 Public Dataset Review",
                "",
                "- Wrote WESAD, DEAP, and SEED mapping review artifacts.",
                "- RD-3 does not add dataset loaders or download raw participant data.",
                "- Public affect labels are response/profile labels, not diagnosis labels.",
            ]
        )
    if write_pilot_preflight:
        lines.extend(
            [
                "",
                "## RD-4 Pilot Preflight",
                "",
                "- Wrote pre-collection pilot manifest, preflight gate, and report.",
                "- RD-4 requires RD-0 through RD-3 evidence before any collection path.",
                "- It does not open hardware, collect real participant data, or support clinical claims.",
            ]
        )
    if write_profile_readiness:
        if profile_readiness is not None:
            lines.extend(
                [
                    "",
                    "## RD-5 Readiness Audit Summary",
                    "",
                    *_profile_readiness_summary_rows(profile_readiness),
                ]
            )
        lines.extend(
            [
                "",
                "## RD-5 Response-Profile Readiness",
                "",
                "- Wrote a readiness audit for future latent response-profile clustering.",
                "- No clustering was performed, no cluster labels were emitted, and no clinical claims are supported.",
            ]
        )
    if gate["failure_reasons"]:
        lines.extend(["", "## Gate Failures", "", *[f"- {reason}" for reason in gate["failure_reasons"]]])
    return "\n".join(lines) + "\n"


def _quality_flag_count_rows(metrics: list[dict[str, object]]) -> list[str]:
    counts: dict[str, int] = {}
    for row in metrics:
        flags = row.get("quality_flags", ())
        if not isinstance(flags, list | tuple):
            continue
        for flag in flags:
            key = str(flag)
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return ["- none"]
    return [f"- {flag}: {count}" for flag, count in sorted(counts.items())]


def _observation_baseline_ladder_rows(result: dict[str, object]) -> list[str]:
    metrics_by_model = result.get("metrics_by_model", {})
    if not isinstance(metrics_by_model, dict):
        return ["- baseline_ladder: unavailable"]
    rows = ["| model_id | role | mse | mae |", "| --- | --- | ---: | ---: |"]
    for model_id in result.get("model_order", ()):
        if not isinstance(model_id, str):
            continue
        metrics = metrics_by_model.get(model_id)
        if not isinstance(metrics, dict):
            continue
        role = "candidate" if model_id == "researchdock_observation_operator" else "baseline"
        rows.append(
            "| {model_id} | {role} | {mse} | {mae} |".format(
                model_id=model_id,
                role=role,
                mse=_report_float(metrics.get("mse")),
                mae=_report_float(metrics.get("mae")),
            )
        )
    return rows


def _observation_split_summary_rows(result: dict[str, object]) -> list[str]:
    split = result.get("split", {})
    if not isinstance(split, dict):
        return ["- split_type: unavailable", "- leakage_passed: False", "- failure_reasons: missing split payload"]
    train_subjects = list(split.get("train_subjects", ()))
    test_subjects = list(split.get("test_subjects", ()))
    subject_overlap = bool(split.get("subject_overlap"))
    failure_reasons: list[str] = []
    if not train_subjects:
        failure_reasons.append("missing train subjects")
    if not test_subjects:
        failure_reasons.append("missing test subjects")
    if subject_overlap:
        failure_reasons.append("subject overlap across train/test split")
    return [
        "- split_type: subject_held_out",
        f"- train_subjects: {len(train_subjects)}",
        f"- test_subjects: {len(test_subjects)}",
        f"- subject_overlap: {subject_overlap}",
        f"- leakage_passed: {not failure_reasons}",
        f"- failure_reasons: {_report_list(failure_reasons)}",
    ]


def _observation_missing_modality_rows(result: dict[str, object]) -> list[str]:
    audit = result.get("missing_modality_audit", {})
    if not isinstance(audit, dict):
        return ["- total_trials: none", "- eligible_trials: none", "- skipped_trials: none"]
    skip_reasons = audit.get("skip_reasons", ())
    rows = [
        f"- total_trials: {audit.get('total_trials')}",
        f"- eligible_trials: {audit.get('eligible_trials')}",
        f"- skipped_trials: {audit.get('skipped_trials')}",
        f"- skip_reasons: {_report_list(skip_reasons)}",
        "",
        "| reason | count |",
        "| --- | ---: |",
    ]
    for reason in ("missing_sensor_packet", "missing_pupil_diameter", "missing_hrv_proxy", "missing_behavior_response"):
        rows.append(f"| {reason} | {int(audit.get(reason, 0))} |")
    return rows


def _profile_readiness_summary_rows(readiness: dict[str, object]) -> list[str]:
    return [
        f"- readiness_scope: {readiness.get('readiness_scope')}",
        f"- ready_for_future_clustering: {readiness.get('ready_for_future_clustering')}",
        f"- clustering_performed: {readiness.get('clustering_performed')}",
        f"- n_metric_rows: {readiness.get('n_metric_rows')}",
        f"- minimum_sessions_required: {readiness.get('minimum_sessions_required')}",
        f"- finite_profile_vectors: {readiness.get('finite_profile_vectors')}",
        f"- failure_reasons: {_report_list(readiness.get('failure_reasons', ()))}",
    ]


def _report_float(value: object) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return "none"


def _evidence_artifact_index(
    *,
    write_session_export: bool,
    run_observation_model: bool,
    write_public_dataset_review: bool,
    write_pilot_preflight: bool,
    write_profile_readiness: bool,
) -> list[str]:
    artifacts = [
        "researchdock_metrics.json",
        "researchdock_data_card.json",
        "researchdock_evidence_gate.json",
        "researchdock_failure_reasons.json",
        "researchdock_report.md",
    ]
    if write_session_export:
        artifacts.extend(
            [
                "researchdock_rd1_protocol.json",
                "researchdock_interface_contract.json",
                "session_export/",
            ]
        )
    if run_observation_model:
        artifacts.extend(
            [
                "researchdock_observation_task.json",
                "researchdock_observation_metrics.json",
                "researchdock_observation_baselines.csv",
                "researchdock_observation_split_audit.json",
                "researchdock_observation_report.md",
            ]
        )
    if write_public_dataset_review:
        artifacts.extend(
            [
                "researchdock_public_dataset_review.json",
                "researchdock_public_dataset_review.md",
            ]
        )
    if write_pilot_preflight:
        artifacts.extend(
            [
                "researchdock_pilot_manifest.json",
                "researchdock_pilot_preflight_gate.json",
                "researchdock_pilot_preflight_report.md",
            ]
        )
    if write_profile_readiness:
        artifacts.extend(
            [
                "researchdock_profile_readiness.json",
                "researchdock_profile_readiness_report.md",
            ]
        )
    return [f"- {artifact}" for artifact in artifacts]


def _write_profile_readiness_artifacts(out_dir: Path, readiness: dict[str, object]) -> dict[str, Path]:
    json_path = write_json(out_dir / "researchdock_profile_readiness.json", readiness)
    report_path = out_dir / "researchdock_profile_readiness_report.md"
    report_path.write_text(_format_profile_readiness_report(readiness), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _format_profile_readiness_report(readiness: dict[str, object]) -> str:
    lines = [
        "# ResearchDock Response-Profile Readiness Audit",
        "",
        f"- readiness_scope: {readiness['readiness_scope']}",
        f"- ready_for_future_clustering: {readiness['ready_for_future_clustering']}",
        f"- clustering_performed: {readiness['clustering_performed']}",
        f"- n_metric_rows: {readiness['n_metric_rows']}",
        f"- minimum_sessions_required: {readiness['minimum_sessions_required']}",
        f"- finite_profile_vectors: {readiness['finite_profile_vectors']}",
        "",
        "No clustering was performed. This artifact does not emit cluster labels and does not support clinical claims.",
        "",
        "## Failure Reasons",
        "",
    ]
    reasons = list(readiness.get("failure_reasons", ()))
    if reasons:
        lines.extend(f"- {reason}" for reason in reasons)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Profile Vector Keys",
            "",
            *[f"- {key}" for key in readiness.get("profile_vector_keys", ())],
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
