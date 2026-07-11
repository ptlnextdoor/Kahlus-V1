from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

from neurotwin.benchmarks.reports import generate_run_report
from neurotwin.eval.paper_demos import PaperDemoConfig, run_identity_probe, run_leakage_demo
from neurotwin.eval.paper_gate import paper_mode_gate_allows_claim
from neurotwin.reports.evidence_gate import read_json_artifact, write_final_prepared_evidence_gate
from neurotwin.reports.model_card import generate_model_card_report
from neurotwin.repro import write_json


A100_PAPER_MODE_ARTIFACTS = (
    "prepared_baseline_suite.json",
    "seed_aggregate.json",
    "seed_aggregate.csv",
    "baseline_failures.json",
    "paper_mode_gate.json",
    "forecast_eligibility.json",
)


@dataclass(frozen=True)
class RunFinalizeConfig:
    run_dir: str | Path
    paper_mode_dir: str | Path | None = None
    event_manifest: str | Path | None = None
    split_manifest: str | Path | None = None
    window_length: int = 128
    stride: int = 128
    seeds: tuple[int, ...] = (0, 1, 2)


def finalize_run(config: RunFinalizeConfig) -> dict[str, Any]:
    run_dir = Path(config.run_dir)
    if not run_dir.exists():
        raise ValueError(f"finalize run-dir does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise ValueError(f"finalize run-dir is not a directory: {run_dir}")

    copied_paper = copy_paper_mode_artifacts(run_dir, config.paper_mode_dir)
    if copied_paper:
        unavailable = run_dir / "paper_mode_artifacts_unavailable.json"
        if unavailable.exists():
            unavailable.unlink()
    else:
        write_paper_mode_unavailable(run_dir, "Phase 1 paper-mode artifacts missing or paper_mode_gate.json did not pass")

    copied_eval_audit = copy_prepared_eval_audit(run_dir, config.event_manifest)
    run_report = generate_run_report(run_dir)
    run_report_path = run_dir / "RUN_REPORT.md"
    run_report_path.write_text(run_report + "\n", encoding="utf-8")
    diagnostics = run_paper_diagnostics(run_dir, config)
    evidence_gate = write_final_prepared_evidence_gate(run_dir)
    model_card_report = generate_model_card_report(run_dir)
    return {
        "run_dir": str(run_dir),
        "paper_mode_artifacts_copied": copied_paper,
        "eval_audit_copied": copied_eval_audit,
        "run_report": str(run_report_path),
        "diagnostics": diagnostics,
        "evidence_gate": str(run_dir / "evidence_gate.json"),
        "evidence_gate_passed": bool(evidence_gate.get("passed")),
        "model_card": str(run_dir / "EEG_MODEL_CARD.md"),
        "model_card_report": model_card_report,
    }


def copy_paper_mode_artifacts(run_dir: Path, paper_mode_dir: str | Path | None) -> bool:
    if paper_mode_dir in (None, ""):
        return False
    source_dir = Path(paper_mode_dir)
    if not paper_mode_gate_passed(source_dir):
        return False
    copied = False
    for name in A100_PAPER_MODE_ARTIFACTS:
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, run_dir / name)
            copied = True
    return copied


def paper_mode_gate_passed(paper_mode_dir: Path) -> bool:
    gate = read_json_artifact(paper_mode_dir / "paper_mode_gate.json")
    return isinstance(gate, dict) and paper_mode_gate_allows_claim(gate)


def write_paper_mode_unavailable(run_dir: Path, reason: str) -> None:
    write_json(
        run_dir / "paper_mode_artifacts_unavailable.json",
        {
            "status": "paper_mode_artifacts_unavailable",
            "reason": reason,
            "scientific_claim_allowed": False,
        },
    )


def copy_prepared_eval_audit(run_dir: Path, event_manifest: str | Path | None) -> bool:
    if event_manifest in (None, ""):
        return False
    event_path = Path(event_manifest)
    if not event_path.is_file():
        return False
    source = event_path.parent / "eval_audit.json"
    if not source.is_file():
        return False
    shutil.copy2(source, run_dir / "eval_audit.json")
    return True


def run_paper_diagnostics(run_dir: Path, config: RunFinalizeConfig) -> dict[str, bool]:
    if config.event_manifest in (None, "") or config.split_manifest in (None, ""):
        return {"leakage_demo": False, "identity_probe": False}
    event_manifest = Path(config.event_manifest)
    split_manifest = Path(config.split_manifest)
    if not event_manifest.is_file() or not split_manifest.is_file():
        return {"leakage_demo": False, "identity_probe": False}
    demo_config = PaperDemoConfig(
        dataset="prepared",
        event_manifest=event_manifest,
        split_manifest=split_manifest,
        out_dir=run_dir,
        window_length=config.window_length,
        stride=config.stride,
        seeds=config.seeds,
        train_steps=1,
    )
    run_leakage_demo(demo_config)
    run_identity_probe(demo_config)
    return {"leakage_demo": True, "identity_probe": True}
