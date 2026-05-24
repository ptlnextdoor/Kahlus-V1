from __future__ import annotations

import json
from pathlib import Path

from neurotwin.benchmarks.registry import competitor_registry
from neurotwin.benchmarks.suite import run_neural_translation_v1_synthetic
from neurotwin.benchmarks.task_specs import default_translation_tasks


def generate_suite_report(suite: str) -> str:
    if suite == "neural_translation_v1":
        payload = run_neural_translation_v1_synthetic(seed=0)
        lines = [
            "# NeuroTwin Neural Translation V1 Suite",
            "",
            "Scope: synthetic-only plumbing report. This is not scientific evidence.",
            "",
            "## Local Baseline Ranking",
            "",
            "| Model | Mean Rank | Tasks Ranked |",
            "| --- | ---: | ---: |",
        ]
        aggregate = payload.get("baseline_suite", {}).get("aggregate", {})  # type: ignore[union-attr]
        for row in aggregate.get("aggregate_rank", []):
            lines.append(f"| {row['model_id']} | {row['mean_rank']:.3f} | {row['tasks_ranked']} |")
        lines.extend(
            [
                "",
                "Required real-data acceptance still requires held-out subject/site/dataset splits, bootstrap CIs, and exact baseline protocols.",
            ]
        )
        return "\n".join(lines)
    if suite != "translation_smoke":
        raise ValueError("Supported suite reports: translation_smoke, neural_translation_v1")

    tasks = default_translation_tasks()
    competitors = competitor_registry()
    lines = [
        "# NeuroTwin Translation Smoke Suite",
        "",
        "Claim under test: leakage-proof Neural Translation, not first multimodal brain foundation model.",
        "Required split discipline: held-out subject/site/dataset before preprocessing, windowing, or augmentation.",
        "",
        "## Required Tasks",
        "",
        "| Task | Inputs | Targets | Metrics |",
        "| --- | --- | --- | --- |",
    ]
    for task in tasks:
        lines.append(
            f"| {task.name} | {', '.join(task.inputs)} | {', '.join(task.targets)} | {', '.join(task.metrics)} |"
        )

    lines.extend(
        [
            "",
            "## Primary Competitors",
            "",
            "| Competitor | Role | License Status |",
            "| --- | --- | --- |",
        ]
    )
    for competitor in competitors:
        lines.append(f"| {competitor.display_name} | {competitor.role} | {competitor.license_status} |")

    lines.extend(
        [
            "",
            "## Leakage Checks",
            "",
            "- Record IDs cannot appear in more than one split.",
            "- Held-out subject/site/dataset keys cannot overlap when a policy requires them.",
            "- Splits are generated from recording-level manifests before preprocessing/windowing.",
            "- Clinical label prediction is secondary and cannot be the headline claim.",
            "",
            "## Current Acceptance Bar",
            "",
            "NeuroTwin must beat the best strong baseline on aggregate rank and at least two modality groups, with bootstrap confidence intervals.",
        ]
    )
    return "\n".join(lines)


def generate_run_report(run_dir: str | Path) -> str:
    path = Path(run_dir)
    lines = ["# NeuroTwin Run Report", "", f"run_dir={path}"]
    for filename in ("config.yaml", "environment.json", "metrics.json", "summary.json"):
        file_path = path / filename
        if not file_path.exists():
            continue
        lines.extend(["", f"## {filename}", ""])
        if file_path.suffix == ".json":
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
                lines.append("```json")
                lines.append(json.dumps(payload, indent=2, sort_keys=True))
                lines.append("```")
            except json.JSONDecodeError:
                lines.append(file_path.read_text(encoding="utf-8"))
        else:
            lines.append("```yaml")
            lines.append(file_path.read_text(encoding="utf-8").rstrip())
            lines.append("```")
    if len(lines) == 3:
        lines.append("")
        lines.append("No run artifacts found.")
    return "\n".join(lines)
