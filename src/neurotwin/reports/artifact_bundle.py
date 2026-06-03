from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


MINIMUM_MODEL_CARD_SOURCE_ARTIFACTS = (
    "prepared_baseline_suite.json",
    "paper_mode_gate.json",
    "evidence_gate.json",
    "eval_audit.json",
    "summary.json",
    "metrics.json",
)


@dataclass(frozen=True)
class ModelCardSourceArtifacts:
    run_dir: Path
    summary: Any
    metrics: Any
    prepared: Any
    eval_audit: Any
    claim_gate: Any
    evidence_gate: Any
    identity_probe: Any
    leakage_demo: Any
    source_artifacts: tuple[str, ...]


def load_model_card_source_artifacts(run_dir: str | Path) -> ModelCardSourceArtifacts:
    path = Path(run_dir)
    if not path.exists():
        raise ValueError(f"model-card run-dir does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"model-card run-dir is not a directory: {path}")

    source_artifacts = tuple(name for name in MINIMUM_MODEL_CARD_SOURCE_ARTIFACTS if (path / name).exists())
    if not source_artifacts:
        required = ", ".join(MINIMUM_MODEL_CARD_SOURCE_ARTIFACTS)
        raise ValueError(f"model-card run-dir lacks required source artifacts: expected at least one of {required}")

    return ModelCardSourceArtifacts(
        run_dir=path,
        summary=read_json_artifact(path / "summary.json"),
        metrics=read_json_artifact(path / "metrics.json"),
        prepared=read_json_artifact(path / "prepared_baseline_suite.json"),
        eval_audit=first_json_artifact(path, ("eval_audit.json", "LEAKAGE_AUDIT.json")),
        claim_gate=first_json_artifact(path, ("paper_mode_gate.json", "CLAIM_GATE.json")),
        evidence_gate=first_json_artifact(path, ("evidence_gate.json",)),
        identity_probe=first_json_artifact(path, ("identity_probe.json", "IDENTITY_PROBE.json")),
        leakage_demo=first_json_artifact(path, ("leakage_demo.json", "LEAKAGE_DEMO.json")),
        source_artifacts=source_artifacts,
    )


def write_paper_artifact_aliases(source: ModelCardSourceArtifacts) -> list[str]:
    path = source.run_dir
    aliases: list[str] = []
    if isinstance(source.eval_audit, dict) and source.eval_audit:
        out = path / "LEAKAGE_AUDIT.json"
        out.write_text(json.dumps(source.eval_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        aliases.append(str(out))
    if isinstance(source.claim_gate, dict) and source.claim_gate:
        out = path / "CLAIM_GATE.json"
        out.write_text(json.dumps(source.claim_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        aliases.append(str(out))
    baseline_payload = source.prepared if isinstance(source.prepared, dict) and source.prepared else source.metrics
    ranking_rows = baseline_ranking_rows(baseline_payload)
    if ranking_rows:
        out = path / "BASELINE_RANKING.csv"
        out.write_text(
            csv_rows(("task_id", "model_id", "metric", "value", "rank"), ranking_rows),
            encoding="utf-8",
        )
        aliases.append(str(out))
    return aliases


def baseline_ranking_rows(payload: Any) -> list[tuple[Any, ...]]:
    if not isinstance(payload, dict):
        return []
    baseline_suite = payload.get("baseline_suite") if isinstance(payload.get("baseline_suite"), dict) else payload
    tasks = baseline_suite.get("tasks", {}) if isinstance(baseline_suite, dict) else {}
    rows: list[tuple[Any, ...]] = []
    if isinstance(tasks, dict):
        for task_id, task_payload in tasks.items():
            if not isinstance(task_payload, dict):
                continue
            for row in task_payload.get("ranking", []):
                if isinstance(row, dict):
                    rows.append((task_id, row.get("model_id", ""), row.get("metric", ""), row.get("value", ""), row.get("rank", "")))
    return rows


def first_json_artifact(path: Path, names: tuple[str, ...]) -> Any:
    for name in names:
        payload = read_json_artifact(path / name)
        if payload:
            return payload
    return {}


def read_json_artifact(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"error": "invalid_json", "path": str(path), "message": str(exc)}
    except OSError as exc:
        return {"error": "read_failed", "path": str(path), "message": str(exc)}


def join_list(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value) if value else "none"
    return str(value) if value not in (None, "") else "unknown"


def format_aggregate_rank(payload: dict[str, Any]) -> str:
    aggregate = payload.get("aggregate", {}) if isinstance(payload, dict) else {}
    rows = aggregate.get("aggregate_rank", []) if isinstance(aggregate, dict) else []
    if not isinstance(rows, list) or not rows:
        return "missing"
    formatted = []
    for row in rows[:5]:
        if isinstance(row, dict):
            formatted.append(f"{row.get('model_id')}:{row.get('mean_rank')}")
    return "; ".join(formatted)


def diagnostic_status(payload: Any) -> str:
    if not isinstance(payload, dict) or not payload:
        return "missing"
    return str(payload.get("interpretation") or payload.get("identity_confounding_risk") or payload.get("probe") or payload.get("demo"))


def append_artifact_errors(lines: list[str], *payloads: Any) -> None:
    errors = [payload for payload in payloads if is_artifact_error(payload)]
    if not errors:
        return
    lines.extend(["", "## Artifact Errors", ""])
    for payload in errors:
        lines.append(f"- {payload.get('error')} {payload.get('path')}: {payload.get('message', '')}")


def is_artifact_error(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("error"), str) and isinstance(payload.get("path"), str)


def csv_rows(header: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(csv_cell(value) for value in row))
    return "\n".join(lines) + "\n"


def csv_cell(value: Any) -> str:
    text = str(value)
    if any(char in text for char in [",", "\"", "\n"]):
        return "\"" + text.replace("\"", "\"\"") + "\""
    return text
