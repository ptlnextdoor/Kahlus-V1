#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a derived Kahlus Labglass artifact index.")
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = build_index(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"indexed {len(payload['reports'])} reports from {args.root}")
    return 0


def build_index(root: Path) -> dict[str, Any]:
    reports = []
    seen: set[str] = set()
    if root.is_file():
        if root.suffix.lower() == ".zip":
            _append_zip_reports(root, reports, seen)
        elif root.suffix.lower() == ".json" and _is_report_name(root.name, str(root)):
            try:
                payload = json.loads(root.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                reports.append(_report_from_payload(payload, path=root, source=None, size_bytes=root.stat().st_size))
        return {
            "generatedAt": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "root": str(root),
            "reports": reports,
        }
    for path in sorted(root.rglob("*.json")):
        if not _is_report_name(path.name, str(path)):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            key = str(path)
            seen.add(key)
            reports.append(_report_from_payload(payload, path=path, source=None, size_bytes=path.stat().st_size))
    for path in sorted(root.rglob("*.zip")):
        _append_zip_reports(path, reports, seen)
    return {
        "generatedAt": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "root": str(root),
        "reports": reports,
    }


def _append_zip_reports(path: Path, reports: list[dict[str, Any]], seen: set[str]) -> None:
    try:
        with ZipFile(path) as zip_file:
            for info in zip_file.infolist():
                if not _is_report_name(Path(info.filename).name, info.filename):
                    continue
                key = f"{path}:{info.filename}"
                if key in seen:
                    continue
                try:
                    payload = json.loads(zip_file.read(info).decode("utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(payload, dict):
                    seen.add(key)
                    reports.append(_report_from_payload(payload, path=path, source=info.filename, size_bytes=info.file_size))
    except BadZipFile:
        return


def _is_report_name(name: str, full_name: str) -> bool:
    lower_name = name.lower()
    lower_full = full_name.lower()
    if lower_name.endswith("_gate_report.json") or lower_name.endswith("_evidence_gate.json") or lower_name in {"evidence_gate.json", "paper_mode_gate.json"}:
        return True
    return lower_name in {"summary.json", "metric_summary.json"} and "/run/" in lower_full


def _report_from_payload(payload: dict[str, Any], *, path: Path, source: str | None, size_bytes: int) -> dict[str, Any]:
    artifact_name = f"{path.name}:{Path(source).name}" if source else path.name
    failures = _failures(payload)
    passed, status, status_kind = _status(payload, failures)
    title = str(payload.get("milestone") or payload.get("dataset") or payload.get("stage") or Path(source or path.name).stem).upper()
    subtitle = artifact_name.replace("_", " ")
    claim_scope = str(payload.get("claim_scope") or payload.get("forecastability_class") or _claim_from_bool(payload) or "artifact scoped")
    metrics, unit = _metrics(payload)
    controls = _controls(payload, metrics)
    return {
        "id": f"{path}:{source or ''}",
        "title": title,
        "subtitle": subtitle,
        "artifactName": artifact_name,
        "sourcePath": f"{path}/{source}" if source else str(path),
        "milestone": title,
        "status": status,
        "statusKind": status_kind,
        "passed": passed,
        "failures": failures,
        "claimScope": claim_scope,
        "validationScope": str(payload.get("validation_scope") or payload.get("run_kind") or payload.get("schema") or "derived_artifact"),
        "publicDataUsed": bool(payload.get("public_data_used", not payload.get("synthetic_only", False))),
        "externalGeneralization": bool(payload.get("external_generalization", False)),
        "nuisanceConditioned": bool(payload.get("nuisance_conditioned", False)),
        "metricUnit": unit,
        "metrics": metrics,
        "controls": controls,
        "gatePredicate": _gate_predicate(payload),
        "metadata": _metadata(payload, path=path, source=source, size_bytes=size_bytes, failures=failures),
    }


def _status(payload: dict[str, Any], failures: list[str]) -> tuple[bool, str, str]:
    for key in ("gate_passed", "passed", "scientific_claim_allowed", "paper_mode_gate_allows_claim"):
        if key in payload:
            passed = bool(payload[key])
            if passed:
                return True, "PASS", "pass"
            if any("underpowered" in failure.lower() for failure in failures):
                return False, "UNDERPOWERED", "warning"
            return False, "FAIL", "fail"
    status = str(payload.get("status", "")).lower()
    if status in {"pass", "passed", "completed"}:
        return True, "RUN", "neutral"
    if failures:
        return False, "WARN", "warning"
    return False, "RUN", "neutral"


def _failures(payload: dict[str, Any]) -> list[str]:
    failures = (
        payload.get("gate_failures")
        or payload.get("sleep_edf_smoke_failures")
        or payload.get("failure_reasons")
        or payload.get("failures")
        or payload.get("violations")
        or payload.get("baseline_failures")
        or []
    )
    if isinstance(failures, dict):
        failures = [f"{key}: {value}" for key, value in failures.items() if value]
    return [str(item) for item in failures]


def _claim_from_bool(payload: dict[str, Any]) -> str | None:
    if "scientific_claim_allowed" in payload:
        return "scientific claim allowed" if payload["scientific_claim_allowed"] else "scientific claim blocked"
    return None


def _metrics(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    if "worlds" in payload:
        rows = []
        for name, world in payload["worlds"].items():
            pic = world.get("pic", {})
            rfs = world.get("integration_feature_residual", world.get("pic_residual", {}))
            rows.append(
                _metric(
                    name,
                    _float(rfs.get("rfs_bits", pic.get("pic_bits", 0.0))),
                    ci_low=_optional_float(rfs.get("rfs_ci_low", pic.get("pic_ci_low"))),
                    ci_high=_optional_float(rfs.get("rfs_ci_high", pic.get("pic_ci_high"))),
                    events=_optional_int(world.get("positive_events")),
                    baseline=str(world.get("gated_baseline_name", "n/a")),
                    kind="rfs",
                )
            )
        return rows, "RFS bits"
    if "synthetic_known_signal" in payload:
        rows = [
            _metric(
                f"horizon {row.get('horizon')}",
                _float(row.get("rfs_bits", 0.0)),
                ci_low=_optional_float(row.get("rfs_ci_low")),
                ci_high=_optional_float(row.get("rfs_ci_high")),
                events=_optional_int(row.get("positive_events")),
                baseline=str(row.get("gated_baseline_name", "n/a")),
                kind="rfs",
            )
            for row in payload["synthetic_known_signal"].get("curve", [])
        ]
        return rows, "RFS bits"
    for key in ("known_signal", "synthetic_sleep_machinery", "real_sleep_edf"):
        block = payload.get(key)
        if isinstance(block, dict):
            rows = [
                _metric(
                    model,
                    _float(block[model].get("rfs_bits", 0.0)),
                    ci_low=_optional_float(block[model].get("rfs_ci_low")),
                    ci_high=_optional_float(block[model].get("rfs_ci_high")),
                    events=_optional_int(block.get("positive_events")),
                    baseline=str(block.get("gated_baseline_name", "n/a")),
                    kind="rfs",
                )
                for model in ("logistic_full", "gbm_full")
                if isinstance(block.get(model), dict)
            ]
            if rows:
                return rows, "RFS bits"
    if isinstance(payload.get("metrics_by_model"), dict):
        rows = []
        for model, metrics in payload["metrics_by_model"].items():
            if isinstance(metrics, dict):
                rows.append(_metric(str(model), _float(metrics.get("test_mse", metrics.get("mse", 0.0))), baseline="reported", kind="mse"))
        return rows, "MSE"
    rows = []
    for key in ("best_eval_mse", "test_mse", "eval_mse", "best_val_mse", "final_val_mse"):
        if key in payload:
            rows.append(_metric(key, _float(payload[key]), baseline="reported", kind="mse"))
    if rows:
        return rows, "MSE"
    return [], "Metric"


def _gate_predicate(payload: dict[str, Any]) -> dict[str, str]:
    predicate = payload.get("gate_predicate")
    if isinstance(predicate, dict):
        return {key: _status_kind(value) for key, value in predicate.items()}
    failures = _failures(payload)
    return {
        "split": "neutral",
        "finite": "neutral",
        "baseline": "warning" if any("baseline" in failure for failure in failures) else "neutral",
        "controls": "warning" if any("control" in failure for failure in failures) else "neutral",
        "power": "warning" if any("underpowered" in failure for failure in failures) else "neutral",
        "scope": "neutral",
    }


def _status_kind(value: Any) -> str:
    text = str(value).lower()
    return text if text in {"pass", "fail", "warning", "neutral"} else "neutral"


def _controls(payload: dict[str, Any], metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    true_value = metrics[0]["value"] if metrics else 0.0
    source = None
    if "worlds" in payload:
        source = payload["worlds"].get("integrated_predictive", {})
    elif "synthetic_known_signal" in payload and payload["synthetic_known_signal"].get("curve"):
        source = payload["synthetic_known_signal"]["curve"][0]
    elif "known_signal" in payload:
        source = payload["known_signal"]
    shuffled = 0.0
    shifted = 0.0
    if isinstance(source, dict):
        shuffled = _float(_nested(source, ("shuffled_target_control", "rfs_bits"), source.get("shuffled_rfs_bits", 0.0)))
        shifted = _float(_nested(source, ("time_shift_control", "rfs_bits"), source.get("time_shift_rfs_bits", 0.0)))
    return [
        {"name": "True labels", "value": true_value, "kind": "true"},
        {"name": "Shuffled", "value": shuffled, "kind": "control"},
        {"name": "Time-shift", "value": shifted, "kind": "control"},
    ]


def _metadata(payload: dict[str, Any], *, path: Path, source: str | None, size_bytes: int, failures: list[str]) -> list[dict[str, Any]]:
    rows = [
        _meta("Artifact", Path(source).name if source else path.name, "pass"),
        _meta("Archive", path.name if source else "loose file", "neutral"),
        _meta("Bytes", str(size_bytes), "neutral"),
        _meta("Failures", str(len(failures)), "pass" if not failures else "warning"),
    ]
    for key in ("dataset", "branch", "run_kind", "schema", "validation_scope", "claim_scope"):
        if key in payload:
            rows.append(_meta(key.replace("_", " ").title(), str(payload[key]), "neutral"))
    return rows


def _metric(
    model: str,
    value: float,
    *,
    ci_low: float | None = None,
    ci_high: float | None = None,
    events: int | None = None,
    baseline: str,
    kind: str,
) -> dict[str, Any]:
    return {"model": model, "value": value, "ciLow": ci_low, "ciHigh": ci_high, "events": events, "baseline": baseline, "kind": kind}


def _meta(label: str, value: str, state: str) -> dict[str, str]:
    return {"label": label, "value": value, "state": state}


def _nested(payload: dict[str, Any], keys: tuple[str, str], default: Any) -> Any:
    first, second = keys
    if isinstance(payload.get(first), dict):
        return payload[first].get(second, default)
    return default


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return _float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
