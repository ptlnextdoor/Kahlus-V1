"""Artifact-bound authority for claim-bearing forecasting evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from neurotwin.data.forecast_contract import FORECAST_PROTOCOL_V2_NONOVERLAP
from neurotwin.repro import hash_file, stable_hash, write_json


ELIGIBILITY_SCHEMA = "neurotwin.forecast_eligibility.v2"
ELIGIBILITY_EVIDENCE_SCHEMA = 2
FILESYSTEM_BUILDER = "neurotwin.forecast_eligibility.filesystem.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ForecastEligibilityDecision:
    claim_eligible: bool
    protocol_id: str | None
    violations: tuple[str, ...]
    checked: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForecastEligibilitySources:
    protocol: str | Path
    source_manifest: str | Path
    transform_lineage: str | Path
    split_audit: str | Path
    firebreak_audit: str | Path
    invalidated_result_registry: str | Path
    result_dependency_ids: tuple[str, ...] = ()


def build_forecast_eligibility_from_sources(
    sources: ForecastEligibilitySources,
) -> dict[str, Any]:
    """Read source artifacts and derive every pass/fail fact from their contents."""

    paths = {
        "protocol": Path(sources.protocol),
        "source_manifest": Path(sources.source_manifest),
        "transform_lineage": Path(sources.transform_lineage),
        "split_audit": Path(sources.split_audit),
        "firebreak_audit": Path(sources.firebreak_audit),
        "invalidated_result_registry": Path(sources.invalidated_result_registry),
    }
    payloads = {name: _read_json_object(path) for name, path in paths.items()}
    source_hashes, source_violations = _verify_source_manifest(
        paths["source_manifest"], payloads["source_manifest"]
    )
    bindings = [
        {
            "artifact": name,
            "file_sha256": hash_file(path),
            "payload_sha256": stable_hash(payloads[name]),
        }
        for name, path in sorted(paths.items())
    ]
    evidence = {
        "builder": FILESYSTEM_BUILDER,
        "artifact_bindings": bindings,
        "artifact_payloads": payloads,
        "source_hashes": source_hashes,
        "source_verification_violations": source_violations,
        "result_dependency_ids": sorted(
            {str(value) for value in sources.result_dependency_ids}
        ),
    }
    return _artifact_from_evidence(evidence)


def build_forecast_eligibility_artifact(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Build an explicitly ineligible legacy artifact from unbound caller data.

    Claim-bearing callers must use :func:`build_forecast_eligibility_from_sources`.
    Keeping this function makes old tooling fail closed instead of silently
    treating caller-supplied booleans as evidence.
    """

    return _artifact_from_evidence(
        {"builder": "unbound_legacy_mapping", "legacy_payload": _json_object(evidence)}
    )


def derive_forecast_eligibility(
    evidence: Mapping[str, Any] | None,
) -> ForecastEligibilityDecision:
    checked = (
        "artifact_bindings",
        "protocol",
        "source_hashes",
        "transform_lineage",
        "split_audit",
        "firebreak_audit",
        "invalidated_result_registry",
    )
    violations: list[str] = []
    if not isinstance(evidence, Mapping):
        return ForecastEligibilityDecision(
            False, None, ("forecast eligibility evidence is missing",), checked
        )
    if evidence.get("builder") != FILESYSTEM_BUILDER:
        return ForecastEligibilityDecision(
            False,
            None,
            ("forecast eligibility was not derived from bound filesystem artifacts",),
            checked,
        )

    payloads = evidence.get("artifact_payloads")
    bindings = evidence.get("artifact_bindings")
    if not isinstance(payloads, Mapping) or not _bindings_match_payloads(
        bindings, payloads
    ):
        violations.append(
            "forecast eligibility artifact bindings are missing or inconsistent"
        )
        payloads = {}

    protocol = payloads.get("protocol")
    protocol_id = protocol.get("protocol_id") if isinstance(protocol, Mapping) else None
    schema_version = (
        protocol.get("schema_version") if isinstance(protocol, Mapping) else None
    )
    if protocol_id != FORECAST_PROTOCOL_V2_NONOVERLAP:
        violations.append(
            f"forecast protocol is missing, unknown, or ineligible: {protocol_id!r}"
        )
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version < 2
    ):
        violations.append("forecast protocol schema_version must be an integer >= 2")

    source_hashes = evidence.get("source_hashes")
    if (
        not isinstance(source_hashes, list)
        or not source_hashes
        or not all(_valid_sha256(value) for value in source_hashes)
    ):
        violations.append("verified raw source SHA-256 values are missing or malformed")
    source_violations = evidence.get("source_verification_violations")
    if not isinstance(source_violations, list) or source_violations:
        violations.append("raw source hash verification did not pass")

    lineage = payloads.get("transform_lineage")
    if not _lineage_complete(lineage):
        violations.append("transform lineage is missing or incomplete")

    split_audit = payloads.get("split_audit")
    split_violations = _split_audit_violations(split_audit)
    violations.extend(split_violations)

    firebreak = payloads.get("firebreak_audit")
    firebreak_violations = _firebreak_violations(firebreak)
    violations.extend(firebreak_violations)

    registry = payloads.get("invalidated_result_registry")
    dependencies = evidence.get("result_dependency_ids")
    invalid_dependencies = _invalid_dependencies(registry, dependencies)
    if invalid_dependencies is None:
        violations.append("invalidated-result registry or dependency list is malformed")
    elif invalid_dependencies:
        violations.append(
            "forecast evidence depends on invalidated historical results: "
            + ",".join(invalid_dependencies)
        )

    return ForecastEligibilityDecision(
        not violations,
        str(protocol_id) if protocol_id is not None else None,
        tuple(violations),
        checked,
    )


def validate_forecast_eligibility_artifact(
    payload: Mapping[str, Any] | None,
) -> ForecastEligibilityDecision:
    if not isinstance(payload, Mapping) or payload.get("schema") != ELIGIBILITY_SCHEMA:
        return derive_forecast_eligibility(None)
    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping):
        return derive_forecast_eligibility(None)
    derived = derive_forecast_eligibility(evidence)
    violations = list(derived.violations)
    if payload.get("evidence_schema_version") != ELIGIBILITY_EVIDENCE_SCHEMA:
        violations.append("forecast eligibility evidence schema is unsupported")
    if payload.get("evidence_sha256") != stable_hash(evidence):
        violations.append("forecast eligibility evidence hash mismatch")
    if payload.get("decision") != _json_object(derived.to_dict()):
        violations.append(
            "persisted forecast eligibility decision does not match re-derived decision"
        )
    return ForecastEligibilityDecision(
        not violations and derived.claim_eligible,
        derived.protocol_id,
        tuple(violations),
        derived.checked,
    )


def write_forecast_eligibility_artifact(
    path: str | Path,
    sources: ForecastEligibilitySources | Mapping[str, Any],
) -> dict[str, Any]:
    artifact = (
        build_forecast_eligibility_from_sources(sources)
        if isinstance(sources, ForecastEligibilitySources)
        else build_forecast_eligibility_artifact(sources)
    )
    write_json(path, artifact)
    return artifact


def forecast_eligibility_sha256(payload: Mapping[str, Any] | None) -> str | None:
    decision = validate_forecast_eligibility_artifact(payload)
    if not decision.claim_eligible or not isinstance(payload, Mapping):
        return None
    value = payload.get("evidence_sha256")
    return str(value) if _valid_sha256(value) else None


def _artifact_from_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _json_object(evidence)
    decision = derive_forecast_eligibility(normalized)
    return {
        "schema": ELIGIBILITY_SCHEMA,
        "evidence_schema_version": ELIGIBILITY_EVIDENCE_SCHEMA,
        "evidence_sha256": stable_hash(normalized),
        "evidence": normalized,
        "decision": _json_object(decision.to_dict()),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"required eligibility artifact does not exist: {path}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"required eligibility artifact is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"required eligibility artifact must contain a JSON object: {path}"
        )
    return payload


def _verify_source_manifest(
    path: Path, payload: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    rows = payload.get("files")
    if not isinstance(rows, list) or not rows:
        return [], ["source manifest has no files"]
    hashes: list[str] = []
    violations: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            violations.append(f"source manifest row {index} is not an object")
            continue
        source = row.get("path")
        expected = row.get("sha256")
        if not isinstance(source, str) or not source or not _valid_sha256(expected):
            violations.append(
                f"source manifest row {index} has invalid path or SHA-256"
            )
            continue
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        if not source_path.is_file():
            violations.append(f"source file is missing: {source}")
            continue
        observed = hash_file(source_path)
        if observed != expected:
            violations.append(f"source SHA-256 mismatch: {source}")
            continue
        hashes.append(observed)
    return sorted(set(hashes)), violations


def _bindings_match_payloads(bindings: Any, payloads: Mapping[str, Any]) -> bool:
    if not isinstance(bindings, list) or not bindings:
        return False
    by_name = {row.get("artifact"): row for row in bindings if isinstance(row, Mapping)}
    required = {
        "protocol",
        "source_manifest",
        "transform_lineage",
        "split_audit",
        "firebreak_audit",
        "invalidated_result_registry",
    }
    if set(payloads) != required or set(by_name) != required:
        return False
    return all(
        _valid_sha256(by_name[name].get("file_sha256"))
        and by_name[name].get("payload_sha256") == stable_hash(payloads[name])
        for name in required
    )


def _lineage_complete(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        return False
    for step in steps:
        if not isinstance(step, Mapping):
            return False
        inputs = step.get("input_hashes")
        outputs = step.get("output_hashes")
        if not _hash_sequence(inputs) or not _hash_sequence(outputs):
            return False
    return True


def _split_audit_violations(payload: Any) -> list[str]:
    if not isinstance(payload, Mapping):
        return ["split audit is missing"]
    violations = payload.get("violations")
    if violations != []:
        return [
            "split audit contains violations or has no explicit empty violation list"
        ]
    failures: list[str] = []
    for field in (
        "subject_overlap_count",
        "recording_overlap_count",
        "session_overlap_count",
    ):
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            failures.append(f"split audit {field} must equal zero")
    return failures


def _firebreak_violations(payload: Any) -> list[str]:
    if not isinstance(payload, Mapping):
        return ["forecast firebreak audit is missing"]
    if payload.get("violations") != []:
        return [
            "forecast firebreak audit contains violations or has no explicit empty violation list"
        ]
    if payload.get("target_overlaps_context") is not False:
        return ["forecast target overlaps context or overlap status is unknown"]
    return []


def _invalid_dependencies(registry: Any, dependencies: Any) -> list[str] | None:
    if not isinstance(registry, Mapping) or not isinstance(dependencies, list):
        return None
    rows = registry.get("results")
    if not isinstance(rows, list):
        return None
    invalid_ids = {
        str(row.get("result_id"))
        for row in rows
        if isinstance(row, Mapping)
        and row.get("status")
        in {"invalid_experiment", "contradicted_by_evidence", "quarantined"}
        and row.get("result_id") is not None
    }
    return sorted(invalid_ids & {str(value) for value in dependencies})


def _hash_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and bool(value)
        and all(_valid_sha256(item) for item in value)
    )


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _json_object(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))
