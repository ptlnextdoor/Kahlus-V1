"""Fail-closed authority for claim-bearing forecasting evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from neurotwin.data.forecast_contract import FORECAST_PROTOCOL_V2_NONOVERLAP


ELIGIBILITY_SCHEMA = "neurotwin.forecast_eligibility.v1"
ELIGIBILITY_EVIDENCE_SCHEMA = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ForecastEligibilityDecision:
    claim_eligible: bool
    protocol_id: str | None
    violations: tuple[str, ...]
    checked: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_forecast_eligibility(evidence: Mapping[str, Any] | None) -> ForecastEligibilityDecision:
    """Derive eligibility from auditable facts; never accept a caller-provided decision."""

    checked = (
        "protocol",
        "source_hashes",
        "transform_lineage",
        "split_audit",
        "firebreak_audit",
        "invalidated_result_registry",
    )
    violations: list[str] = []
    if not isinstance(evidence, Mapping):
        return ForecastEligibilityDecision(False, None, ("forecast eligibility evidence is missing",), checked)

    protocol = evidence.get("protocol")
    protocol_id = protocol.get("protocol_id") if isinstance(protocol, Mapping) else None
    schema_version = protocol.get("schema_version") if isinstance(protocol, Mapping) else None
    if protocol_id != FORECAST_PROTOCOL_V2_NONOVERLAP:
        violations.append(f"forecast protocol is missing, unknown, or ineligible: {protocol_id!r}")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version < 2:
        violations.append("forecast protocol schema_version must be an integer >= 2")

    source_hashes = evidence.get("source_hashes")
    if not isinstance(source_hashes, list) or not source_hashes or not all(_valid_sha256(value) for value in source_hashes):
        violations.append("verified raw source SHA-256 values are missing or malformed")
    if evidence.get("source_hash_verification_passed") is not True:
        violations.append("raw source hash verification did not pass")

    lineage_hash = evidence.get("transform_lineage_hash")
    if not _valid_sha256(lineage_hash):
        violations.append("transform lineage hash is missing or malformed")
    if evidence.get("transform_lineage_complete") is not True:
        violations.append("transform lineage is incomplete")

    split_audit = evidence.get("split_audit")
    if not _audit_passes(split_audit):
        violations.append("split audit is missing or failed")
    else:
        for field in ("subject_overlap_count", "recording_overlap_count", "session_overlap_count"):
            value = split_audit.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value != 0:
                violations.append(f"split audit {field} must equal zero")

    firebreak = evidence.get("firebreak_audit")
    if not _audit_passes(firebreak):
        violations.append("forecast firebreak audit is missing or failed")
    elif firebreak.get("target_overlaps_context") is not False:
        violations.append("forecast target overlaps context or overlap status is unknown")

    invalid_ids = evidence.get("invalidated_result_ids")
    if not isinstance(invalid_ids, list):
        violations.append("invalidated_result_ids must be an explicit list")
    elif invalid_ids:
        violations.append("forecast evidence depends on invalidated historical results")

    return ForecastEligibilityDecision(not violations, str(protocol_id) if protocol_id is not None else None, tuple(violations), checked)


def build_forecast_eligibility_artifact(evidence: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _json_object(evidence)
    decision = derive_forecast_eligibility(normalized)
    return {
        "schema": ELIGIBILITY_SCHEMA,
        "evidence_schema_version": ELIGIBILITY_EVIDENCE_SCHEMA,
        "evidence_sha256": _canonical_sha256(normalized),
        "evidence": normalized,
        "decision": _json_object(decision.to_dict()),
    }


def validate_forecast_eligibility_artifact(payload: Mapping[str, Any] | None) -> ForecastEligibilityDecision:
    """Re-derive and compare a persisted decision so stale or edited artifacts fail."""

    if not isinstance(payload, Mapping) or payload.get("schema") != ELIGIBILITY_SCHEMA:
        return derive_forecast_eligibility(None)
    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping):
        return derive_forecast_eligibility(None)
    derived = derive_forecast_eligibility(evidence)
    violations = list(derived.violations)
    if payload.get("evidence_schema_version") != ELIGIBILITY_EVIDENCE_SCHEMA:
        violations.append("forecast eligibility evidence schema is unsupported")
    if payload.get("evidence_sha256") != _canonical_sha256(evidence):
        violations.append("forecast eligibility evidence hash mismatch")
    if payload.get("decision") != _json_object(derived.to_dict()):
        violations.append("persisted forecast eligibility decision does not match re-derived decision")
    return ForecastEligibilityDecision(not violations and derived.claim_eligible, derived.protocol_id, tuple(violations), derived.checked)


def write_forecast_eligibility_artifact(path: str | Path, evidence: Mapping[str, Any]) -> dict[str, Any]:
    artifact = build_forecast_eligibility_artifact(evidence)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _audit_passes(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("passed") is True and value.get("violations") == []


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_object(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(dict(value), sort_keys=True))
    if not isinstance(normalized, dict):
        raise TypeError("forecast eligibility evidence must serialize to a JSON object")
    return normalized
