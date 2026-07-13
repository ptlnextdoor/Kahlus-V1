"""Sealed CAP registry boundary for the HNPH external evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from neurotwin.forecastability import PhysicalRecordRegistry, PhysicalSignalRecord


CAP_DATASET_ID = "cap-sleep-database"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CapSealError(ValueError):
    """Raised when a CAP external record would bypass its independent test seal."""


@dataclass(frozen=True)
class CapSealedManifest:
    """The minimum custodian-owned metadata needed before CAP records may enter a registry."""

    source_manifest_sha256: str
    custodian_id: str
    evaluator_id: str
    sealed: bool = True
    opened: bool = False

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.source_manifest_sha256.lower()):
            raise ValueError("source_manifest_sha256 must be a 64-character SHA-256 value")
        if not self.custodian_id.strip() or not self.evaluator_id.strip():
            raise ValueError("custodian_id and evaluator_id must be non-empty")
        if self.custodian_id == self.evaluator_id:
            raise ValueError("CAP custodian and evaluator must be independent roles")
        if self.opened and not self.sealed:
            raise ValueError("an opened CAP manifest must retain the fact that it was sealed")


def build_cap_registry(
    records: Iterable[PhysicalSignalRecord],
    *,
    manifest: CapSealedManifest,
) -> PhysicalRecordRegistry:
    """Admit CAP records only after an independent custodian explicitly opens the seal."""

    if not manifest.sealed or not manifest.opened:
        raise CapSealError("CAP external records remain sealed until the custodian opens the frozen evaluation")
    rows = tuple(records)
    if not rows:
        raise CapSealError("opened CAP registry requires at least one physical record")
    foreign = [record.record_id for record in rows if record.dataset_id != CAP_DATASET_ID]
    if foreign:
        raise CapSealError("CAP registry contains records outside the frozen CAP dataset")
    return PhysicalRecordRegistry.from_records(rows)
