"""Schemas for Kahlus-EM Stage 0 (no-human artifact audit).

SAFETY: Stage 0 is metadata + passive-logging scaffolding only. No stimulation, no high
voltage, no plasma/coils, no human protocol, no clinical claim. All "EM context" fields use
arbitrary synthetic units and describe environment/device conditions, never a delivered dose.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class IdleRecordingMetadata:
    """Metadata for an idle (no-task) EEG recording. Subject is a phantom/dummy by default."""

    recording_id: str
    fs_hz: float
    n_channels: int
    duration_s: float
    device: str = "unspecified"
    montage: str = "unspecified"
    modality: str = "eeg"
    subject: str = "none_phantom"
    is_phantom: bool = True
    notes: str = ""

    def validate(self) -> "IdleRecordingMetadata":
        if self.fs_hz <= 0:
            raise ValueError("fs_hz must be positive")
        if int(self.n_channels) < 1:
            raise ValueError("n_channels must be >= 1")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhantomRecordingSchema:
    """Schema for a dummy/phantom recording (no brain present)."""

    recording_id: str
    fs_hz: float
    n_channels: int
    duration_s: float
    phantom_type: str = "resistor_phantom"  # e.g. resistor_phantom, saline, dummy_head
    device: str = "unspecified"
    notes: str = ""

    def validate(self) -> "PhantomRecordingSchema":
        if self.fs_hz <= 0 or self.duration_s <= 0 or int(self.n_channels) < 1:
            raise ValueError("invalid phantom recording schema dimensions")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoomEnvironmentLog:
    """A single room/device/environment passive-logging entry."""

    timestamp: str
    room_id: str
    device_id: str
    temperature_c: float | None = None
    humidity_pct: float | None = None
    mains_freq_hz: float | None = None
    line_noise_uv: float | None = None
    nearby_devices: list[str] = field(default_factory=list)
    geomagnetic_index: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EMContext:
    """Environment/device condition label for an audit comparison.

    ``field_strength_arb`` is an ARBITRARY synthetic magnitude for differentiating audit
    conditions in simulation; it is not a physical dose and is never delivered to a human.
    """

    condition_label: str  # e.g. "baseline" or "perturbed_environment"
    description: str = ""
    em_source: str = "none"  # "none" | "synthetic_field"
    field_strength_arb: float = 0.0
    involves_human: bool = False

    def validate(self) -> "EMContext":
        if self.involves_human:
            raise ValueError("Kahlus-EM Stage 0 forbids human involvement")
        if self.em_source not in {"none", "synthetic_field"}:
            raise ValueError("em_source must be 'none' or 'synthetic_field' in Stage 0")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
