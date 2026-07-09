from __future__ import annotations

import unittest

from neurotwin.data.manifest_io import record_from_dict, record_to_dict
from neurotwin.data.split_manifest import RecordingRecord
from neurotwin.forecastability import (
    EvidenceDecision,
    LeadGeometry,
    PhysicalRecordRegistry,
    PhysicalSignalRecord,
    QualityInterval,
    StateTargetSpec,
)


def _record(record_id: str = "sleep-001") -> PhysicalSignalRecord:
    return PhysicalSignalRecord(
        record_id=record_id,
        subject_id="subject-001",
        session_id="night-1",
        dataset_id="sleep_edf_expanded",
        site_id=None,
        modality="eeg",
        sampling_rate_hz=100.0,
        physical_unit="uV",
        duration_s=120.0,
        leads=(
            LeadGeometry(
                lead_id="Fpz-Cz",
                positive_xyz_m=(0.0, 0.1, 0.2),
                negative_xyz_m=(0.0, 0.0, 0.1),
                reference_kind="bipolar",
                position_source="fixture",
            ),
        ),
        quality_intervals=(
            QualityInterval(start_s=0.0, end_s=30.0, state="valid"),
            QualityInterval(start_s=30.0, end_s=40.0, state="artifact", reason="movement"),
            QualityInterval(start_s=40.0, end_s=120.0, state="valid"),
        ),
        raw_source_uri="physionet://sleep-edf/sleep-001.edf",
        source_sha256="a" * 64,
        annotation_uri="physionet://sleep-edf/sleep-001-hypnogram.edf",
    )


class ForecastabilityContractTests(unittest.TestCase):
    def test_physical_record_exposes_only_declared_valid_intervals(self) -> None:
        record = _record()
        self.assertEqual(record.valid_intervals_s, ((0.0, 30.0), (40.0, 120.0)))

    def test_physical_record_rejects_duplicate_leads_and_overlapping_quality_intervals(self) -> None:
        lead = LeadGeometry(
            lead_id="C3",
            positive_xyz_m=None,
            negative_xyz_m=None,
            reference_kind="unknown",
            position_source="unknown",
        )
        with self.assertRaisesRegex(ValueError, "lead IDs"):
            PhysicalSignalRecord(
                **{
                    **_record().__dict__,
                    "leads": (lead, lead),
                }
            )
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            PhysicalSignalRecord(
                **{
                    **_record().__dict__,
                    "quality_intervals": (
                        QualityInterval(start_s=0.0, end_s=20.0, state="valid"),
                        QualityInterval(start_s=10.0, end_s=30.0, state="artifact", reason="fixture"),
                    ),
                }
            )

    def test_state_target_spec_rejects_overlapping_bands(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            StateTargetSpec(
                version="state-v1",
                bands_hz=((1.0, 8.0), (4.0, 12.0)),
                target_window_s=4.0,
                include_aperiodic=True,
                include_spatial_covariance=False,
                include_complex_spectrum=False,
            )

    def test_evidence_decision_rejects_contradictory_claims(self) -> None:
        with self.assertRaisesRegex(ValueError, "both allowed and blocked"):
            EvidenceDecision(
                protocol_version="0.1.0",
                gate_passed=False,
                outcome_class="invalid_experiment",
                failed_requirements=("R3",),
                allowed_claims=("forecasting",),
                blocked_claims=("forecasting",),
            )

    def test_registry_round_trip_preserves_manifest_sidecar(self) -> None:
        registry = PhysicalRecordRegistry.from_records((_record(),))
        restored = PhysicalRecordRegistry.from_dict(registry.to_dict())
        self.assertEqual(restored.by_id("sleep-001"), _record())

        generic = RecordingRecord(
            record_id="sleep-001",
            modality="eeg",
            dataset="sleep_edf_expanded",
            subject_id="subject-001",
            session_id="night-1",
            site_id="unknown_site",
            start_time=0.0,
            end_time=120.0,
            metadata=_record().manifest_metadata(),
        )
        self.assertEqual(record_from_dict(record_to_dict(generic)).metadata, generic.metadata)
        self.assertEqual(generic.metadata["physical_record_schema_version"], "hnph_physical_record_v1")


if __name__ == "__main__":
    unittest.main()
