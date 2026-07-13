from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest

from neurotwin.adapters.sleep_edf import (
    SleepEdfAdapterError,
    SleepEdfPair,
    build_sleep_edf_person_split,
    build_sleep_edf_registry,
)
from neurotwin.forecastability import (
    CausalPreprocessingSpec,
    TransitionLeadBand,
    TransitionTargetSpec,
    audit_forecast_firebreak,
    build_natural_transition_targets,
)


@dataclass(frozen=True)
class _Signal:
    label: str
    sampling_frequency: float
    physical_dimension: str


@dataclass(frozen=True)
class _Annotation:
    onset: float
    duration: float | None
    text: str


@dataclass(frozen=True)
class _Edf:
    duration: float
    signals: tuple[_Signal, ...]
    annotations: tuple[_Annotation, ...]


def _reader(path: Path) -> _Edf:
    if path.name.endswith("-PSG.edf"):
        return _Edf(
            duration=120.0,
            signals=(
                _Signal("EEG Fpz-Cz", 100.0, "uV"),
                _Signal("EEG Pz-Oz", 100.0, "uV"),
            ),
            annotations=(),
        )
    return _Edf(
        duration=120.0,
        signals=(),
        annotations=(
            _Annotation(0.0, 30.0, "Sleep stage W"),
            _Annotation(30.0, 60.0, "Sleep stage 2"),
            _Annotation(90.0, 30.0, "Sleep stage R"),
        ),
    )


def _pair(root: Path, record_id: str, subject_id: str, session_id: str) -> SleepEdfPair:
    psg = root / f"{record_id}-PSG.edf"
    hypnogram = root / f"{record_id}C-Hypnogram.edf"
    psg.write_bytes(b"psg-" + record_id.encode())
    hypnogram.write_bytes(b"hyp-" + record_id.encode())
    return SleepEdfPair(
        record_id=record_id,
        subject_id=subject_id,
        session_id=session_id,
        psg_path=psg,
        hypnogram_path=hypnogram,
        raw_source_uri=f"physionet://sleep-edf/sleep-cassette/{psg.name}",
        annotation_uri=f"physionet://sleep-edf/sleep-cassette/{hypnogram.name}",
    )


class SleepEdfAdapterTests(unittest.TestCase):
    def test_builds_registry_rk_epochs_redacted_cards_and_person_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = build_sleep_edf_registry(
                (
                    _pair(root, "SC4001E0", "SC4001", "E0"),
                    _pair(root, "SC4001E1", "SC4001", "E1"),
                    _pair(root, "SC4010E0", "SC4010", "E0"),
                ),
                reader=_reader,
            )
            split = build_sleep_edf_person_split(bundle, seed=4)

            self.assertEqual(len(bundle.registry.records), 3)
            self.assertEqual([epoch.macrostate for epoch in bundle.stage_epochs[:4]], ["Wake", "NREM", "NREM", "REM"])
            self.assertTrue(all(card["annotation_sha256"] for card in bundle.data_cards))
            self.assertNotIn(str(root), json.dumps(bundle.data_cards))
            split_subjects = [
                {record.subject_id for record in group}
                for group in (split.train, split.val, split.test)
            ]
            self.assertEqual(sum("SC4001" in group for group in split_subjects), 1)

    def test_rejects_off_grid_annotation_instead_of_silently_rounding(self) -> None:
        def off_grid_reader(path: Path) -> _Edf:
            if path.name.endswith("-PSG.edf"):
                return _reader(path)
            return _Edf(120.0, (), (_Annotation(1.0, 30.0, "Sleep stage W"),))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(SleepEdfAdapterError, "off the natural epoch grid"):
                build_sleep_edf_registry((_pair(root, "SC4001E0", "SC4001", "E0"),), reader=off_grid_reader)

    def test_tiny_edf_path_reaches_audited_natural_transition_anchors(self) -> None:
        def long_reader(path: Path) -> _Edf:
            if path.name.endswith("-PSG.edf"):
                return _Edf(1200.0, (_Signal("EEG Fpz-Cz", 100.0, "uV"), _Signal("EEG Pz-Oz", 100.0, "uV")), ())
            return _Edf(
                1200.0,
                (),
                (_Annotation(0.0, 750.0, "Sleep stage 2"), _Annotation(750.0, 450.0, "Sleep stage R")),
            )

        spec = TransitionTargetSpec(
            version="hnph-v0.2",
            cadence_s=30.0,
            context_s=600.0,
            stable_destination_epochs=2,
            lead_bands=(TransitionLeadBand("B2", 120.0, 300.0),),
            primary_band_id="B2",
        )
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_sleep_edf_registry((_pair(Path(tmp), "SC4001E0", "SC4001", "E0"),), reader=long_reader)
            targets = build_natural_transition_targets(bundle.stage_epochs, spec)
            report = audit_forecast_firebreak(
                (target.anchor for target in targets),
                bundle.registry,
                CausalPreprocessingSpec("context_only", "none", 0.0),
            )

        self.assertTrue(report.passed, report.violations)
        self.assertEqual(next(target for target in targets if target.target_id == "SC4001E0:000019:B2").outcome, "REM")


if __name__ == "__main__":
    unittest.main()
