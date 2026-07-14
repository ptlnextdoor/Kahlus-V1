from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neurotwin.adapters.edf_common import (
    EdfReadError,
    build_edf_data_card,
    build_physical_record_from_edf,
    normalize_edf_physical_unit,
    read_edf_header,
    write_edf_data_card,
)
from neurotwin.forecastability import QualityInterval

try:
    import edfio
except ImportError:  # pragma: no cover - exercised by environments without the optional EDF extra.
    edfio = None


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


class EdfCommonAdapterTests(unittest.TestCase):
    def _header(self, edf: _Edf):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.edf"
            path.write_bytes(b"header-only-fixture")
            return read_edf_header(path, reader=lambda _path: edf)

    def test_normalizes_only_known_voltage_units(self) -> None:
        self.assertEqual(normalize_edf_physical_unit("µV"), "uV")
        self.assertEqual(normalize_edf_physical_unit(" mV "), "mV")
        self.assertIsNone(normalize_edf_physical_unit("degreeC"))

    def test_header_and_physical_record_keep_local_paths_out_of_data_card(self) -> None:
        header = self._header(
            _Edf(
                duration=60.0,
                signals=(
                    _Signal("EEG Fpz-Cz", 100.0, "µV"),
                    _Signal("EEG Pz-Oz", 100.0, "uV"),
                ),
                annotations=(_Annotation(0.0, 30.0, "Sleep stage W"),),
            )
        )
        record = build_physical_record_from_edf(
            header,
            record_id="SC4001E0",
            subject_id="SC4001",
            session_id="E0",
            dataset_id="sleep_edf_expanded",
            site_id=None,
            raw_source_uri="physionet://sleep-edfx/sleep-cassette/SC4001E0-PSG.edf",
            annotation_uri="physionet://sleep-edfx/sleep-cassette/SC4001EC-Hypnogram.edf",
            quality_intervals=(QualityInterval(0.0, 60.0, "valid"),),
        )
        card = build_edf_data_card(header, record)
        self.assertEqual(card["physical_unit"], "uV")
        self.assertEqual(card["annotation_count"], 1)
        self.assertFalse(card["local_source_path_recorded"])
        self.assertNotIn("fixture.edf", json.dumps(card))
        with tempfile.TemporaryDirectory() as tmp:
            out = write_edf_data_card(Path(tmp) / "card.json", header, record)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8")), card)

    def test_rejects_ambiguous_units_and_mixed_sampling_rates(self) -> None:
        mixed_units = self._header(
            _Edf(
                duration=60.0,
                signals=(_Signal("EEG 1", 100.0, "uV"), _Signal("ECG", 100.0, "mV")),
                annotations=(),
            )
        )
        with self.assertRaisesRegex(EdfReadError, "shared voltage unit"):
            build_physical_record_from_edf(
                mixed_units,
                record_id="rec",
                subject_id="sub",
                session_id="ses",
                dataset_id="fixture",
                site_id=None,
                raw_source_uri="fixture://rec.edf",
                annotation_uri=None,
                quality_intervals=(QualityInterval(0.0, 60.0, "valid"),),
            )

    def test_redacts_local_source_uris_from_data_cards(self) -> None:
        header = self._header(
            _Edf(
                duration=60.0,
                signals=(_Signal("EEG Fpz-Cz", 100.0, "uV"),),
                annotations=(),
            )
        )
        record = build_physical_record_from_edf(
            header,
            record_id="rec",
            subject_id="sub",
            session_id="ses",
            dataset_id="fixture",
            site_id=None,
            raw_source_uri="file:///private/raw/rec.edf",
            annotation_uri="/private/raw/rec-hypnogram.edf",
            quality_intervals=(QualityInterval(0.0, 60.0, "valid"),),
        )

        card = build_edf_data_card(header, record)
        self.assertIsNone(card["raw_source_uri"])
        self.assertIsNone(card["annotation_uri"])
        self.assertNotIn("/private/raw", json.dumps(card))

    @unittest.skipIf(edfio is None, "edfio is required for the real EDF reader fixture")
    def test_default_reader_extracts_real_edf_header_without_loading_signal_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.edf"
            signal = edfio.EdfSignal(
                np.linspace(-50.0, 50.0, 100, dtype=np.float64),
                100.0,
                label="EEG Fpz-Cz",
                physical_dimension="uV",
                physical_range=(-100.0, 100.0),
            )
            edfio.Edf(
                (signal,),
                annotations=(edfio.EdfAnnotation(0.0, 1.0, "fixture annotation"),),
            ).write(path)
            header = read_edf_header(path)

        self.assertEqual(header.signals[0].label, "EEG Fpz-Cz")
        self.assertEqual(header.signals[0].canonical_physical_unit, "uV")
        self.assertEqual(header.annotations[0].text, "fixture annotation")
        mixed_rates = self._header(
            _Edf(
                duration=60.0,
                signals=(_Signal("EEG 1", 100.0, "uV"), _Signal("EEG 2", 128.0, "uV")),
                annotations=(),
            )
        )
        with self.assertRaisesRegex(EdfReadError, "shared sampling rate"):
            build_physical_record_from_edf(
                mixed_rates,
                record_id="rec",
                subject_id="sub",
                session_id="ses",
                dataset_id="fixture",
                site_id=None,
                raw_source_uri="fixture://rec.edf",
                annotation_uri=None,
                quality_intervals=(QualityInterval(0.0, 60.0, "valid"),),
            )


if __name__ == "__main__":
    unittest.main()
