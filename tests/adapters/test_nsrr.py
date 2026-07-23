from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurotwin.adapters.nsrr import epoch_arousal_mask, parse_nsrr_arousal_events


class NsrrAdapterTests(unittest.TestCase):
    def test_parse_arousal_events_from_xml(self) -> None:
        xml = """<?xml version="1.0"?>
        <PSGAnnotation>
          <ScoredEvents>
            <ScoredEvent>
              <Start>30.0</Start>
              <Duration>5.0</Duration>
              <EventConcept>Arousal ()</EventConcept>
            </ScoredEvent>
            <ScoredEvent>
              <Start>120.0</Start>
              <Duration>3.0</Duration>
              <EventConcept>Stage 2 sleep</EventConcept>
            </ScoredEvent>
          </ScoredEvents>
        </PSGAnnotation>
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test-nsrr.xml"
            path.write_text(xml, encoding="utf-8")
            events = parse_nsrr_arousal_events(path)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0][0], 30.0)
        self.assertAlmostEqual(events[0][1], 5.0)

    def test_epoch_arousal_mask(self) -> None:
        mask = epoch_arousal_mask([(25.0, 10.0)], n_epochs=4, epoch_seconds=30.0)
        self.assertEqual(int(mask[0]), 1)
        self.assertEqual(int(mask[1]), 1)
        self.assertEqual(int(mask[2]), 0)


if __name__ == "__main__":
    unittest.main()
