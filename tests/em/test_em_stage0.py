import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurotwin.em import (
    EMContext,
    RoomEnvironmentLog,
    RoomEMFLogger,
    build_em_artifact_audit_gate,
    channel_artifact_features,
    compute_psd,
    fetch_geomagnetic,
    run_artifact_audit,
    synthesize_idle_recording,
)


class EMStage0Tests(unittest.TestCase):
    def test_em_context_forbids_human(self):
        EMContext(condition_label="baseline").validate()  # ok
        with self.assertRaises(ValueError):
            EMContext(condition_label="x", involves_human=True).validate()

    def test_psd_and_features_finite_shapes(self):
        signal = synthesize_idle_recording(seed=0, n_channels=6, n_samples=512, fs_hz=128.0)
        freqs, psd = compute_psd(signal, 128.0)
        self.assertEqual(psd.shape[0], 6)
        self.assertEqual(psd.shape[1], freqs.shape[0])
        feats = channel_artifact_features(signal, 128.0, line_freq_hz=60.0)
        for key in ("rms", "broadband_power", "line_noise_ratio", "kurtosis"):
            self.assertEqual(feats[key].shape, (6,))
            self.assertTrue(np.isfinite(feats[key]).all())

    def test_artifact_audit_detects_synthetic_environment_effect(self):
        baseline = synthesize_idle_recording(seed=0, n_channels=8, n_samples=1024, fs_hz=256.0, em_field_strength_arb=0.0)
        perturbed = synthesize_idle_recording(seed=1, n_channels=8, n_samples=1024, fs_hz=256.0, em_field_strength_arb=0.5)
        report = run_artifact_audit(
            baseline,
            perturbed,
            fs_hz=256.0,
            line_freq_hz=60.0,
            baseline_context=EMContext(condition_label="baseline", em_source="none"),
            condition_context=EMContext(condition_label="perturbed", em_source="synthetic_field", field_strength_arb=0.5),
        )
        self.assertEqual(report["branch"], "em")
        self.assertTrue(report["response"]["finite"])
        self.assertTrue(report["response"]["environment_effect_detected"])

    def test_em_gate_blocks_claim(self):
        baseline = synthesize_idle_recording(seed=0, n_channels=4, n_samples=512, fs_hz=128.0)
        perturbed = synthesize_idle_recording(seed=1, n_channels=4, n_samples=512, fs_hz=128.0, em_field_strength_arb=0.3)
        report = run_artifact_audit(
            baseline, perturbed, fs_hz=128.0, line_freq_hz=60.0,
            baseline_context=EMContext(condition_label="baseline"),
            condition_context=EMContext(condition_label="perturbed", em_source="synthetic_field", field_strength_arb=0.3),
        )
        gate = build_em_artifact_audit_gate(report)
        self.assertEqual(gate["branch"], "em")
        self.assertFalse(gate["scientific_claim_allowed"])

    def test_geomagnetic_offline_fallback(self):
        result = fetch_geomagnetic(None)
        self.assertEqual(result["status"], "not_fetched")
        self.assertFalse(result["network_access"])
        self.assertEqual(result["records"], [])

    def test_geomagnetic_missing_local_file(self):
        result = fetch_geomagnetic("/nonexistent/path/geomag.json")
        self.assertEqual(result["status"], "not_fetched")
        self.assertFalse(result["network_access"])

    def test_room_logger_append_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = RoomEMFLogger(Path(tmp) / "log.jsonl")
            logger.log(RoomEnvironmentLog(timestamp="t0", room_id="lab", device_id="amp", line_noise_uv=1.2))
            logger.log(RoomEnvironmentLog(timestamp="t1", room_id="lab", device_id="amp", line_noise_uv=1.3))
            rows = logger.read_all()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["room_id"], "lab")


if __name__ == "__main__":
    unittest.main()
