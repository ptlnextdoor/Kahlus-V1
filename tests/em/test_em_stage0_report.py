import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurotwin.em import (
    EMContext,
    artifact_severity_summary,
    build_stage0_report,
    contamination_map,
    format_stage0_report_md,
    run_artifact_audit,
    synthesize_idle_recording,
    write_stage0_report,
)
from neurotwin.em.artifact_severity import EEG_BANDS
from neurotwin.gates import evaluate_gate


def _audit(seed=0, n_channels=8, n_samples=1024, fs_hz=256.0, line_freq=60.0, strength=0.5):
    baseline = synthesize_idle_recording(
        seed=seed, n_channels=n_channels, n_samples=n_samples, fs_hz=fs_hz, em_field_strength_arb=0.0
    )
    perturbed = synthesize_idle_recording(
        seed=seed + 1, n_channels=n_channels, n_samples=n_samples, fs_hz=fs_hz, em_field_strength_arb=strength
    )
    report = run_artifact_audit(
        baseline,
        perturbed,
        fs_hz=fs_hz,
        line_freq_hz=line_freq,
        baseline_context=EMContext(condition_label="baseline", em_source="none"),
        condition_context=EMContext(condition_label="perturbed_environment", em_source="synthetic_field", field_strength_arb=strength),
    )
    return report, {"baseline": baseline, "perturbed_environment": perturbed}, fs_hz, line_freq


def _bundle(seed=0):
    report, conditions, fs_hz, line_freq = _audit(seed=seed)
    return build_stage0_report(
        audit_report=report, conditions=conditions, fs_hz=fs_hz, line_freq_hz=line_freq, seed=seed
    )


class EMStage0ReportTests(unittest.TestCase):
    def test_severity_finite_and_deterministic(self):
        report, conditions, fs_hz, line_freq = _audit(seed=3)
        cmap = contamination_map(conditions, fs_hz=fs_hz, line_freq_hz=line_freq)
        kwargs = dict(fs_hz=fs_hz, line_freq_hz=line_freq, cmap=cmap, condition_label="perturbed_environment")
        s1 = artifact_severity_summary(conditions["baseline"], conditions["perturbed_environment"], **kwargs)
        s2 = artifact_severity_summary(conditions["baseline"], conditions["perturbed_environment"], **kwargs)
        self.assertTrue(s1["finite"])
        self.assertTrue(np.isfinite(s1["overall_artifact_severity"]))
        self.assertGreaterEqual(s1["overall_artifact_severity"], 0.0)
        self.assertLess(s1["overall_artifact_severity"], 1.0)
        self.assertEqual(s1["overall_artifact_severity"], s2["overall_artifact_severity"])
        self.assertIn(s1["verdict"], {"pass", "warn", "fail"})

    def test_contamination_map_shape_correct(self):
        report, conditions, fs_hz, line_freq = _audit(seed=1, n_channels=6)
        cmap = contamination_map(conditions, fs_hz=fs_hz, line_freq_hz=line_freq)
        self.assertEqual(cmap["n_channels"], 6)
        self.assertEqual(cmap["bands"], list(EEG_BANDS.keys()) + ["line"])
        for band in cmap["bands"]:
            self.assertEqual(len(cmap["conditions"]["baseline"]["band_power"][band]), 6)
            self.assertEqual(len(cmap["conditions"]["perturbed_environment"]["band_power"][band]), 6)
            self.assertIn(band, cmap["delta_vs_baseline"]["perturbed_environment"]["band_power_mean_abs_delta"])
        self.assertIn("no_brain", cmap["attribution"])

    def test_markdown_report_written_with_safety_red_lines(self):
        bundle = _bundle(seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_stage0_report(tmp, bundle)
            md_path = Path(paths["stage0_report_md"])
            self.assertEqual(md_path.name, "stage0_artifact_report.md")
            self.assertTrue(md_path.exists())
            text = md_path.read_text(encoding="utf-8")
        for red_line in (
            "No human subject",
            "No stimulation",
            "No high voltage",
            "No God Helmet",
            "No consciousness claim",
            "No causal EM claim",
            "No predictive human EEG claim",
        ):
            self.assertIn(red_line, text)
        for section in (
            "## Run metadata",
            "## Device / environment log summary",
            "## PSD / channel artifact summary",
            "## Contamination map",
            "## Severity score",
            "## Gate result",
            "## Pass / fail recommendation",
            "## Safety disclaimer",
        ):
            self.assertIn(section, text)

    def test_narrow_scope_can_pass_broad_blocked(self):
        bundle = _bundle(seed=0)
        gate = bundle["gate"]
        self.assertEqual(gate["schema"], "kahlus.unified_evidence_gate.v1")
        self.assertEqual(gate["claim_scope"], "em_stage0_artifact_audit")
        self.assertTrue(gate["scientific_claim_allowed"])
        # Broad EM / consciousness scopes stay blocked even with otherwise-passing inputs.
        for broad in ("em_affects_consciousness", "human_eeg_modulation", "causal_em_effect"):
            broad_gate = evaluate_gate(
                branch="em",
                dataset="em_stage0_artifact_audit",
                split_audit_passed=True,
                baseline_table_present=True,
                finite_metrics=True,
                calibration_checked=True,
                claim_scope=broad,
            )
            self.assertFalse(broad_gate["scientific_claim_allowed"])
            self.assertTrue(any("too broad" in r for r in broad_gate["failure_reasons"]))

    def test_report_uses_shared_falsification_core(self):
        bundle = _bundle(seed=2)
        report = bundle["report"]
        # Shared-core report contract: diagnostics list, falsification flag, embedded gate.
        self.assertEqual(report["schema"], "kahlus.em_stage0_artifact_report.v1")
        self.assertIn("diagnostics", report)
        self.assertIn("falsification_passed", report)
        self.assertEqual(report["falsification_passed"], report["scientific_claim_allowed"])
        self.assertEqual(report["evidence_gate"], bundle["gate"])
        names = {d["name"] for d in report["diagnostics"]}
        self.assertEqual(
            names,
            {"severity_finite", "contamination_map_well_formed", "audit_finite", "no_human_attribution"},
        )

    def test_human_involvement_still_forbidden(self):
        with self.assertRaises(ValueError):
            EMContext(condition_label="x", involves_human=True).validate()

    def test_gate_json_round_trips_to_dossier_schema(self):
        bundle = _bundle(seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_stage0_report(tmp, bundle)
            gate = json.loads(Path(paths["evidence_gate"]).read_text(encoding="utf-8"))
        for field_name in (
            "schema",
            "branch",
            "dataset",
            "split_audit_passed",
            "baseline_table_present",
            "finite_metrics",
            "calibration_checked",
            "claim_scope",
            "scientific_claim_allowed",
            "failure_reasons",
        ):
            self.assertIn(field_name, gate)
        self.assertEqual(gate["branch"], "em")


if __name__ == "__main__":
    unittest.main()
