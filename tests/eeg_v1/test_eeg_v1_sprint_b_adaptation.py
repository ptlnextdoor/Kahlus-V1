import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurotwin.eeg_v1 import (
    EEG_V1_ADAPTATION_CLAIM_SCOPE,
    audit_eeg_v1_split,
    build_eeg_v1_adaptation_gate,
    build_fewshot_adaptation_task,
    make_synthetic_eeg_v1_dataset,
    run_fewshot_adaptation,
    write_fewshot_adaptation_artifacts,
)


class EEGV1SprintBAdaptationTests(unittest.TestCase):
    def test_fewshot_task_has_support_and_query_for_heldout_subjects(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=11, n_subjects=10, sessions_per_subject=2)
        task = build_fewshot_adaptation_task(dataset, window_length=8, forecast_horizon=1, support_windows=5)

        self.assertGreater(task.pretrain_x.shape[0], 0)
        self.assertGreater(len(task.subjects), 0)
        for subject, split in task.subject_splits.items():
            with self.subTest(subject=subject):
                self.assertEqual(split.support_x.shape[0], 5)
                self.assertGreater(split.query_x.shape[0], 0)
                self.assertEqual(split.support_x.shape[1:], split.query_x.shape[1:])

    def test_adaptation_runner_reports_baselines_before_adapter_methods(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=4, n_subjects=9, sessions_per_subject=2)
        task = build_fewshot_adaptation_task(dataset, window_length=8, forecast_horizon=1, support_windows=4)
        result = run_fewshot_adaptation(task, seed=0, pretrain_steps=2, adapt_steps=2)

        methods = [row["method"] for row in result["adaptation_table"]]
        self.assertEqual(methods[:2], ["support_persistence", "support_ridge"])
        for required in ("linear_probe", "bottleneck_adapter", "full_finetune"):
            self.assertIn(required, methods)
        for metrics in result["metrics_by_method"].values():
            for value in metrics.values():
                self.assertTrue(np.isfinite(float(value)))

    def test_adaptation_report_lists_gate_failures_when_blocked(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=5, n_subjects=9, sessions_per_subject=2)
        task = build_fewshot_adaptation_task(dataset, window_length=8, forecast_horizon=1, support_windows=4)
        result = run_fewshot_adaptation(task, seed=0, pretrain_steps=2, adapt_steps=2)
        split_audit = dict(audit_eeg_v1_split(dataset, split_type="subject_held_out"))
        split_audit["leakage_passed"] = False
        split_audit["failure_reasons"] = ["forced split audit failure for report coverage"]

        with tempfile.TemporaryDirectory() as tmp:
            write_fewshot_adaptation_artifacts(tmp, task=task, result=result, split_audit=split_audit)
            gate = json.loads((Path(tmp) / "adaptation_evidence_gate.json").read_text(encoding="utf-8"))
            report = (Path(tmp) / "adaptation_report.md").read_text(encoding="utf-8")

        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertIn("split audit did not pass", gate["failure_reasons"])
        self.assertIn("## Evidence Gate Failures", report)
        self.assertIn("- split audit did not pass", report)
        self.assertIn("## Split Audit Failures", report)
        self.assertIn("- forced split audit failure for report coverage", report)

    def test_adaptation_gate_allows_only_narrow_scope(self):
        allowed = build_eeg_v1_adaptation_gate(
            dataset="synthetic_eeg_v1",
            split_audit_passed=True,
            finite_metrics=True,
            baseline_table_present=True,
            support_windows=4,
            query_windows=20,
        )
        self.assertTrue(allowed["scientific_claim_allowed"], allowed["failure_reasons"])
        self.assertEqual(allowed["claim_scope"], EEG_V1_ADAPTATION_CLAIM_SCOPE)
        self.assertIn("gate_criteria", allowed)
        self.assertEqual(
            allowed["gate_criteria"],
            {
                "min_support_windows": 1,
                "min_query_windows": 1,
                "requires_split_audit_passed": True,
                "requires_baseline_table_present": True,
                "requires_finite_metrics": True,
                "requires_calibration_checked": True,
                "allowed_claim_scope": EEG_V1_ADAPTATION_CLAIM_SCOPE,
            },
        )

        blocked = build_eeg_v1_adaptation_gate(
            dataset="synthetic_eeg_v1",
            split_audit_passed=True,
            finite_metrics=True,
            baseline_table_present=True,
            support_windows=4,
            query_windows=20,
            claim_scope="adapter_beats_all_subjects",
        )
        self.assertFalse(blocked["scientific_claim_allowed"])

    def test_adaptation_script_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_adaptation.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--pretrain-steps",
                    "2",
                    "--adapt-steps",
                    "2",
                    "--support-windows",
                    "4",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("claim_scope=eeg_fewshot_adaptation_benchmark_ready", result.stdout)
            self.assertIn("target_units=normalized_eeg_fixture_units", result.stdout)
            self.assertIn("target_std=", result.stdout)
            self.assertIn("target_variance=", result.stdout)
            self.assertIn("best_method_rmse_relative_to_target_std=", result.stdout)
            self.assertIn("best_support_baseline=", result.stdout)
            self.assertIn("best_adaptation_method=", result.stdout)
            self.assertIn("adaptation_vs_best_support_baseline_mse_delta=", result.stdout)
            self.assertIn("adaptation_beats_best_support_baseline=", result.stdout)
            self.assertIn("adaptation_subject_rows=", result.stdout)
            self.assertIn("adaptation_subject_metrics=", result.stdout)
            self.assertIn("subjects_where_adaptation_beats_best_support_baseline=", result.stdout)
            self.assertIn("adaptation_subject_baseline_gap_summary=", result.stdout)
            self.assertIn("adaptation_verification=", result.stdout)
            self.assertIn("adaptation_checksum_manifest=", result.stdout)
            self.assertIn(
                f"checksum_audit_command=PYTHONPATH=src python3 scripts/audit_eeg_v1_adaptation_checksums.py --artifact-dir {tmp}",
                result.stdout,
            )
            expected = {
                "adaptation_metrics.json",
                "adaptation_table.csv",
                "adaptation_subject_metrics.csv",
                "adaptation_evidence_gate.json",
                "adaptation_run_config.json",
                "adaptation_dataset_summary.json",
                "adaptation_split_audit.json",
                "adaptation_failure_reasons.json",
                "adaptation_target_scale_context.json",
                "adaptation_baseline_gap_summary.json",
                "adaptation_subject_baseline_gap_summary.json",
                "adaptation_verification.json",
                "adaptation_checksum_manifest.json",
                "adaptation_report.md",
            }
            self.assertTrue(expected.issubset({p.name for p in Path(tmp).iterdir()}))
            payload = json.loads((Path(tmp) / "adaptation_metrics.json").read_text(encoding="utf-8"))
            gate = json.loads((Path(tmp) / "adaptation_evidence_gate.json").read_text(encoding="utf-8"))
            run_config = json.loads((Path(tmp) / "adaptation_run_config.json").read_text(encoding="utf-8"))
            dataset_summary = json.loads((Path(tmp) / "adaptation_dataset_summary.json").read_text(encoding="utf-8"))
            split_audit = json.loads((Path(tmp) / "adaptation_split_audit.json").read_text(encoding="utf-8"))
            failures = json.loads((Path(tmp) / "adaptation_failure_reasons.json").read_text(encoding="utf-8"))
            target_scale = json.loads((Path(tmp) / "adaptation_target_scale_context.json").read_text(encoding="utf-8"))
            baseline_gap = json.loads((Path(tmp) / "adaptation_baseline_gap_summary.json").read_text(encoding="utf-8"))
            subject_gap = json.loads((Path(tmp) / "adaptation_subject_baseline_gap_summary.json").read_text(encoding="utf-8"))
            verification = json.loads((Path(tmp) / "adaptation_verification.json").read_text(encoding="utf-8"))
            checksum_manifest = json.loads((Path(tmp) / "adaptation_checksum_manifest.json").read_text(encoding="utf-8"))
            report = (Path(tmp) / "adaptation_report.md").read_text(encoding="utf-8")
            checksum_rows = {row["path"]: row for row in checksum_manifest["artifacts"]}
            metrics_bytes = (Path(tmp) / "adaptation_metrics.json").read_bytes()
            self.assertEqual(checksum_manifest["schema"], "kahlus.eeg_v1_fewshot_adaptation_checksums.v1")
            self.assertEqual(checksum_manifest["algorithm"], "sha256")
            self.assertIn("adaptation_metrics.json", checksum_rows)
            self.assertIn("adaptation_report.md", checksum_rows)
            self.assertIn("adaptation_verification.json", checksum_rows)
            self.assertNotIn("adaptation_checksum_manifest.json", checksum_rows)
            self.assertEqual(checksum_rows["adaptation_metrics.json"]["bytes"], len(metrics_bytes))
            self.assertEqual(checksum_rows["adaptation_metrics.json"]["sha256"], hashlib.sha256(metrics_bytes).hexdigest())
            self.assertEqual(verification["schema"], "kahlus.eeg_v1_fewshot_adaptation_verification.v1")
            self.assertEqual(verification["checksum_manifest"], "adaptation_checksum_manifest.json")
            self.assertEqual(
                verification["checksum_audit_command"],
                f"PYTHONPATH=src python3 scripts/audit_eeg_v1_adaptation_checksums.py --artifact-dir {tmp}",
            )
            self.assertFalse(verification["a100_jobs_launched"])
            self.assertEqual(verification["execution_lane"], "local_cpu_or_single_process_only")
            self.assertIn("support_ridge", payload["metrics_by_method"])
            self.assertIn("bottleneck_adapter", payload["metrics_by_method"])
            self.assertIn(baseline_gap["best_support_baseline"], {"support_persistence", "support_ridge"})
            self.assertIn(baseline_gap["best_adaptation_method"], {"linear_probe", "bottleneck_adapter", "full_finetune"})
            self.assertIn("adaptation_vs_best_support_baseline_mse_delta", baseline_gap)
            self.assertIn("adaptation_beats_best_support_baseline", baseline_gap)
            self.assertEqual(subject_gap["n_subjects"], dataset_summary["n_adaptation_subjects"])
            self.assertIn("subjects_where_adaptation_beats_best_support_baseline", subject_gap)
            self.assertEqual(len(subject_gap["subjects"]), dataset_summary["n_adaptation_subjects"])
            first_subject_gap = subject_gap["subjects"][0]
            self.assertIn(first_subject_gap["best_support_baseline"], {"support_persistence", "support_ridge"})
            self.assertIn(first_subject_gap["best_adaptation_method"], {"linear_probe", "bottleneck_adapter", "full_finetune"})
            self.assertIn("adaptation_vs_best_support_baseline_mse_delta", first_subject_gap)
            self.assertIn("adaptation_beats_best_support_baseline", first_subject_gap)
            self.assertEqual(target_scale["target_units"], "normalized_eeg_fixture_units")
            self.assertGreater(target_scale["target_std"], 0.0)
            self.assertGreater(target_scale["target_variance"], 0.0)
            self.assertIn("mse_relative_to_target_variance", target_scale["methods"]["support_ridge"])
            self.assertIn("rmse_relative_to_target_std", target_scale["methods"]["support_ridge"])
            self.assertTrue(np.isfinite(target_scale["methods"]["support_ridge"]["rmse_relative_to_target_std"]))
            for field in ("seed", "pretrain_steps", "adapt_steps", "query_windows"):
                self.assertIn(field, run_config)
            self.assertEqual(run_config["seed"], payload["seed"])
            self.assertEqual(run_config["pretrain_steps"], payload["pretrain_steps"])
            self.assertEqual(run_config["adapt_steps"], payload["adapt_steps"])
            self.assertEqual(run_config["query_windows"], payload["query_windows"])
            self.assertEqual(split_audit["split_type"], "subject_held_out")
            self.assertTrue(split_audit["leakage_passed"], split_audit["failure_reasons"])
            self.assertFalse(split_audit["subject_overlap"])
            self.assertEqual(failures["gate_failures"], gate["failure_reasons"])
            self.assertEqual(failures["split_audit_failures"], split_audit["failure_reasons"])
            self.assertIn("gate_criteria", gate)
            self.assertEqual(gate["gate_criteria"]["min_support_windows"], 1)
            self.assertEqual(gate["gate_criteria"]["min_query_windows"], 1)
            self.assertTrue(gate["gate_criteria"]["requires_split_audit_passed"])
            self.assertTrue(gate["gate_criteria"]["requires_baseline_table_present"])
            self.assertTrue(gate["gate_criteria"]["requires_finite_metrics"])
            self.assertTrue(gate["gate_criteria"]["requires_calibration_checked"])
            self.assertEqual(gate["gate_criteria"]["allowed_claim_scope"], EEG_V1_ADAPTATION_CLAIM_SCOPE)
            self.assertEqual(dataset_summary["dataset"], payload["dataset"])
            self.assertEqual(dataset_summary["n_adaptation_subjects"], 2)
            self.assertEqual(dataset_summary["n_query_windows"], payload["query_windows"])
            self.assertEqual(dataset_summary["support_windows_per_subject"], payload["support_windows"])
            self.assertGreater(dataset_summary["n_pretrain_windows"], 0)
            self.assertEqual(dataset_summary["n_channels"], 6)
            self.assertEqual(dataset_summary["method_count"], len(payload["methods"]))
            self.assertIn("## Dataset Summary", report)
            self.assertIn("## Artifact Index", report)
            self.assertIn("| artifact | purpose |", report)
            for artifact in sorted(expected - {"adaptation_report.md"}):
                self.assertIn(f"| {artifact} |", report)
            self.assertLess(report.index("## Artifact Index"), report.index("## Run Config"))
            self.assertIn("## Checksum Audit", report)
            self.assertIn("adaptation_checksum_manifest.json", report)
            self.assertIn("adaptation_verification.json", report)
            self.assertIn("scripts/audit_eeg_v1_adaptation_checksums.py --artifact-dir <artifact-dir>", report)
            self.assertLess(report.index("## Artifact Index"), report.index("## Checksum Audit"))
            self.assertLess(report.index("## Checksum Audit"), report.index("## Run Config"))
            self.assertIn(f"- n_pretrain_windows: {dataset_summary['n_pretrain_windows']}", report)
            self.assertIn("- n_adaptation_subjects: 2", report)
            self.assertIn(f"- n_query_windows: {payload['query_windows']}", report)
            self.assertIn("- n_channels: 6", report)
            self.assertIn(f"- method_count: {len(payload['methods'])}", report)
            self.assertIn("## Split Audit", report)
            self.assertIn("- split_type: subject_held_out", report)
            self.assertIn("- leakage_passed: True", report)
            self.assertIn("- subject_overlap: False", report)
            self.assertIn("## Evidence Gate Criteria", report)
            self.assertIn("- min_support_windows: 1", report)
            self.assertIn("- min_query_windows: 1", report)
            self.assertIn("- requires_split_audit_passed: True", report)
            self.assertIn("- requires_baseline_table_present: True", report)
            self.assertIn("- requires_finite_metrics: True", report)
            self.assertIn("- requires_calibration_checked: True", report)
            self.assertIn(f"- allowed_claim_scope: {EEG_V1_ADAPTATION_CLAIM_SCOPE}", report)
            self.assertIn("## Run Config", report)
            self.assertIn(f"- seed: {payload['seed']}", report)
            self.assertIn(f"- pretrain_steps: {payload['pretrain_steps']}", report)
            self.assertIn(f"- adapt_steps: {payload['adapt_steps']}", report)
            self.assertIn(f"- window_length: {dataset_summary['window_length']}", report)
            self.assertIn(f"- forecast_horizon: {dataset_summary['forecast_horizon']}", report)
            self.assertIn(f"- support_windows: {payload['support_windows']}", report)
            self.assertIn(f"- query_windows: {payload['query_windows']}", report)
            self.assertLess(report.index("## Run Config"), report.index("## Method Order"))
            self.assertIn("## Target Scale Context", report)
            self.assertIn("- target_units: normalized_eeg_fixture_units", report)
            self.assertIn("rmse_relative_to_target_std", report)
            self.assertIn("## Baseline Gap Summary", report)
            self.assertIn("- best_support_baseline:", report)
            self.assertIn("- adaptation_vs_best_support_baseline_mse_delta:", report)
            self.assertIn("- interpretation: Baselines are results; adaptation methods must beat the best support baseline before any adapter-win discussion.", report)
            self.assertIn("## Per-Subject Baseline Gap Summary", report)
            self.assertIn("- subjects_where_adaptation_beats_best_support_baseline:", report)
            self.assertIn("| subject_id | best_support_baseline | best_adaptation_method | adaptation_vs_best_support_baseline_mse_delta | adaptation_beats_best_support_baseline |", report)
            self.assertIn("## Metric Breakdown Summary", report)
            self.assertIn("- adaptation_subject_rows:", report)
            self.assertIn("- detailed_sidecars: adaptation_subject_metrics.csv", report)
            self.assertIn("## Method Order", report)
            self.assertLess(report.index("## Method Order"), report.index("## Ranking"))
            method_rows = [
                "| 1 | support_persistence | baseline |",
                "| 2 | support_ridge | baseline |",
                "| 3 | linear_probe | adaptation |",
                "| 4 | bottleneck_adapter | adaptation |",
                "| 5 | full_finetune | adaptation |",
            ]
            positions = [report.index(row) for row in method_rows]
            self.assertEqual(positions, sorted(positions))
            self.assertTrue(gate["scientific_claim_allowed"], gate["failure_reasons"])

    def test_adaptation_checksum_audit_script_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_adaptation.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--pretrain-steps",
                    "2",
                    "--adapt-steps",
                    "2",
                    "--support-windows",
                    "4",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("a100_jobs_launched=false", run.stdout)

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_adaptation_checksums.py",
                    "--artifact-dir",
                    tmp,
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )
            self.assertEqual(audit.returncode, 0, audit.stderr + audit.stdout)
            audit_payload = json.loads(audit.stdout)
            self.assertTrue(audit_payload["passed"], audit_payload)
            self.assertEqual(audit_payload["artifacts_checked"], 13)

            metrics_path = Path(tmp) / "adaptation_metrics.json"
            metrics_path.write_text(metrics_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            tampered = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_adaptation_checksums.py",
                    "--artifact-dir",
                    tmp,
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(tampered.returncode, 0)
            tampered_payload = json.loads(tampered.stdout)
            self.assertFalse(tampered_payload["passed"], tampered_payload)
            self.assertIn("checksum_mismatch:adaptation_metrics.json", tampered_payload["failure_reasons"])

    def test_adaptation_checksum_audit_rejects_invalid_verification_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_adaptation.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--pretrain-steps",
                    "2",
                    "--adapt-steps",
                    "2",
                    "--support-windows",
                    "4",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            verification_path = Path(tmp) / "adaptation_verification.json"
            verification = json.loads(verification_path.read_text(encoding="utf-8"))
            verification["a100_jobs_launched"] = True
            verification_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_adaptation_checksums.py",
                    "--artifact-dir",
                    tmp,
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(audit.returncode, 0)
            payload = json.loads(audit.stdout)
            self.assertFalse(payload["passed"], payload)
            self.assertIn("checksum_mismatch:adaptation_verification.json", payload["failure_reasons"])
            self.assertIn("invalid_verification_a100_jobs_launched", payload["failure_reasons"])

    def test_adaptation_checksum_audit_requires_verification_sidecar_manifest_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_adaptation.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--pretrain-steps",
                    "2",
                    "--adapt-steps",
                    "2",
                    "--support-windows",
                    "4",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            manifest_path = Path(tmp) / "adaptation_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"] = [
                row for row in manifest["artifacts"] if row["path"] != "adaptation_verification.json"
            ]
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_adaptation_checksums.py",
                    "--artifact-dir",
                    tmp,
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(audit.returncode, 0)
            payload = json.loads(audit.stdout)
            self.assertFalse(payload["passed"], payload)
            self.assertIn(
                "missing_manifest_entry:adaptation_verification.json",
                payload["failure_reasons"],
            )

    def test_adaptation_checksum_audit_requires_all_required_artifact_manifest_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_adaptation.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--pretrain-steps",
                    "2",
                    "--adapt-steps",
                    "2",
                    "--support-windows",
                    "4",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            manifest_path = Path(tmp) / "adaptation_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"] = [
                row for row in manifest["artifacts"] if row["path"] != "adaptation_metrics.json"
            ]
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_adaptation_checksums.py",
                    "--artifact-dir",
                    tmp,
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(audit.returncode, 0)
            payload = json.loads(audit.stdout)
            self.assertFalse(payload["passed"], payload)
            self.assertIn(
                "missing_manifest_entry:adaptation_metrics.json",
                payload["failure_reasons"],
            )

    def test_adaptation_checksum_audit_rejects_duplicate_manifest_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_adaptation.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--pretrain-steps",
                    "2",
                    "--adapt-steps",
                    "2",
                    "--support-windows",
                    "4",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            manifest_path = Path(tmp) / "adaptation_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            verification_rows = [
                row for row in manifest["artifacts"] if row["path"] == "adaptation_verification.json"
            ]
            self.assertEqual(len(verification_rows), 1)
            manifest["artifacts"].append(dict(verification_rows[0]))
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_adaptation_checksums.py",
                    "--artifact-dir",
                    tmp,
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(audit.returncode, 0)
            payload = json.loads(audit.stdout)
            self.assertFalse(payload["passed"], payload)
            self.assertIn(
                "duplicate_manifest_entry:adaptation_verification.json",
                payload["failure_reasons"],
            )

    def test_adaptation_checksum_audit_rejects_unexpected_manifest_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_adaptation.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--pretrain-steps",
                    "2",
                    "--adapt-steps",
                    "2",
                    "--support-windows",
                    "4",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            manifest_path = Path(tmp) / "adaptation_checksum_manifest.json"
            unexpected_path = Path(tmp) / "unexpected_extra_evidence.json"
            unexpected_payload = b'{"note":"unexpected manifest expansion"}\n'
            unexpected_path.write_bytes(unexpected_payload)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"].append(
                {
                    "path": unexpected_path.name,
                    "bytes": len(unexpected_payload),
                    "sha256": hashlib.sha256(unexpected_payload).hexdigest(),
                }
            )
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_adaptation_checksums.py",
                    "--artifact-dir",
                    tmp,
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(audit.returncode, 0)
            payload = json.loads(audit.stdout)
            self.assertFalse(payload["passed"], payload)
            self.assertIn(
                "unexpected_manifest_entry:unexpected_extra_evidence.json",
                payload["failure_reasons"],
            )

    def test_adaptation_script_reports_invalid_task_config_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_adaptation.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--support-windows",
                    "1000",
                    "--pretrain-steps",
                    "2",
                    "--adapt-steps",
                    "2",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

        combined = result.stderr + result.stdout
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("EEG v1 adaptation task configuration invalid:", combined)
        self.assertIn("few-shot adaptation task requires train subjects and held-out query windows", combined)
        self.assertNotIn("Traceback", combined)

    def test_hbn_local_fixture_adaptation_marks_local_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hbn"
            out = Path(tmp) / "out"
            root.mkdir()
            rows = []
            time = np.arange(80, dtype=np.float32)
            for idx in range(9):
                signal = np.stack(
                    [
                        np.sin(time / 5.0 + idx),
                        np.cos(time / 7.0 + idx * 0.2),
                    ],
                    axis=1,
                ).astype(np.float32)
                path = root / f"sub-{idx:03d}.npy"
                np.save(path, signal)
                rows.append(
                    json.dumps(
                        {
                            "path": path.name,
                            "subject_id": f"sub-{idx:03d}",
                            "session_id": "ses-00",
                            "sampling_rate": 128.0,
                        }
                    )
                )
            (root / "manifest.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_adaptation.py",
                    "--dataset",
                    "hbn_eeg",
                    "--data-root",
                    str(root),
                    "--out-dir",
                    str(out),
                    "--seed",
                    "0",
                    "--pretrain-steps",
                    "1",
                    "--adapt-steps",
                    "1",
                    "--support-windows",
                    "4",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("data_source=hbn_eeg_local", result.stdout)
            payload = json.loads((out / "adaptation_metrics.json").read_text(encoding="utf-8"))
            run_config = json.loads((out / "adaptation_run_config.json").read_text(encoding="utf-8"))
            dataset_summary = json.loads((out / "adaptation_dataset_summary.json").read_text(encoding="utf-8"))
            report = (out / "adaptation_report.md").read_text(encoding="utf-8")
            self.assertEqual(payload["source"], "hbn_eeg_local")
            self.assertEqual(payload["benchmark_status"], "local_manifest_not_public_hbn_benchmark")
            self.assertEqual(run_config["data_source"], "hbn_eeg_local")
            self.assertEqual(run_config["benchmark_status"], "local_manifest_not_public_hbn_benchmark")
            self.assertIn("sampling_rate_hz", run_config)
            self.assertEqual(run_config["sampling_rate_hz"], 128.0)
            self.assertEqual(dataset_summary["data_source"], "hbn_eeg_local")
            self.assertIn("sampling_rate_hz", dataset_summary)
            self.assertEqual(dataset_summary["sampling_rate_hz"], 128.0)
            self.assertIn("source: hbn_eeg_local", report)
            self.assertIn("- sampling_rate_hz: 128", report)
            self.assertIn("local manifest; not a public HBN benchmark result", report)


if __name__ == "__main__":
    unittest.main()
