import json
import os
import subprocess
import sys
import tempfile
import unittest
import csv
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from neurotwin.eeg_v1 import (
    EEG_V1_CLAIM_SCOPE,
    audit_eeg_v1_split,
    build_eeg_v1_gate,
    build_future_forecasting_task,
    load_hbn_eeg_local_dataset,
    make_synthetic_eeg_v1_dataset,
    run_eeg_v1_autocorrelation_diagnostics,
    run_eeg_v1_baselines,
    smoothness_loss,
    write_eeg_v1_artifacts,
)
from neurotwin.eeg_v1.reporting import DEFAULT_EEG_V1_MODELS


class EEGV1SprintATests(unittest.TestCase):
    def test_synthetic_fixture_is_deterministic_and_subject_held_out(self):
        left = make_synthetic_eeg_v1_dataset(seed=7)
        right = make_synthetic_eeg_v1_dataset(seed=7)

        self.assertEqual([b.recording_id for b in left.batches], [b.recording_id for b in right.batches])
        self.assertTrue(np.array_equal(left.batches[0].signal, right.batches[0].signal))

        train = set(left.split_subjects["train"])
        val = set(left.split_subjects["val"])
        test = set(left.split_subjects["test"])
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)

    def test_future_forecasting_task_and_split_audit(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=3, n_subjects=9, sessions_per_subject=1)
        task = build_future_forecasting_task(dataset, window_length=8, forecast_horizon=2)
        # stride=8 only thins consecutive windows; input↔target still overlap for H=2, W=8.
        wide_stride = build_future_forecasting_task(dataset, window_length=8, forecast_horizon=2, stride=8)
        audit = audit_eeg_v1_split(dataset, split_type="subject_held_out")

        self.assertEqual(task.task_id, "future_state_forecasting")
        self.assertEqual(task.source_modality, "eeg")
        self.assertEqual(task.x_train.shape[1:], task.y_train.shape[1:])
        self.assertEqual(task.metadata["forecast_horizon"], 2)
        self.assertEqual(task.metadata["window_stride"], 1)
        self.assertEqual(wide_stride.metadata["window_stride"], 8)
        self.assertLess(wide_stride.x_train.shape[0], task.x_train.shape[0])
        self.assertIsNotNone(task.metric_mask)
        # Strictly-future positions for H=2, W=8 are the last two target indices.
        self.assertTrue(np.all(task.metric_mask[:, -2:, :]))
        self.assertFalse(np.any(task.metric_mask[:, :-2, :]))
        self.assertTrue(audit["leakage_passed"], audit["failure_reasons"])
        self.assertFalse(audit["subject_overlap"])
        self.assertFalse(audit["window_overlap"])

    def test_autocorrelation_diagnostics_explain_easy_smoke_task(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=0)
        diagnostics = run_eeg_v1_autocorrelation_diagnostics(
            dataset,
            seed=0,
            window_length=8,
            forecast_horizon=1,
            train_steps=2,
            model_ids=("persistence", "linear_ridge"),
        )

        by_id = {row["diagnostic_id"]: row for row in diagnostics["diagnostics"]}
        for required in (
            "short_horizon_overlap",
            "long_horizon",
            "non_overlapping_windows",
            "shuffled_target_control",
            "delta_prediction",
            "subject_held_out_split",
            "stimulus_task_held_out_split",
        ):
            self.assertIn(required, by_id)

        self.assertEqual(by_id["short_horizon_overlap"]["status"], "completed")
        self.assertEqual(by_id["long_horizon"]["status"], "completed")
        self.assertEqual(by_id["non_overlapping_windows"]["window_stride"], 8)
        self.assertEqual(by_id["stimulus_task_held_out_split"]["status"], "not_applicable_missing_labels")
        self.assertTrue(by_id["subject_held_out_split"]["leakage_passed"])
        self.assertGreater(
            by_id["shuffled_target_control"]["best_mse"],
            by_id["short_horizon_overlap"]["best_mse"],
        )
        self.assertIn("autocorrelation_warning", diagnostics["summary"])
        self.assertIn("delta_prediction_best_mse", diagnostics["summary"])
        self.assertEqual(
            diagnostics["summary"]["delta_prediction_best_mse"],
            by_id["delta_prediction"]["best_mse"],
        )
        self.assertIn("delta_prediction_delta_vs_short", diagnostics["summary"])

    def test_a3_requires_tiny_ssm_and_train_only_shuffled_target_control(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=0)
        task = build_future_forecasting_task(dataset, window_length=8, forecast_horizon=1)
        result = run_eeg_v1_baselines(task, seed=0, train_steps=2, model_ids=DEFAULT_EEG_V1_MODELS)
        diagnostics = run_eeg_v1_autocorrelation_diagnostics(
            dataset,
            seed=0,
            window_length=8,
            forecast_horizon=1,
            train_steps=2,
        )

        self.assertIn("tiny_ssm", result["metrics_by_model"])
        self.assertNotIn("ssm_fallback", DEFAULT_EEG_V1_MODELS)
        by_id = {row["diagnostic_id"]: row for row in diagnostics["diagnostics"]}
        shuffled = by_id["shuffled_target_control"]
        self.assertEqual(shuffled["status"], "completed")
        self.assertTrue(shuffled["train_targets_shuffled"])
        self.assertFalse(shuffled["validation_targets_shuffled"])
        self.assertFalse(shuffled["test_targets_shuffled"])
        self.assertEqual(shuffled["shuffle_boundary"], "train_split_only")
        self.assertEqual(shuffled["shuffle_seed"], 1701)
        self.assertEqual(shuffled["shuffle_seed_source"], "diagnostic_seed_plus_1701")
        self.assertIn("tiny_ssm", shuffled["metrics_by_model"])
        self.assertIn("tiny_ssm_mse", diagnostics["summary"])
        self.assertIn("shuffled_target_control_mse", diagnostics["summary"])

    def test_a3_gate_fails_when_tiny_ssm_or_shuffled_control_is_missing(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=0)
        task = build_future_forecasting_task(dataset, window_length=8, forecast_horizon=1)
        split_audit = audit_eeg_v1_split(dataset, split_type="subject_held_out")
        missing_ssm_result = run_eeg_v1_baselines(
            task,
            seed=0,
            train_steps=2,
            model_ids=("persistence", "linear_ridge", "autoregressive_ridge"),
        )
        diagnostics = run_eeg_v1_autocorrelation_diagnostics(
            dataset,
            seed=0,
            window_length=8,
            forecast_horizon=1,
            train_steps=2,
        )
        diagnostics_without_shuffle = {
            **diagnostics,
            "diagnostics": [
                row for row in diagnostics["diagnostics"] if row["diagnostic_id"] != "shuffled_target_control"
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            write_eeg_v1_artifacts(
                tmp,
                task=task,
                baseline_result=missing_ssm_result,
                split_audit=split_audit,
                models=("persistence", "linear_ridge", "autoregressive_ridge"),
                train_steps=2,
                seed=0,
                autocorrelation_diagnostics=diagnostics_without_shuffle,
            )
            gate = json.loads((Path(tmp) / "evidence_gate.json").read_text(encoding="utf-8"))
            report = (Path(tmp) / "diagnostic_report.md").read_text(encoding="utf-8")

        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertIn("required first-class baseline missing: tiny_ssm", gate["failure_reasons"])
        self.assertIn("required negative control missing: shuffled_target_control", gate["failure_reasons"])
        self.assertIn("requires_first_class_ssm_baseline: tiny_ssm", report)
        self.assertIn("requires_negative_control: shuffled_target_control", report)

    def test_a4_gate_fails_when_shuffled_target_control_stays_too_close(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=0)
        task = build_future_forecasting_task(dataset, window_length=8, forecast_horizon=1)
        split_audit = audit_eeg_v1_split(dataset, split_type="subject_held_out")
        result = run_eeg_v1_baselines(
            task,
            seed=0,
            train_steps=2,
            model_ids=("persistence", "linear_ridge", "tiny_ssm"),
        )
        diagnostics = run_eeg_v1_autocorrelation_diagnostics(
            dataset,
            seed=0,
            window_length=8,
            forecast_horizon=1,
            train_steps=2,
            model_ids=("persistence", "linear_ridge", "tiny_ssm"),
        )
        unsafe_diagnostics = json.loads(json.dumps(diagnostics))
        unsafe_diagnostics["summary"]["shuffled_control_degrades"] = False
        unsafe_diagnostics["summary"]["shuffled_target_close_to_real_baselines"] = True

        with tempfile.TemporaryDirectory() as tmp:
            write_eeg_v1_artifacts(
                tmp,
                task=task,
                baseline_result=result,
                split_audit=split_audit,
                models=("persistence", "linear_ridge", "tiny_ssm"),
                train_steps=2,
                seed=0,
                autocorrelation_diagnostics=unsafe_diagnostics,
            )
            gate = json.loads((Path(tmp) / "evidence_gate.json").read_text(encoding="utf-8"))
            report = (Path(tmp) / "diagnostic_report.md").read_text(encoding="utf-8")

        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertTrue(gate["gate_criteria"]["requires_shuffled_target_degradation"])
        self.assertTrue(gate["gate_criteria"]["requires_shuffled_target_not_close_to_real_baselines"])
        self.assertIn("shuffled-target negative control did not degrade", gate["failure_reasons"])
        self.assertIn(
            "shuffled-target negative control is too close to real baseline performance",
            gate["failure_reasons"],
        )
        self.assertIn("- requires_shuffled_target_degradation: True", report)
        self.assertIn("- requires_shuffled_target_not_close_to_real_baselines: True", report)

    def test_stimulus_task_split_diagnostic_audits_label_overlap(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=0, n_subjects=9, sessions_per_subject=1)
        split_labels = {
            "train": "stimulus_train_only",
            "val": "stimulus_val_only",
            "test": "stimulus_test_only",
        }
        labels_by_record: dict[str, str] = {}
        split_parts = {}
        for split_name in ("train", "val", "test"):
            updated_records = []
            for record in getattr(dataset.split_manifest, split_name):
                label = split_labels[split_name]
                labels_by_record[record.record_id] = label
                updated_records.append(
                    replace(
                        record,
                        stimulus_id=label,
                        metadata={**record.metadata, "stimulus_id": label},
                    )
                )
            split_parts[split_name] = updated_records
        labelled_batches = tuple(
            replace(
                batch,
                metadata={**batch.metadata, "stimulus_id": labels_by_record[batch.recording_id]},
            )
            for batch in dataset.batches
        )
        labelled_dataset = replace(
            dataset,
            batches=labelled_batches,
            records=tuple(record for split_name in ("train", "val", "test") for record in split_parts[split_name]),
            split_manifest=replace(dataset.split_manifest, **split_parts),
        )

        diagnostics = run_eeg_v1_autocorrelation_diagnostics(
            labelled_dataset,
            seed=0,
            window_length=8,
            forecast_horizon=1,
            train_steps=2,
            model_ids=("persistence", "linear_ridge"),
        )

        by_id = {row["diagnostic_id"]: row for row in diagnostics["diagnostics"]}
        diagnostic = by_id["stimulus_task_held_out_split"]
        self.assertEqual(diagnostic["status"], "completed")
        self.assertTrue(diagnostic["leakage_passed"], diagnostic["failure_reasons"])
        self.assertFalse(diagnostic["label_overlap"])
        self.assertEqual(diagnostic["observed_label_keys"], ["stimulus_id"])
        self.assertEqual(diagnostic["train_labels"], ["stimulus_train_only"])
        self.assertEqual(diagnostic["val_labels"], ["stimulus_val_only"])
        self.assertEqual(diagnostic["test_labels"], ["stimulus_test_only"])

    def test_diagnostic_report_lists_stimulus_task_split_audit(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=0, n_subjects=9, sessions_per_subject=1)
        split_labels = {
            "train": "shared_stimulus",
            "val": "stimulus_val_only",
            "test": "shared_stimulus",
        }
        labels_by_record: dict[str, str] = {}
        split_parts = {}
        for split_name in ("train", "val", "test"):
            updated_records = []
            for record in getattr(dataset.split_manifest, split_name):
                label = split_labels[split_name]
                labels_by_record[record.record_id] = label
                updated_records.append(
                    replace(
                        record,
                        stimulus_id=label,
                        metadata={**record.metadata, "stimulus_id": label},
                    )
                )
            split_parts[split_name] = updated_records
        labelled_batches = tuple(
            replace(
                batch,
                metadata={**batch.metadata, "stimulus_id": labels_by_record[batch.recording_id]},
            )
            for batch in dataset.batches
        )
        labelled_dataset = replace(
            dataset,
            batches=labelled_batches,
            records=tuple(record for split_name in ("train", "val", "test") for record in split_parts[split_name]),
            split_manifest=replace(dataset.split_manifest, **split_parts),
        )
        task = build_future_forecasting_task(labelled_dataset, window_length=8, forecast_horizon=1)
        result = run_eeg_v1_baselines(task, seed=0, train_steps=2, model_ids=("persistence", "linear_ridge", "tiny_ssm"))
        split_audit = audit_eeg_v1_split(labelled_dataset, split_type="subject_held_out")
        diagnostics = run_eeg_v1_autocorrelation_diagnostics(
            labelled_dataset,
            seed=0,
            window_length=8,
            forecast_horizon=1,
            train_steps=2,
            model_ids=("persistence", "linear_ridge", "tiny_ssm"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            write_eeg_v1_artifacts(
                tmp,
                task=task,
                baseline_result=result,
                split_audit=split_audit,
                models=("persistence", "linear_ridge", "tiny_ssm"),
                train_steps=2,
                seed=0,
                autocorrelation_diagnostics=diagnostics,
            )
            report = (Path(tmp) / "diagnostic_report.md").read_text(encoding="utf-8")

        self.assertIn("## Stimulus/Task Split Audit", report)
        self.assertIn("- status: completed", report)
        self.assertIn("- leakage_passed: False", report)
        self.assertIn("- label_overlap: True", report)
        self.assertIn("- observed_label_keys: stimulus_id", report)
        self.assertIn("- train_labels: shared_stimulus", report)
        self.assertIn("- val_labels: stimulus_val_only", report)
        self.assertIn("- test_labels: shared_stimulus", report)
        self.assertIn("- failure_reasons: train_test_label_overlap:shared_stimulus", report)

    def test_baseline_checksum_audit_rejects_extra_diagnostic_report_stimulus_task_audit_line(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=0, n_subjects=9, sessions_per_subject=1)
        split_labels = {
            "train": "shared_stimulus",
            "val": "stimulus_val_only",
            "test": "shared_stimulus",
        }
        labels_by_record: dict[str, str] = {}
        split_parts = {}
        for split_name in ("train", "val", "test"):
            updated_records = []
            for record in getattr(dataset.split_manifest, split_name):
                label = split_labels[split_name]
                labels_by_record[record.record_id] = label
                updated_records.append(
                    replace(
                        record,
                        stimulus_id=label,
                        metadata={**record.metadata, "stimulus_id": label},
                    )
                )
            split_parts[split_name] = updated_records
        labelled_batches = tuple(
            replace(
                batch,
                metadata={**batch.metadata, "stimulus_id": labels_by_record[batch.recording_id]},
            )
            for batch in dataset.batches
        )
        labelled_dataset = replace(
            dataset,
            batches=labelled_batches,
            records=tuple(record for split_name in ("train", "val", "test") for record in split_parts[split_name]),
            split_manifest=replace(dataset.split_manifest, **split_parts),
        )
        task = build_future_forecasting_task(labelled_dataset, window_length=8, forecast_horizon=1)
        result = run_eeg_v1_baselines(task, seed=0, train_steps=2, model_ids=("persistence", "linear_ridge", "tiny_ssm"))
        split_audit = audit_eeg_v1_split(labelled_dataset, split_type="subject_held_out")
        diagnostics = run_eeg_v1_autocorrelation_diagnostics(
            labelled_dataset,
            seed=0,
            window_length=8,
            forecast_horizon=1,
            train_steps=2,
            model_ids=("persistence", "linear_ridge", "tiny_ssm"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            write_eeg_v1_artifacts(
                tmp,
                task=task,
                baseline_result=result,
                split_audit=split_audit,
                models=("persistence", "linear_ridge", "tiny_ssm"),
                train_steps=2,
                seed=0,
                autocorrelation_diagnostics=diagnostics,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            marker = "- failure_reasons: train_test_label_overlap:shared_stimulus"
            self.assertIn(marker, report)
            report_path.write_text(
                report.replace(marker, "- reviewer_override: stimulus split passed\n" + marker),
                encoding="utf-8",
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
            "diagnostic_report_mismatch:stimulus_task_split_audit_section",
            payload["failure_reasons"],
        )

    def test_stimulus_task_split_overlap_blocks_evidence_gate(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=0, n_subjects=9, sessions_per_subject=1)
        split_labels = {
            "train": "shared_stimulus",
            "val": "stimulus_val_only",
            "test": "shared_stimulus",
        }
        labels_by_record: dict[str, str] = {}
        split_parts = {}
        for split_name in ("train", "val", "test"):
            updated_records = []
            for record in getattr(dataset.split_manifest, split_name):
                label = split_labels[split_name]
                labels_by_record[record.record_id] = label
                updated_records.append(
                    replace(
                        record,
                        stimulus_id=label,
                        metadata={**record.metadata, "stimulus_id": label},
                    )
                )
            split_parts[split_name] = updated_records
        labelled_batches = tuple(
            replace(
                batch,
                metadata={**batch.metadata, "stimulus_id": labels_by_record[batch.recording_id]},
            )
            for batch in dataset.batches
        )
        labelled_dataset = replace(
            dataset,
            batches=labelled_batches,
            records=tuple(record for split_name in ("train", "val", "test") for record in split_parts[split_name]),
            split_manifest=replace(dataset.split_manifest, **split_parts),
        )
        task = build_future_forecasting_task(labelled_dataset, window_length=8, forecast_horizon=1)
        result = run_eeg_v1_baselines(task, seed=0, train_steps=2, model_ids=("persistence", "linear_ridge", "tiny_ssm"))
        split_audit = audit_eeg_v1_split(labelled_dataset, split_type="subject_held_out")
        diagnostics = run_eeg_v1_autocorrelation_diagnostics(
            labelled_dataset,
            seed=0,
            window_length=8,
            forecast_horizon=1,
            train_steps=2,
            model_ids=("persistence", "linear_ridge", "tiny_ssm"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            write_eeg_v1_artifacts(
                tmp,
                task=task,
                baseline_result=result,
                split_audit=split_audit,
                models=("persistence", "linear_ridge", "tiny_ssm"),
                train_steps=2,
                seed=0,
                autocorrelation_diagnostics=diagnostics,
            )
            gate = json.loads((Path(tmp) / "evidence_gate.json").read_text(encoding="utf-8"))
            failures = json.loads((Path(tmp) / "failure_reasons.json").read_text(encoding="utf-8"))
            report = (Path(tmp) / "diagnostic_report.md").read_text(encoding="utf-8")

        expected = "stimulus/task label split audit did not pass: train_test_label_overlap:shared_stimulus"
        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertIn(expected, gate["failure_reasons"])
        self.assertEqual(failures["diagnostic_failures"], [expected])
        self.assertIn("## Gate Failures", report)
        self.assertIn(f"- {expected}", report)

    def test_diagnostic_report_lists_split_audit_failures_when_blocked(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=6, n_subjects=9, sessions_per_subject=2)
        task = build_future_forecasting_task(dataset, window_length=8, forecast_horizon=1)
        result = run_eeg_v1_baselines(
            task,
            seed=0,
            train_steps=2,
            model_ids=("persistence", "linear_ridge"),
        )
        split_audit = dict(audit_eeg_v1_split(dataset, split_type="subject_held_out"))
        split_audit["leakage_passed"] = False
        split_audit["failure_reasons"] = ["forced baseline split audit failure for report coverage"]

        with tempfile.TemporaryDirectory() as tmp:
            write_eeg_v1_artifacts(
                tmp,
                task=task,
                baseline_result=result,
                split_audit=split_audit,
                models=("persistence", "linear_ridge"),
                train_steps=2,
                seed=0,
            )
            gate = json.loads((Path(tmp) / "evidence_gate.json").read_text(encoding="utf-8"))
            failures = json.loads((Path(tmp) / "failure_reasons.json").read_text(encoding="utf-8"))
            report = (Path(tmp) / "diagnostic_report.md").read_text(encoding="utf-8")

        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertIn("split audit did not pass", gate["failure_reasons"])
        self.assertEqual(failures["gate_failures"], gate["failure_reasons"])
        self.assertIn("split_audit_failures", failures)
        self.assertEqual(
            failures["split_audit_failures"],
            ["forced baseline split audit failure for report coverage"],
        )
        self.assertIn("## Gate Failures", report)
        self.assertIn("- split audit did not pass", report)
        self.assertIn("## Split Audit Failures", report)
        self.assertIn("- forced baseline split audit failure for report coverage", report)

    def test_diagnostic_report_lists_baseline_failures(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=6, n_subjects=9, sessions_per_subject=2)
        task = build_future_forecasting_task(dataset, window_length=8, forecast_horizon=1)
        result = run_eeg_v1_baselines(
            task,
            seed=0,
            train_steps=2,
            model_ids=("persistence", "missing_baseline"),
        )
        split_audit = audit_eeg_v1_split(dataset, split_type="subject_held_out")

        with tempfile.TemporaryDirectory() as tmp:
            write_eeg_v1_artifacts(
                tmp,
                task=task,
                baseline_result=result,
                split_audit=split_audit,
                models=("persistence", "missing_baseline"),
                train_steps=2,
                seed=0,
            )
            failures = json.loads((Path(tmp) / "failure_reasons.json").read_text(encoding="utf-8"))
            report = (Path(tmp) / "diagnostic_report.md").read_text(encoding="utf-8")

        self.assertEqual(failures["baseline_failures"][0]["model_id"], "missing_baseline")
        self.assertIn("## Baseline Failures", report)
        self.assertIn("- missing_baseline: unknown requested baseline model_id", report)

    def test_baseline_checksum_audit_rejects_extra_diagnostic_report_failure_lines(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=6, n_subjects=9, sessions_per_subject=2)
        task = build_future_forecasting_task(dataset, window_length=8, forecast_horizon=1)
        result = run_eeg_v1_baselines(
            task,
            seed=0,
            train_steps=2,
            model_ids=("persistence", "missing_baseline"),
        )
        split_audit = dict(audit_eeg_v1_split(dataset, split_type="subject_held_out"))
        split_audit["leakage_passed"] = False
        split_audit["failure_reasons"] = ["forced split failure for exact-section audit"]

        with tempfile.TemporaryDirectory() as tmp:
            write_eeg_v1_artifacts(
                tmp,
                task=task,
                baseline_result=result,
                split_audit=split_audit,
                models=("persistence", "missing_baseline"),
                train_steps=2,
                seed=0,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("## Gate Failures", report)
            self.assertIn("## Split Audit Failures", report)
            self.assertIn("## Baseline Failures", report)
            report = report.replace("## Split Audit Failures", "- fabricated reviewer gate failure\n\n## Split Audit Failures")
            report = report.replace("## Baseline Failures", "- fabricated reviewer split failure\n\n## Baseline Failures")
            report = report + "- fabricated_baseline: reviewer-only failure\n"
            report_path.write_text(report, encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:gate_failures_section", payload["failure_reasons"])
        self.assertIn("diagnostic_report_mismatch:split_audit_failures_section", payload["failure_reasons"])
        self.assertIn("diagnostic_report_mismatch:baseline_failures_section", payload["failure_reasons"])

    def test_diagnostic_report_lists_autocorrelation_diagnostic_reasons(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=7, n_subjects=9, sessions_per_subject=2)
        task = build_future_forecasting_task(dataset, window_length=8, forecast_horizon=1)
        result = run_eeg_v1_baselines(task, seed=0, train_steps=2, model_ids=("persistence", "linear_ridge"))
        split_audit = audit_eeg_v1_split(dataset, split_type="subject_held_out")
        diagnostics = run_eeg_v1_autocorrelation_diagnostics(
            dataset,
            seed=0,
            window_length=8,
            forecast_horizon=1,
            train_steps=2,
            model_ids=("persistence", "linear_ridge"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            write_eeg_v1_artifacts(
                tmp,
                task=task,
                baseline_result=result,
                split_audit=split_audit,
                models=("persistence", "linear_ridge"),
                train_steps=2,
                seed=0,
                autocorrelation_diagnostics=diagnostics,
            )
            report = (Path(tmp) / "diagnostic_report.md").read_text(encoding="utf-8")

        self.assertIn(
            "| diagnostic | status | best_model | best_mse | persistence_mse | linear_ridge_mse | tiny_ssm_mse | reason |",
            report,
        )
        self.assertIn("- caveat: Low persistence or ridge MSE is autocorrelation evidence, not brain-state understanding.", report)
        self.assertIn("short_horizon_overlap | completed", report)
        self.assertIn("persistence", report)
        self.assertIn("linear_ridge", report)
        self.assertIn(
            "| stimulus_task_held_out_split | not_applicable_missing_labels |  |  |  |  |  | dataset has no stimulus/task label metadata to hold out |",
            report,
        )

    def test_diagnostic_report_lists_baseline_first_method_order(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=8, n_subjects=9, sessions_per_subject=2)
        task = build_future_forecasting_task(dataset, window_length=8, forecast_horizon=1)
        models = ("persistence", "linear_ridge", "neurotwin")
        result = run_eeg_v1_baselines(task, seed=0, train_steps=2, model_ids=models)
        split_audit = audit_eeg_v1_split(dataset, split_type="subject_held_out")

        with tempfile.TemporaryDirectory() as tmp:
            write_eeg_v1_artifacts(
                tmp,
                task=task,
                baseline_result=result,
                split_audit=split_audit,
                models=models,
                train_steps=2,
                seed=0,
            )
            report = (Path(tmp) / "diagnostic_report.md").read_text(encoding="utf-8")

        self.assertIn("## Method Order", report)
        self.assertLess(report.index("## Method Order"), report.index("## Baseline Ranking"))
        rows = [
            "| 1 | persistence | baseline |",
            "| 2 | linear_ridge | baseline |",
            "| 3 | neurotwin | main_model |",
        ]
        positions = [report.index(row) for row in rows]
        self.assertEqual(positions, sorted(positions))

    def test_smoothness_loss_is_finite(self):
        y_hat = np.arange(2 * 6 * 3, dtype=np.float64).reshape(2, 6, 3)
        self.assertTrue(np.isfinite(smoothness_loss(y_hat)))
        self.assertEqual(smoothness_loss(np.ones((2, 2, 3))), 0.0)

    def test_gate_allows_only_narrow_eeg_scope(self):
        allowed = build_eeg_v1_gate(
            dataset="synthetic_fixture",
            split_audit_passed=True,
            baseline_table_present=True,
            finite_metrics=True,
            forecast_horizon=1,
            split_type="subject_held_out",
            claim_scope=EEG_V1_CLAIM_SCOPE,
        )
        self.assertTrue(allowed["scientific_claim_allowed"], allowed["failure_reasons"])
        self.assertIn("gate_criteria", allowed)
        self.assertEqual(
            allowed["gate_criteria"],
            {
                "min_forecast_horizon": 1,
                "allowed_split_types": ["session_held_out", "subject_held_out"],
                "requires_split_audit_passed": True,
                "requires_baseline_table_present": True,
                "requires_finite_metrics": True,
                "requires_calibration_checked": True,
                "allowed_claim_scope": EEG_V1_CLAIM_SCOPE,
            },
        )

        for broad_scope in (
            "diagnosis",
            "treatment",
            "depression_detection",
            "epilepsy_detection",
            "foundation_model",
            "sota",
            "kahlus_v2_success",
            "kahlus_v3_success",
        ):
            blocked = build_eeg_v1_gate(
                dataset="synthetic_fixture",
                split_audit_passed=True,
                baseline_table_present=True,
                finite_metrics=True,
                forecast_horizon=1,
                split_type="subject_held_out",
                claim_scope=broad_scope,
            )
            self.assertFalse(blocked["scientific_claim_allowed"])
            self.assertTrue(blocked["failure_reasons"])

    def test_script_writes_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,autoregressive_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("claim_scope=eeg_future_forecasting_benchmark_ready", result.stdout)
            self.assertIn("autocorrelation_short_horizon_best_mse=", result.stdout)
            self.assertIn("autocorrelation_shuffled_target_best_mse=", result.stdout)
            self.assertIn("autocorrelation_long_horizon_delta_vs_short=", result.stdout)
            self.assertIn("autocorrelation_non_overlap_delta_vs_short=", result.stdout)
            self.assertIn("autocorrelation_delta_prediction_delta_vs_short=", result.stdout)
            self.assertIn("target_units=normalized_eeg_fixture_units", result.stdout)
            self.assertIn("best_baseline_rmse_relative_to_target_std=", result.stdout)
            self.assertIn("model_win_claim_allowed=False", result.stdout)
            self.assertIn("model_win_status=blocked_by_autocorrelation_baseline", result.stdout)
            self.assertIn("baseline_verification=", result.stdout)
            self.assertIn("a100_jobs_launched=false", result.stdout)
            self.assertIn("baseline_checksum_manifest=", result.stdout)
            self.assertIn(
                f"checksum_audit_command=PYTHONPATH=src python3 scripts/audit_eeg_v1_baseline_checksums.py --artifact-dir {tmp}",
                result.stdout,
            )
            expected = {
                "metrics.json",
                "metrics.csv",
                "baseline_table.json",
                "baseline_table.csv",
                "split_audit.json",
                "evidence_gate.json",
                "run_config.json",
                "dataset_summary.json",
                "failure_reasons.json",
                "diagnostic_report.md",
                "per_subject_metrics.csv",
                "per_channel_metrics.csv",
                "per_horizon_metrics.csv",
                "autocorrelation_diagnostics.json",
                "autocorrelation_diagnostics.csv",
                "target_scale_context.json",
                "baseline_verification.json",
            }
            self.assertTrue(expected.issubset({p.name for p in Path(tmp).iterdir()}))
            metrics = json.loads((Path(tmp) / "metrics.json").read_text(encoding="utf-8"))
            gate = json.loads((Path(tmp) / "evidence_gate.json").read_text(encoding="utf-8"))
            run_config = json.loads((Path(tmp) / "run_config.json").read_text(encoding="utf-8"))
            dataset_summary = json.loads((Path(tmp) / "dataset_summary.json").read_text(encoding="utf-8"))
            diagnostics = json.loads((Path(tmp) / "autocorrelation_diagnostics.json").read_text(encoding="utf-8"))
            target_scale = json.loads((Path(tmp) / "target_scale_context.json").read_text(encoding="utf-8"))
            diagnostic_report = (Path(tmp) / "diagnostic_report.md").read_text(encoding="utf-8")
            per_subject_text = (Path(tmp) / "per_subject_metrics.csv").read_text(encoding="utf-8")
            per_subject_rows = list(csv.DictReader(per_subject_text.splitlines()))
            self.assertIn("best_baseline_gap", metrics)
            self.assertFalse(metrics["model_win_claim_allowed"])
            self.assertEqual(metrics["model_win_status"], "blocked_by_autocorrelation_baseline")
            self.assertIn("best baseline is linear_ridge", metrics["model_win_claim_failure_reasons"])
            self.assertEqual(target_scale["target_units"], "normalized_eeg_fixture_units")
            self.assertGreater(target_scale["target_std"], 0.0)
            self.assertGreater(target_scale["target_variance"], 0.0)
            self.assertIn("mse_relative_to_target_variance", target_scale["models"]["linear_ridge"])
            self.assertIn("rmse_relative_to_target_std", target_scale["models"]["linear_ridge"])
            self.assertTrue(np.isfinite(target_scale["models"]["linear_ridge"]["rmse_relative_to_target_std"]))
            self.assertEqual(dataset_summary["dataset"], metrics["dataset"])
            self.assertEqual(dataset_summary["data_source"], metrics["source"])
            self.assertEqual(dataset_summary["n_channels"], 6)
            self.assertGreater(dataset_summary["n_train_windows"], 0)
            self.assertEqual(dataset_summary["split_type"], "subject_held_out")
            self.assertIn("shuffled_target_control", {row["diagnostic_id"] for row in diagnostics["diagnostics"]})
            self.assertIn("autocorrelation_warning", diagnostics["summary"])
            self.assertIn("gate_criteria", gate)
            self.assertEqual(gate["gate_criteria"]["min_forecast_horizon"], 1)
            self.assertEqual(gate["gate_criteria"]["allowed_split_types"], ["session_held_out", "subject_held_out"])
            self.assertTrue(gate["gate_criteria"]["requires_split_audit_passed"])
            self.assertTrue(gate["gate_criteria"]["requires_baseline_table_present"])
            self.assertTrue(gate["gate_criteria"]["requires_finite_metrics"])
            self.assertTrue(gate["gate_criteria"]["requires_calibration_checked"])
            self.assertEqual(gate["gate_criteria"]["allowed_claim_scope"], EEG_V1_CLAIM_SCOPE)
            self.assertFalse(gate["model_win_claim_allowed"])
            self.assertEqual(gate["model_win_status"], "blocked_by_autocorrelation_baseline")
            self.assertIn("best baseline is linear_ridge", gate["model_win_claim_failure_reasons"])
            self.assertIn("long_horizon_delta_vs_short", diagnostic_report)
            self.assertIn("non_overlap_delta_vs_short", diagnostic_report)
            self.assertIn("delta_prediction_delta_vs_short", diagnostic_report)
            self.assertIn("shuffled_control_degrades", diagnostic_report)
            self.assertIn("## Artifact Index", diagnostic_report)
            self.assertIn("| artifact | purpose |", diagnostic_report)
            self.assertIn("| baseline_verification.json |", diagnostic_report)
            self.assertIn("| baseline_checksum_manifest.json |", diagnostic_report)
            self.assertIn("## Checksum Audit", diagnostic_report)
            self.assertIn("baseline_checksum_manifest.json", diagnostic_report)
            self.assertIn("baseline_verification.json", diagnostic_report)
            self.assertIn("scripts/audit_eeg_v1_baseline_checksums.py --artifact-dir <artifact-dir>", diagnostic_report)
            self.assertLess(diagnostic_report.index("## Artifact Index"), diagnostic_report.index("## Checksum Audit"))
            self.assertLess(diagnostic_report.index("## Checksum Audit"), diagnostic_report.index("## Run Config"))
            self.assertIn("## Target Scale Context", diagnostic_report)
            self.assertIn("- target_units: normalized_eeg_fixture_units", diagnostic_report)
            self.assertIn("rmse_relative_to_target_std", diagnostic_report)
            self.assertIn("## Baseline Gap Summary", diagnostic_report)
            self.assertIn("- persistence_gap:", diagnostic_report)
            self.assertIn("- ridge_gap:", diagnostic_report)
            self.assertIn("- best_baseline_gap:", diagnostic_report)
            self.assertIn("## Model Win Claim Status", diagnostic_report)
            self.assertIn("- model_win_claim_allowed: False", diagnostic_report)
            self.assertIn("- model_win_status: blocked_by_autocorrelation_baseline", diagnostic_report)
            self.assertIn("- best baseline is linear_ridge", diagnostic_report)
            self.assertIn("## Metric Breakdown Summary", diagnostic_report)
            self.assertIn("- per_subject_rows:", diagnostic_report)
            self.assertIn("- per_channel_rows:", diagnostic_report)
            self.assertIn("- per_horizon_rows:", diagnostic_report)
            self.assertIn("## Run Config", diagnostic_report)
            self.assertIn(f"- seed: {run_config['seed']}", diagnostic_report)
            self.assertIn(f"- train_steps: {run_config['train_steps']}", diagnostic_report)
            self.assertIn("- models: persistence, linear_ridge, autoregressive_ridge, tiny_ssm", diagnostic_report)
            self.assertIn(f"- window_length: {run_config['window_length']}", diagnostic_report)
            self.assertIn(f"- forecast_horizon: {run_config['forecast_horizon']}", diagnostic_report)
            self.assertIn(f"- data_source: {run_config['data_source']}", diagnostic_report)
            self.assertIn(f"- benchmark_status: {run_config['benchmark_status']}", diagnostic_report)
            self.assertIn(f"- selection_policy: {run_config['selection_policy']}", diagnostic_report)
            self.assertIn(f"- claim_scope: {run_config['claim_scope']}", diagnostic_report)
            self.assertIn("## Evidence Gate Criteria", diagnostic_report)
            self.assertLess(diagnostic_report.index("## Run Config"), diagnostic_report.index("## Evidence Gate Criteria"))
            self.assertIn("- min_forecast_horizon: 1", diagnostic_report)
            self.assertIn("- allowed_split_types: session_held_out, subject_held_out", diagnostic_report)
            self.assertIn("- requires_split_audit_passed: True", diagnostic_report)
            self.assertIn("- requires_baseline_table_present: True", diagnostic_report)
            self.assertIn("- requires_finite_metrics: True", diagnostic_report)
            self.assertIn("- requires_calibration_checked: True", diagnostic_report)
            self.assertIn(f"- allowed_claim_scope: {EEG_V1_CLAIM_SCOPE}", diagnostic_report)
            self.assertGreaterEqual(len(metrics["baseline_ranking"]), 1)
            self.assertIn("subject_id", per_subject_text)
            self.assertIn("persistence", per_subject_text)
            persistence_mses = {
                round(float(row["mse"]), 10)
                for row in per_subject_rows
                if row["model_id"] == "persistence"
            }
            self.assertGreater(len(persistence_mses), 1)
            self.assertTrue(gate["scientific_claim_allowed"], gate["failure_reasons"])

    def test_script_writes_baseline_verification_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )

            verification_path = Path(tmp) / "baseline_verification.json"
            self.assertTrue(verification_path.exists())
            verification = json.loads(verification_path.read_text(encoding="utf-8"))
            checksum_manifest = json.loads((Path(tmp) / "baseline_checksum_manifest.json").read_text(encoding="utf-8"))
            checksum_rows = {row["path"]: row for row in checksum_manifest["artifacts"]}
            verification_bytes = verification_path.read_bytes()

            self.assertEqual(verification["schema"], "kahlus.eeg_v1_baseline_verification.v1")
            self.assertEqual(verification["execution_lane"], "local_cpu_or_single_process_only")
            self.assertFalse(verification["a100_jobs_launched"])
            self.assertEqual(verification["checksum_manifest"], "baseline_checksum_manifest.json")
            self.assertEqual(
                verification["checksum_audit_command"],
                f"PYTHONPATH=src python3 scripts/audit_eeg_v1_baseline_checksums.py --artifact-dir {tmp}",
            )
            self.assertIn("baseline_verification.json", checksum_rows)
            self.assertEqual(checksum_rows["baseline_verification.json"]["bytes"], len(verification_bytes))
            self.assertEqual(
                checksum_rows["baseline_verification.json"]["sha256"],
                hashlib.sha256(verification_bytes).hexdigest(),
            )

    def test_script_writes_baseline_checksum_manifest_for_evidence_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )

            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "kahlus.eeg_v1_baseline_checksums.v1")
            self.assertEqual(manifest["algorithm"], "sha256")
            rows = manifest["artifacts"]
            paths = {row["path"] for row in rows}
            self.assertNotIn("baseline_checksum_manifest.json", paths)
            self.assertTrue(
                {
                    "metrics.json",
                    "baseline_table.json",
                    "baseline_table.csv",
                    "split_audit.json",
                    "evidence_gate.json",
                    "diagnostic_report.md",
                    "autocorrelation_diagnostics.json",
                    "target_scale_context.json",
                }.issubset(paths)
            )
            for row in rows:
                artifact = Path(tmp) / row["path"]
                payload = artifact.read_bytes()
                self.assertEqual(row["bytes"], len(payload), row["path"])
                self.assertEqual(row["sha256"], hashlib.sha256(payload).hexdigest(), row["path"])

    def test_baseline_checksum_audit_script_passes_clean_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
                    "--artifact-dir",
                    tmp,
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

            self.assertEqual(audit.returncode, 0, audit.stderr + audit.stdout)
            payload = json.loads(audit.stdout)
            self.assertTrue(payload["passed"], payload)
            self.assertEqual(payload["schema"], "kahlus.eeg_v1_baseline_checksum_audit.v1")
            self.assertEqual(payload["failure_reasons"], [])
            self.assertGreaterEqual(payload["artifacts_checked"], 8)

    def test_baseline_checksum_audit_requires_core_artifact_manifest_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"] = [
                row for row in manifest["artifacts"] if row["path"] != "metrics.json"
            ]
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
            self.assertIn("missing_manifest_entry:metrics.json", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_invalid_verification_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            verification_path = Path(tmp) / "baseline_verification.json"
            verification = json.loads(verification_path.read_text(encoding="utf-8"))
            verification["a100_jobs_launched"] = True
            verification_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
            self.assertIn("checksum_mismatch:baseline_verification.json", payload["failure_reasons"])
            self.assertIn("invalid_verification_a100_jobs_launched", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_invalid_failure_reasons_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            sidecar_path = Path(tmp) / "failure_reasons.json"
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar.pop("diagnostic_failures")
            sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = sidecar_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "failure_reasons.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
            self.assertIn("missing_failure_reasons_field:diagnostic_failures", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_failure_reasons_gate_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            sidecar_path = Path(tmp) / "failure_reasons.json"
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["gate_failures"] = ["fabricated gate failure"]
            sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = sidecar_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "failure_reasons.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
            self.assertIn("failure_reasons_gate_failures_mismatch", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_failure_reasons_split_mismatch(self):
        dataset = make_synthetic_eeg_v1_dataset(seed=6, n_subjects=9, sessions_per_subject=2)
        task = build_future_forecasting_task(dataset, window_length=8, forecast_horizon=1)
        result = run_eeg_v1_baselines(task, seed=0, train_steps=2, model_ids=("persistence", "linear_ridge"))
        split_audit = dict(audit_eeg_v1_split(dataset, split_type="subject_held_out"))
        split_audit["leakage_passed"] = False
        split_audit["failure_reasons"] = ["forced split audit failure for sidecar consistency"]

        with tempfile.TemporaryDirectory() as tmp:
            write_eeg_v1_artifacts(
                tmp,
                task=task,
                baseline_result=result,
                split_audit=split_audit,
                models=("persistence", "linear_ridge"),
                train_steps=2,
                seed=0,
            )
            sidecar_path = Path(tmp) / "failure_reasons.json"
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["split_audit_failures"] = []
            sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = sidecar_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "failure_reasons.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("failure_reasons_split_audit_failures_mismatch", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_failure_reasons_baseline_and_diagnostic_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,unknown_model",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            sidecar_path = Path(tmp) / "failure_reasons.json"
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            self.assertTrue(sidecar["baseline_failures"])
            self.assertTrue(sidecar["diagnostic_failures"])
            sidecar["baseline_failures"] = []
            sidecar["diagnostic_failures"] = []
            sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = sidecar_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "failure_reasons.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("failure_reasons_baseline_failures_mismatch", payload["failure_reasons"])
        self.assertIn("failure_reasons_diagnostic_failures_mismatch", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_model_win_metrics_gate_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm,neurotwin",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            metrics_path = Path(tmp) / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["model_win_claim_allowed"] = True
            metrics["model_win_status"] = "allowed_model_beats_baselines"
            metrics["model_win_claim_failure_reasons"] = []
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = metrics_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "metrics.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("model_win_metrics_gate_mismatch:model_win_claim_allowed", payload["failure_reasons"])
        self.assertIn("model_win_metrics_gate_mismatch:model_win_status", payload["failure_reasons"])
        self.assertIn("model_win_metrics_gate_mismatch:model_win_claim_failure_reasons", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_gate_finite_metrics_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            gate_path = Path(tmp) / "evidence_gate.json"
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            gate["finite_metrics"] = False
            gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = gate_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "evidence_gate.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("evidence_gate_mismatch:finite_metrics", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_gate_baseline_table_present_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            gate_path = Path(tmp) / "evidence_gate.json"
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            gate["baseline_table_present"] = False
            gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = gate_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "evidence_gate.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("evidence_gate_mismatch:baseline_table_present", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_gate_split_audit_passed_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            gate_path = Path(tmp) / "evidence_gate.json"
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            gate["split_audit_passed"] = False
            gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = gate_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "evidence_gate.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("evidence_gate_mismatch:split_audit_passed", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_gate_dataset_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            gate_path = Path(tmp) / "evidence_gate.json"
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            gate["dataset"] = "public_benchmark"
            gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = gate_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "evidence_gate.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("evidence_gate_mismatch:dataset", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_gate_branch_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            gate_path = Path(tmp) / "evidence_gate.json"
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            gate["branch"] = "v2"
            gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = gate_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "evidence_gate.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("evidence_gate_mismatch:branch", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_gate_claim_scope_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            broad_scope = "diagnoses_depression"
            gate_path = Path(tmp) / "evidence_gate.json"
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            gate["claim_scope"] = broad_scope
            gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            run_config_path = Path(tmp) / "run_config.json"
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            run_config["claim_scope"] = broad_scope
            run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            report = report.replace(
                f"- claim_scope: {EEG_V1_CLAIM_SCOPE}",
                f"- claim_scope: {broad_scope}",
            )
            report_path.write_text(report, encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            changed_paths = {
                "diagnostic_report.md": report_path,
                "evidence_gate.json": gate_path,
                "run_config.json": run_config_path,
            }
            for row in manifest["artifacts"]:
                if row["path"] in changed_paths:
                    payload = changed_paths[row["path"]].read_bytes()
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("evidence_gate_mismatch:claim_scope", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_gate_allowed_claim_scope_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            broad_scope = "diagnoses_depression"
            gate_path = Path(tmp) / "evidence_gate.json"
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            gate["gate_criteria"]["allowed_claim_scope"] = broad_scope
            gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            report = report.replace(
                f"- allowed_claim_scope: {EEG_V1_CLAIM_SCOPE}",
                f"- allowed_claim_scope: {broad_scope}",
            )
            report_path.write_text(report, encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            changed_paths = {
                "diagnostic_report.md": report_path,
                "evidence_gate.json": gate_path,
            }
            for row in manifest["artifacts"]:
                if row["path"] in changed_paths:
                    payload = changed_paths[row["path"]].read_bytes()
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("evidence_gate_mismatch:allowed_claim_scope", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_gate_required_control_criteria_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            gate_path = Path(tmp) / "evidence_gate.json"
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            gate["gate_criteria"]["required_first_class_baselines"] = []
            gate["gate_criteria"]["required_negative_controls"] = []
            gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            report = report.replace("- requires_first_class_ssm_baseline: tiny_ssm", "- requires_first_class_ssm_baseline: ")
            report = report.replace("- requires_negative_control: shuffled_target_control", "- requires_negative_control: ")
            report_path.write_text(report, encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            changed_paths = {
                "diagnostic_report.md": report_path,
                "evidence_gate.json": gate_path,
            }
            for row in manifest["artifacts"]:
                if row["path"] in changed_paths:
                    payload = changed_paths[row["path"]].read_bytes()
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("evidence_gate_mismatch:required_first_class_baselines", payload["failure_reasons"])
        self.assertIn("evidence_gate_mismatch:required_negative_controls", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_gate_shuffled_control_criteria_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            gate_path = Path(tmp) / "evidence_gate.json"
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            gate["gate_criteria"]["requires_shuffled_target_degradation"] = False
            gate["gate_criteria"]["requires_shuffled_target_not_close_to_real_baselines"] = False
            gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            report = report.replace("- requires_shuffled_target_degradation: True", "- requires_shuffled_target_degradation: False")
            report = report.replace(
                "- requires_shuffled_target_not_close_to_real_baselines: True",
                "- requires_shuffled_target_not_close_to_real_baselines: False",
            )
            report_path.write_text(report, encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            changed_paths = {
                "diagnostic_report.md": report_path,
                "evidence_gate.json": gate_path,
            }
            for row in manifest["artifacts"]:
                if row["path"] in changed_paths:
                    payload = changed_paths[row["path"]].read_bytes()
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("evidence_gate_mismatch:requires_shuffled_target_degradation", payload["failure_reasons"])
        self.assertIn(
            "evidence_gate_mismatch:requires_shuffled_target_not_close_to_real_baselines",
            payload["failure_reasons"],
        )

    def test_baseline_checksum_audit_rejects_gate_core_criteria_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            gate_path = Path(tmp) / "evidence_gate.json"
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            gate["gate_criteria"]["min_forecast_horizon"] = 0
            gate["gate_criteria"]["allowed_split_types"] = ["random_window_split"]
            gate["gate_criteria"]["requires_split_audit_passed"] = False
            gate["gate_criteria"]["requires_baseline_table_present"] = False
            gate["gate_criteria"]["requires_finite_metrics"] = False
            gate["gate_criteria"]["requires_calibration_checked"] = False
            gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            report = report.replace("- min_forecast_horizon: 1", "- min_forecast_horizon: 0")
            report = report.replace(
                "- allowed_split_types: session_held_out, subject_held_out",
                "- allowed_split_types: random_window_split",
            )
            report = report.replace("- requires_split_audit_passed: True", "- requires_split_audit_passed: False")
            report = report.replace(
                "- requires_baseline_table_present: True",
                "- requires_baseline_table_present: False",
            )
            report = report.replace("- requires_finite_metrics: True", "- requires_finite_metrics: False")
            report = report.replace(
                "- requires_calibration_checked: True",
                "- requires_calibration_checked: False",
            )
            report_path.write_text(report, encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            changed_paths = {
                "diagnostic_report.md": report_path,
                "evidence_gate.json": gate_path,
            }
            for row in manifest["artifacts"]:
                if row["path"] in changed_paths:
                    payload = changed_paths[row["path"]].read_bytes()
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("evidence_gate_mismatch:min_forecast_horizon", payload["failure_reasons"])
        self.assertIn("evidence_gate_mismatch:allowed_split_types", payload["failure_reasons"])
        self.assertIn("evidence_gate_mismatch:requires_split_audit_passed", payload["failure_reasons"])
        self.assertIn("evidence_gate_mismatch:requires_baseline_table_present", payload["failure_reasons"])
        self.assertIn("evidence_gate_mismatch:requires_finite_metrics", payload["failure_reasons"])
        self.assertIn("evidence_gate_mismatch:requires_calibration_checked", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_coordinated_false_model_win_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for artifact_name in ("metrics.json", "evidence_gate.json"):
                artifact_path = Path(tmp) / artifact_name
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                payload["model_win_claim_allowed"] = True
                payload["model_win_status"] = "allowed_model_beats_baselines"
                payload["model_win_claim_failure_reasons"] = []
                artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                artifact_bytes = artifact_path.read_bytes()
                for row in manifest["artifacts"]:
                    if row["path"] == artifact_name:
                        row["bytes"] = len(artifact_bytes)
                        row["sha256"] = hashlib.sha256(artifact_bytes).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("model_win_recomputed_mismatch:model_win_claim_allowed", payload["failure_reasons"])
        self.assertIn("model_win_recomputed_mismatch:model_win_status", payload["failure_reasons"])
        self.assertIn("model_win_recomputed_mismatch:model_win_claim_failure_reasons", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_best_baseline_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for artifact_name in ("metrics.json", "evidence_gate.json"):
                artifact_path = Path(tmp) / artifact_name
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                if artifact_name == "metrics.json":
                    payload["best_baseline"] = "tiny_ssm"
                    payload["best_baseline_mse"] = payload["metrics_by_model"]["tiny_ssm"]["mse"]
                    payload["best_baseline_gap"] = -0.25
                    payload["kahlus_beats_best_baseline"] = True
                payload["model_win_claim_allowed"] = True
                payload["model_win_status"] = "allowed_model_beats_baselines"
                payload["model_win_claim_failure_reasons"] = []
                artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                artifact_bytes = artifact_path.read_bytes()
                for row in manifest["artifacts"]:
                    if row["path"] == artifact_name:
                        row["bytes"] = len(artifact_bytes)
                        row["sha256"] = hashlib.sha256(artifact_bytes).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("baseline_summary_recomputed_mismatch:best_baseline", payload["failure_reasons"])
        self.assertIn("baseline_summary_recomputed_mismatch:best_baseline_mse", payload["failure_reasons"])
        self.assertIn("baseline_summary_recomputed_mismatch:best_baseline_gap", payload["failure_reasons"])
        self.assertIn("baseline_summary_recomputed_mismatch:kahlus_beats_best_baseline", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_baseline_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            metrics_path = Path(tmp) / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["baseline_ranking"] = [
                {"model_id": "tiny_ssm", "metric": "mse", "value": 0.001, "rank": 1},
                {"model_id": "linear_ridge", "metric": "mse", "value": metrics["metrics_by_model"]["linear_ridge"]["mse"], "rank": 2},
            ]
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = metrics_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "metrics.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("baseline_ranking_recomputed_mismatch", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_baseline_table_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            table_path = Path(tmp) / "baseline_table.json"
            table = json.loads(table_path.read_text(encoding="utf-8"))
            table["rows"][0]["mse"] = 0.001
            table["ranking"] = [{"model_id": "tiny_ssm", "metric": "mse", "value": 0.001, "rank": 1}]
            table_path.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = table_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "baseline_table.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("baseline_table_json_rows_recomputed_mismatch", payload["failure_reasons"])
        self.assertIn("baseline_table_json_ranking_recomputed_mismatch", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_baseline_table_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            table_path = Path(tmp) / "baseline_table.csv"
            rows = list(csv.DictReader(table_path.read_text(encoding="utf-8").splitlines()))
            rows[0]["mse"] = "0.001"
            with table_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("model_id", "mse", "mae", "r2", "pearsonr", "status"))
                writer.writeheader()
                writer.writerows(rows)
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = table_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "baseline_table.csv":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("baseline_table_csv_rows_recomputed_mismatch", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_metrics_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            metrics_csv_path = Path(tmp) / "metrics.csv"
            rows = list(csv.DictReader(metrics_csv_path.read_text(encoding="utf-8").splitlines()))
            rows[0]["value"] = "0.001"
            with metrics_csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("model_id", "metric", "value"))
                writer.writeheader()
                writer.writerows(rows)
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = metrics_csv_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "metrics.csv":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("metrics_csv_recomputed_mismatch", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_granular_metric_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for filename, fieldnames in (
                ("per_subject_metrics.csv", ("subject_id", "model_id", "mse", "mae", "pearsonr", "r2", "n_windows")),
                ("per_channel_metrics.csv", ("channel", "model_id", "mse", "mae", "pearsonr", "r2")),
                ("per_horizon_metrics.csv", ("horizon_index", "model_id", "mse", "mae", "pearsonr", "r2")),
            ):
                path = Path(tmp) / filename
                rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
                rows[0]["mse"] = "0.001"
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                payload = path.read_bytes()
                for row in manifest["artifacts"]:
                    if row["path"] == filename:
                        row["bytes"] = len(payload)
                        row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("per_subject_metrics_csv_recomputed_mismatch", payload["failure_reasons"])
        self.assertIn("per_channel_metrics_csv_recomputed_mismatch", payload["failure_reasons"])
        self.assertIn("per_horizon_metrics_csv_recomputed_mismatch", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_diagnostic_report_metric_breakdown_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            metrics = json.loads((Path(tmp) / "metrics.json").read_text(encoding="utf-8"))
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            for field, metric_key in (
                ("per_subject_rows", "per_subject_metrics"),
                ("per_channel_rows", "per_channel_metrics"),
                ("per_horizon_rows", "per_horizon_metrics"),
            ):
                original = f"- {field}: {len(metrics[metric_key])}"
                self.assertIn(original, report)
                report = report.replace(original, f"- {field}: 999")
            report_path.write_text(report, encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:per_subject_rows", payload["failure_reasons"])
        self.assertIn("diagnostic_report_mismatch:per_channel_rows", payload["failure_reasons"])
        self.assertIn("diagnostic_report_mismatch:per_horizon_rows", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_extra_diagnostic_report_metric_breakdown_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            marker = "\n\n## Evidence Gate Criteria"
            self.assertIn(marker, report)
            report_path.write_text(
                report.replace(marker, "\n- fake_sidecar_rows: 999\n\n## Evidence Gate Criteria"),
                encoding="utf-8",
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:metric_breakdown_section", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_diagnostic_report_metric_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            original = "- detailed_sidecars: per_subject_metrics.csv, per_channel_metrics.csv, per_horizon_metrics.csv"
            self.assertIn(original, report)
            report_path.write_text(report.replace(original, "- detailed_sidecars: metrics.csv"), encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:detailed_sidecars", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_diagnostic_report_dataset_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            summary = json.loads((Path(tmp) / "dataset_summary.json").read_text(encoding="utf-8"))
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            originals = {
                "dataset_summary_n_test_windows": f"- n_test_windows: {summary['n_test_windows']}",
                "dataset_summary_n_channels": f"- n_channels: {summary['n_channels']}",
            }
            for original in originals.values():
                self.assertIn(original, report)
                report = report.replace(original, original.rsplit(" ", 1)[0] + " 999")
            marker = "\n\n## Target Scale Context"
            self.assertIn(marker, report)
            report_path.write_text(
                report.replace(marker, "\n- fake_dataset_windows: 999\n\n## Target Scale Context"),
                encoding="utf-8",
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:dataset_summary_n_test_windows", payload["failure_reasons"])
        self.assertIn("diagnostic_report_mismatch:dataset_summary_n_channels", payload["failure_reasons"])
        self.assertIn("diagnostic_report_mismatch:dataset_summary_section", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_autocorr_diagnostics_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            diagnostics_csv_path = Path(tmp) / "autocorrelation_diagnostics.csv"
            rows = list(csv.DictReader(diagnostics_csv_path.read_text(encoding="utf-8").splitlines()))
            rows[0]["best_mse"] = "0.001"
            with diagnostics_csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "diagnostic_id",
                        "status",
                        "window_length",
                        "forecast_horizon",
                        "window_stride",
                        "n_train_windows",
                        "n_test_windows",
                        "best_model",
                        "best_mse",
                        "persistence_mse",
                        "linear_ridge_mse",
                        "tiny_ssm_mse",
                        "leakage_passed",
                        "reason",
                    ),
                )
                writer.writeheader()
                writer.writerows(rows)
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = diagnostics_csv_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "autocorrelation_diagnostics.csv":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("autocorrelation_diagnostics_csv_recomputed_mismatch", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_extra_diagnostic_report_summary_header_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            marker = "\n\n## Artifact Index"
            self.assertIn(marker, report)
            report_path.write_text(
                report.replace(marker, "\n- fake_model_win_ready: True\n\n## Artifact Index"),
                encoding="utf-8",
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:summary_header_section", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_diagnostic_report_claim_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            report = report.replace("- scientific_claim_allowed: True", "- scientific_claim_allowed: False")
            report = report.replace("- best_baseline: linear_ridge", "- best_baseline: tiny_ssm")
            report = report.replace("- kahlus_beats_best_baseline: False", "- kahlus_beats_best_baseline: True")
            report_path.write_text(report, encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:best_baseline", payload["failure_reasons"])
        self.assertIn("diagnostic_report_mismatch:kahlus_beats_best_baseline", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_duplicate_diagnostic_report_claim_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            duplicate_line = "- best_baseline: linear_ridge"
            self.assertEqual(report.splitlines().count(duplicate_line), 1)
            report_path.write_text(report + duplicate_line + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_duplicate:best_baseline", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_extra_diagnostic_report_claim_boundary_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            marker = (
                "- It does not support diagnosis, treatment, epilepsy detection, depression detection, "
                "foundation-model, SOTA, v2, or v3 claims."
            )
            self.assertIn(marker, report)
            report_path.write_text(
                report.replace(marker, marker + "\n- It diagnoses depression from EEG forecasting."),
                encoding="utf-8",
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:claim_boundaries_section", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_diagnostic_report_gate_criteria(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            report = report.replace("- min_forecast_horizon: 1", "- min_forecast_horizon: 0")
            report = report.replace(
                "- allowed_split_types: session_held_out, subject_held_out",
                "- allowed_split_types: random_window_split",
            )
            report = report.replace("- requires_split_audit_passed: True", "- requires_split_audit_passed: False")
            report = report.replace(
                "- requires_first_class_ssm_baseline: tiny_ssm",
                "- requires_first_class_ssm_baseline: ",
            )
            report = report.replace(
                "- requires_negative_control: shuffled_target_control",
                "- requires_negative_control: ",
            )
            report_path.write_text(report, encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:min_forecast_horizon", payload["failure_reasons"])
        self.assertIn("diagnostic_report_mismatch:allowed_split_types", payload["failure_reasons"])
        self.assertIn("diagnostic_report_mismatch:requires_split_audit_passed", payload["failure_reasons"])
        self.assertIn("diagnostic_report_mismatch:required_first_class_baselines", payload["failure_reasons"])
        self.assertIn("diagnostic_report_mismatch:required_negative_controls", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_extra_diagnostic_report_gate_criteria_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            marker = "\n\n## Autocorrelation Diagnostics"
            self.assertIn(marker, report)
            report_path.write_text(
                report.replace(marker, "\n- fake_gate_passed: True\n\n## Autocorrelation Diagnostics"),
                encoding="utf-8",
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:gate_criteria_section", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_diagnostic_report_model_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            report = report.replace("- model_win_claim_allowed: False", "- model_win_claim_allowed: True")
            report = report.replace(
                "- model_win_status: blocked_by_autocorrelation_baseline",
                "- model_win_status: allowed_model_beats_baselines",
            )
            report = report.replace("- best baseline is linear_ridge", "- report claims model beat baselines")
            report_path.write_text(report, encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:model_win_claim_allowed", payload["failure_reasons"])
        self.assertIn("diagnostic_report_mismatch:model_win_status", payload["failure_reasons"])
        self.assertIn("diagnostic_report_mismatch:model_win_claim_failure_reason", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_extra_diagnostic_report_model_win_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("- best baseline is linear_ridge\n", report)
            report = report.replace(
                "- best baseline is linear_ridge\n",
                "- best baseline is linear_ridge\n- report claims model beat baselines\n",
            )
            report_path.write_text(report, encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:model_win_claim_failure_reasons", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_extra_diagnostic_report_model_win_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            marker = "- model_win_claim_failure_reasons:"
            self.assertIn(marker, report)
            report_path.write_text(
                report.replace(marker, "- calibrated_model_win_override: True\n" + marker),
                encoding="utf-8",
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:model_win_section", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_diagnostic_report_target_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            target_scale = json.loads((Path(tmp) / "target_scale_context.json").read_text(encoding="utf-8"))
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            report = report.replace(
                f"- target_units: {target_scale['target_units']}",
                "- target_units: raw_microvolts",
            )
            report = report.replace(
                f"- target_std: {target_scale['target_std']:.6g}",
                "- target_std: 0.001",
            )
            report = report.replace(
                f"- target_variance: {target_scale['target_variance']:.6g}",
                "- target_variance: 0.001",
            )
            report_path.write_text(report, encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:target_units", payload["failure_reasons"])
        self.assertIn("diagnostic_report_mismatch:target_std", payload["failure_reasons"])
        self.assertIn("diagnostic_report_mismatch:target_variance", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_extra_diagnostic_report_target_scale_header_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            marker = "\n\n| model_id | rmse | rmse_relative_to_target_std | mse_relative_to_target_variance |"
            self.assertIn(marker, report)
            report_path.write_text(
                report.replace(
                    marker,
                    "\n- fake_scale_note: model rows look normalized\n\n"
                    "| model_id | rmse | rmse_relative_to_target_std | mse_relative_to_target_variance |",
                ),
                encoding="utf-8",
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:target_scale_header_lines", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_diagnostic_report_target_scale_model_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            target_scale = json.loads((Path(tmp) / "target_scale_context.json").read_text(encoding="utf-8"))
            values = target_scale["models"]["linear_ridge"]
            original_line = (
                "| "
                f"linear_ridge | "
                f"{values['rmse']:.6g} | "
                f"{values['rmse_relative_to_target_std']:.6g} | "
                f"{values['mse_relative_to_target_variance']:.6g} |"
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            self.assertIn(original_line, report)
            report_path.write_text(report.replace(original_line, "| linear_ridge | 0.001 | 0.001 | 0.001 |"), encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:target_scale_model:linear_ridge", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_extra_diagnostic_report_target_scale_model_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            target_scale = json.loads((Path(tmp) / "target_scale_context.json").read_text(encoding="utf-8"))
            values = target_scale["models"]["linear_ridge"]
            original_line = (
                "| "
                f"linear_ridge | "
                f"{values['rmse']:.6g} | "
                f"{values['rmse_relative_to_target_std']:.6g} | "
                f"{values['mse_relative_to_target_variance']:.6g} |"
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            self.assertIn(original_line + "\n", report)
            report_path.write_text(
                report.replace(original_line + "\n", original_line + "\n| linear_ridge | 0.001 | 0.001 | 0.001 |\n"),
                encoding="utf-8",
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:target_scale_model_rows", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_diagnostic_report_autocorr_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            diagnostics = json.loads((Path(tmp) / "autocorrelation_diagnostics.json").read_text(encoding="utf-8"))
            summary = diagnostics["summary"]
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            report = report.replace(
                f"| short_horizon_best_mse | {summary['short_horizon_best_mse']:.6g} |",
                "| short_horizon_best_mse | 0.001 |",
            )
            report = report.replace(
                f"| persistence_or_ridge_dominates | {summary['persistence_or_ridge_dominates']} |",
                "| persistence_or_ridge_dominates | False |",
            )
            report = report.replace(
                f"| shuffled_target_close_to_real_baselines | {summary['shuffled_target_close_to_real_baselines']} |",
                "| shuffled_target_close_to_real_baselines | True |",
            )
            report = report.replace(
                f"| shuffled_control_degrades | {summary['shuffled_control_degrades']} |",
                "| shuffled_control_degrades | False |",
            )
            report = report.replace(
                f"| long_horizon_delta_vs_short | {summary['long_horizon_delta_vs_short']:.6g} |",
                "| long_horizon_delta_vs_short | 0.001 |",
            )
            report = report.replace(
                f"| non_overlap_delta_vs_short | {summary['non_overlap_delta_vs_short']:.6g} |",
                "| non_overlap_delta_vs_short | 0.001 |",
            )
            report = report.replace(
                f"| delta_prediction_delta_vs_short | {summary['delta_prediction_delta_vs_short']:.6g} |",
                "| delta_prediction_delta_vs_short | 0.001 |",
            )
            report = report.replace(
                f"| verdict | {summary['verdict']} |",
                "| verdict | autocorrelation_not_detected |",
            )
            report_path.write_text(report, encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:autocorrelation_short_horizon_best_mse", payload["failure_reasons"])
        self.assertIn(
            "diagnostic_report_mismatch:autocorrelation_persistence_or_ridge_dominates",
            payload["failure_reasons"],
        )
        self.assertIn(
            "diagnostic_report_mismatch:autocorrelation_shuffled_target_close_to_real_baselines",
            payload["failure_reasons"],
        )
        self.assertIn(
            "diagnostic_report_mismatch:autocorrelation_shuffled_control_degrades",
            payload["failure_reasons"],
        )
        self.assertIn(
            "diagnostic_report_mismatch:autocorrelation_long_horizon_delta_vs_short",
            payload["failure_reasons"],
        )
        self.assertIn(
            "diagnostic_report_mismatch:autocorrelation_non_overlap_delta_vs_short",
            payload["failure_reasons"],
        )
        self.assertIn(
            "diagnostic_report_mismatch:autocorrelation_delta_prediction_delta_vs_short",
            payload["failure_reasons"],
        )
        self.assertIn("diagnostic_report_mismatch:autocorrelation_verdict", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_extra_diagnostic_report_autocorr_summary_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            marker = "\n\n\n### Baseline Dominance"
            self.assertIn(marker, report)
            report_path.write_text(
                report.replace(marker, "\n| fake_autocorr_cleared | True |\n\n\n### Baseline Dominance"),
                encoding="utf-8",
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:autocorrelation_summary_rows", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_extra_diagnostic_report_autocorr_intro_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            marker = "\n\n### Summary"
            self.assertIn(marker, report)
            report_path.write_text(
                report.replace(marker, "\n- autocorrelation_cleared: True\n\n### Summary"),
                encoding="utf-8",
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:autocorrelation_intro_lines", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_extra_diagnostic_report_autocorr_dominance_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            marker = "\n\n| diagnostic | status | best_model | best_mse | persistence_mse | linear_ridge_mse | tiny_ssm_mse | reason |"
            self.assertIn(marker, report)
            report_path.write_text(
                report.replace(marker, "\n- autocorrelation_cleared: True" + marker),
                encoding="utf-8",
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:autocorrelation_dominance_lines", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_diagnostic_report_autocorr_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            diagnostics = json.loads((Path(tmp) / "autocorrelation_diagnostics.json").read_text(encoding="utf-8"))
            row = next(item for item in diagnostics["diagnostics"] if item["diagnostic_id"] == "shuffled_target_control")
            def report_value(value):
                return "" if value is None else f"{float(value):.6g}"

            original_line = (
                f"| {row['diagnostic_id']} | {row['status']} | {row.get('best_model', '')} | "
                f"{row['best_mse']:.6g} | {report_value(row['persistence_mse'])} | "
                f"{report_value(row['linear_ridge_mse'])} | {report_value(row['tiny_ssm_mse'])} | "
                f"{row.get('reason', '')} |"
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            self.assertIn(original_line, report)
            report = report.replace(
                original_line,
                original_line.replace("control_degraded", "control_passed").replace(f"{row['best_mse']:.6g}", "0.001"),
            )
            report_path.write_text(report, encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for manifest_row in manifest["artifacts"]:
                if manifest_row["path"] == "diagnostic_report.md":
                    manifest_row["bytes"] = len(payload)
                    manifest_row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
            "diagnostic_report_mismatch:autocorrelation_row:shuffled_target_control",
            payload["failure_reasons"],
        )

    def test_baseline_checksum_audit_rejects_extra_diagnostic_report_autocorr_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            marker = "\n\n## Claim Boundaries"
            self.assertIn(marker, report)
            report_path.write_text(
                report.replace(
                    marker,
                    "\n| fake_control | completed | fake_model | 0.001 |  |  |  |  |\n\n## Claim Boundaries",
                ),
                encoding="utf-8",
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for manifest_row in manifest["artifacts"]:
                if manifest_row["path"] == "diagnostic_report.md":
                    manifest_row["bytes"] = len(payload)
                    manifest_row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:autocorrelation_rows", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_diagnostic_report_artifact_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            artifact_line = (
                "| autocorrelation_diagnostics.json | optional autocorrelation control diagnostics |"
            )
            self.assertIn(artifact_line, report)
            report_path.write_text(report.replace(artifact_line + "\n", ""), encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
            "diagnostic_report_mismatch:artifact_index:autocorrelation_diagnostics.json",
            payload["failure_reasons"],
        )

    def test_baseline_checksum_audit_rejects_extra_diagnostic_report_artifact_index_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            marker = "\n\n## Checksum Audit"
            self.assertIn(marker, report)
            report_path.write_text(
                report.replace(marker, "\n| fake_artifact.json | fake reviewer artifact |\n" + marker),
                encoding="utf-8",
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:artifact_index_rows", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_diagnostic_report_verification_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            report = report.replace(
                "- execution_lane: local_cpu_or_single_process_only",
                "- execution_lane: a100_cluster",
            )
            report = report.replace("- a100_jobs_launched: False", "- a100_jobs_launched: True")
            report_path.write_text(report, encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:execution_lane", payload["failure_reasons"])
        self.assertIn("diagnostic_report_mismatch:a100_jobs_launched", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_extra_diagnostic_report_checksum_audit_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            marker = "\n\n## Method Order"
            self.assertIn(marker, report)
            report_path.write_text(
                report.replace(marker, "\n- fake_cluster_ready: True\n\n## Method Order"),
                encoding="utf-8",
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:checksum_audit_section", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_diagnostic_report_method_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,neurotwin",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            report = report.replace("| 1 | persistence | baseline |", "| 1 | neurotwin | main_model |")
            report = report.replace("| 3 | neurotwin | main_model |", "| 3 | persistence | baseline |")
            report_path.write_text(report, encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:method_order:persistence", payload["failure_reasons"])
        self.assertIn("diagnostic_report_mismatch:method_order:neurotwin", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_extra_diagnostic_report_method_order_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,neurotwin",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            report = report.replace(
                "| 3 | neurotwin | main_model |\n\n## Baseline Ranking",
                "| 3 | neurotwin | main_model |\n| 4 | fake_ssm | baseline |\n\n## Baseline Ranking",
            )
            report_path.write_text(report, encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:method_order_rows", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_diagnostic_report_run_config_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            run_config = json.loads((Path(tmp) / "run_config.json").read_text(encoding="utf-8"))
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            start = report.index("## Run Config")
            end = report.index("## Target Scale Context", start)
            section = report[start:end]
            sampling_rate = run_config.get("sampling_rate_hz")
            sampling_rate_text = "" if sampling_rate is None else f"{float(sampling_rate):.6g}"
            replacements = {
                f"- seed: {run_config['seed']}": "- seed: 999",
                f"- train_steps: {run_config['train_steps']}": "- train_steps: 999",
                f"- models: {', '.join(run_config['models'])}": "- models: neurotwin",
                f"- window_length: {run_config['window_length']}": "- window_length: 999",
                f"- forecast_horizon: {run_config['forecast_horizon']}": "- forecast_horizon: 999",
                f"- sampling_rate_hz: {sampling_rate_text}": "- sampling_rate_hz: 999",
                f"- data_source: {run_config['data_source']}": "- data_source: public_eeg",
                f"- benchmark_status: {run_config['benchmark_status']}": "- benchmark_status: public_benchmark",
                f"- selection_policy: {run_config['selection_policy']}": "- selection_policy: best_model_wins",
                f"- claim_scope: {run_config['claim_scope']}": "- claim_scope: model_beats_baselines",
            }
            for original, replacement in replacements.items():
                section = section.replace(original, replacement)
            report_path.write_text(report[:start] + section + report[end:], encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        for field in (
            "run_config_seed",
            "run_config_train_steps",
            "run_config_models",
            "run_config_window_length",
            "run_config_forecast_horizon",
            "run_config_sampling_rate_hz",
            "run_config_data_source",
            "run_config_benchmark_status",
            "run_config_selection_policy",
            "run_config_claim_scope",
            "run_config_section",
        ):
            self.assertIn(f"diagnostic_report_mismatch:{field}", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_diagnostic_report_baseline_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            metrics = json.loads((Path(tmp) / "metrics.json").read_text(encoding="utf-8"))
            best = metrics["baseline_ranking"][0]
            original_line = f"| {best['rank']} | {best['model_id']} | {best['value']:.6g} |"
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            self.assertIn(original_line, report)
            report_path.write_text(report.replace(original_line, "| 1 | neurotwin | 0.001 |"), encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
            f"diagnostic_report_mismatch:baseline_ranking:{best['model_id']}",
            payload["failure_reasons"],
        )

    def test_baseline_checksum_audit_rejects_extra_diagnostic_report_baseline_ranking_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            metrics = json.loads((Path(tmp) / "metrics.json").read_text(encoding="utf-8"))
            last = metrics["baseline_ranking"][-1]
            last_line = f"| {last['rank']} | {last['model_id']} | {last['value']:.6g} |"
            report = report_path.read_text(encoding="utf-8")
            self.assertIn(last_line, report)
            report = report.replace(last_line, f"{last_line}\n| 99 | fake_model | 0.0001 |")
            report_path.write_text(report, encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:baseline_ranking_rows", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_diagnostic_report_baseline_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            metrics = json.loads((Path(tmp) / "metrics.json").read_text(encoding="utf-8"))
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")

            def report_value(value):
                return "" if value is None else f"{float(value):.6g}"

            for field in ("persistence_gap", "ridge_gap", "best_baseline_gap"):
                expected_line = f"- {field}: {report_value(metrics.get(field))}"
                self.assertIn(expected_line, report)
                report = report.replace(expected_line, f"- {field}: 999")
            report_path.write_text(report, encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:persistence_gap", payload["failure_reasons"])
        self.assertIn("diagnostic_report_mismatch:ridge_gap", payload["failure_reasons"])
        self.assertIn("diagnostic_report_mismatch:best_baseline_gap", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_extra_diagnostic_report_baseline_gap_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = Path(tmp) / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            marker = "\n\n## Model Win Claim Status"
            self.assertIn(marker, report)
            report_path.write_text(
                report.replace(marker, "\n- calibrated_gap_win: True\n\n## Model Win Claim Status"),
                encoding="utf-8",
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("diagnostic_report_mismatch:baseline_gap_section", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_target_scale_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            target_scale_path = Path(tmp) / "target_scale_context.json"
            target_scale = json.loads(target_scale_path.read_text(encoding="utf-8"))
            target_scale["models"]["linear_ridge"]["rmse"] = 0.001
            target_scale["models"]["linear_ridge"]["rmse_relative_to_target_std"] = 0.001
            target_scale["models"]["linear_ridge"]["mse_relative_to_target_variance"] = 0.001
            target_scale_path.write_text(json.dumps(target_scale, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = target_scale_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "target_scale_context.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("target_scale_context_mismatch:linear_ridge:rmse", payload["failure_reasons"])
        self.assertIn(
            "target_scale_context_mismatch:linear_ridge:rmse_relative_to_target_std",
            payload["failure_reasons"],
        )
        self.assertIn(
            "target_scale_context_mismatch:linear_ridge:mse_relative_to_target_variance",
            payload["failure_reasons"],
        )

    def test_baseline_checksum_audit_rejects_target_scale_variance_std_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            metrics = json.loads((Path(tmp) / "metrics.json").read_text(encoding="utf-8"))
            target_scale_path = Path(tmp) / "target_scale_context.json"
            target_scale = json.loads(target_scale_path.read_text(encoding="utf-8"))
            target_scale["target_variance"] = float(target_scale["target_variance"]) * 10.0
            for model_id, values in metrics["metrics_by_model"].items():
                target_scale["models"][model_id]["mse_relative_to_target_variance"] = (
                    values["mse"] / target_scale["target_variance"]
                )
            target_scale_path.write_text(json.dumps(target_scale, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = target_scale_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "target_scale_context.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("target_scale_context_mismatch:target_variance", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_dataset_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            summary_path = Path(tmp) / "dataset_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["benchmark_status"] = "public_benchmark"
            summary["split_type"] = "random_window_split"
            summary["n_test_subjects"] = 99
            summary["window_length"] = 99
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = summary_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "dataset_summary.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("dataset_summary_mismatch:benchmark_status", payload["failure_reasons"])
        self.assertIn("dataset_summary_mismatch:split_type", payload["failure_reasons"])
        self.assertIn("dataset_summary_mismatch:n_test_subjects", payload["failure_reasons"])
        self.assertIn("dataset_summary_mismatch:window_length", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_dataset_summary_granular_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            summary_path = Path(tmp) / "dataset_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["n_test_windows"] += 1
            summary["n_channels"] += 1
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = summary_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "dataset_summary.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("dataset_summary_mismatch:n_test_windows", payload["failure_reasons"])
        self.assertIn("dataset_summary_mismatch:n_channels", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_run_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            run_config_path = Path(tmp) / "run_config.json"
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            run_config["dataset"] = "hbn_eeg"
            run_config["benchmark_status"] = "public_benchmark"
            run_config["claim_scope"] = "model_beats_baselines"
            run_config["models"] = ["persistence", "linear_ridge"]
            run_config["window_length"] = 99
            run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = run_config_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "run_config.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("run_config_mismatch:dataset", payload["failure_reasons"])
        self.assertIn("run_config_mismatch:benchmark_status", payload["failure_reasons"])
        self.assertIn("run_config_mismatch:claim_scope", payload["failure_reasons"])
        self.assertIn("run_config_mismatch:models", payload["failure_reasons"])
        self.assertIn("run_config_mismatch:window_length", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_tampered_split_audit_subject_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            split_path = Path(tmp) / "split_audit.json"
            split_audit = json.loads(split_path.read_text(encoding="utf-8"))
            split_audit["train_subjects"][0] = split_audit["test_subjects"][0]
            split_audit["subject_overlap"] = False
            split_audit["leakage_passed"] = True
            split_path.write_text(json.dumps(split_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = split_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "split_audit.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("split_audit_mismatch:subject_overlap", payload["failure_reasons"])
        self.assertIn("split_audit_mismatch:leakage_passed", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_autocorr_gate_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            diagnostics_path = Path(tmp) / "autocorrelation_diagnostics.json"
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            diagnostics["summary"]["shuffled_control_degrades"] = False
            diagnostics["summary"]["shuffled_target_close_to_real_baselines"] = True
            diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = diagnostics_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "autocorrelation_diagnostics.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("autocorrelation_gate_mismatch:shuffled_control_degrades", payload["failure_reasons"])
        self.assertIn("autocorrelation_gate_mismatch:shuffled_target_close_to_real_baselines", payload["failure_reasons"])

    def test_baseline_checksum_audit_requires_autocorr_manifest_entry_when_gate_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"] = [
                row for row in manifest["artifacts"] if row["path"] != "autocorrelation_diagnostics.json"
            ]
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
        self.assertIn("gate_requires_manifest_entry:autocorrelation_diagnostics.json", payload["failure_reasons"])

    def test_baseline_checksum_audit_rejects_missing_required_autocorr_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            diagnostics_path = Path(tmp) / "autocorrelation_diagnostics.json"
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            diagnostics["diagnostics"] = [
                row for row in diagnostics["diagnostics"] if row.get("diagnostic_id") != "shuffled_target_control"
            ]
            diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = diagnostics_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "autocorrelation_diagnostics.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
            "autocorrelation_gate_mismatch:required_negative_control:shuffled_target_control",
            payload["failure_reasons"],
        )

    def test_baseline_checksum_audit_rejects_non_train_only_shuffled_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            diagnostics_path = Path(tmp) / "autocorrelation_diagnostics.json"
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            for row in diagnostics["diagnostics"]:
                if row.get("diagnostic_id") == "shuffled_target_control":
                    row["shuffle_boundary"] = "all_splits"
                    row["validation_targets_shuffled"] = True
                    row["test_targets_shuffled"] = True
            diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = diagnostics_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "autocorrelation_diagnostics.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
            "autocorrelation_gate_mismatch:shuffled_target_control_not_train_split_only",
            payload["failure_reasons"],
        )

    def test_baseline_checksum_audit_rejects_missing_shuffled_control_seed_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            diagnostics_path = Path(tmp) / "autocorrelation_diagnostics.json"
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            for row in diagnostics["diagnostics"]:
                if row.get("diagnostic_id") == "shuffled_target_control":
                    row.pop("shuffle_seed", None)
                    row.pop("shuffle_seed_source", None)
            diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = diagnostics_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "autocorrelation_diagnostics.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
            "autocorrelation_gate_mismatch:shuffled_target_control_missing_seed_provenance",
            payload["failure_reasons"],
        )

    def test_baseline_checksum_audit_rejects_shuffled_control_seed_run_config_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            diagnostics_path = Path(tmp) / "autocorrelation_diagnostics.json"
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            for row in diagnostics["diagnostics"]:
                if row.get("diagnostic_id") == "shuffled_target_control":
                    row["shuffle_seed"] = 99999
            diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = diagnostics_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "autocorrelation_diagnostics.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
            "autocorrelation_gate_mismatch:shuffled_target_control_seed_run_config_mismatch",
            payload["failure_reasons"],
        )

    def test_baseline_checksum_audit_rejects_autocorr_row_metric_field_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            diagnostics_path = Path(tmp) / "autocorrelation_diagnostics.json"
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            for row in diagnostics["diagnostics"]:
                if row.get("diagnostic_id") == "short_horizon_overlap":
                    row["metrics_by_model"]["tiny_ssm"]["mse"] = 0.001
            diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = diagnostics_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "autocorrelation_diagnostics.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
            "autocorrelation_gate_mismatch:diagnostic_metric:short_horizon_overlap:tiny_ssm_mse",
            payload["failure_reasons"],
        )
        self.assertIn(
            "autocorrelation_gate_mismatch:diagnostic_best_model:short_horizon_overlap",
            payload["failure_reasons"],
        )

    def test_baseline_checksum_audit_rejects_autocorr_summary_mse_row_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            diagnostics_path = Path(tmp) / "autocorrelation_diagnostics.json"
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            diagnostics["summary"]["short_horizon_best_mse"] = 0.001
            diagnostics["summary"]["tiny_ssm_mse"] = 0.002
            diagnostics["summary"]["shuffled_target_control_mse"] = 0.003
            diagnostics["summary"]["shuffled_target_best_mse"] = 0.004
            diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = diagnostics_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "autocorrelation_diagnostics.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
            "autocorrelation_gate_mismatch:summary_short_horizon_best_mse",
            payload["failure_reasons"],
        )
        self.assertIn(
            "autocorrelation_gate_mismatch:summary_tiny_ssm_mse",
            payload["failure_reasons"],
        )
        self.assertIn(
            "autocorrelation_gate_mismatch:summary_shuffled_target_control_mse",
            payload["failure_reasons"],
        )
        self.assertIn(
            "autocorrelation_gate_mismatch:summary_shuffled_target_best_mse",
            payload["failure_reasons"],
        )

    def test_baseline_checksum_audit_rejects_autocorr_summary_delta_row_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            diagnostics_path = Path(tmp) / "autocorrelation_diagnostics.json"
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            diagnostics["summary"]["long_horizon_delta_vs_short"] = 0.001
            diagnostics["summary"]["non_overlap_delta_vs_short"] = 0.002
            diagnostics["summary"]["delta_prediction_delta_vs_short"] = 0.003
            diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = diagnostics_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "autocorrelation_diagnostics.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
            "autocorrelation_gate_mismatch:summary_long_horizon_delta_vs_short",
            payload["failure_reasons"],
        )
        self.assertIn(
            "autocorrelation_gate_mismatch:summary_non_overlap_delta_vs_short",
            payload["failure_reasons"],
        )
        self.assertIn(
            "autocorrelation_gate_mismatch:summary_delta_prediction_delta_vs_short",
            payload["failure_reasons"],
        )

    def test_baseline_checksum_audit_rejects_autocorr_summary_boolean_row_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge,tiny_ssm",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            diagnostics_path = Path(tmp) / "autocorrelation_diagnostics.json"
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            diagnostics["summary"]["shuffled_control_degrades"] = False
            diagnostics["summary"]["persistence_or_ridge_dominates"] = False
            diagnostics["summary"]["shuffled_target_close_to_real_baselines"] = True
            diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = diagnostics_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "autocorrelation_diagnostics.json":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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
            "autocorrelation_gate_mismatch:summary_shuffled_control_degrades",
            payload["failure_reasons"],
        )
        self.assertIn(
            "autocorrelation_gate_mismatch:summary_persistence_or_ridge_dominates",
            payload["failure_reasons"],
        )
        self.assertIn(
            "autocorrelation_gate_mismatch:summary_shuffled_target_close_to_real_baselines",
            payload["failure_reasons"],
        )

    def test_baseline_checksum_audit_rejects_unexpected_manifest_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--seed",
                    "0",
                    "--train-steps",
                    "2",
                    "--models",
                    "persistence,linear_ridge",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            manifest_path = Path(tmp) / "baseline_checksum_manifest.json"
            unexpected_path = Path(tmp) / "unexpected_extra_evidence.json"
            unexpected_payload = b'{"note":"unexpected baseline manifest expansion"}\n'
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
                    "scripts/audit_eeg_v1_baseline_checksums.py",
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

    def test_hbn_missing_data_root_fails_clearly(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_eeg_v1_baselines.py",
                "--dataset",
                "hbn_eeg",
                "--out-dir",
                tempfile.gettempdir(),
            ],
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "PYTHONPATH": "src"},
            text=True,
            capture_output=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("HBN-EEG data root not found. Provide --data-root.", result.stderr + result.stdout)

    def test_script_reports_invalid_task_config_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "synthetic_fixture",
                    "--out-dir",
                    tmp,
                    "--window-length",
                    "1000",
                    "--models",
                    "persistence",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

        combined = result.stderr + result.stdout
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("EEG v1 task configuration invalid:", combined)
        self.assertIn("future forecasting task requires nonempty train and test windows", combined)
        self.assertNotIn("Traceback", combined)

    def test_hbn_manifest_rejects_relative_paths_outside_data_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hbn"
            root.mkdir()
            outside = Path(tmp) / "outside.npy"
            np.save(outside, np.ones((20, 2), dtype=np.float32))
            (root / "manifest.jsonl").write_text(
                json.dumps({"path": "../outside.npy", "subject_id": "sub-001"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "escapes data root"):
                load_hbn_eeg_local_dataset(root)

    def test_hbn_manifest_rejects_unsupported_signal_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hbn"
            root.mkdir()
            signal = root / "signal.txt"
            signal.write_text("not a numpy signal\n", encoding="utf-8")
            (root / "manifest.jsonl").write_text(
                json.dumps({"path": "signal.txt", "subject_id": "sub-001"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unsupported signal file extension"):
                load_hbn_eeg_local_dataset(root)

    def test_hbn_manifest_rejects_malformed_json_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hbn"
            root.mkdir()
            (root / "manifest.jsonl").write_text('{"path": "signal.npy"\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "manifest line 1 is not valid JSON"):
                load_hbn_eeg_local_dataset(root)

    def test_hbn_manifest_rejects_non_object_json_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hbn"
            root.mkdir()
            (root / "manifest.jsonl").write_text(json.dumps(["signal.npy", "sub-001"]) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "manifest line 1 must be a JSON object"):
                load_hbn_eeg_local_dataset(root)

    def test_hbn_manifest_rejects_nonpositive_sampling_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hbn"
            root.mkdir()
            np.save(root / "signal.npy", np.ones((20, 2), dtype=np.float32))
            (root / "manifest.jsonl").write_text(
                json.dumps({"path": "signal.npy", "subject_id": "sub-001", "sampling_rate": 0.0}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "sampling_rate must be finite and > 0"):
                load_hbn_eeg_local_dataset(root)

    def test_hbn_manifest_rejects_inconsistent_sampling_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hbn"
            root.mkdir()
            np.save(root / "sub-001.npy", np.ones((20, 2), dtype=np.float32))
            np.save(root / "sub-002.npy", np.ones((20, 2), dtype=np.float32))
            (root / "manifest.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {"path": "sub-001.npy", "subject_id": "sub-001", "sampling_rate": 128.0}
                        ),
                        json.dumps(
                            {"path": "sub-002.npy", "subject_id": "sub-002", "sampling_rate": 256.0}
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "consistent sampling_rate"):
                load_hbn_eeg_local_dataset(root)

    def test_hbn_manifest_rejects_nonfinite_signal_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hbn"
            root.mkdir()
            signal = np.ones((20, 2), dtype=np.float32)
            signal[0, 0] = np.nan
            np.save(root / "signal.npy", signal)
            (root / "manifest.jsonl").write_text(
                json.dumps({"path": "signal.npy", "subject_id": "sub-001"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "signal contains NaN or Inf"):
                load_hbn_eeg_local_dataset(root)

    def test_hbn_manifest_rejects_empty_time_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hbn"
            root.mkdir()
            np.save(root / "signal.npy", np.ones((0, 2), dtype=np.float32))
            (root / "manifest.jsonl").write_text(
                json.dumps({"path": "signal.npy", "subject_id": "sub-001"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "signal must have at least one time sample and one channel"):
                load_hbn_eeg_local_dataset(root)

    def test_hbn_manifest_rejects_empty_channel_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hbn"
            root.mkdir()
            np.save(root / "signal.npy", np.ones((20, 0), dtype=np.float32))
            (root / "manifest.jsonl").write_text(
                json.dumps({"path": "signal.npy", "subject_id": "sub-001"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "signal must have at least one time sample and one channel"):
                load_hbn_eeg_local_dataset(root)

    def test_hbn_manifest_rejects_inconsistent_channel_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hbn"
            root.mkdir()
            np.save(root / "sub-001.npy", np.ones((20, 2), dtype=np.float32))
            np.save(root / "sub-002.npy", np.ones((20, 3), dtype=np.float32))
            (root / "manifest.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"path": "sub-001.npy", "subject_id": "sub-001"}),
                        json.dumps({"path": "sub-002.npy", "subject_id": "sub-002"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "consistent channel count"):
                load_hbn_eeg_local_dataset(root)

    def test_hbn_manifest_rejects_channel_names_length_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hbn"
            root.mkdir()
            np.save(root / "signal.npy", np.ones((20, 2), dtype=np.float32))
            (root / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "path": "signal.npy",
                        "subject_id": "sub-001",
                        "channel_names": ["Fz"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "channel_names length"):
                load_hbn_eeg_local_dataset(root)

    def test_hbn_manifest_rejects_inconsistent_channel_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hbn"
            root.mkdir()
            np.save(root / "sub-001.npy", np.ones((20, 2), dtype=np.float32))
            np.save(root / "sub-002.npy", np.ones((20, 2), dtype=np.float32))
            (root / "manifest.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "path": "sub-001.npy",
                                "subject_id": "sub-001",
                                "channel_names": ["Fz", "Cz"],
                            }
                        ),
                        json.dumps(
                            {
                                "path": "sub-002.npy",
                                "subject_id": "sub-002",
                                "channel_names": ["Fz", "Pz"],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "consistent channel_names"):
                load_hbn_eeg_local_dataset(root)

    def test_hbn_manifest_rejects_duplicate_record_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hbn"
            root.mkdir()
            np.save(root / "sub-001.npy", np.ones((20, 2), dtype=np.float32))
            np.save(root / "sub-002.npy", np.ones((20, 2), dtype=np.float32))
            (root / "manifest.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {"path": "sub-001.npy", "subject_id": "sub-001", "record_id": "rec-duplicate"}
                        ),
                        json.dumps(
                            {"path": "sub-002.npy", "subject_id": "sub-002", "record_id": "rec-duplicate"}
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate record_id"):
                load_hbn_eeg_local_dataset(root)

    def test_hbn_manifest_rejects_missing_subject_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hbn"
            root.mkdir()
            np.save(root / "signal.npy", np.ones((20, 2), dtype=np.float32))
            (root / "manifest.jsonl").write_text(
                json.dumps({"path": "signal.npy"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing subject_id"):
                load_hbn_eeg_local_dataset(root)

    def test_hbn_manifest_rejects_blank_subject_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hbn"
            root.mkdir()
            np.save(root / "signal.npy", np.ones((20, 2), dtype=np.float32))
            (root / "manifest.jsonl").write_text(
                json.dumps({"path": "signal.npy", "subject_id": "   "}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing subject_id"):
                load_hbn_eeg_local_dataset(root)

    def test_hbn_manifest_rejects_blank_optional_text_fields(self):
        for field in ("session_id", "site_id", "record_id"):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / "hbn"
                    root.mkdir()
                    np.save(root / "signal.npy", np.ones((20, 2), dtype=np.float32))
                    row = {"path": "signal.npy", "subject_id": "sub-001", field: "   "}
                    (root / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

                    with self.assertRaisesRegex(ValueError, f"missing {field}"):
                        load_hbn_eeg_local_dataset(root)

    def test_hbn_local_fixture_script_marks_local_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hbn"
            out = Path(tmp) / "out"
            root.mkdir()
            rows = []
            time = np.arange(48, dtype=np.float32)
            for idx in range(6):
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
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "hbn_eeg",
                    "--data-root",
                    str(root),
                    "--out-dir",
                    str(out),
                    "--seed",
                    "0",
                    "--train-steps",
                    "1",
                    "--models",
                    "persistence,linear_ridge",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("data_source=hbn_eeg_local", result.stdout)
            run_config = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
            dataset_summary = json.loads((out / "dataset_summary.json").read_text(encoding="utf-8"))
            report = (out / "diagnostic_report.md").read_text(encoding="utf-8")
            self.assertEqual(run_config["data_source"], "hbn_eeg_local")
            self.assertEqual(run_config["benchmark_status"], "local_manifest_not_public_hbn_benchmark")
            self.assertIn("sampling_rate_hz", run_config)
            self.assertIn("sampling_rate_hz", dataset_summary)
            self.assertEqual(run_config["sampling_rate_hz"], 128.0)
            self.assertEqual(dataset_summary["sampling_rate_hz"], 128.0)
            self.assertIn("source: hbn_eeg_local", report)
            self.assertIn("- sampling_rate_hz: 128", report)
            self.assertIn("local manifest; not a public HBN benchmark result", report)

    def test_baseline_checksum_audit_rejects_tampered_hbn_local_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hbn"
            out = Path(tmp) / "out"
            root.mkdir()
            rows = []
            time = np.arange(48, dtype=np.float32)
            for idx in range(6):
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

            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_eeg_v1_baselines.py",
                    "--dataset",
                    "hbn_eeg",
                    "--data-root",
                    str(root),
                    "--out-dir",
                    str(out),
                    "--seed",
                    "0",
                    "--train-steps",
                    "1",
                    "--models",
                    "persistence,linear_ridge",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
                check=True,
            )
            report_path = out / "diagnostic_report.md"
            report = report_path.read_text(encoding="utf-8")
            report = report.replace(
                "This run used a user-provided HBN-style local manifest; not a public HBN benchmark result.",
                "This run is a public HBN benchmark result.",
            )
            report_path.write_text(report, encoding="utf-8")
            manifest_path = out / "baseline_checksum_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload = report_path.read_bytes()
            for row in manifest["artifacts"]:
                if row["path"] == "diagnostic_report.md":
                    row["bytes"] = len(payload)
                    row["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            audit = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_eeg_v1_baseline_checksums.py",
                    "--artifact-dir",
                    str(out),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(audit.returncode, 0)
        payload = json.loads(audit.stdout)
        self.assertFalse(payload["passed"], payload)
        self.assertIn("diagnostic_report_mismatch:hbn_local_boundary_section", payload["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
