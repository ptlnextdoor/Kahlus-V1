import json
import hashlib
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

from neurotwin.adapters.synthetic import (
    make_synthetic_event_batches,
    make_synthetic_multimodal_event_batches,
    make_synthetic_multimodal_recordings,
    make_synthetic_recordings,
)
from neurotwin.data.event_io import save_event_batches
from neurotwin.data.manifest_io import save_split_manifest
from neurotwin.data.split_manifest import build_split_manifest
from neurotwin.reports.evidence_gate import build_prepared_evidence_gate, write_final_prepared_evidence_gate
from neurotwin.training.command import TrainingCommandConfig, run_training_command
from neurotwin.training.prepared import PreparedBatchSampler, PreparedTrainingConfig, PreparedTrainingRunPaths, run_prepared_training
from neurotwin.training.prepared_loop import _normalize_model_type


class PreparedTrainingTests(unittest.TestCase):
    def _prepared_dir(self, root: Path) -> Path:
        prepared = root / "prepared"
        records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=1, modalities=("eeg", "fmri"))
        batches = make_synthetic_event_batches(n_subjects=6, sessions_per_subject=1, modalities=("eeg", "fmri"))
        split = build_split_manifest(records, policy="subject", seed=0)
        save_split_manifest(split, prepared / "split_manifest.json")
        save_event_batches(batches, prepared)
        return prepared

    def _multimodal_prepared_dir(self, root: Path) -> Path:
        prepared = root / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        feature_path = prepared / "stimulus_features.bin"
        feature_bytes = b"real precomputed stimulus features"
        feature_path.write_bytes(feature_bytes)
        feature_hash = hashlib.sha256(feature_bytes).hexdigest()
        records = make_synthetic_multimodal_recordings(n_subjects=6, sessions_per_subject=1, include_unpaired=True)
        batches = make_synthetic_multimodal_event_batches(n_subjects=6, sessions_per_subject=1, include_unpaired=True)
        for batch in batches:
            if batch.modality == "fmri" and batch.stimulus_embedding is not None:
                batch.metadata.update(
                    {
                        "require_real_stimulus": True,
                        "stimulus_feature_source": "sentence_transformer_audio_video_cache",
                        "stimulus_feature_modalities": ["text", "audio", "video"],
                        "stimulus_feature_hash": feature_hash,
                        "stimulus_feature_path": str(feature_path),
                        "stimulus_feature_status": "real_precomputed",
                    }
                )
        split = build_split_manifest(records, policy="subject", seed=0)
        save_split_manifest(split, prepared / "split_manifest.json")
        save_event_batches(batches, prepared)
        return prepared

    def test_rank_aware_training_batch_indices_split_distributed_work(self):
        rank0_sampler = PreparedBatchSampler(
            num_samples=32,
            batch_size=4,
            gradient_accumulation_steps=2,
            rank=0,
            world_size=2,
        )
        rank1_sampler = PreparedBatchSampler(
            num_samples=32,
            batch_size=4,
            gradient_accumulation_steps=2,
            rank=1,
            world_size=2,
        )
        rank0 = rank0_sampler.indices(step=0, micro_step=0)
        rank1 = rank1_sampler.indices(step=0, micro_step=0)
        next_rank0 = rank0_sampler.indices(step=0, micro_step=1)
        next_rank1 = rank1_sampler.indices(step=0, micro_step=1)

        self.assertEqual(rank0, [0, 1, 2, 3])
        self.assertEqual(rank1, [4, 5, 6, 7])
        self.assertEqual(next_rank0, [8, 9, 10, 11])
        self.assertEqual(next_rank1, [12, 13, 14, 15])
        self.assertTrue(set(rank0).isdisjoint(rank1))
        self.assertTrue(set(next_rank0).isdisjoint(next_rank1))

    def test_rank_aware_training_batch_indices_wrap_distributed_batches(self):
        rank0 = PreparedBatchSampler(
            num_samples=10,
            batch_size=4,
            gradient_accumulation_steps=1,
            rank=0,
            world_size=2,
        ).indices(step=1, micro_step=0)
        rank1 = PreparedBatchSampler(
            num_samples=10,
            batch_size=4,
            gradient_accumulation_steps=1,
            rank=1,
            world_size=2,
        ).indices(step=1, micro_step=0)

        self.assertEqual(rank0, [8, 9, 0, 1])
        self.assertEqual(rank1, [2, 3, 4, 5])
        self.assertEqual(len(rank0), 4)
        self.assertEqual(len(rank1), 4)
        self.assertTrue(set(rank0).isdisjoint(rank1))

    def test_single_rank_training_batch_indices_keep_sequential_tail(self):
        indices = PreparedBatchSampler(
            num_samples=10,
            batch_size=4,
            gradient_accumulation_steps=1,
            rank=0,
            world_size=1,
        ).indices(step=2, micro_step=0)

        self.assertEqual(indices, [8, 9])

    def test_prepared_training_config_honors_nested_training_batch_size(self):
        config = PreparedTrainingConfig.from_mapping(
            {
                "event_manifest": "event_manifest.json",
                "split_manifest": "split_manifest.json",
                "training": {"batch_size": 7},
                "model": {"latent_dim": 16},
            }
        )

        self.assertEqual(config.batch_size, 7)
        self.assertEqual(config.max_grad_norm, 1.0)

    def test_prepared_training_config_honors_top_level_schedule_keys(self):
        config = PreparedTrainingConfig.from_mapping(
            {
                "event_manifest": "event_manifest.json",
                "split_manifest": "split_manifest.json",
                "eval_every_steps": 3,
                "checkpoint_every_steps": 5,
                "model": {"latent_dim": 16},
            }
        )

        self.assertEqual(config.eval_every_steps, 3)
        self.assertEqual(config.checkpoint_every_steps, 5)

    def test_prepared_training_config_accepts_nfc_fields_and_aliases(self):
        config = PreparedTrainingConfig.from_mapping(
            {
                "event_manifest": "event_manifest.json",
                "split_manifest": "split_manifest.json",
                "model": {
                    "type": "nfc",
                    "latent_dim": 16,
                    "pair_rank": 3,
                    "use_pair_kernel": False,
                    "use_observation_operator": False,
                    "use_uncertainty": True,
                },
            }
        )

        self.assertEqual(config.model.type, "nfc")
        self.assertFalse(config.model.use_pair_kernel)
        self.assertFalse(config.model.use_observation_operator)
        self.assertTrue(config.model.use_uncertainty)
        self.assertEqual(_normalize_model_type("nfc"), "NeuralFieldCompiler")
        self.assertEqual(_normalize_model_type("neural_field_compiler"), "NeuralFieldCompiler")

    def test_prepared_training_config_forwards_explicit_forecast_protocol(self):
        config = PreparedTrainingConfig.from_mapping(
            {
                "event_manifest": "event_manifest.json",
                "split_manifest": "split_manifest.json",
                "forecast_task": {
                    "protocol_id": "kahlus.forecast.v2_nonoverlap",
                    "schema_version": 2,
                    "context_seconds": 1.0,
                    "target_seconds": 0.5,
                    "gap_seconds": 0.25,
                    "stride_seconds": 0.5,
                    "claim_eligible": True,
                },
            }
        )

        self.assertIsNotNone(config.forecast_task)
        self.assertEqual(config.suite_config().forecast_task, config.forecast_task)

    def test_prepared_training_cleans_up_distributed_group_on_failure(self):
        config = {
            "event_manifest": "missing-event-manifest.json",
            "split_manifest": "missing-split-manifest.json",
            "task": "future_state_forecasting",
            "window_size": 8,
            "steps": 1,
            "model": {"latent_dim": 16, "n_layers": 1},
        }
        with (
            mock.patch("neurotwin.training.prepared.maybe_init_process_group", return_value=(True, "gloo")),
            mock.patch("neurotwin.training.prepared.cleanup_process_group") as cleanup,
            mock.patch("neurotwin.training.prepared.load_event_batches", side_effect=RuntimeError("load failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                run_prepared_training(config)

        cleanup.assert_called_once_with()

    def test_prepared_training_writes_checkpoint_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = self._prepared_dir(root)
            checkpoint = root / "checkpoint.pt"
            metrics_csv = root / "metrics.csv"

            result = run_prepared_training(
                {
                    "event_manifest": str(prepared / "event_manifest.json"),
                    "split_manifest": str(prepared / "split_manifest.json"),
                    "task": "future_state_forecasting",
                    "window_size": 8,
                    "stride": 8,
                    "steps": 2,
                    "batch_size": 4,
                    "gradient_accumulation_steps": 2,
                    "max_grad_norm": 0.5,
                    "model": {"latent_dim": 16, "n_layers": 1, "subject_adapter_dim": 4},
                },
                checkpoint_path=checkpoint,
                metrics_csv_path=metrics_csv,
            )

            self.assertEqual(result.status, "completed_prepared_training")
            self.assertEqual(result.task_id, "future_state_forecasting")
            self.assertTrue(result.synthetic_only)
            self.assertEqual(result.gradient_accumulation_steps, 2)
            self.assertEqual(result.max_grad_norm, 0.5)
            self.assertEqual(result.completed_steps, 2)
            self.assertEqual(result.selection_split, "val")
            self.assertEqual(result.report_split, "test")
            self.assertTrue(result.best_val_mse >= 0.0)
            self.assertTrue(result.test_mse >= 0.0)
            self.assertFalse(result.distributed_initialized)
            self.assertEqual(result.world_size, 1)
            self.assertTrue(checkpoint.exists())
            self.assertTrue(metrics_csv.exists())

            resumed = run_prepared_training(
                {
                    "event_manifest": str(prepared / "event_manifest.json"),
                    "split_manifest": str(prepared / "split_manifest.json"),
                    "task": "future_state_forecasting",
                    "window_size": 8,
                    "stride": 8,
                    "steps": 1,
                    "batch_size": 4,
                    "model": {"latent_dim": 16, "n_layers": 1, "subject_adapter_dim": 4},
                },
                checkpoint_path=root / "checkpoint_resumed.pt",
                resume_path=checkpoint,
            )

            self.assertEqual(resumed.start_step, 2)
            self.assertEqual(resumed.completed_steps, 3)
            self.assertEqual(resumed.resumed_from, str(checkpoint))

    def test_prepared_training_accepts_grouped_run_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = self._prepared_dir(root)
            paths = PreparedTrainingRunPaths(
                checkpoint_path=root / "checkpoint.pt",
                metrics_csv_path=root / "metrics.csv",
            )

            result = run_prepared_training(
                {
                    "event_manifest": str(prepared / "event_manifest.json"),
                    "split_manifest": str(prepared / "split_manifest.json"),
                    "task": "future_state_forecasting",
                    "window_size": 8,
                    "stride": 8,
                    "steps": 1,
                    "batch_size": 4,
                    "model": {"latent_dim": 16, "n_layers": 1, "subject_adapter_dim": 4},
                },
                paths=paths,
            )

            self.assertEqual(result.status, "completed_prepared_training")
            self.assertTrue((root / "checkpoint.pt").exists())
            self.assertTrue((root / "metrics.csv").exists())

    def test_prepared_training_rejects_mixed_run_path_styles(self):
        with self.assertRaisesRegex(ValueError, "either paths=PreparedTrainingRunPaths"):
            run_prepared_training(
                {
                    "event_manifest": "event_manifest.json",
                    "split_manifest": "split_manifest.json",
                    "model": {"latent_dim": 16},
                },
                paths=PreparedTrainingRunPaths(),
                checkpoint_path="checkpoint.pt",
            )

    def test_prepared_training_rejects_unknown_task_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepared = self._prepared_dir(Path(tmp))

            with self.assertRaisesRegex(ValueError, "No runnable prepared training task matched"):
                run_prepared_training(
                    {
                        "event_manifest": str(prepared / "event_manifest.json"),
                        "split_manifest": str(prepared / "split_manifest.json"),
                        "task": "not_a_prepared_task",
                        "window_size": 8,
                        "stride": 8,
                        "steps": 1,
                        "batch_size": 4,
                        "model": {"latent_dim": 16, "n_layers": 1, "subject_adapter_dim": 4},
                    }
                )

    def test_prepared_training_runs_all_neural_translation_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = self._prepared_dir(root)
            checkpoint = root / "checkpoint.pt"
            best_checkpoint = root / "checkpoint_best.pt"
            metrics_jsonl = root / "metrics.jsonl"

            result = run_prepared_training(
                {
                    "event_manifest": str(prepared / "event_manifest.json"),
                    "split_manifest": str(prepared / "split_manifest.json"),
                    "task": "neural_translation_v1",
                    "window_size": 8,
                    "stride": 8,
                    "steps": 1,
                    "batch_size": 4,
                    "training": {"eval_every_steps": 1, "checkpoint_every_steps": 1},
                    "model": {
                        "latent_dim": 16,
                        "n_layers": 1,
                        "backbone": "ssm_fallback",
                        "encoder": "auto",
                        "subject_adapter_dim": 4,
                        "adapter_mode": "disabled",
                    },
                },
                checkpoint_path=checkpoint,
                best_checkpoint_path=best_checkpoint,
                metrics_jsonl_path=metrics_jsonl,
            )

            task_ids = {row["task_id"] for row in result.task_results}
            self.assertEqual(result.task_id, "neural_translation_v1")
            self.assertIn("future_state_forecasting", task_ids)
            self.assertIn("masked_neural_reconstruction", task_ids)
            self.assertIn("cross_modal_translation", task_ids)
            self.assertTrue(checkpoint.exists())
            self.assertTrue(best_checkpoint.exists())
            self.assertTrue(metrics_jsonl.exists())

    def test_prepared_training_tracks_best_validation_checkpoint_separately_from_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = self._prepared_dir(root)
            checkpoint = root / "checkpoint.pt"
            best_checkpoint = root / "checkpoint_best.pt"
            metrics_jsonl = root / "metrics.jsonl"
            val_mse_values = [0.4, 0.3, 0.8]

            def fake_evaluate(model, task, x_test, y_test, precision, prefix, batch_size):
                if prefix == "val":
                    value = val_mse_values.pop(0)
                    return {
                        "val_mse": value,
                        "val_mae": value,
                        "val_pearsonr": 0.0,
                        "val_spearmanr": 0.0,
                        "val_r2": 0.0,
                    }
                return {
                    "test_mse": 0.12,
                    "test_mae": 0.12,
                    "test_pearsonr": 0.0,
                    "test_spearmanr": 0.0,
                    "test_r2": 0.0,
                }

            with mock.patch("neurotwin.training.prepared_loop.evaluate_task", side_effect=fake_evaluate):
                result = run_prepared_training(
                    {
                        "event_manifest": str(prepared / "event_manifest.json"),
                        "split_manifest": str(prepared / "split_manifest.json"),
                        "task": "future_state_forecasting",
                        "window_size": 8,
                        "stride": 8,
                        "steps": 2,
                        "batch_size": 4,
                        "training": {"eval_every_steps": 1},
                        "model": {"latent_dim": 16, "n_layers": 1, "subject_adapter_dim": 4},
                    },
                    checkpoint_path=checkpoint,
                    best_checkpoint_path=best_checkpoint,
                    metrics_jsonl_path=metrics_jsonl,
                )

            self.assertEqual(result.best_step, 2)
            self.assertAlmostEqual(result.best_val_mse, 0.3)
            self.assertAlmostEqual(result.final_val_mse, 0.8)
            self.assertNotEqual(result.best_val_mse, result.final_val_mse)
            self.assertEqual(result.best_checkpoint_path, str(best_checkpoint))
            self.assertEqual(result.final_checkpoint_path, str(checkpoint))
            best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(best_payload["best_step"], 2)
            self.assertEqual(best_payload["checkpoint_selection_metric"], "val_mse")
            final_rows = [
                json.loads(line)
                for line in metrics_jsonl.read_text(encoding="utf-8").splitlines()
                if json.loads(line).get("phase") == "final"
            ]
            self.assertEqual(len(final_rows), 1)
            self.assertEqual(final_rows[0]["checkpoint_role"], "selected_best")
            self.assertAlmostEqual(final_rows[0]["val_mse"], 0.3)
            self.assertAlmostEqual(final_rows[0]["best_val_mse"], 0.3)
            self.assertAlmostEqual(final_rows[0]["final_val_mse"], 0.8)

    def test_prepared_training_runs_pair_operator_stimulus_fmri_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = self._multimodal_prepared_dir(root)

            result = run_prepared_training(
                {
                    "event_manifest": str(prepared / "event_manifest.json"),
                    "split_manifest": str(prepared / "split_manifest.json"),
                    "task": "stimulus_to_fmri_response",
                    "window_size": 8,
                    "stride": 8,
                    "steps": 1,
                    "batch_size": 4,
                    "model": {
                        "type": "NeuroTwinPairOperator",
                        "latent_dim": 16,
                        "n_layers": 1,
                        "pair_rank": 3,
                        "projection_dim": 8,
                    },
                }
            )

            self.assertEqual(result.status, "completed_prepared_training")
            self.assertEqual(result.task_id, "stimulus_to_fmri_response")
            self.assertEqual(result.task_results[0]["model_config"]["type"], "NeuroTwinPairOperator")
            self.assertEqual(result.stimulus_evidence["status"], "real_stimulus_features")
            self.assertTrue(result.stimulus_evidence["hash_verified"])
            self.assertFalse(result.quarantined_tasks)

    def test_pair_operator_command_writes_finite_uncertainty_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = self._multimodal_prepared_dir(root)
            config_path = root / "pair_operator_uncertainty.yaml"
            run_root = root / "runs"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "experiment": "pair_operator_uncertainty",
                        "data": {
                            "event_manifest": str(prepared / "event_manifest.json"),
                            "split_manifest": str(prepared / "split_manifest.json"),
                        },
                        "task": "stimulus_to_fmri_response",
                        "window_size": 8,
                        "stride": 8,
                        "steps": 1,
                        "batch_size": 4,
                        "model": {
                            "type": "NeuroTwinPairOperator",
                            "latent_dim": 16,
                            "n_layers": 1,
                            "pair_rank": 3,
                            "projection_dim": 8,
                            "use_uncertainty_head": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            command = run_training_command(TrainingCommandConfig(str(config_path), run_root=str(run_root)))

            self.assertEqual(command.exit_code, 0, command.error)
            rows = (run_root / "pair_operator_uncertainty" / "uncertainty_calibration.csv").read_text(encoding="utf-8").splitlines()

        header = rows[0].split(",")
        values = rows[1].split(",")
        row = dict(zip(header, values))
        self.assertEqual(row["status"], "finite")
        self.assertTrue(math.isfinite(float(row["mean_uncertainty"])))
        self.assertTrue(math.isfinite(float(row["error_uncertainty_correlation"])))

    def test_nonfinite_task_metrics_are_quarantined_and_best_checkpoint_skips_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = self._prepared_dir(root)

            def fake_evaluate(model, task, x_test, y_test, precision, prefix, batch_size):
                key = f"{prefix}_mse"
                return {
                    key: float("nan"),
                    f"{prefix}_mae": float("nan"),
                    f"{prefix}_pearsonr": float("nan"),
                    f"{prefix}_spearmanr": float("nan"),
                    f"{prefix}_r2": float("nan"),
                }

            with mock.patch("neurotwin.training.prepared_loop.evaluate_task", side_effect=fake_evaluate):
                result = run_prepared_training(
                    {
                        "event_manifest": str(prepared / "event_manifest.json"),
                        "split_manifest": str(prepared / "split_manifest.json"),
                        "task": "future_state_forecasting",
                        "window_size": 8,
                        "stride": 8,
                        "steps": 1,
                        "batch_size": 4,
                        "model": {"latent_dim": 16, "n_layers": 1},
                    }
                )

            self.assertEqual(result.task_results[0]["status"], "quarantined_nonfinite")
            self.assertTrue(result.quarantined_tasks)
            self.assertIsNone(result.best_task_id)
            self.assertIsNone(result.best_eval_mse)

    def test_nonfinite_training_loss_skips_optimizer_step_and_quarantines_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = self._prepared_dir(root)
            metrics_jsonl = root / "metrics.jsonl"

            def nan_predict(model, task, xb, precision):
                del model, precision
                return torch.full((xb.shape[0], *task.y_train.shape[1:]), float("nan"), device=xb.device)

            with mock.patch("neurotwin.training.prepared_loop.predict", side_effect=nan_predict):
                result = run_prepared_training(
                    {
                        "event_manifest": str(prepared / "event_manifest.json"),
                        "split_manifest": str(prepared / "split_manifest.json"),
                        "task": "future_state_forecasting",
                        "window_size": 8,
                        "stride": 8,
                        "steps": 2,
                        "batch_size": 4,
                        "model": {"latent_dim": 16, "n_layers": 1},
                    },
                    metrics_jsonl_path=metrics_jsonl,
                )

            rows = [json.loads(line) for line in metrics_jsonl.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(result.task_results[0]["status"], "quarantined_nonfinite")
            self.assertTrue(result.quarantined_tasks)
            self.assertIsNone(result.task_results[0]["final_loss"])
            self.assertTrue(any(row.get("phase") == "nonfinite_loss" and row.get("optimizer_step_skipped") for row in rows))

    def test_nonfinite_gradient_skips_optimizer_step_and_quarantines_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = self._prepared_dir(root)
            metrics_jsonl = root / "metrics.jsonl"

            with mock.patch("neurotwin.training.prepared_loop._clip_or_measure_grad_norm", return_value=float("nan")):
                result = run_prepared_training(
                    {
                        "event_manifest": str(prepared / "event_manifest.json"),
                        "split_manifest": str(prepared / "split_manifest.json"),
                        "task": "future_state_forecasting",
                        "window_size": 8,
                        "stride": 8,
                        "steps": 2,
                        "batch_size": 4,
                        "model": {"latent_dim": 16, "n_layers": 1},
                    },
                    metrics_jsonl_path=metrics_jsonl,
                )

            rows = [json.loads(line) for line in metrics_jsonl.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(result.task_results[0]["status"], "quarantined_nonfinite")
            self.assertTrue(result.quarantined_tasks)
            self.assertIsNone(result.task_results[0]["final_loss"])
            self.assertTrue(any(row.get("phase") == "nonfinite_gradient" and row.get("optimizer_step_skipped") for row in rows))

    def test_prepared_evidence_gate_fails_quarantined_required_task(self):
        gate = build_prepared_evidence_gate(
            {
                "scientific_claim_allowed": True,
                "quarantined_tasks": [{"task_id": "masked_neural_reconstruction", "reason": "nonfinite metric"}],
                "task_results": [
                    {
                        "task_id": "masked_neural_reconstruction",
                        "best_val_mse": None,
                        "test_mse": None,
                    }
                ],
            }
        )

        self.assertFalse(gate["passed"])
        self.assertIn("required task quarantined: masked_neural_reconstruction", gate["failures"])

    def test_final_evidence_gate_uses_completed_run_dir_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            summary = {
                "scientific_claim_allowed": True,
                "synthetic_only": False,
                "real_data_smoke": False,
                "status": "completed",
                "quarantined_tasks": [],
                "task_results": [
                    {
                        "task_id": "future_state_forecasting",
                        "best_val_mse": 0.2,
                        "test_mse": 0.3,
                    }
                ],
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (run_dir / "eval_audit.json").write_text(json.dumps({"passed": True}), encoding="utf-8")

            missing_gate = write_final_prepared_evidence_gate(run_dir)

            self.assertFalse(missing_gate["passed"])
            self.assertIn("baseline ranking artifact missing or unavailable", missing_gate["failures"])
            self.assertIn("paper_mode_gate.json missing or not passed", missing_gate["failures"])

            (run_dir / "paper_mode_gate.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
            (run_dir / "prepared_baseline_suite.json").write_text(
                json.dumps(
                    {
                        "baseline_catalog": [{"model_id": "linear_ridge", "status": "local_baseline"}],
                        "tasks": {
                            "future_state_forecasting": {
                                "ranking": [{"model_id": "linear_ridge", "metric": "mse", "value": 0.3, "rank": 1}]
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            final_gate = write_final_prepared_evidence_gate(run_dir)
            rewritten_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertTrue(final_gate["passed"])
        self.assertEqual(final_gate["stage"], "final")
        self.assertTrue(final_gate["checks"]["baseline_ranking_present"])
        self.assertTrue(final_gate["checks"]["competitor_reproduction_status_present"])
        self.assertNotIn("evidence_gate_passed", rewritten_summary)
        self.assertNotIn("evidence_gate_failures", rewritten_summary)

    def test_final_evidence_gate_csv_ranking_parsing_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_minimal_final_gate_inputs(run_dir)
            tables = run_dir / "tables"
            tables.mkdir()
            (tables / "baseline_ranking.csv").write_text(
                "task_id,model_id,metric,value,rank\nbaseline_ranking_unavailable,,status,no colocated baseline rankings found,\n",
                encoding="utf-8",
            )

            unavailable_gate = write_final_prepared_evidence_gate(run_dir)
            (tables / "baseline_ranking.csv").write_text("task,model\nfuture_state_forecasting,linear_ridge\n", encoding="utf-8")
            malformed_gate = write_final_prepared_evidence_gate(run_dir)

        self.assertFalse(unavailable_gate["checks"]["baseline_ranking_present"])
        self.assertFalse(malformed_gate["checks"]["baseline_ranking_present"])

    def test_final_evidence_gate_ignores_csv_real_ranking_without_structured_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_minimal_final_gate_inputs(run_dir)
            tables = run_dir / "tables"
            tables.mkdir()
            (tables / "baseline_ranking.csv").write_text(
                "task_id,model_id,metric,value,rank\nfuture_state_forecasting,linear_ridge,mse,0.3,1\n",
                encoding="utf-8",
            )

            gate = write_final_prepared_evidence_gate(run_dir)

        self.assertFalse(gate["checks"]["baseline_ranking_present"])
        self.assertFalse(gate["passed"])
        self.assertIn("baseline ranking artifact missing or unavailable", gate["failures"])

    def _write_minimal_final_gate_inputs(self, run_dir: Path) -> None:
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "scientific_claim_allowed": True,
                    "status": "completed",
                    "quarantined_tasks": [],
                    "task_results": [{"task_id": "future_state_forecasting", "best_val_mse": 0.2, "test_mse": 0.3}],
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "eval_audit.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
        (run_dir / "paper_mode_gate.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
        (run_dir / "prepared_baseline_suite.json").write_text(
            json.dumps({"baseline_catalog": [{"model_id": "linear_ridge", "status": "local_baseline"}], "tasks": {}}),
            encoding="utf-8",
        )

    def test_objective_weights_are_recorded_for_multitask_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = self._prepared_dir(root)

            result = run_prepared_training(
                {
                    "event_manifest": str(prepared / "event_manifest.json"),
                    "split_manifest": str(prepared / "split_manifest.json"),
                    "task": "neural_translation_v1",
                    "window_size": 8,
                    "stride": 8,
                    "steps": 1,
                    "batch_size": 4,
                    "training": {"objective_weights": {"masked_neural_reconstruction": 0.25}},
                    "model": {"latent_dim": 16, "n_layers": 1, "subject_adapter_dim": 4},
                }
            )

            weights = {row["task_id"]: row["objective_weight"] for row in result.task_results}
            self.assertEqual(weights["masked_neural_reconstruction"], 0.25)
            self.assertEqual(weights["future_state_forecasting"], 1.0)

    def test_prepared_training_uses_batched_eval_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = self._prepared_dir(root)
            import neurotwin.training.prepared_metrics as prepared_metrics

            observed_batch_sizes = []
            original_predict = prepared_metrics.predict

            def recording_predict(model, task, x, precision="fp32"):
                observed_batch_sizes.append(int(x.shape[0]))
                return original_predict(model, task, x, precision=precision)

            with mock.patch.object(prepared_metrics, "predict", side_effect=recording_predict):
                result = run_prepared_training(
                    {
                        "event_manifest": str(prepared / "event_manifest.json"),
                        "split_manifest": str(prepared / "split_manifest.json"),
                        "task": "future_state_forecasting",
                        "window_size": 8,
                        "stride": 8,
                        "steps": 1,
                        "batch_size": 4,
                        "eval_batch_size": 2,
                        "precision": "bf16",
                        "model": {"latent_dim": 16, "n_layers": 1, "subject_adapter_dim": 4},
                    }
                )

            self.assertEqual(result.status, "completed_prepared_training")
            self.assertTrue(observed_batch_sizes)
            self.assertLessEqual(max(observed_batch_sizes), 4)
            self.assertIn(2, observed_batch_sizes)

    def test_train_cli_uses_prepared_path_when_manifests_are_configured(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = self._prepared_dir(root)
            config_path = root / "prepared_train.yaml"
            run_root = root / "runs"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "experiment": "prepared_train",
                        "data": {
                            "event_manifest": str(prepared / "event_manifest.json"),
                            "split_manifest": str(prepared / "split_manifest.json"),
                        },
                        "task": "masked_neural_reconstruction",
                        "window_size": 8,
                        "stride": 8,
                        "steps": 2,
                        "batch_size": 4,
                        "seed": 0,
                        "model": {"latent_dim": 16, "n_layers": 1, "subject_adapter_dim": 4},
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "train",
                    "--config",
                    str(config_path),
                    "--run-root",
                    str(run_root),
                ],
                check=True,
                text=True,
                capture_output=True,
                env=env,
            )

            run_dir = run_root / "prepared_train"
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("completed_prepared_training", result.stdout)
            self.assertEqual(summary["status"], "completed_prepared_training")
            self.assertEqual(summary["event_manifest"], str(prepared / "event_manifest.json"))
            self.assertFalse(summary["distributed_initialized"])
            self.assertEqual(metrics["task_id"], "masked_neural_reconstruction")
            self.assertIn("completed_steps", summary)
            self.assertEqual(summary["selection_split"], "val")
            self.assertEqual(summary["report_split"], "test")
            self.assertIn("scientific_claim_allowed", summary)
            self.assertFalse(summary["scientific_claim_allowed"])
            self.assertIn("source_commit_missing", summary)
            self.assertEqual(summary["run"]["mode"], "direct")
            self.assertIn("checkpoint.pt", [entry["filename"] for entry in summary["checkpoint_manifest"]])
            self.assertIn("final_val_mse", summary)
            self.assertIn("best_step", summary)
            self.assertIn("best_checkpoint_path", summary)
            self.assertIn("final_checkpoint_path", summary)
            self.assertEqual(summary["checkpoint_selection_metric"], "val_mse")
            self.assertEqual(summary["checkpoint_selection_mode"], "min")
            self.assertTrue((run_dir / "checkpoint_manifest.json").exists())
            self.assertTrue((run_dir / "eval_audit.json").exists())
            self.assertTrue((run_dir / "metrics.csv").exists())
            self.assertTrue((run_dir / "checkpoint.pt").exists())
            self.assertTrue((run_dir / "evidence_gate_provisional.json").exists())
            self.assertTrue((run_dir / "diagnostic_report_provisional.md").exists())
            self.assertTrue((run_dir / "pair_operator_ablation.csv").exists())
            self.assertTrue((run_dir / "uncertainty_calibration.csv").exists())
            self.assertFalse((run_dir / "evidence_gate.json").exists())
            evidence_gate = json.loads((run_dir / "evidence_gate_provisional.json").read_text(encoding="utf-8"))
            self.assertFalse(evidence_gate["passed"])
            self.assertFalse(evidence_gate["scientific_claim_allowed"])
            self.assertIn("summary.json scientific_claim_allowed is false", evidence_gate["failures"])
            self.assertIn(
                "not_pair_operator_run",
                (run_dir / "pair_operator_ablation.csv").read_text(encoding="utf-8"),
            )

    def test_train_cli_keeps_claims_disabled_for_non_synthetic_non_smoke_runs(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "prepared"
            records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=1, modalities=("eeg", "fmri"))
            batches = make_synthetic_event_batches(n_subjects=6, sessions_per_subject=1, modalities=("eeg", "fmri"))
            for batch in batches:
                batch.metadata.pop("synthetic", None)
            split = build_split_manifest(records, policy="subject", seed=0)
            save_split_manifest(split, prepared / "split_manifest.json")
            save_event_batches(batches, prepared)
            config_path = root / "prepared_train_realish.yaml"
            run_root = root / "runs"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "experiment": "prepared_train_realish",
                        "data": {
                            "event_manifest": str(prepared / "event_manifest.json"),
                            "split_manifest": str(prepared / "split_manifest.json"),
                        },
                        "task": "masked_neural_reconstruction",
                        "window_size": 8,
                        "stride": 8,
                        "steps": 1,
                        "batch_size": 4,
                        "seed": 0,
                        "model": {"latent_dim": 16, "n_layers": 1, "subject_adapter_dim": 4},
                    }
                ),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "train",
                    "--config",
                    str(config_path),
                    "--run-root",
                    str(run_root),
                ],
                check=True,
                text=True,
                capture_output=True,
                env=env,
            )

            summary = json.loads((run_root / "prepared_train_realish" / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["synthetic_only"])
            self.assertFalse(summary["real_data_smoke"])
            self.assertFalse(summary["scientific_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
