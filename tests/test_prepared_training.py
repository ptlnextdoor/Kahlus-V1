import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from neurotwin.adapters.synthetic import make_synthetic_event_batches, make_synthetic_recordings
from neurotwin.data.event_io import save_event_batches
from neurotwin.data.manifest_io import save_split_manifest
from neurotwin.data.split_manifest import build_split_manifest
from neurotwin.training.prepared import PreparedBatchSampler, run_prepared_training


class PreparedTrainingTests(unittest.TestCase):
    def _prepared_dir(self, root: Path) -> Path:
        prepared = root / "prepared"
        records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=1, modalities=("eeg", "fmri"))
        batches = make_synthetic_event_batches(n_subjects=6, sessions_per_subject=1, modalities=("eeg", "fmri"))
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
                    "model": {"latent_dim": 16, "n_layers": 1, "subject_adapter_dim": 4},
                },
                checkpoint_path=checkpoint,
                metrics_csv_path=metrics_csv,
            )

            self.assertEqual(result.status, "completed_prepared_training")
            self.assertEqual(result.task_id, "future_state_forecasting")
            self.assertTrue(result.synthetic_only)
            self.assertEqual(result.gradient_accumulation_steps, 2)
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
            import neurotwin.training.prepared as prepared_training

            observed_batch_sizes = []
            original_predict = prepared_training._predict

            def recording_predict(model, task, x, precision="fp32"):
                observed_batch_sizes.append(int(x.shape[0]))
                return original_predict(model, task, x, precision=precision)

            with mock.patch.object(prepared_training, "_predict", side_effect=recording_predict):
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
            self.assertTrue((run_dir / "checkpoint_manifest.json").exists())
            self.assertTrue((run_dir / "metrics.csv").exists())
            self.assertTrue((run_dir / "checkpoint.pt").exists())

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
