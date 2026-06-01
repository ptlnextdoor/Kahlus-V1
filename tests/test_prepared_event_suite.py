import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurotwin.adapters.synthetic import (
    make_synthetic_event_batches,
    make_synthetic_multimodal_event_batches,
    make_synthetic_multimodal_recordings,
    make_synthetic_recordings,
)
from neurotwin.benchmarks.prepared_suite import (
    PreparedSuiteConfig,
    _aggregate_seed_ranks,
    build_prepared_window_tasks,
    format_prepared_baseline_report,
    run_prepared_baseline_suite,
    run_prepared_baseline_suite_multi_seed,
)
from neurotwin.data.event_io import event_manifest_summary, load_event_batches, save_event_batches
from neurotwin.data.split_manifest import build_split_manifest
from neurotwin.data.manifest_io import save_split_manifest
from neurotwin.eval.command import EvalCommandConfig, run_eval_command
from neurotwin.eval.paper_gate import CANONICAL_REQUIRED_SEEDS


class PreparedEventSuiteTests(unittest.TestCase):
    def test_event_batches_roundtrip_through_manifest(self):
        batches = make_synthetic_event_batches(n_subjects=2, sessions_per_subject=1, modalities=("eeg", "fmri"))
        batches[0].metadata.update(
            {
                "task_id": "synthetic_task",
                "sampling_rate": 128.0,
                "source_hash": "source-hash",
                "preprocessing_hash": "prep-hash",
                "split_assignment": "train",
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = save_event_batches(batches, tmp)
            loaded = load_event_batches(manifest_path)
            summary = event_manifest_summary(manifest_path)

        self.assertEqual(len(loaded), len(batches))
        self.assertEqual(loaded[0].signal.shape, batches[0].signal.shape)
        self.assertEqual(summary["schema"], "neurotwin.event_manifest.v2")
        self.assertIn("eeg", summary["modalities"])
        self.assertIn("fmri", summary["modalities"])
        self.assertIn("synthetic_task", summary["task_ids"])
        self.assertEqual(loaded[0].task_id, "synthetic_task")
        self.assertEqual(loaded[0].sampling_rate, 128.0)
        self.assertEqual(loaded[0].source_hash, "source-hash")
        self.assertEqual(loaded[0].preprocessing_hash, "prep-hash")
        self.assertEqual(loaded[0].split_assignment, "train")

    def test_v1_event_manifest_still_loads(self):
        batches = make_synthetic_event_batches(n_subjects=1, sessions_per_subject=1, modalities=("eeg",))
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = save_event_batches(batches, tmp)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["schema"] = "neurotwin.event_manifest.v1"
            for row in payload["events"]:
                for key in (
                    "recording_id",
                    "dataset_id",
                    "task_id",
                    "sampling_rate",
                    "time_start",
                    "time_end",
                    "source_hash",
                    "preprocessing_hash",
                    "split_assignment",
                ):
                    row.pop(key, None)
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = load_event_batches(manifest_path)
            summary = event_manifest_summary(manifest_path)

        self.assertEqual(summary["schema"], "neurotwin.event_manifest.v1")
        self.assertEqual(len(loaded), len(batches))
        self.assertTrue(loaded[0].recording_id)

    def test_prepared_window_tasks_use_split_manifest(self):
        records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=1, modalities=("eeg", "fmri"))
        batches = make_synthetic_event_batches(n_subjects=6, sessions_per_subject=1, modalities=("eeg", "fmri"))
        split = build_split_manifest(records, policy="subject", seed=0)

        tasks, skipped = build_prepared_window_tasks(batches, split, window_length=8, stride=8)

        self.assertFalse([row for row in skipped if row["task_id"] == "all"])
        self.assertIn("future_state_forecasting", {task.task_id for task in tasks})
        self.assertIn("cross_modal_translation", {task.task_id for task in tasks})

    def test_synthetic_multimodal_smoke_builds_cross_modal_task(self):
        records = make_synthetic_multimodal_recordings(n_subjects=6, sessions_per_subject=1, include_unpaired=True)
        batches = make_synthetic_multimodal_event_batches(n_subjects=6, sessions_per_subject=1, include_unpaired=True)
        split = build_split_manifest(records, policy="subject", seed=0)

        tasks, skipped = build_prepared_window_tasks(batches, split, window_length=8, stride=8)

        task_ids = {task.task_id for task in tasks}
        self.assertIn("future_state_forecasting", task_ids)
        self.assertIn("masked_neural_reconstruction", task_ids)
        self.assertIn("cross_modal_translation", task_ids)
        self.assertIn("stimulus_to_fmri_response", task_ids)
        self.assertTrue(any(batch.modality == "behavior" for batch in batches))
        self.assertTrue(any(batch.modality == "stimulus" for batch in batches))
        self.assertTrue(any(batch.sampling_rate == 0.5 for batch in batches if batch.modality == "fmri"))
        self.assertTrue(any(not batch.mask.all() for batch in batches if batch.modality in {"eeg", "fmri"}))
        self.assertFalse([row for row in skipped if row["task_id"] == "all"])

    def test_eeg_only_prepared_data_skips_stimulus_fmri_task(self):
        records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=1, modalities=("eeg",))
        batches = make_synthetic_event_batches(n_subjects=6, sessions_per_subject=1, modalities=("eeg",))
        split = build_split_manifest(records, policy="subject", seed=0)

        tasks, skipped = build_prepared_window_tasks(batches, split, window_length=8, stride=8)

        self.assertNotIn("stimulus_to_fmri_response", {task.task_id for task in tasks})
        self.assertIn(
            {"task_id": "stimulus_to_fmri_response", "reason": "need fMRI train/test windows with aligned stimulus embeddings"},
            skipped,
        )

    def test_prepared_baseline_suite_runs_tribe_style_on_stimulus_fmri(self):
        with tempfile.TemporaryDirectory() as tmp:
            prep_dir = Path(tmp) / "prepared"
            records = make_synthetic_multimodal_recordings(n_subjects=6, sessions_per_subject=1, include_unpaired=True)
            batches = make_synthetic_multimodal_event_batches(n_subjects=6, sessions_per_subject=1, include_unpaired=True)
            split = build_split_manifest(records, policy="subject", seed=0)
            save_split_manifest(split, prep_dir / "split_manifest.json")
            save_event_batches(batches, prep_dir)

            payload = run_prepared_baseline_suite(
                PreparedSuiteConfig(
                    event_manifest=prep_dir / "event_manifest.json",
                    split_manifest=prep_dir / "split_manifest.json",
                    train_steps=1,
                ),
            )

        task = payload["tasks"]["stimulus_to_fmri_response"]
        metrics = task["metrics_by_model"]["tribe_style"]
        self.assertIn("mse_ci_low", metrics)
        self.assertIn("mse_ci_high", metrics)
        self.assertIn("tribe_style", {row["model_id"] for row in task["ranking"]})
        self.assertTrue(
            any(
                row["model_id"] == "tribe_style" and row["status"] == "clean_room_approximation"
                for row in payload["baseline_catalog"]
            )
        )
        report = format_prepared_baseline_report(payload)
        self.assertIn("tribe_style: clean_room_approximation", report)
        self.assertIn("not an exact TRIBE v2 reproduction", report)

    def test_prepared_baseline_suite_and_cli_artifacts(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        with tempfile.TemporaryDirectory() as tmp:
            prep_dir = Path(tmp) / "prepared"
            eval_dir = Path(tmp) / "eval"
            prepare = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "data",
                    "prepare",
                    "--dataset",
                    "synthetic",
                    "--split",
                    "subject",
                    "--out-dir",
                    str(prep_dir),
                ],
                check=True,
                text=True,
                capture_output=True,
                env=env,
            )
            self.assertIn("event_manifest=", prepare.stdout)

            payload = run_prepared_baseline_suite(
                PreparedSuiteConfig(
                    event_manifest=prep_dir / "event_manifest.json",
                    split_manifest=prep_dir / "split_manifest.json",
                    train_steps=1,
                ),
                out_dir=eval_dir,
            )
            self.assertEqual(payload["scope"]["status"], "prepared-synthetic")
            self.assertTrue((eval_dir / "prepared_baseline_suite.json").exists())
            self.assertTrue((eval_dir / "baseline_failures.json").exists())
            self.assertIn("few_shot_subject_adaptation", payload["tasks"])
            self.assertIn("dataset_site_generalization", payload["tasks"])
            self.assertEqual(payload["paper_mode_contract"]["required_seeds"], list(CANONICAL_REQUIRED_SEEDS))
            with self.assertRaises(TypeError):
                PreparedSuiteConfig(
                    event_manifest=prep_dir / "event_manifest.json",
                    split_manifest=prep_dir / "split_manifest.json",
                    required_seeds=(0,),  # type: ignore[call-arg]
                )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "eval",
                    "--suite",
                    "neural_translation_v1",
                    "--event-manifest",
                    str(prep_dir / "event_manifest.json"),
                    "--split-manifest",
                    str(prep_dir / "split_manifest.json"),
                    "--out-dir",
                    str(eval_dir),
                    "--train-steps",
                    "1",
                ],
                check=True,
                text=True,
                capture_output=True,
                env=env,
            )

            artifact = json.loads((eval_dir / "prepared_baseline_suite.json").read_text(encoding="utf-8"))
            self.assertIn("Prepared Baseline Suite", result.stdout)
            self.assertIn("ci95=", result.stdout)
            self.assertIn("baseline_catalog", result.stdout)
            self.assertIn("few_shot_subject_adaptation", result.stdout)
            self.assertIn("dataset_site_generalization", result.stdout)
            future = artifact["tasks"]["future_state_forecasting"]["metrics_by_model"]["linear_ridge"]
            self.assertIn("mse_ci_low", future)
            self.assertIn("mse_ci_high", future)
            self.assertTrue(artifact["aggregate"]["aggregate_rank"])

            paper_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "eval",
                    "--suite",
                    "neural_translation_v1",
                    "--event-manifest",
                    str(prep_dir / "event_manifest.json"),
                    "--split-manifest",
                    str(prep_dir / "split_manifest.json"),
                    "--out-dir",
                    str(eval_dir),
                    "--train-steps",
                    "1",
                    "--paper-mode",
                ],
                text=True,
                capture_output=True,
                env=env,
            )
            self.assertNotEqual(paper_result.returncode, 0)
            self.assertIn("paper_mode_gate=True", paper_result.stdout)
            self.assertIn("paper_mode_passed=False", paper_result.stdout)
            self.assertIn("missing 1,2", paper_result.stdout)
            self.assertTrue((eval_dir / "paper_mode_gate.json").exists())

            paper_pass = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "eval",
                    "--suite",
                    "neural_translation_v1",
                    "--event-manifest",
                    str(prep_dir / "event_manifest.json"),
                    "--split-manifest",
                    str(prep_dir / "split_manifest.json"),
                    "--out-dir",
                    str(eval_dir),
                    "--train-steps",
                    "1",
                    "--paper-mode",
                    "--seeds",
                    "0",
                    "1",
                    "2",
                ],
                check=True,
                text=True,
                capture_output=True,
                env=env,
            )
            paper_artifact = json.loads((eval_dir / "prepared_baseline_suite.json").read_text(encoding="utf-8"))
            self.assertIn("paper_mode_passed=True", paper_pass.stdout)
            self.assertEqual(paper_artifact["seeds"], [0, 1, 2])
            self.assertEqual([row["seed"] for row in paper_artifact["seed_results"]], [0, 1, 2])
            self.assertTrue(paper_artifact["seed_aggregate"])
            self.assertNotEqual(paper_artifact["tasks"], paper_artifact["seed_results"][0]["tasks"])
            self.assertIn("representative_seed_tasks", paper_artifact)
            self.assertTrue(all(task["status"] == "seed_aggregated" for task in paper_artifact["tasks"].values()))
            self.assertIn("status=seed_aggregated", paper_pass.stdout)
            self.assertTrue((eval_dir / "seed_aggregate.json").exists())
            self.assertTrue((eval_dir / "seed_aggregate.csv").exists())

    def test_prepared_baseline_multi_seed_satisfies_paper_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            prep_dir = Path(tmp) / "prepared"
            eval_dir = Path(tmp) / "eval"
            records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=1, modalities=("eeg", "fmri"))
            batches = make_synthetic_event_batches(n_subjects=6, sessions_per_subject=1, modalities=("eeg", "fmri"))
            split = build_split_manifest(records, policy="subject", seed=0)
            save_split_manifest(split, prep_dir / "split_manifest.json")
            save_event_batches(batches, prep_dir)

            single = run_prepared_baseline_suite(
                PreparedSuiteConfig(
                    event_manifest=prep_dir / "event_manifest.json",
                    split_manifest=prep_dir / "split_manifest.json",
                    train_steps=1,
                ),
            )
            self.assertEqual(single["seeds"], [0])

            payload = run_prepared_baseline_suite_multi_seed(
                PreparedSuiteConfig(
                    event_manifest=prep_dir / "event_manifest.json",
                    split_manifest=prep_dir / "split_manifest.json",
                    train_steps=1,
                ),
                out_dir=eval_dir,
            )

            self.assertTrue(payload["paper_mode_gate"]["passed"], payload["paper_mode_gate"]["violations"])
            self.assertEqual(tuple(payload["paper_mode_gate"]["observed_seeds"]), (0, 1, 2))
            self.assertEqual([row["seed"] for row in payload["seed_results"]], [0, 1, 2])
            self.assertTrue(payload["aggregate"]["aggregate_rank"])
            self.assertTrue(payload["seed_aggregate"])
            self.assertNotEqual(payload["tasks"], payload["seed_results"][0]["tasks"])
            self.assertIn("representative_seed_tasks", payload)
            self.assertTrue(all(task["status"] == "seed_aggregated" for task in payload["tasks"].values()))
            report = format_prepared_baseline_report(payload)
            self.assertIn("status=seed_aggregated", report)
            example = payload["seed_aggregate"][0]
            for key in ("task_id", "model_id", "metric", "mean", "std", "ci_low", "ci_high", "n_seeds"):
                self.assertIn(key, example)
            self.assertTrue((eval_dir / "prepared_baseline_suite.json").exists())
            self.assertTrue((eval_dir / "seed_aggregate.json").exists())
            self.assertTrue((eval_dir / "seed_aggregate.csv").exists())

    def test_aggregate_seed_ranks_preserves_fractional_concrete_ranks(self):
        rows = _aggregate_seed_ranks(
            [
                {
                    "seed": 0,
                    "tasks": {
                        "task_a": {
                            "ranking": [
                                {"model_id": "model_a", "rank": 1.0},
                                {"model_id": "model_b", "rank": 2.0},
                            ]
                        },
                        "task_b": {
                            "ranking": [
                                {"model_id": "model_a", "rank": 1.2},
                                {"model_id": "model_b", "rank": 1.8},
                            ]
                        },
                    },
                    "aggregate": {
                        "aggregate_rank": [
                            {"model_id": "model_a", "mean_rank": 9.0},
                            {"model_id": "model_b", "mean_rank": 9.0},
                        ]
                    },
                }
            ]
        )

        by_model = {row["model_id"]: row for row in rows}
        self.assertAlmostEqual(by_model["model_a"]["mean_rank"], 1.1)
        self.assertAlmostEqual(by_model["model_b"]["mean_rank"], 1.9)
        self.assertEqual([row["model_id"] for row in rows], ["model_a", "model_b"])
        self.assertEqual(by_model["model_a"]["tasks_ranked"], 2)
        self.assertEqual(by_model["model_a"]["n_seeds"], 1)

    def test_eval_command_service_returns_prepared_paper_mode_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            prep_dir = Path(tmp) / "prepared"
            eval_dir = Path(tmp) / "eval"
            records = make_synthetic_recordings(n_subjects=6, sessions_per_subject=1, modalities=("eeg", "fmri"))
            batches = make_synthetic_event_batches(n_subjects=6, sessions_per_subject=1, modalities=("eeg", "fmri"))
            split = build_split_manifest(records, policy="subject", seed=0)
            save_split_manifest(split, prep_dir / "split_manifest.json")
            save_event_batches(batches, prep_dir)

            result = run_eval_command(
                EvalCommandConfig(
                    suite="neural_translation_v1",
                    event_manifest=prep_dir / "event_manifest.json",
                    split_manifest=prep_dir / "split_manifest.json",
                    out_dir=eval_dir,
                    train_steps=1,
                    paper_mode=True,
                )
            )

            self.assertEqual(result.exit_code, 1)
            self.assertIn("eval_audit_passed=True", result.output)
            self.assertIn("paper_mode_passed=False", result.output)
            self.assertTrue((eval_dir / "prepared_baseline_suite.json").exists())
            self.assertTrue((eval_dir / "paper_mode_gate.json").exists())


if __name__ == "__main__":
    unittest.main()
