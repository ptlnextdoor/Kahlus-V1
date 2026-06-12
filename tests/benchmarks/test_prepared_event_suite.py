import json
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurotwin.adapters.synthetic import (
    make_synthetic_event_batches,
    make_synthetic_multimodal_event_batches,
    make_synthetic_multimodal_recordings,
    make_synthetic_recordings,
)
from neurotwin.benchmarks.prepared_suite import (
    PreparedSuiteConfig,
    _subject_adaptation_from_windows,
    build_prepared_window_tasks,
    format_prepared_baseline_report,
    run_prepared_baseline_suite,
)
from neurotwin.data.prepared_tasks import prepared_windows_by_split
from neurotwin.data.event_io import event_manifest_summary, load_event_batches, save_event_batches
from neurotwin.data.schemas import NeuralEventBatch
from neurotwin.data.split_manifest import build_split_manifest
from neurotwin.data.manifest_io import save_split_manifest
from neurotwin.eval.command import EvalCommandConfig, run_eval_command
from neurotwin.contracts.paper_mode import CANONICAL_REQUIRED_SEEDS
from neurotwin.eval.paper_contracts import build_paper_mode_evidence
from neurotwin.eval.paper_gate import validate_paper_mode_payload
from neurotwin.eval.prepared_paper_mode import _aggregate_seed_ranks, run_prepared_baseline_suite_multi_seed


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

    def test_prepared_windows_can_be_capped_per_split(self):
        records = make_synthetic_recordings(n_subjects=9, sessions_per_subject=2, modalities=("fmri",))
        batches = make_synthetic_event_batches(n_subjects=9, sessions_per_subject=2, modalities=("fmri",), n_time=80)
        split = build_split_manifest(records, policy="subject", seed=0)

        windows = prepared_windows_by_split(
            batches,
            split,
            window_length=8,
            stride=4,
            max_windows_per_split=3,
        )

        self.assertEqual({key: len(value) for key, value in windows.items()}, {"train": 3, "val": 3, "test": 3})

    def test_prepared_baseline_suite_reports_window_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            prep_dir = Path(tmp) / "prepared"
            records = make_synthetic_recordings(n_subjects=9, sessions_per_subject=2, modalities=("fmri",))
            batches = make_synthetic_event_batches(n_subjects=9, sessions_per_subject=2, modalities=("fmri",), n_time=80)
            split = build_split_manifest(records, policy="subject", seed=0)
            save_split_manifest(split, prep_dir / "split_manifest.json")
            save_event_batches(batches, prep_dir)

            payload = run_prepared_baseline_suite(
                PreparedSuiteConfig(
                    event_manifest=prep_dir / "event_manifest.json",
                    split_manifest=prep_dir / "split_manifest.json",
                    train_steps=1,
                    max_windows_per_split=3,
                ),
            )

        self.assertEqual(payload["prepared_data"]["max_windows_per_split"], 3)
        report = format_prepared_baseline_report(payload)
        self.assertIn("max_windows_per_split=3", report)

    def test_prepared_baseline_suite_can_filter_models_for_debug_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            prep_dir = Path(tmp) / "prepared"
            records = make_synthetic_recordings(n_subjects=9, sessions_per_subject=2, modalities=("fmri",))
            batches = make_synthetic_event_batches(n_subjects=9, sessions_per_subject=2, modalities=("fmri",), n_time=80)
            split = build_split_manifest(records, policy="subject", seed=0)
            save_split_manifest(split, prep_dir / "split_manifest.json")
            save_event_batches(batches, prep_dir)

            payload = run_prepared_baseline_suite(
                PreparedSuiteConfig(
                    event_manifest=prep_dir / "event_manifest.json",
                    split_manifest=prep_dir / "split_manifest.json",
                    train_steps=1,
                    max_windows_per_split=3,
                    model_ids=("train_mean", "linear_ridge"),
                ),
            )

        ranked_models = {
            row["model_id"]
            for task in payload["tasks"].values()
            for row in task.get("ranking", [])
        }
        self.assertEqual(ranked_models, {"train_mean", "linear_ridge"})
        self.assertNotIn("mlp", ranked_models)
        self.assertNotIn("pair_operator", ranked_models)

    def test_eval_suite_cli_config_passes_filtered_debug_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            prep_dir = Path(tmp) / "prepared"
            out_dir = Path(tmp) / "out"
            records = make_synthetic_recordings(n_subjects=9, sessions_per_subject=2, modalities=("fmri",))
            batches = make_synthetic_event_batches(n_subjects=9, sessions_per_subject=2, modalities=("fmri",), n_time=80)
            split = build_split_manifest(records, policy="subject", seed=0)
            save_split_manifest(split, prep_dir / "split_manifest.json")
            save_event_batches(batches, prep_dir)

            result = run_eval_command(
                EvalCommandConfig(
                    suite="neural_translation_v1",
                    event_manifest=prep_dir / "event_manifest.json",
                    split_manifest=prep_dir / "split_manifest.json",
                    out_dir=out_dir,
                    train_steps=1,
                    max_windows_per_split=3,
                    baseline_model_ids=("train_mean", "linear_ridge"),
                )
            )

            self.assertEqual(result.exit_code, 0)
            payload = json.loads((out_dir / "prepared_baseline_suite.json").read_text(encoding="utf-8"))
        ranked_models = {
            row["model_id"]
            for task in payload["tasks"].values()
            for row in task.get("ranking", [])
        }
        self.assertEqual(ranked_models, {"train_mean", "linear_ridge"})

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
        stimulus_evidence = payload["prepared_data"]["stimulus_evidence"]
        self.assertIn("mse_ci_low", metrics)
        self.assertIn("mse_ci_high", metrics)
        self.assertIn("tribe_style", {row["model_id"] for row in task["ranking"]})
        self.assertIn("brainvista_style", {row["model_id"] for row in task["ranking"]})
        self.assertIn("pair_operator", {row["model_id"] for row in task["ranking"]})
        self.assertEqual(stimulus_evidence["status"], "plumbing_only")
        self.assertFalse(stimulus_evidence["claim_eligible"])
        self.assertTrue(
            any(
                row["model_id"] == "tribe_style" and row["status"] == "clean_room_approximation"
                for row in payload["baseline_catalog"]
            )
        )
        report = format_prepared_baseline_report(payload)
        self.assertIn("tribe_style: clean_room_approximation", report)
        self.assertIn("not an exact TRIBE v2 reproduction", report)
        self.assertIn("stimulus_evidence=plumbing_only claim_eligible=False", report)

    def test_real_stimulus_metadata_marks_stimulus_task_claim_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            prep_dir = Path(tmp) / "prepared"
            feature_path = Path(tmp) / "stimulus_features.bin"
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
                            "stimulus_feature_path": str(feature_path),
                            "stimulus_feature_modalities": ["text", "audio", "video"],
                            "stimulus_feature_hash": feature_hash,
                            "stimulus_feature_status": "real_precomputed",
                        }
                    )
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

        stimulus_evidence = payload["prepared_data"]["stimulus_evidence"]
        self.assertEqual(stimulus_evidence["status"], "real_stimulus_features")
        self.assertTrue(stimulus_evidence["claim_eligible"])
        self.assertEqual(stimulus_evidence["modalities"], ["audio", "text", "video"])
        self.assertTrue(stimulus_evidence["hash_verified"])
        self.assertTrue(stimulus_evidence["source_artifact_hash_verified"])

    def test_file_uri_stimulus_artifact_hash_marks_claim_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            feature_path = Path(tmp) / "stimulus_features.bin"
            feature_bytes = b"real precomputed stimulus features via file uri"
            feature_path.write_bytes(feature_bytes)
            feature_hash = hashlib.sha256(feature_bytes).hexdigest()

            stimulus_evidence = self._stimulus_evidence_for_metadata(
                tmp,
                {
                    "require_real_stimulus": True,
                    "stimulus_feature_source": "sentence_transformer_audio_video_cache",
                    "stimulus_feature_uri": f"file://{feature_path}",
                    "stimulus_feature_modalities": ["text", "audio", "video"],
                    "stimulus_feature_hash": feature_hash,
                    "stimulus_feature_status": "real_precomputed",
                },
            )

        self.assertEqual(stimulus_evidence["status"], "real_stimulus_features")
        self.assertTrue(stimulus_evidence["claim_eligible"])
        self.assertTrue(stimulus_evidence["source_artifact_hash_verified"])

    def test_stimulus_verified_flag_with_hash_mismatch_is_not_claim_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            feature_path = Path(tmp) / "stimulus_features.bin"
            feature_path.write_bytes(b"actual real precomputed stimulus features")
            declared_hash = hashlib.sha256(b"different stimulus features").hexdigest()

            stimulus_evidence = self._stimulus_evidence_for_metadata(
                tmp,
                {
                    "require_real_stimulus": True,
                    "stimulus_feature_source": "sentence_transformer_audio_video_cache",
                    "stimulus_feature_path": str(feature_path),
                    "stimulus_feature_modalities": ["text", "audio", "video"],
                    "stimulus_feature_hash": declared_hash,
                    "stimulus_feature_hash_verified": True,
                    "stimulus_feature_status": "real_precomputed",
                },
            )

        self.assertEqual(stimulus_evidence["status"], "hash_mismatch")
        self.assertFalse(stimulus_evidence["claim_eligible"])
        self.assertFalse(stimulus_evidence["hash_verified"])
        self.assertFalse(stimulus_evidence["source_artifact_hash_verified"])
        self.assertIn("stimulus_feature_hash is not verified", stimulus_evidence["failure_reasons"])

    def test_embedding_hash_match_does_not_verify_mismatched_source_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            prep_dir = Path(tmp) / "prepared"
            feature_path = Path(tmp) / "stimulus_features.bin"
            feature_path.write_bytes(b"mismatched source artifact bytes")
            records = make_synthetic_multimodal_recordings(n_subjects=6, sessions_per_subject=1, include_unpaired=True)
            batches = make_synthetic_multimodal_event_batches(n_subjects=6, sessions_per_subject=1, include_unpaired=True)
            declared_hash = ""
            for batch in batches:
                if batch.modality == "fmri" and batch.stimulus_embedding is not None:
                    embedding = np.full(batch.stimulus_embedding.shape, 0.125, dtype=np.float32)
                    batch.stimulus_embedding = embedding
                    declared_hash = hashlib.sha256(np.ascontiguousarray(embedding[:8]).tobytes()).hexdigest()
                    batch.metadata.update(
                        {
                            "require_real_stimulus": True,
                            "stimulus_feature_source": "sentence_transformer_audio_video_cache",
                            "stimulus_feature_path": str(feature_path),
                            "stimulus_feature_modalities": ["text", "audio", "video"],
                            "stimulus_feature_hash": declared_hash,
                            "stimulus_feature_status": "real_precomputed",
                        }
                    )
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

        stimulus_evidence = payload["prepared_data"]["stimulus_evidence"]
        self.assertEqual(stimulus_evidence["status"], "hash_mismatch")
        self.assertFalse(stimulus_evidence["claim_eligible"])
        self.assertFalse(stimulus_evidence["source_artifact_hash_verified"])
        self.assertFalse(stimulus_evidence["hash_verified"])
        self.assertEqual(stimulus_evidence["stimulus_embedding_hash"], declared_hash)
        self.assertIn("stimulus_feature_hash is not verified", stimulus_evidence["failure_reasons"])

    def test_stimulus_verified_flag_with_missing_artifact_is_not_claim_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_feature_path = Path(tmp) / "missing_stimulus_features.bin"
            declared_hash = hashlib.sha256(b"real precomputed stimulus features").hexdigest()

            stimulus_evidence = self._stimulus_evidence_for_metadata(
                tmp,
                {
                    "require_real_stimulus": True,
                    "stimulus_feature_source": "sentence_transformer_audio_video_cache",
                    "stimulus_feature_path": str(missing_feature_path),
                    "stimulus_feature_modalities": ["text", "audio", "video"],
                    "stimulus_feature_hash": declared_hash,
                    "stimulus_feature_hash_verified": True,
                    "stimulus_feature_status": "real_precomputed",
                },
            )

        self.assertEqual(stimulus_evidence["status"], "unverified")
        self.assertFalse(stimulus_evidence["claim_eligible"])
        self.assertFalse(stimulus_evidence["hash_verified"])
        self.assertIn("stimulus feature source artifact is not verifiable", stimulus_evidence["failure_reasons"])

    def test_transcript_hash_stimulus_features_remain_plumbing_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            feature_path = Path(tmp) / "stimulus_features.bin"
            feature_bytes = b"hash-derived placeholder"
            feature_path.write_bytes(feature_bytes)

            stimulus_evidence = self._stimulus_evidence_for_metadata(
                tmp,
                {
                    "require_real_stimulus": True,
                    "stimulus_feature_source": "transcript_hash",
                    "stimulus_feature_path": str(feature_path),
                    "stimulus_feature_modalities": ["text"],
                    "stimulus_feature_hash": hashlib.sha256(feature_bytes).hexdigest(),
                    "stimulus_feature_status": "real_precomputed",
                },
            )

        self.assertEqual(stimulus_evidence["status"], "plumbing_only")
        self.assertFalse(stimulus_evidence["claim_eligible"])
        self.assertIn("stimulus_feature_source looks hash-derived", stimulus_evidence["failure_reasons"])

    def test_require_real_stimulus_without_hash_or_source_is_not_claim_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            prep_dir = Path(tmp) / "prepared"
            records = make_synthetic_multimodal_recordings(n_subjects=6, sessions_per_subject=1, include_unpaired=True)
            batches = make_synthetic_multimodal_event_batches(n_subjects=6, sessions_per_subject=1, include_unpaired=True)
            for batch in batches:
                if batch.modality == "fmri" and batch.stimulus_embedding is not None:
                    batch.metadata.update(
                        {
                            "require_real_stimulus": True,
                            "stimulus_feature_source": "sentence_transformer_audio_video_cache",
                            "stimulus_feature_modalities": ["text", "audio", "video"],
                            "stimulus_feature_status": "real_precomputed",
                        }
                    )
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

        stimulus_evidence = payload["prepared_data"]["stimulus_evidence"]
        self.assertEqual(stimulus_evidence["status"], "unverified")
        self.assertFalse(stimulus_evidence["claim_eligible"])
        self.assertIn("stimulus_feature_hash is missing", stimulus_evidence["failure_reasons"])
        self.assertIn("stimulus feature path/manifest/uri is missing", stimulus_evidence["failure_reasons"])

    def _stimulus_evidence_for_metadata(self, tmp: str, metadata: dict[str, object]) -> dict[str, object]:
        prep_dir = Path(tmp) / "prepared"
        records = make_synthetic_multimodal_recordings(n_subjects=6, sessions_per_subject=1, include_unpaired=True)
        batches = make_synthetic_multimodal_event_batches(n_subjects=6, sessions_per_subject=1, include_unpaired=True)
        for batch in batches:
            if batch.modality == "fmri" and batch.stimulus_embedding is not None:
                batch.metadata.update(metadata)
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
        stimulus_evidence = payload["prepared_data"]["stimulus_evidence"]
        self.assertIsInstance(stimulus_evidence, dict)
        return stimulus_evidence

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
                    **{"required_seeds": (0,)},
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
            self.assertIn("missing 0,1,2", paper_result.stdout)
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
            for seed_result in paper_artifact["seed_results"]:
                for task_result in seed_result["tasks"].values():
                    metrics = task_result.get("metrics", {})
                    self.assertFalse(any(str(key).endswith(("_ci_low", "_ci_high")) for key in metrics))
            self.assertIn("status=seed_aggregated", paper_pass.stdout)
            self.assertTrue((eval_dir / "seed_aggregate.json").exists())
            self.assertTrue((eval_dir / "seed_aggregate.csv").exists())

    def test_subject_adaptation_support_minutes_use_selected_subject_rate(self):
        def window(subject_id: str, sampling_rate: float, offset: float) -> NeuralEventBatch:
            return NeuralEventBatch(
                modality="eeg",
                dataset="synthetic",
                subject_id=subject_id,
                session_id="session-0",
                site_id="site-0",
                time=np.arange(4, dtype=np.float32),
                signal=np.full((4, 2), offset, dtype=np.float32),
                mask=np.ones((4, 2), dtype=bool),
                stimulus_embedding=None,
                behavior={},
                space_index=np.arange(2),
                metadata={"sampling_rate": sampling_rate, "record_id": f"{subject_id}-{offset}"},
            )

        result = _subject_adaptation_from_windows(
            {
                "train": [window("train-subject", 10.0, 0.0)],
                "val": [],
                "test": [
                    window("z-first-test-window", 1000.0, 1.0),
                    window("a-selected-subject", 10.0, 2.0),
                    window("a-selected-subject", 10.0, 3.0),
                ],
            }
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.metrics["k1_support_minutes"], 4.0 / 10.0 / 60.0)

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

            gate = validate_paper_mode_payload(payload, audit_report={"passed": True})
            self.assertTrue(gate.passed, gate.violations)
            self.assertEqual(gate.observed_seeds, (0, 1, 2))
            self.assertEqual([row["seed"] for row in payload["seed_results"]], [0, 1, 2])
            self.assertTrue(payload["aggregate"]["aggregate_rank"])
            self.assertTrue(payload["seed_aggregate"])
            self.assertNotEqual(payload["tasks"], payload["seed_results"][0]["tasks"])
            self.assertIn("representative_seed_tasks", payload)
            self.assertTrue(all(task["status"] == "seed_aggregated" for task in payload["tasks"].values()))
            for seed_result in payload["seed_results"]:
                for task_result in seed_result["tasks"].values():
                    metrics = task_result.get("metrics", {})
                    self.assertFalse(any(str(key).endswith(("_ci_low", "_ci_high")) for key in metrics))
            report = format_prepared_baseline_report(payload)
            self.assertIn("status=seed_aggregated", report)
            example = payload["seed_aggregate"][0]
            for key in ("task_id", "model_id", "metric", "mean", "std", "ci_low", "ci_high", "n_seeds"):
                self.assertIn(key, example)
            self.assertTrue((eval_dir / "prepared_baseline_suite.json").exists())
            self.assertTrue((eval_dir / "seed_aggregate.json").exists())
            self.assertTrue((eval_dir / "seed_aggregate.csv").exists())

    def test_paper_mode_producer_and_gate_share_aggregate_rank_contract(self):
        def seed_record(seed: int, rank: float) -> dict[str, object]:
            mse = 0.1 + seed * 0.01
            return {
                "seed": seed,
                "tasks": {
                    "future_state_forecasting": {
                        "ranking": [{"model_id": "linear_ridge", "metric": "mse", "value": mse, "rank": rank}],
                        "metrics_by_model": {
                            "linear_ridge": {
                                "mse": mse,
                                "mse_ci_low": mse - 0.01,
                                "mse_ci_high": mse + 0.01,
                            }
                        },
                    }
                },
            }

        evidence = build_paper_mode_evidence(
            [seed_record(0, 1.0), seed_record(1, 1.2), seed_record(2, 1.4)],
            required_seeds=CANONICAL_REQUIRED_SEEDS,
            require_ci=True,
        )
        payload = {
            "aggregate": {
                "selection_metric": "mse",
                "higher_is_better": False,
                "aggregate_rank": [row.to_dict() for row in evidence.aggregate_rank],
            },
            "seed_results": list(evidence.seed_results),
            "seed_aggregate": [row.to_dict() for row in evidence.seed_aggregate],
        }

        gate = validate_paper_mode_payload(payload, audit_report={"passed": True})

        self.assertTrue(gate.passed, gate.violations)
        self.assertAlmostEqual(payload["aggregate"]["aggregate_rank"][0]["mean_rank"], 1.2)

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

    def test_eval_command_runs_leakage_demo_and_identity_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            leakage = run_eval_command(
                EvalCommandConfig(
                    eval_command="leakage-demo",
                    out_dir=out_dir,
                    seed=0,
                )
            )
            identity = run_eval_command(
                EvalCommandConfig(
                    eval_command="identity-probe",
                    out_dir=out_dir,
                    seed=0,
                )
            )

            self.assertEqual(leakage.exit_code, 0)
            self.assertIn("eval_leakage_demo=True", leakage.output)
            self.assertIn("bad_segment_split", leakage.output)
            self.assertFalse(leakage.payload["scientific_claim_allowed"])
            self.assertEqual(leakage.payload["observed_seeds"], [0])
            self.assertEqual(leakage.payload["evidence_status"], "single_seed_non_paper")
            self.assertFalse(leakage.payload["paper_demo_gate"]["passed"])
            bad = [row for row in leakage.payload["comparisons"] if row["split_id"] == "bad_segment_split"][0]
            self.assertEqual(bad["status"], "negative_control")
            self.assertGreater(bad["subject_overlap"], 0)
            self.assertTrue((out_dir / "leakage_demo.json").exists())
            self.assertTrue((out_dir / "LEAKAGE_DEMO.json").exists())

            self.assertEqual(identity.exit_code, 0)
            self.assertIn("eval_identity_probe=True", identity.output)
            self.assertIn("identity_confounding_risk=", identity.output)
            self.assertFalse(identity.payload["scientific_claim_allowed"])
            self.assertEqual(identity.payload["observed_seeds"], [0])
            self.assertEqual(identity.payload["evidence_status"], "single_seed_non_paper")
            self.assertFalse(identity.payload["paper_demo_gate"]["passed"])
            self.assertGreater(identity.payload["window_split_probe"]["subject_overlap"], 0)
            self.assertTrue((out_dir / "identity_probe.json").exists())
            self.assertTrue((out_dir / "IDENTITY_PROBE.json").exists())

    def test_eval_command_runs_multi_seed_paper_demos(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            leakage = run_eval_command(
                EvalCommandConfig(
                    eval_command="leakage-demo",
                    out_dir=out_dir,
                    seeds=(0, 1, 2),
                    train_steps=1,
                )
            )
            identity = run_eval_command(
                EvalCommandConfig(
                    eval_command="identity-probe",
                    out_dir=out_dir,
                    seeds=(0, 1, 2),
                    train_steps=1,
                )
            )

            self.assertEqual(leakage.exit_code, 0)
            self.assertEqual(leakage.payload["observed_seeds"], [0, 1, 2])
            self.assertEqual(len(leakage.payload["seed_results"]), 3)
            self.assertTrue(leakage.payload["paper_demo_gate"]["passed"])
            self.assertNotIn("seed", leakage.payload)
            self.assertNotIn("comparisons", leakage.payload)
            self.assertIn("representative_seed_result", leakage.payload)
            self.assertLess(leakage.output.index("seed_aggregate="), leakage.output.index("representative_split_result="))
            self.assertTrue(
                any(
                    row["split_id"] == "bad_segment_split" and row["metric"] == "mse" and row["n_seeds"] == 3
                    for row in leakage.payload["seed_aggregate"]
                )
            )

            self.assertEqual(identity.exit_code, 0)
            self.assertEqual(identity.payload["observed_seeds"], [0, 1, 2])
            self.assertEqual(len(identity.payload["seed_results"]), 3)
            self.assertTrue(identity.payload["paper_demo_gate"]["passed"])
            self.assertNotIn("seed", identity.payload)
            self.assertNotIn("window_split_probe", identity.payload)
            self.assertIn("representative_seed_result", identity.payload)
            self.assertLess(identity.output.index("seed_aggregate="), identity.output.index("representative_window_split_accuracy="))
            self.assertTrue(any(row["metric"] == "accuracy" and row["n_seeds"] == 3 for row in identity.payload["seed_aggregate"]))

    def test_eval_command_seed_reaches_synthetic_suites(self):
        smoke_a = run_eval_command(EvalCommandConfig(suite="translation_smoke", seed=3))
        smoke_b = run_eval_command(EvalCommandConfig(suite="translation_smoke", seed=4))
        v1_a = run_eval_command(EvalCommandConfig(suite="neural_translation_v1", seed=3))
        v1_b = run_eval_command(EvalCommandConfig(suite="neural_translation_v1", seed=4))

        self.assertNotEqual(smoke_a.output, smoke_b.output)
        self.assertNotEqual(v1_a.payload["future_state_forecasting"]["metrics"], v1_b.payload["future_state_forecasting"]["metrics"])


if __name__ == "__main__":
    unittest.main()
