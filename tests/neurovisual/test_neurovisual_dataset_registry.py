import json
import tarfile
from hashlib import sha256
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurotwin.neurovisual import (
    build_metadata_only_adapter_plan,
    build_metadata_query_plan,
    build_registry_verification_summary,
    build_seed_dataset_registry,
    build_split_audit_plan,
    validate_registry_entry,
)


class NeurovisualDatasetRegistryTests(unittest.TestCase):
    def test_registry_entries_validate_required_fields_and_scores(self):
        registry = build_seed_dataset_registry()
        names = {entry["dataset_name"] for entry in registry["entries"]}

        self.assertIn("HBN-EEG / EEG Foundation Challenge", names)
        self.assertIn("CHB-MIT Scalp EEG Database", names)
        self.assertIn("TUSZ / TUH EEG Seizure Corpus", names)
        for entry in registry["entries"]:
            validate_registry_entry(entry)
            self.assertIn(entry["neurovisual_relevance_score"], (0, 1, 2, 3))
            if entry["neurovisual_relevance_score"] == 3:
                self.assertTrue(
                    entry["subjective_symptom_annotations_available"]
                    or "clinician-coded neurovisual phenotype" in entry["notes"].lower()
                )

    def test_openneuro_leads_remain_unverified_without_confirmed_metadata(self):
        registry = build_seed_dataset_registry()
        openneuro_entries = [entry for entry in registry["entries"] if "OpenNeuro" in entry["dataset_name"]]

        self.assertTrue(openneuro_entries)
        for entry in openneuro_entries:
            self.assertEqual(entry["verification_status"], "unverified")
            self.assertNotIn("ds003838", entry["source_url_or_identifier"])
            self.assertNotIn("ds003478", entry["source_url_or_identifier"])

    def test_registry_verification_summary_tracks_sources_and_priorities(self):
        registry = build_seed_dataset_registry(date_checked="2026-06-20")
        summary = build_registry_verification_summary(registry)

        self.assertEqual(summary["schema"], "kahlus.nv1.registry_verification_summary.v1")
        self.assertEqual(summary["metadata_retrieval_date"], "2026-06-20")
        self.assertFalse(summary["execution"]["bulk_dataset_download"])
        self.assertFalse(summary["execution"]["a100_jobs_launched"])
        self.assertEqual(summary["counts"]["confirmed"], 3)
        self.assertEqual(summary["counts"]["unverified"], 2)
        self.assertEqual(summary["counts"]["rejected"], 1)
        self.assertEqual(
            [row["dataset_name"] for row in summary["top_adapter_priorities"]],
            [
                "HBN-EEG / EEG Foundation Challenge",
                "CHB-MIT Scalp EEG Database",
                "TUSZ / TUH EEG Seizure Corpus",
            ],
        )
        self.assertTrue(summary["openneuro_verification_results"])
        for row in summary["openneuro_verification_results"]:
            self.assertEqual(row["verification_status"], "unverified")
            self.assertEqual(row["confirmed_accession"], None)

    def test_registry_exposes_metadata_source_urls_separately(self):
        registry = build_seed_dataset_registry(date_checked="2026-06-20")
        summary = build_registry_verification_summary(registry)

        for entry in registry["entries"]:
            self.assertIn("metadata_source_url", entry)
            self.assertTrue(str(entry["metadata_source_url"]).strip(), entry["dataset_name"])
        self.assertIn("metadata_source_urls", summary)
        self.assertIn("https://physionet.org/content/chbmit/1.0.0/", summary["metadata_source_urls"])
        self.assertIn("https://arxiv.org/abs/2506.19141", summary["metadata_source_urls"])
        for row in summary["openneuro_verification_results"]:
            self.assertEqual(row["metadata_source_url"], "UNVERIFIED")

    def test_metadata_only_adapter_plan_declares_split_and_audit_strategy(self):
        registry = build_seed_dataset_registry(date_checked="2026-06-20")
        plan = build_metadata_only_adapter_plan(registry)

        self.assertEqual(plan["schema"], "kahlus.nv1.adapter_plan.v1")
        self.assertFalse(plan["execution"]["bulk_dataset_download"])
        self.assertFalse(plan["execution"]["a100_jobs_launched"])
        self.assertFalse(plan["execution"]["adapters_implemented"])
        self.assertEqual(
            [row["dataset_name"] for row in plan["planned_adapters"]],
            [
                "HBN-EEG / EEG Foundation Challenge",
                "CHB-MIT Scalp EEG Database",
                "TUSZ / TUH EEG Seizure Corpus",
            ],
        )
        for row in plan["planned_adapters"]:
            self.assertIn("local_manifest_required_fields", row)
            self.assertIn("split_audit_strategy", row)
            self.assertIn("leakage_risks", row)
            self.assertEqual(row["implementation_status"], "planned_not_implemented")

    def test_split_audit_plan_requires_subject_heldout_and_baselines_before_models(self):
        registry = build_seed_dataset_registry(date_checked="2026-06-20")
        split_plan = build_split_audit_plan(registry)

        self.assertEqual(split_plan["schema"], "kahlus.nv1.split_audit_plan.v1")
        self.assertFalse(split_plan["execution"]["split_audit_executed"])
        self.assertFalse(split_plan["execution"]["baselines_run"])
        self.assertFalse(split_plan["execution"]["models_run"])
        self.assertEqual(
            [row["dataset_name"] for row in split_plan["dataset_split_plans"]],
            [
                "HBN-EEG / EEG Foundation Challenge",
                "CHB-MIT Scalp EEG Database",
                "TUSZ / TUH EEG Seizure Corpus",
            ],
        )
        for row in split_plan["dataset_split_plans"]:
            self.assertIn("subject_id", row["required_split_keys"])
            self.assertIn("session_id", row["required_split_keys"])
            self.assertIn("baseline_ladder_before_model", row["gates_before_model"])
            self.assertIn("subject_overlap", row["leakage_checks"])
            self.assertEqual(row["implementation_status"], "planned_not_executed")

    def test_metadata_query_plan_records_catalog_searches_without_confirming_accessions(self):
        registry = build_seed_dataset_registry(date_checked="2026-06-20")
        query_plan = build_metadata_query_plan(registry)

        self.assertEqual(query_plan["schema"], "kahlus.nv1.metadata_query_plan.v1")
        self.assertFalse(query_plan["execution"]["bulk_dataset_download"])
        self.assertFalse(query_plan["execution"]["a100_jobs_launched"])
        self.assertFalse(query_plan["execution"]["metadata_queries_executed"])
        required_fields = query_plan["required_verification_fields"]
        for field in (
            "accession_number",
            "subject_count",
            "task_description",
            "modalities",
            "license_or_access_terms",
            "metadata_source_url",
            "date_checked",
        ):
            self.assertIn(field, required_fields)
        target_ids = {target["target_id"] for target in query_plan["query_targets"]}
        self.assertIn("openneuro_nemar_multimodal_working_memory", target_ids)
        self.assertIn("openneuro_nemar_bdi_reward_selection", target_ids)
        self.assertIn("moabb_neurovisual_relevant_eeg", target_ids)
        self.assertIn("eegdash_public_multimodal_eeg", target_ids)
        for target in query_plan["query_targets"]:
            self.assertEqual(target["verification_status"], "planned_not_executed")
            self.assertEqual(target["confirmed_accession"], None)
            self.assertNotIn("ds003838", json.dumps(target))
            self.assertNotIn("ds003478", json.dumps(target))

    def test_registry_builder_writes_artifacts_without_download_or_a100(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_neurovisual_dataset_registry.py",
                    "--out",
                    tmp,
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            registry = json.loads((Path(tmp) / "neurovisual_dataset_registry.json").read_text(encoding="utf-8"))
            summary = json.loads(
                (Path(tmp) / "neurovisual_registry_verification_summary.json").read_text(encoding="utf-8")
            )
            adapter_plan = json.loads((Path(tmp) / "neurovisual_adapter_plan.json").read_text(encoding="utf-8"))
            claim_gate = json.loads((Path(tmp) / "neurovisual_registry_claim_gate.json").read_text(encoding="utf-8"))
            query_plan = json.loads((Path(tmp) / "neurovisual_metadata_query_plan.json").read_text(encoding="utf-8"))
            manifest_schema = json.loads(
                (Path(tmp) / "neurovisual_local_manifest_schema.json").read_text(encoding="utf-8")
            )
            split_audit_plan = json.loads((Path(tmp) / "neurovisual_split_audit_plan.json").read_text(encoding="utf-8"))
            synthetic_split_manifest = json.loads(
                (Path(tmp) / "neurovisual_synthetic_split_manifest.json").read_text(encoding="utf-8")
            )
            synthetic_split_audit = json.loads(
                (Path(tmp) / "neurovisual_synthetic_split_audit.json").read_text(encoding="utf-8")
            )
            evidence_manifest = json.loads(
                (Path(tmp) / "neurovisual_registry_evidence_manifest.json").read_text(encoding="utf-8")
            )
            report = (Path(tmp) / "neurovisual_dataset_registry.md").read_text(encoding="utf-8")
            expected_artifacts = {}
            for artifact_name in (
                "neurovisual_dataset_registry.json",
                "neurovisual_registry_verification_summary.json",
                "neurovisual_adapter_plan.json",
                "neurovisual_registry_claim_gate.json",
                "neurovisual_metadata_query_plan.json",
                "neurovisual_local_manifest_schema.json",
                "neurovisual_split_audit_plan.json",
                "neurovisual_synthetic_split_manifest.json",
                "neurovisual_synthetic_split_audit.json",
                "neurovisual_dataset_registry.md",
            ):
                artifact_path = Path(tmp) / artifact_name
                expected_artifacts[artifact_name] = {
                    "sha256": sha256(artifact_path.read_bytes()).hexdigest(),
                    "size_bytes": artifact_path.stat().st_size,
                }

        self.assertFalse(registry["execution"]["bulk_dataset_download"])
        self.assertFalse(registry["execution"]["a100_jobs_launched"])
        self.assertEqual(summary["counts"]["confirmed"], 3)
        self.assertIn("openneuro_verification_results", summary)
        self.assertFalse(adapter_plan["execution"]["adapters_implemented"])
        self.assertEqual(claim_gate["claim_scope"], "dataset_registry_ready")
        self.assertTrue(claim_gate["passed"], claim_gate)
        self.assertEqual(claim_gate["blocked_claims_found"], [])
        self.assertIn("predicts_seizure", claim_gate["blocked_claim_terms"])
        self.assertEqual(query_plan["schema"], "kahlus.nv1.metadata_query_plan.v1")
        self.assertFalse(query_plan["execution"]["metadata_queries_executed"])
        self.assertEqual(manifest_schema["schema"], "kahlus.nv1.local_manifest_schema.v1")
        self.assertIn("CHB-MIT Scalp EEG Database", manifest_schema["allowed_confirmed_dataset_names"])
        self.assertEqual(split_audit_plan["schema"], "kahlus.nv1.split_audit_plan.v1")
        self.assertFalse(split_audit_plan["execution"]["split_audit_executed"])
        self.assertEqual(len(synthetic_split_manifest), 3)
        self.assertEqual({row["split_name"] for row in synthetic_split_manifest}, {"train", "validation", "test"})
        self.assertTrue(
            all(str(row["signal_path"]).startswith("USER_PROVIDED_") for row in synthetic_split_manifest),
            synthetic_split_manifest,
        )
        self.assertEqual(synthetic_split_audit["schema"], "kahlus.nv1.local_split_audit.v1")
        self.assertTrue(synthetic_split_audit["passed"], synthetic_split_audit)
        self.assertTrue(synthetic_split_audit["execution"]["split_audit_executed"])
        self.assertFalse(synthetic_split_audit["execution"]["raw_file_existence_checked"])
        self.assertFalse(synthetic_split_audit["execution"]["adapters_implemented"])
        self.assertEqual(synthetic_split_audit["split_counts"], {"train": 1, "validation": 1, "test": 1})
        self.assertEqual(evidence_manifest["schema"], "kahlus.nv1.registry_evidence_manifest.v1")
        self.assertFalse(evidence_manifest["execution"]["bulk_dataset_download"])
        self.assertFalse(evidence_manifest["execution"]["a100_jobs_launched"])
        self.assertFalse(evidence_manifest["execution"]["cluster_jobs_launched"])
        manifest_artifacts = {row["path"]: row for row in evidence_manifest["artifacts"]}
        for artifact_name in (
            "neurovisual_dataset_registry.json",
            "neurovisual_registry_verification_summary.json",
            "neurovisual_adapter_plan.json",
            "neurovisual_registry_claim_gate.json",
            "neurovisual_metadata_query_plan.json",
            "neurovisual_local_manifest_schema.json",
            "neurovisual_split_audit_plan.json",
            "neurovisual_synthetic_split_manifest.json",
            "neurovisual_synthetic_split_audit.json",
            "neurovisual_dataset_registry.md",
        ):
            self.assertEqual(manifest_artifacts[artifact_name]["sha256"], expected_artifacts[artifact_name]["sha256"])
            self.assertEqual(
                manifest_artifacts[artifact_name]["size_bytes"],
                expected_artifacts[artifact_name]["size_bytes"],
            )
        self.assertIn("no bulk dataset download", report.lower())
        self.assertIn("adapter_priority", report)

    def test_registry_bundle_audit_passes_clean_bundle_and_fails_tampered_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_neurovisual_dataset_registry.py",
                    "--out",
                    tmp,
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )
            self.assertEqual(build_result.returncode, 0, build_result.stderr + build_result.stdout)

            audit_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_neurovisual_registry_bundle.py",
                    "--artifact-dir",
                    tmp,
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )
            self.assertEqual(audit_result.returncode, 0, audit_result.stderr + audit_result.stdout)
            audit_payload = json.loads(audit_result.stdout)
            self.assertTrue(audit_payload["passed"], audit_payload)
            self.assertEqual(audit_payload["schema"], "kahlus.nv1.registry_bundle_audit.v1")
            self.assertFalse(audit_payload["execution"]["bulk_dataset_download"])
            self.assertFalse(audit_payload["execution"]["a100_jobs_launched"])
            self.assertFalse(audit_payload["execution"]["cluster_jobs_launched"])
            self.assertEqual(audit_payload["failures"], [])
            self.assertIn("neurovisual_metadata_query_plan.json", audit_payload["verified_artifacts"])
            self.assertIn("neurovisual_local_manifest_schema.json", audit_payload["verified_artifacts"])
            self.assertIn("neurovisual_split_audit_plan.json", audit_payload["verified_artifacts"])
            self.assertIn("neurovisual_synthetic_split_manifest.json", audit_payload["verified_artifacts"])
            self.assertIn("neurovisual_synthetic_split_audit.json", audit_payload["verified_artifacts"])

            report_path = Path(tmp) / "neurovisual_dataset_registry.md"
            report_path.write_text(report_path.read_text(encoding="utf-8") + "\nTAMPERED\n", encoding="utf-8")
            tampered_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_neurovisual_registry_bundle.py",
                    "--artifact-dir",
                    tmp,
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(tampered_result.returncode, 0, tampered_result.stdout)
            tampered_payload = json.loads(tampered_result.stdout)
            self.assertFalse(tampered_payload["passed"], tampered_payload)
            self.assertIn("checksum_mismatch:neurovisual_dataset_registry.md", tampered_payload["failures"])

    def test_fixture_replay_smoke_builds_registry_and_replays_split_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_neurovisual_fixture_replay.py",
                    "--out-dir",
                    tmp,
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            evidence_path = Path(tmp) / "neurovisual_fixture_replay_evidence.json"
            self.assertTrue(evidence_path.exists(), result.stdout)
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["schema"], "kahlus.nv1.fixture_replay_smoke.v1")
            self.assertTrue(evidence["passed"], evidence)
            self.assertEqual(evidence["registry_build_returncode"], 0)
            self.assertEqual(evidence["split_cli_returncode"], 0)
            self.assertTrue(evidence["split_audit_passed"], evidence)
            self.assertEqual(evidence["split_counts"], {"train": 1, "validation": 1, "test": 1})
            self.assertFalse(evidence["execution"]["bulk_dataset_download"])
            self.assertFalse(evidence["execution"]["a100_jobs_launched"])
            self.assertFalse(evidence["execution"]["raw_file_existence_checked"])
            self.assertTrue((Path(tmp) / "registry_package" / "neurovisual_dataset_registry.json").exists())
            self.assertTrue((Path(tmp) / "registry_package" / "neurovisual_synthetic_split_manifest.json").exists())

    def test_handoff_manifest_summarizes_replay_evidence_under_claim_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay_dir = root / "fixture_replay"
            replay_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_neurovisual_fixture_replay.py",
                    "--out-dir",
                    str(replay_dir),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )
            self.assertEqual(replay_result.returncode, 0, replay_result.stderr + replay_result.stdout)

            handoff_path = root / "neurovisual_handoff_manifest.json"
            handoff_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_neurovisual_handoff_manifest.py",
                    "--registry-package-dir",
                    str(replay_dir / "registry_package"),
                    "--fixture-replay-evidence",
                    str(replay_dir / "neurovisual_fixture_replay_evidence.json"),
                    "--out",
                    str(handoff_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

            self.assertEqual(handoff_result.returncode, 0, handoff_result.stderr + handoff_result.stdout)
            handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
            self.assertEqual(handoff["schema"], "kahlus.nv1.handoff_manifest.v1")
            self.assertTrue(handoff["passed"], handoff)
            self.assertEqual(handoff["claim_gate"]["claim_scope"], "dataset_registry_ready")
            self.assertTrue(handoff["claim_gate"]["passed"], handoff["claim_gate"])
            self.assertEqual(handoff["claim_gate"]["blocked_claims_found"], [])
            self.assertEqual(handoff["registry_summary"]["confirmed_datasets"], 3)
            self.assertEqual(handoff["fixture_replay_summary"]["split_counts"], {"train": 1, "validation": 1, "test": 1})
            self.assertFalse(handoff["execution"]["bulk_dataset_download"])
            self.assertFalse(handoff["execution"]["a100_jobs_launched"])
            self.assertFalse(handoff["execution"]["adapters_implemented"])
            self.assertFalse(handoff["execution"]["baselines_run"])
            artifact_names = {artifact["path"] for artifact in handoff["input_artifacts"]}
            self.assertIn("neurovisual_dataset_registry.json", artifact_names)
            self.assertIn("neurovisual_fixture_replay_evidence.json", artifact_names)

    def test_handoff_manifest_audit_passes_clean_manifest_and_fails_tampered_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay_dir = root / "fixture_replay"
            replay_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_neurovisual_fixture_replay.py",
                    "--out-dir",
                    str(replay_dir),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )
            self.assertEqual(replay_result.returncode, 0, replay_result.stderr + replay_result.stdout)
            handoff_path = root / "neurovisual_handoff_manifest.json"
            handoff_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_neurovisual_handoff_manifest.py",
                    "--registry-package-dir",
                    str(replay_dir / "registry_package"),
                    "--fixture-replay-evidence",
                    str(replay_dir / "neurovisual_fixture_replay_evidence.json"),
                    "--out",
                    str(handoff_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )
            self.assertEqual(handoff_result.returncode, 0, handoff_result.stderr + handoff_result.stdout)

            audit_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_neurovisual_handoff_manifest.py",
                    "--handoff",
                    str(handoff_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )
            self.assertEqual(audit_result.returncode, 0, audit_result.stderr + audit_result.stdout)
            audit_payload = json.loads(audit_result.stdout)
            self.assertTrue(audit_payload["passed"], audit_payload)
            self.assertEqual(audit_payload["schema"], "kahlus.nv1.handoff_manifest_audit.v1")
            self.assertEqual(audit_payload["failures"], [])
            self.assertIn("neurovisual_dataset_registry.json", audit_payload["verified_artifacts"])
            self.assertFalse(audit_payload["execution"]["bulk_dataset_download"])
            self.assertFalse(audit_payload["execution"]["a100_jobs_launched"])

            registry_path = replay_dir / "registry_package" / "neurovisual_dataset_registry.json"
            registry_path.write_text(registry_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            tampered_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_neurovisual_handoff_manifest.py",
                    "--handoff",
                    str(handoff_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(tampered_result.returncode, 0, tampered_result.stdout)
            tampered_payload = json.loads(tampered_result.stdout)
            self.assertFalse(tampered_payload["passed"], tampered_payload)
            self.assertIn("checksum_mismatch:neurovisual_dataset_registry.json", tampered_payload["failures"])

    def test_local_evidence_gate_runs_replay_handoff_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_neurovisual_local_evidence_gate.py",
                    "--out-dir",
                    tmp,
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            gate_path = Path(tmp) / "neurovisual_local_evidence_gate.json"
            self.assertTrue(gate_path.exists(), result.stdout)
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertEqual(gate["schema"], "kahlus.nv1.local_evidence_gate.v1")
            self.assertTrue(gate["passed"], gate)
            self.assertEqual(gate["fixture_replay_returncode"], 0)
            self.assertEqual(gate["handoff_manifest_returncode"], 0)
            self.assertEqual(gate["handoff_audit_returncode"], 0)
            self.assertTrue(gate["fixture_replay_passed"], gate)
            self.assertTrue(gate["handoff_manifest_passed"], gate)
            self.assertTrue(gate["handoff_audit_passed"], gate)
            self.assertFalse(gate["execution"]["bulk_dataset_download"])
            self.assertFalse(gate["execution"]["a100_jobs_launched"])
            self.assertFalse(gate["execution"]["adapters_implemented"])
            self.assertFalse(gate["execution"]["baselines_run"])
            self.assertTrue((Path(tmp) / "fixture_replay" / "registry_package").exists())
            self.assertTrue((Path(tmp) / "neurovisual_handoff_manifest.json").exists())
            report_path = Path(tmp) / "neurovisual_local_evidence_gate.md"
            self.assertTrue(report_path.exists(), result.stdout)
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("# NV-1 Local Evidence Gate", report)
            self.assertIn("passed: true", report)
            self.assertIn("bulk_dataset_download: false", report)
            self.assertIn("a100_jobs_launched: false", report)
            self.assertIn("does not approve adapters, baselines, models, clinical claims, or A100 execution", report)

    def test_local_evidence_bundle_package_contains_gate_outputs_and_no_raw_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/package_neurovisual_local_evidence_bundle.py",
                    "--out-dir",
                    tmp,
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            manifest_path = Path(tmp) / "neurovisual_local_evidence_bundle_manifest.json"
            archive_path = Path(tmp) / "neurovisual_local_evidence_bundle.tar.gz"
            self.assertTrue(manifest_path.exists(), result.stdout)
            self.assertTrue(archive_path.exists(), result.stdout)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "kahlus.nv1.local_evidence_bundle_manifest.v1")
            self.assertTrue(manifest["passed"], manifest)
            self.assertFalse(manifest["execution"]["bulk_dataset_download"])
            self.assertFalse(manifest["execution"]["a100_jobs_launched"])
            self.assertFalse(manifest["execution"]["raw_private_data_included"])
            self.assertFalse(manifest["execution"]["checkpoints_included"])
            self.assertIn("neurovisual_local_evidence_gate.json", {row["path"] for row in manifest["artifacts"]})
            self.assertIn("neurovisual_local_evidence_gate.md", {row["path"] for row in manifest["artifacts"]})
            self.assertEqual(manifest["archive"]["path"], archive_path.name)
            with tarfile.open(archive_path, "r:gz") as archive:
                names = set(archive.getnames())
            self.assertIn("neurovisual_local_evidence_bundle/neurovisual_local_evidence_gate.json", names)
            self.assertIn("neurovisual_local_evidence_bundle/neurovisual_local_evidence_gate.md", names)
            self.assertIn("neurovisual_local_evidence_bundle/neurovisual_handoff_manifest.json", names)
            self.assertFalse(any(name.endswith((".edf", ".fif", ".bdf", ".set", ".ckpt", ".pt", ".pth")) for name in names))

    def test_requirement_coverage_audit_maps_prompt_requirements_to_generated_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_neurovisual_requirement_coverage.py",
                    "--out-dir",
                    tmp,
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={"PYTHONPATH": "src"},
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            audit_path = Path(tmp) / "neurovisual_requirement_coverage_audit.json"
            report_path = Path(tmp) / "neurovisual_requirement_coverage_audit.md"
            self.assertTrue(audit_path.exists(), result.stdout)
            self.assertTrue(report_path.exists(), result.stdout)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["schema"], "kahlus.nv1.requirement_coverage_audit.v1")
            self.assertTrue(audit["passed"], audit)
            self.assertEqual(audit["failures"], [])
            self.assertEqual(audit["requirements"]["ontology"]["status"], "covered")
            self.assertEqual(audit["requirements"]["intake_smoke"]["status"], "covered")
            self.assertEqual(audit["requirements"]["condition_matrix"]["status"], "covered")
            self.assertEqual(audit["requirements"]["dataset_registry"]["status"], "covered")
            self.assertEqual(audit["requirements"]["claim_gate"]["status"], "covered")
            self.assertEqual(audit["requirements"]["documentation"]["status"], "covered")
            self.assertEqual(audit["requirements"]["deferred_model_experiment_documented_only"]["status"], "covered")
            self.assertFalse(audit["execution"]["bulk_dataset_download"])
            self.assertFalse(audit["execution"]["a100_jobs_launched"])
            self.assertFalse(audit["execution"]["cluster_jobs_launched"])
            self.assertFalse(audit["execution"]["adapters_implemented"])
            self.assertFalse(audit["execution"]["baselines_run"])
            self.assertFalse(audit["execution"]["models_run"])
            self.assertFalse(audit["score_3_assignments"])
            self.assertEqual(
                audit["top_adapter_priorities"],
                [
                    "HBN-EEG / EEG Foundation Challenge",
                    "CHB-MIT Scalp EEG Database",
                    "TUSZ / TUH EEG Seizure Corpus",
                ],
            )
            self.assertEqual(
                {row["verification_status"] for row in audit["openneuro_verification_results"]},
                {"unverified"},
            )
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("# NV-1 Requirement Coverage Audit", report)
            self.assertIn("passed: true", report)
            self.assertIn("deferred model experiment remains documentation-only", report)


if __name__ == "__main__":
    unittest.main()
