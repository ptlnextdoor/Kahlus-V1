import importlib.util
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


EVIDENCE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "package_a100_evidence_bundle.py"
_SPEC = importlib.util.spec_from_file_location("package_a100_evidence_bundle", EVIDENCE_SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
evidence = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = evidence
_SPEC.loader.exec_module(evidence)


class EvidenceBundleArtifactTests(unittest.TestCase):
    def _create_a100_evidence_fixture(
        self,
        root: Path,
        environment_json: str | None = '{"run":{"slurm":{"job_id":"123"}}}\n',
        summary_json: str = (
            '{"status":"completed_prepared_training","completed_steps":50,'
            '"real_data_smoke":true,"scientific_claim_allowed":false}\n'
        ),
        docker_run_env: str | None = (
            "DOCKER_LOG_PATH={persistent}/logs/neurotwin-a100-docker-20260531T000000Z.log\n"
            "DOCKER_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel\n"
        ),
    ) -> Path:
        persistent = root / "persistent"
        run = persistent / "runs" / "moabb_a100_smoke"
        prepared = persistent / "prepared" / "moabb_benchmark"
        logs = persistent / "logs"
        (run / "tables").mkdir(parents=True)
        (run / "figures").mkdir(parents=True)
        prepared.mkdir(parents=True)
        logs.mkdir(parents=True)
        files = {
            persistent / "gpu_preflight.json": (
                '{"passed":true,"visible_gpu_count":6,'
                '"CUDA_VISIBLE_DEVICES":"0,1,2,3,4,5","WORLD_SIZE":"6"}\n'
            ),
            run / "summary.json": summary_json,
            run / "metrics.json": "{}\n",
            run / "metrics.csv": "metric,value\n",
            run / "metrics.jsonl": "{}\n",
            run / "config.yaml": "experiment: moabb_a100_smoke\n",
            run / "split_manifest.json": "{}\n",
            run / "tables" / "metrics_flat.csv": "metric,value\n",
            run / "figures" / "metric_summary.json": "{}\n",
            prepared / "eval_audit.json": '{"passed":true}\n',
            prepared / "data_manifest.json": "{}\n",
            prepared / "event_manifest.json": "{}\n",
            prepared / "split_manifest.json": "{}\n",
            prepared / "leakage_report.json": "{}\n",
            logs / "neurotwin-a100-full-123.out": "ok\n",
            logs / "neurotwin-a100-full-123.err": "\n",
            logs / "neurotwin-a100-full-999.out": "old\n",
            logs / "neurotwin-a100-full-999.err": "old\n",
            logs / "neurotwin-a100-docker-20260531T000000Z.log": "current docker log\n",
            logs / "neurotwin-a100-docker-20260530T000000Z.log": "old docker log\n",
            logs / "other-project.out": "unrelated\n",
        }
        if docker_run_env is not None:
            files[persistent / "docker_run.env"] = docker_run_env.format(persistent=persistent)
        if environment_json is not None:
            files[run / "environment.json"] = environment_json
        for path, body in files.items():
            path.write_text(body, encoding="utf-8")
        (run / "checkpoint.pt").write_bytes(b"checkpoint")
        (run / "checkpoint_best.pt").write_bytes(b"checkpoint")
        (prepared / "events").mkdir()
        (prepared / "events" / "raw_event.npz").write_bytes(b"raw")
        (logs / "pw.txt").write_text("secret\n", encoding="utf-8")
        (logs / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
        (logs / "ssh.key").write_text("secret\n", encoding="utf-8")
        (logs / "runner.tar.gz").write_bytes(b"runner")
        return persistent

    def test_evidence_helpers_select_current_slurm_job_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            persistent = self._create_a100_evidence_fixture(Path(tmp))

            job_id = evidence.current_slurm_job_id(persistent / "runs" / "moabb_a100_smoke")

        self.assertEqual(job_id, "123")

    def test_evidence_helpers_fall_back_to_summary_job_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            persistent = self._create_a100_evidence_fixture(
                Path(tmp),
                environment_json="{}\n",
                summary_json='{"run":{"slurm":{"job_id":"summary-123"}}}\n',
            )

            job_id = evidence.current_slurm_job_id(persistent / "runs" / "moabb_a100_smoke")

        self.assertEqual(job_id, "summary-123")

    def test_evidence_helpers_reject_unsafe_current_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            persistent = self._create_a100_evidence_fixture(root)
            logs = persistent / "logs"

            self.assertIsNone(evidence.safe_child(logs, "../other-project.out"))
            self.assertIsNone(evidence.safe_resolved_path(logs, root / "outside.log"))
            self.assertIsNone(evidence.safe_resolved_path(logs, Path("/tmp/neurotwin-a100-docker-20260531T000000Z.log")))

    def test_evidence_helpers_select_current_docker_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            persistent = self._create_a100_evidence_fixture(Path(tmp))

            log_path = evidence.current_docker_log_path(persistent, persistent / "runs" / "moabb_a100_smoke")

        self.assertIsNotNone(log_path)
        self.assertEqual(log_path.name, "neurotwin-a100-docker-20260531T000000Z.log")

    def test_evidence_helpers_reject_forbidden_bundle_paths(self):
        for name in (
            "pw.txt",
            ".env",
            ".env.local",
            "checkpoint.pt",
            "raw.npz",
            "secret.txt",
            "ssh.key",
            "run/tables/secret-folder/metrics.json",
            "run/figures/password-cache/plot.json",
        ):
            with self.subTest(name=name):
                self.assertTrue(evidence.is_forbidden_bundle_path(Path(name)))
        self.assertFalse(evidence.is_forbidden_bundle_path(Path("docker_run.env")))
        self.assertFalse(evidence.is_forbidden_bundle_path(Path("metrics.json")))

    def test_evidence_helpers_allow_safe_absolute_source_files_under_secret_ancestors(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "password-parent" / "repo"
            repo_root.mkdir(parents=True)
            handoff = repo_root / "README_HANDOFF.md"
            handoff.write_text("# handoff\n", encoding="utf-8")
            payload = repo_root / "environment.json"
            payload.write_text('{"run":{"slurm":{"job_id":"123"}}}\n', encoding="utf-8")

            self.assertTrue(evidence.is_readable_source_file(handoff))
            loaded = evidence.load_json(payload)

        self.assertEqual(loaded["run"]["slurm"]["job_id"], "123")

    def test_write_readmes_copies_safe_handoff_from_secret_named_repo_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_root = root / "secret-parent" / "repo"
            repo_root.mkdir(parents=True)
            expected = "# copied handoff\n"
            (repo_root / "README_HANDOFF.md").write_text(expected, encoding="utf-8")
            (repo_root / "README_HANDOFF.md.in").write_text("fallback ${FULL_SHA}\n", encoding="utf-8")
            stage_root = root / "stage"
            stage_root.mkdir()

            evidence.write_readmes(
                evidence.EvidenceBundleConfig(
                    persistent_root=root / "persistent",
                    zip_path=root / "bundle.zip",
                    evidence_name="evidence",
                    repo_root=repo_root,
                    full_sha="abcdef1234567890",
                ),
                stage_root,
            )
            self.assertEqual((stage_root / "README_HANDOFF.md").read_text(encoding="utf-8"), expected)

    def test_copy_tree_files_filters_using_full_bundle_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tables_source = root / "tables_source"
            figures_source = root / "figures_source"
            stage_root = root / "stage"
            (tables_source / "secret-folder").mkdir(parents=True)
            (tables_source / "safe").mkdir(parents=True)
            (figures_source / "password-cache").mkdir(parents=True)
            (figures_source / "safe").mkdir(parents=True)
            (tables_source / "secret-folder" / "metrics.json").write_text('{"hidden":true}\n', encoding="utf-8")
            (tables_source / "safe" / "metrics.json").write_text('{"ok":true}\n', encoding="utf-8")
            (figures_source / "password-cache" / "plot.json").write_text('{"hidden":true}\n', encoding="utf-8")
            (figures_source / "safe" / "plot.json").write_text('{"ok":true}\n', encoding="utf-8")

            evidence.copy_tree_files(tables_source, stage_root, Path("run") / "tables")
            evidence.copy_tree_files(figures_source, stage_root, Path("run") / "figures")
            self.assertFalse((stage_root / "run" / "tables" / "secret-folder" / "metrics.json").exists())
            self.assertFalse((stage_root / "run" / "figures" / "password-cache" / "plot.json").exists())
            self.assertTrue((stage_root / "run" / "tables" / "safe" / "metrics.json").exists())
            self.assertTrue((stage_root / "run" / "figures" / "safe" / "plot.json").exists())

    def test_copy_bundle_file_rejects_invalid_staged_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            source.write_text('{"ok":true}\n', encoding="utf-8")
            stage_root = root / "stage"

            with self.assertRaises(ValueError):
                evidence.copy_bundle_file(source, stage_root, Path("/absolute/path.json"))
            with self.assertRaises(ValueError):
                evidence.copy_bundle_file(source, stage_root, Path("run") / ".." / "escape.json")

    def test_evidence_helpers_write_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("a\n", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "b.txt").write_text("b\n", encoding="utf-8")

            evidence.write_checksums(root)
            checksum = (root / "handoff-SHA256SUMS").read_text(encoding="utf-8")

        self.assertIn("a.txt", checksum)
        self.assertIn("nested/b.txt", checksum)
        self.assertNotIn("handoff-SHA256SUMS", checksum)

    def _package_a100_evidence_fixture(self, persistent: Path, root: Path) -> set[str]:
        rel_names, _evidence_root = self._package_a100_evidence_fixture_with_root(persistent, root)
        return rel_names

    def _package_a100_evidence_fixture_with_root(self, persistent: Path, root: Path) -> tuple[set[str], Path]:
        result = subprocess.run(
            ["bash", "scripts/package_a100_evidence_bundle.sh", str(persistent), str(root / "out")],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

        zips = sorted((root / "out").glob("neurotwin-a100-results-*-evidence.zip"))
        self.assertEqual(len(zips), 1)
        with zipfile.ZipFile(zips[0], "r") as archive:
            names = set(archive.namelist())
            extract_root = root / "evidence"
            archive.extractall(extract_root)

        roots = {name.split("/", 1)[0] for name in names if name}
        self.assertEqual(len(roots), 1)
        evidence_root = extract_root / roots.pop()
        checksum = subprocess.run(
            ["shasum", "-a", "256", "-c", "handoff-SHA256SUMS"],
            cwd=evidence_root,
            text=True,
            capture_output=True,
        )
        self.assertEqual(checksum.returncode, 0, checksum.stderr + checksum.stdout)
        return {name.split("/", 1)[1] for name in names}, evidence_root

    def test_package_a100_evidence_bundle_excludes_checkpoints_and_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            persistent = self._create_a100_evidence_fixture(root)
            rel_names = self._package_a100_evidence_fixture(persistent, root)
            required = {
                "COMMIT_HASH.txt",
                "README_HANDOFF.md",
                "README_SEND_TO_FRIEND.md",
                "handoff-SHA256SUMS",
                "run/summary.json",
                "run/metrics.json",
                "run/metrics.csv",
                "run/metrics.jsonl",
                "run/config.yaml",
                "run/environment.json",
                "run/gpu_preflight.json",
                "run/docker_run.env",
                "run/split_manifest.json",
                "run/tables/metrics_flat.csv",
                "run/figures/metric_summary.json",
                "prepared/eval_audit.json",
                "prepared/data_manifest.json",
                "prepared/event_manifest.json",
                "prepared/split_manifest.json",
                "prepared/leakage_report.json",
                "logs/neurotwin-a100-full-123.out",
                "logs/neurotwin-a100-full-123.err",
                "logs/neurotwin-a100-docker-20260531T000000Z.log",
            }
            for rel in required:
                self.assertIn(rel, rel_names)
            self.assertNotIn("logs/neurotwin-a100-full-999.out", rel_names)
            self.assertNotIn("logs/neurotwin-a100-full-999.err", rel_names)
            self.assertNotIn("logs/neurotwin-a100-docker-20260530T000000Z.log", rel_names)
            self.assertNotIn("logs/other-project.out", rel_names)
            for rel in rel_names:
                self.assertFalse(rel.endswith((".pt", ".npz", ".tar.gz", ".zip", ".pem", ".key")), rel)
                self.assertNotIn("pw.txt", rel)
                if rel != "run/docker_run.env":
                    self.assertNotIn(".env", rel)
            _, evidence_root = self._package_a100_evidence_fixture_with_root(persistent, root / "second")
            handoff_readme = (evidence_root / "README_HANDOFF.md").read_text(encoding="utf-8")
            self.assertIn("This handoff contains a runnable NeuroTwin A100 runner tarball", handoff_readme)
            self.assertIn("bash scripts/run_docker_6gpu.sh", handoff_readme)
            self.assertIn("README_AGENT_DEPLOY.md", handoff_readme)
            self.assertIn("scripts/docker_a100_inner.sh", handoff_readme)
            self.assertNotIn("Evidence bundle for commit", handoff_readme)
            self.assertNotIn("bash scripts/docker_a100_inner.sh", handoff_readme)
            self.assertNotIn("python -m neurotwin.cli cluster materialize-config", handoff_readme)

    def test_package_a100_evidence_bundle_falls_back_to_summary_job_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            persistent = self._create_a100_evidence_fixture(
                root,
                environment_json="{}\n",
                summary_json='{"run":{"slurm":{"job_id":"123"}}}\n',
            )
            rel_names = self._package_a100_evidence_fixture(persistent, root)

        self.assertIn("logs/neurotwin-a100-full-123.out", rel_names)
        self.assertIn("logs/neurotwin-a100-full-123.err", rel_names)
        self.assertIn("logs/neurotwin-a100-docker-20260531T000000Z.log", rel_names)
        self.assertNotIn("logs/neurotwin-a100-full-999.out", rel_names)
        self.assertNotIn("logs/other-project.out", rel_names)

    def test_package_a100_evidence_bundle_without_log_metadata_includes_no_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            persistent = self._create_a100_evidence_fixture(
                root,
                environment_json="{}\n",
                docker_run_env=None,
            )
            rel_names = self._package_a100_evidence_fixture(persistent, root)

        self.assertFalse(any(rel.startswith("logs/") for rel in rel_names))

    def test_package_a100_evidence_bundle_unsafe_job_id_includes_no_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            persistent = self._create_a100_evidence_fixture(
                root,
                environment_json='{"run":{"slurm":{"job_id":"../other-project"}}}\n',
                docker_run_env=None,
            )
            rel_names = self._package_a100_evidence_fixture(persistent, root)

        self.assertFalse(any(rel.startswith("logs/") for rel in rel_names))

    def test_package_a100_evidence_bundle_unsafe_docker_log_path_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            persistent = self._create_a100_evidence_fixture(
                root,
                environment_json="{}\n",
                docker_run_env="DOCKER_LOG_PATH=/tmp/neurotwin-a100-docker-20260531T000000Z.log\n",
            )
            rel_names = self._package_a100_evidence_fixture(persistent, root)

        self.assertFalse(any(rel.startswith("logs/") for rel in rel_names))

    def test_package_a100_evidence_bundle_excludes_secret_directories_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            persistent = self._create_a100_evidence_fixture(root)
            run = persistent / "runs" / "moabb_a100_smoke"
            logs = persistent / "logs"
            (run / "tables" / "secret-folder").mkdir(parents=True)
            (run / "tables" / "secret-folder" / "metrics.json").write_text('{"hidden":true}\n', encoding="utf-8")
            (run / "figures" / "password-cache").mkdir(parents=True)
            (run / "figures" / "password-cache" / "plot.json").write_text('{"hidden":true}\n', encoding="utf-8")
            safe_target = run / "tables" / "safe.txt"
            safe_target.write_text("safe\n", encoding="utf-8")
            try:
                (run / "tables" / "symlinked.txt").symlink_to(safe_target)
                (logs / "neurotwin-a100-full-123.out").unlink()
                (logs / "neurotwin-a100-full-123.out").symlink_to(safe_target)
            except OSError as exc:
                self.skipTest(f"symlink support unavailable: {exc}")

            rel_names = self._package_a100_evidence_fixture(persistent, root)

        self.assertNotIn("run/tables/secret-folder/metrics.json", rel_names)
        self.assertNotIn("run/figures/password-cache/plot.json", rel_names)
        self.assertNotIn("run/tables/symlinked.txt", rel_names)
        self.assertNotIn("logs/neurotwin-a100-full-123.out", rel_names)
