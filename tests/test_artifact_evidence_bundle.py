import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path


class EvidenceBundleArtifactTests(unittest.TestCase):
    def _create_a100_evidence_fixture(
        self,
        root: Path,
        environment_json: str | None = '{"run":{"slurm":{"job_id":"123"}}}\n',
        summary_json: str = (
            '{"status":"completed_prepared_training","completed_steps":50,'
            '"real_data_smoke":true,"scientific_claim_allowed":false}\n'
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
            logs / "other-project.out": "unrelated\n",
        }
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

    def _package_a100_evidence_fixture(self, persistent: Path, root: Path) -> set[str]:
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
        return {name.split("/", 1)[1] for name in names}

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
            }
            for rel in required:
                self.assertIn(rel, rel_names)
            self.assertNotIn("logs/neurotwin-a100-full-999.out", rel_names)
            self.assertNotIn("logs/neurotwin-a100-full-999.err", rel_names)
            self.assertNotIn("logs/other-project.out", rel_names)
            for rel in rel_names:
                self.assertFalse(rel.endswith((".pt", ".npz", ".tar.gz", ".zip", ".pem", ".key")), rel)
                self.assertNotIn("pw.txt", rel)
                self.assertNotIn(".env", rel)

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
        self.assertNotIn("logs/neurotwin-a100-full-999.out", rel_names)
        self.assertNotIn("logs/other-project.out", rel_names)

    def test_package_a100_evidence_bundle_without_job_id_includes_no_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            persistent = self._create_a100_evidence_fixture(root, environment_json="{}\n")
            rel_names = self._package_a100_evidence_fixture(persistent, root)

        self.assertFalse(any(rel.startswith("logs/") for rel in rel_names))

    def test_package_a100_evidence_bundle_unsafe_job_id_includes_no_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            persistent = self._create_a100_evidence_fixture(
                root,
                environment_json='{"run":{"slurm":{"job_id":"../other-project"}}}\n',
            )
            rel_names = self._package_a100_evidence_fixture(persistent, root)

        self.assertFalse(any(rel.startswith("logs/") for rel in rel_names))
