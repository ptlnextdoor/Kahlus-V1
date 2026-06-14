import json
import tempfile
import unittest
from pathlib import Path

from neurotwin.runtime.distributed import DistributedInfo
from neurotwin.training_v3 import KTMTrainConfig, train_ktm, write_training_bundle

_INFO = DistributedInfo(rank=0, local_rank=0, world_size=1)

_BUNDLE_FILES = (
    "metrics.json",
    "baseline_table.json",
    "baseline_table.csv",
    "evidence_gate.json",
    "model_card.json",
    "data_card.json",
    "run_config.json",
    "failure_reasons.json",
)


class TrainingBundleTests(unittest.TestCase):
    def _run(self, out: Path) -> dict:
        cfg = KTMTrainConfig(mode="cpu_smoke", n_episodes=32, steps=120,
                             eval_every_steps=40, checkpoint_every_steps=40, seed=0)
        artifacts = train_ktm(cfg, out_dir=out, dist_info=_INFO)
        write_training_bundle(out, cfg=cfg, artifacts=artifacts)
        return artifacts

    def test_all_bundle_files_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._run(out)
            for name in _BUNDLE_FILES:
                self.assertTrue((out / name).exists(), name)
            self.assertTrue((out / "checkpoints").is_dir())
            self.assertTrue(list((out / "checkpoints").glob("*.pt")))

    def test_harness_scope_passes_recovery_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._run(out)
            gate = json.loads((out / "evidence_gate.json").read_text())
            self.assertEqual(gate["claim_scope"], "synthetic_ktm_training_harness")
            self.assertTrue(gate["scientific_claim_allowed"])

            card = json.loads((out / "model_card.json").read_text())
            self.assertEqual(card["claim_scope"], "synthetic_ktm_training_harness")
            self.assertTrue(card["scientific_claim_allowed"])
            # Untrained-vs-baselines: the tiny KTM must not be claimed to beat baselines, and
            # when it does not, the stronger recovery scope must stay blocked.
            if not card["ktm_beats_baselines"]:
                self.assertFalse(card["recovery_claim_allowed"])

            metrics = json.loads((out / "metrics.json").read_text())
            self.assertIn("recovery_claim_allowed", metrics)
            self.assertIn("ktm_vs_baselines", metrics)

            # The comparison must be locked under a matched baseline budget and carry full
            # budget provenance, so a recovery claim can never flip on a budget artifact.
            comparison = metrics["ktm_vs_baselines"]
            for key in ("budget_matched", "comparison_locked", "relative_improvement", "margin"):
                self.assertIn(key, comparison)
            self.assertTrue(comparison["budget_matched"])
            budget = comparison["budget"]
            self.assertEqual(budget["baseline_budget_policy"], "matched_optimizer_steps")
            # Smoke leaves baseline_train_steps unset → auto-matches the KTM step budget (120).
            self.assertEqual(budget["baseline_train_steps"], 120)
            self.assertEqual(budget["ktm_train_steps"], 120)

    def test_baseline_table_includes_ktm_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._run(out)
            table = json.loads((out / "baseline_table.json").read_text())
            model_ids = {row["model_id"] for row in table["rows"]}
            self.assertIn("ktm_torch", model_ids)
            self.assertIn("ridge", model_ids)


if __name__ == "__main__":
    unittest.main()
