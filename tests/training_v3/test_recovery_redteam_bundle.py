import json
import tempfile
import unittest
from pathlib import Path

from neurotwin.runtime.distributed import DistributedInfo
from neurotwin.training_v3 import KTMTrainConfig, train_ktm, write_training_bundle

_INFO = DistributedInfo(rank=0, local_rank=0, world_size=1)
_CFG = KTMTrainConfig(
    mode="cpu_smoke", n_episodes=24, steps=10, eval_every_steps=5,
    checkpoint_every_steps=5, seed=0, recovery_margin=0.05,
)


class SingleRunRecoveryBlockedTests(unittest.TestCase):
    """A single training run must NEVER earn recovery — red-team discipline in the bundle path."""

    def _run(self, out):
        artifacts = train_ktm(_CFG, out_dir=out, dist_info=_INFO)
        write_training_bundle(out, cfg=_CFG, artifacts=artifacts, config_path=None)

    def test_single_run_recovery_blocked_with_redteam_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(Path(tmp))
            metrics = json.loads((Path(tmp) / "metrics.json").read_text(encoding="utf-8"))
            card = json.loads((Path(tmp) / "model_card.json").read_text(encoding="utf-8"))

        # Recovery scope blocked at the gate level.
        self.assertFalse(metrics["recovery_claim_allowed"])
        self.assertFalse(card["recovery_claim_allowed"])

        # Red-team dossier blocks a lone run: one seed, no alternate generator family.
        rt = metrics["recovery_redteam"]
        self.assertFalse(rt["recovery_allowed"])
        self.assertIn("single_seed_only", rt["blocker_reasons"])
        self.assertIn("architecture_affinity_not_tested", rt["blocker_reasons"])

    def test_single_run_uses_symmetric_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(Path(tmp))
            metrics = json.loads((Path(tmp) / "metrics.json").read_text(encoding="utf-8"))
        parity = metrics["selection_parity"]
        self.assertEqual(parity["selection_policy"], "symmetric_best_val")
        self.assertEqual(parity["ktm_checkpoint_policy"], "best_val")
        self.assertEqual(parity["baseline_checkpoint_policy"], "best_val")
        self.assertEqual(parity["same_eval_metric"], "mse")

    def test_harness_scope_still_allowed(self):
        # The infrastructure (harness) scope may pass even while recovery is blocked.
        with tempfile.TemporaryDirectory() as tmp:
            self._run(Path(tmp))
            gate = json.loads((Path(tmp) / "evidence_gate.json").read_text(encoding="utf-8"))
        self.assertEqual(gate["claim_scope"], "synthetic_ktm_training_harness")


if __name__ == "__main__":
    unittest.main()
