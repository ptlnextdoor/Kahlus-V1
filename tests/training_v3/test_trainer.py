import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from neurotwin.models.ktm import TorchKTM
from neurotwin.runtime.distributed import DistributedInfo
from neurotwin.training_v3 import KTMTrainConfig, evaluate_ktm, train_ktm
from neurotwin.training_v3.checkpoint import apply_resume, load_resume, resume_start_step

_INFO = DistributedInfo(rank=0, local_rank=0, world_size=1)


def _cfg(**overrides):
    base = dict(mode="cpu_smoke", n_episodes=32, steps=200, eval_every_steps=50,
                checkpoint_every_steps=50, seed=0)
    base.update(overrides)
    return KTMTrainConfig(**base)


class TrainerSmokeTests(unittest.TestCase):
    def test_cpu_smoke_decreases_val_loss(self):
        artifacts = train_ktm(_cfg(), dist_info=_INFO)
        self.assertLess(artifacts.val_after, artifacts.val_before)
        self.assertTrue(artifacts.loss_decreased)
        self.assertFalse(artifacts.aborted)
        self.assertEqual(artifacts.failure_reasons, [])

    def test_checkpoint_save_resume_reproduces(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cfg = _cfg()
            artifacts = train_ktm(cfg, out_dir=out, dist_info=_INFO)
            last = out / "checkpoints" / "last.pt"
            self.assertTrue(last.exists())
            self.assertTrue((out / "checkpoints" / "best.pt").exists())

            original = evaluate_ktm(
                artifacts.model, artifacts.bundle,
                artifacts.bundle.splits.test_episodes, "cpu",
            )["trajectory"]["mse"]

            checkpoint = load_resume(last, "cpu")
            self.assertEqual(resume_start_step(checkpoint), cfg.steps)
            fresh = TorchKTM(cfg.to_model_config())
            apply_resume(checkpoint, model=fresh)
            restored = evaluate_ktm(
                fresh, artifacts.bundle, artifacts.bundle.splits.test_episodes, "cpu"
            )["trajectory"]["mse"]
            self.assertAlmostEqual(original, restored, places=6)

    def test_progress_and_status_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            train_ktm(_cfg(steps=60), out_dir=out, dist_info=_INFO)
            progress = (out / "progress.jsonl").read_text().splitlines()
            self.assertTrue(progress)
            events = [json.loads(line)["event"] for line in progress]
            self.assertIn("run_started", events)
            self.assertIn("val_after", events)
            self.assertIn("step", events)
            status = json.loads((out / "run_status.json").read_text())
            self.assertEqual(status["status"], "training_complete")
            self.assertEqual(status["completed_steps"], 60)

    def test_failure_writes_status_failed_with_phase(self):
        # A crash mid-run must still leave a forensic trail: run_status flips to failed and
        # records the phase it died in (here: data construction).
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            boom = mock.patch(
                "neurotwin.training_v3.trainer.build_transition_gym",
                side_effect=RuntimeError("synthetic boom"),
            )
            with boom, self.assertRaises(RuntimeError):
                train_ktm(_cfg(steps=20), out_dir=out, dist_info=_INFO)
            status = json.loads((out / "run_status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["phase"], "data")
            self.assertEqual(status["error_type"], "RuntimeError")

    def test_loss_explosion_guard_aborts_via_real_config(self):
        # No debug flags: a tight explosion factor + small noisy minibatches makes a real
        # spike trip the guard, exercising the production abort path.
        artifacts = train_ktm(
            _cfg(steps=120, batch_size=4, loss_explosion_factor=1.0001),
            dist_info=_INFO,
        )
        self.assertTrue(artifacts.aborted)
        self.assertTrue(any("loss explosion" in r for r in artifacts.failure_reasons))


if __name__ == "__main__":
    unittest.main()
