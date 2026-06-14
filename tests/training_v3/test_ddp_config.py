import unittest

import torch

from neurotwin.runtime.distributed import DistributedInfo
from neurotwin.training_v3 import build_torchrun_command
from neurotwin.training_v3.trainer import resolve_device


class DDPConfigTests(unittest.TestCase):
    def test_torchrun_command_built_without_launch(self):
        cmd = build_torchrun_command(
            config_path="configs/train/ktm_a100_micro.yaml",
            out_dir="$RUN_ROOT/ktm_micro_sweep",
            nproc=8,
        )
        self.assertEqual(cmd[0], "torchrun")
        self.assertIn("--nproc_per_node=8", cmd)
        self.assertIn("scripts/run_ktm_train.py", cmd)
        self.assertIn("--mode", cmd)
        self.assertEqual(cmd[cmd.index("--mode") + 1], "ddp")
        # Building the command must not initialize any process group.
        self.assertFalse(torch.distributed.is_initialized())

    def test_resolve_device_cpu_modes(self):
        info = DistributedInfo(rank=0, local_rank=0, world_size=1)
        self.assertEqual(resolve_device("cpu_smoke", info).type, "cpu")
        if not torch.cuda.is_available():
            self.assertEqual(resolve_device("single_gpu", info).type, "cpu")
            self.assertEqual(resolve_device("ddp", info).type, "cpu")

    def test_default_nproc_is_eight(self):
        cmd = build_torchrun_command(config_path="c.yaml", out_dir="o")
        self.assertIn("--nproc_per_node=8", cmd)


if __name__ == "__main__":
    unittest.main()
