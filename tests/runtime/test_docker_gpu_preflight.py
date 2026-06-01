import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "docker_gpu_preflight.py"
_SPEC = importlib.util.spec_from_file_location("docker_gpu_preflight", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
docker_gpu_preflight = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(docker_gpu_preflight)


class DockerGpuPreflightTests(unittest.TestCase):
    def _run_preflight(
        self,
        *,
        cuda_available: bool,
        visible_count: int,
        expected_count: str = "2",
    ) -> tuple[int, dict[str, object]]:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "gpu_preflight.json"
            with (
                mock.patch.dict(
                    docker_gpu_preflight.os.environ,
                    {
                        "GPU_COUNT": expected_count,
                        "DOCKER_IMAGE": "neurotwin:test",
                        "HOST_GPU_IDS": "1,2",
                        "CUDA_VISIBLE_DEVICES": "0,1",
                        "LOCAL_RANK": "0",
                        "RANK": "0",
                        "WORLD_SIZE": "1",
                        "NPROC_PER_NODE": expected_count,
                    },
                    clear=True,
                ),
                mock.patch.object(docker_gpu_preflight, "_nccl_version", return_value="2.18.1"),
                mock.patch.object(docker_gpu_preflight.torch.cuda, "is_available", return_value=cuda_available),
                mock.patch.object(docker_gpu_preflight.torch.cuda, "device_count", return_value=visible_count),
                mock.patch.object(
                    docker_gpu_preflight.torch.cuda,
                    "get_device_name",
                    side_effect=lambda index: f"A100-{index}",
                ),
            ):
                exit_code = docker_gpu_preflight.main(["docker_gpu_preflight.py", str(out)])
            return exit_code, json.loads(out.read_text(encoding="utf-8"))

    def test_preflight_passes_when_cuda_count_matches(self):
        exit_code, payload = self._run_preflight(cuda_available=True, visible_count=2)

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["passed"])
        self.assertTrue(payload["cuda_available"])
        self.assertEqual(payload["expected_gpu_count"], 2)
        self.assertEqual(payload["visible_gpu_count"], 2)
        self.assertEqual(payload["visible_gpu_names"], ["A100-0", "A100-1"])
        self.assertEqual(payload["docker_image"], "neurotwin:test")

    def test_preflight_returns_no_cuda_code(self):
        exit_code, payload = self._run_preflight(cuda_available=False, visible_count=0, expected_count="1")

        self.assertEqual(exit_code, 10)
        self.assertFalse(payload["passed"])
        self.assertFalse(payload["cuda_available"])
        self.assertEqual(payload["visible_gpu_count"], 0)

    def test_preflight_returns_wrong_count_code(self):
        exit_code, payload = self._run_preflight(cuda_available=True, visible_count=1, expected_count="2")

        self.assertEqual(exit_code, 11)
        self.assertFalse(payload["passed"])
        self.assertTrue(payload["cuda_available"])
        self.assertEqual(payload["expected_gpu_count"], 2)
        self.assertEqual(payload["visible_gpu_count"], 1)

    def test_preflight_rejects_invalid_gpu_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "gpu_preflight.json"
            with mock.patch.dict(docker_gpu_preflight.os.environ, {"GPU_COUNT": "six"}, clear=True):
                with self.assertRaisesRegex(SystemExit, "GPU_COUNT must be an integer"):
                    docker_gpu_preflight.main(["docker_gpu_preflight.py", str(out)])


if __name__ == "__main__":
    unittest.main()
