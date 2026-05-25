import unittest
from pathlib import Path


class ResearchArtifactTests(unittest.TestCase):
    def test_h100_configs_scripts_and_paper_docs_exist(self):
        required = [
            "configs/train/moabb_debug.yaml",
            "configs/train/moabb_h100.yaml",
            "configs/train/bids_debug.yaml",
            "configs/train/neurotwin_v1_h100.yaml",
            "configs/eval/neural_translation_v1.yaml",
            "scripts/slurm/train_h100.sh",
            "scripts/slurm/eval_h100.sh",
            "scripts/slurm/sweep_h100.sh",
            "docs/CLAIMS.md",
            "docs/H100_RUNBOOK.md",
            "docs/paper/outline.md",
            "docs/paper/limitations.md",
        ]

        for path in required:
            with self.subTest(path=path):
                self.assertTrue(Path(path).exists(), path)

    def test_claims_doc_blocks_forbidden_claims(self):
        claims = Path("docs/CLAIMS.md").read_text(encoding="utf-8")

        self.assertIn("Do not claim", claims)
        self.assertIn("clinical digital twin", claims)
        self.assertIn("Synthetic smoke tests validate plumbing only", claims)
