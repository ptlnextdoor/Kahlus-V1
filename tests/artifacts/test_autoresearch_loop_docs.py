import unittest
from pathlib import Path


class AutoresearchLoopDocsTests(unittest.TestCase):
    def test_autoresearch_runbook_exists_and_points_to_required_sources(self):
        runbook = Path("docs/AUTORESEARCH_LOOP.md")
        self.assertTrue(runbook.exists())
        text = runbook.read_text(encoding="utf-8")

        for required in (
            "AGENTS.md",
            "docs/CLAIMS.md",
            "docs/ROADMAP.md",
            "docs/research/neurotwin_project_state.md",
            "docs/research/neurotwin-technical-report.md",
            "A100_CLUSTER_AGENT_RUNBOOK.md",
            "MAX_GPUS=4",
            "actual `NxA100`",
            "docs/templates/autoresearch_iteration.md",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)

    def test_autoresearch_runbook_preserves_claim_boundaries(self):
        text = Path("docs/AUTORESEARCH_LOOP.md").read_text(encoding="utf-8")

        for forbidden_boundary in (
            "clinical product",
            "seizure predictor",
            "depression classifier",
            "brain foundation model",
            "proven model-superiority result",
            "NeuroTwin beats baselines",
            "NFC is proven",
            "diagnostic model",
            "treatment predictor",
            "exact TRIBE v2 reproduction",
            "exact BrainVista reproduction",
            "A100 success if no evidence bundle exists",
            "scientific_claim_allowed=true",
            "Never print, copy, upload, archive, or commit `pw.txt`",
            "no more than four free GPUs",
        ):
            with self.subTest(forbidden_boundary=forbidden_boundary):
                self.assertIn(forbidden_boundary, text)

    def test_autoresearch_runbook_requires_kahlus_gate_fields(self):
        text = Path("docs/AUTORESEARCH_LOOP.md").read_text(encoding="utf-8")

        for required in (
            "GatePass = C_split",
            "C_baseline",
            "C_controls",
            "C_power",
            "C_claim_scope",
            "`claim_scope`",
            "`stop_reason`",
            "Aggregate metrics must never be promoted",
            "MAX_GPUS=4 by default",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)

    def test_exercise_depression_gate_is_evidence_first_and_cautious(self):
        text = Path("docs/AUTORESEARCH_LOOP.md").read_text(encoding="utf-8")

        for required in (
            "Exercise And Depression Evidence Gate",
            "systematic reviews",
            "meta-analyses",
            "randomized controlled trials",
            "PMID:41500513",
            "PMID:38355154",
            "PMID:36796860",
            "PMID:26473759",
            "may help reduce depressive symptoms",
            "cures depression",
            "replaces medication or therapy",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)

    def test_iteration_template_contains_verification_and_claim_gate(self):
        template = Path("docs/templates/autoresearch_iteration.md")
        self.assertTrue(template.exists())
        text = template.read_text(encoding="utf-8")

        for required in (
            "Required First Reads",
            "Research Evidence",
            "Feature Spec",
            "Verification",
            "Kahlus Gate Predicate",
            "Cluster Protocol",
            '"claim_scope": ""',
            '"stop_reason": ""',
            '"gate_pass": false',
            '"c_baseline": false',
            '"c_controls": false',
            '"c_power": false',
            '"scientific_claim_allowed": false',
            '"clinical_claim_allowed": false',
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)
