import unittest

from neurotwin.stf import (
    REQUIRED_STF_BASELINES_BY_TASK,
    REQUIRED_STF_NEGATIVE_CONTROLS,
    REQUIRED_STF_SPLITS,
    REQUIRED_STF_TASKS,
    STF_CLAIM_SCOPE,
    audit_stf_benchmark_contract,
    build_stf_gate,
    stf_benchmark_contract,
)


class STFBenchmarkContractTests(unittest.TestCase):
    def test_contract_declares_required_tasks_baselines_controls_and_splits(self):
        contract = stf_benchmark_contract()

        self.assertEqual(contract["claim_scope"], STF_CLAIM_SCOPE)
        self.assertEqual(contract["required_tasks"], list(REQUIRED_STF_TASKS))
        self.assertEqual(
            contract["required_negative_controls"],
            list(REQUIRED_STF_NEGATIVE_CONTROLS),
        )
        self.assertEqual(contract["required_splits"], list(REQUIRED_STF_SPLITS))
        self.assertEqual(
            contract["required_baselines_by_task"],
            {task_id: list(baselines) for task_id, baselines in REQUIRED_STF_BASELINES_BY_TASK.items()},
        )
        self.assertIn("diagnosis", contract["blocked_claim_terms"])
        self.assertIn("stimulation", contract["blocked_claim_terms"])

    def test_contract_audit_rejects_missing_time_split_negative_control_and_baseline(self):
        audit = audit_stf_benchmark_contract(
            declared_tasks=REQUIRED_STF_TASKS,
            baselines_by_task={
                task_id: baselines for task_id, baselines in REQUIRED_STF_BASELINES_BY_TASK.items()
            }
            | {"patient_held_out_event_risk_forecasting": ("cycle_time_of_day", "event_frequency")},
            negative_controls=("shuffled_target_control",),
            split_types=("patient_held_out",),
        )

        self.assertFalse(audit["passed"])
        self.assertIn(
            "required baseline missing for patient_held_out_event_risk_forecasting: logistic_ridge",
            audit["failure_reasons"],
        )
        self.assertIn(
            "required negative control missing: time_shifted_label_control",
            audit["failure_reasons"],
        )
        self.assertIn("required split missing: time_held_out", audit["failure_reasons"])

    def test_stf_gate_passes_only_for_narrow_complete_benchmark_definition(self):
        gate = build_stf_gate(
            dataset="stf_public_fixture",
            declared_tasks=REQUIRED_STF_TASKS,
            baselines_by_task=REQUIRED_STF_BASELINES_BY_TASK,
            negative_controls=REQUIRED_STF_NEGATIVE_CONTROLS,
            split_types=REQUIRED_STF_SPLITS,
            split_audit_passed=True,
            baseline_table_present=True,
            finite_metrics=True,
            calibration_checked=True,
        )

        self.assertTrue(gate["scientific_claim_allowed"], gate["failure_reasons"])
        self.assertEqual(gate["branch"], "stf")
        self.assertEqual(gate["claim_scope"], STF_CLAIM_SCOPE)
        self.assertTrue(gate["benchmark_contract_audit"]["passed"])

    def test_stf_gate_blocks_clinical_claim_scope(self):
        gate = build_stf_gate(
            dataset="stf_public_fixture",
            declared_tasks=REQUIRED_STF_TASKS,
            baselines_by_task=REQUIRED_STF_BASELINES_BY_TASK,
            negative_controls=REQUIRED_STF_NEGATIVE_CONTROLS,
            split_types=REQUIRED_STF_SPLITS,
            split_audit_passed=True,
            baseline_table_present=True,
            finite_metrics=True,
            calibration_checked=True,
            claim_scope="epilepsy_diagnosis_predictor",
        )

        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertIn(
            "claim scope too broad: 'epilepsy_diagnosis_predictor' is not in the narrow synthetic allowlist",
            gate["failure_reasons"],
        )
        self.assertIn(
            "blocked clinical/device claim term in scope: 'epilepsy_diagnosis_predictor'",
            gate["failure_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
