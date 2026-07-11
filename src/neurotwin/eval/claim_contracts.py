from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping

from neurotwin.repro import stable_hash


class ClaimKind(str, Enum):
    FORECASTING = "forecasting"
    SLEEP_TRANSITION_FORECASTING = "sleep_transition_forecasting"
    RECONSTRUCTION = "reconstruction"
    TRANSLATION = "translation"
    RESPONSE_PREDICTION = "response_prediction"
    SYNTHETIC_DIAGNOSTIC = "synthetic_diagnostic"


@dataclass(frozen=True)
class TaskClaimContract:
    task_id: str
    claim_kind: ClaimKind
    required_metric_fields: tuple[str, ...]
    required_baselines: tuple[str, ...]
    required_controls: tuple[str, ...] = ()
    requires_forecast_eligibility: bool = False
    scientific_claim_eligible: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["claim_kind"] = self.claim_kind.value
        return payload


_FORECAST_BASELINES = (
    "persistence",
    "linear_ridge",
    "autoregressive_ridge",
)


TASK_CLAIM_CONTRACTS: dict[str, TaskClaimContract] = {
    "future_state_forecasting": TaskClaimContract(
        task_id="future_state_forecasting",
        claim_kind=ClaimKind.FORECASTING,
        required_metric_fields=("test_mse", "best_val_mse"),
        required_baselines=_FORECAST_BASELINES,
        required_controls=("random_permutation",),
        requires_forecast_eligibility=True,
    ),
    "future_eeg_forecasting": TaskClaimContract(
        task_id="future_eeg_forecasting",
        claim_kind=ClaimKind.FORECASTING,
        required_metric_fields=("test_mse", "best_val_mse"),
        required_baselines=_FORECAST_BASELINES,
        required_controls=("random_permutation",),
        requires_forecast_eligibility=True,
    ),
    "longer_horizon_eeg_forecasting": TaskClaimContract(
        task_id="longer_horizon_eeg_forecasting",
        claim_kind=ClaimKind.FORECASTING,
        required_metric_fields=("test_mse", "best_val_mse"),
        required_baselines=_FORECAST_BASELINES,
        required_controls=("random_permutation",),
        requires_forecast_eligibility=True,
    ),
    "patient_held_out_event_risk_forecasting": TaskClaimContract(
        task_id="patient_held_out_event_risk_forecasting",
        claim_kind=ClaimKind.FORECASTING,
        required_metric_fields=("test_brier_score",),
        required_baselines=("moving_average", "patient_history"),
        required_controls=("time_shift_control",),
        requires_forecast_eligibility=True,
    ),
    "oracle_conditional_stable_sleep_transition": TaskClaimContract(
        task_id="oracle_conditional_stable_sleep_transition",
        claim_kind=ClaimKind.SLEEP_TRANSITION_FORECASTING,
        required_metric_fields=(
            "external_log_skill_bits",
            "external_brier_score",
            "chief_comparator_brier_score",
        ),
        required_baselines=(
            "empirical_transition",
            "markov_transition",
            "semi_markov_competing_risk",
        ),
        required_controls=(
            "shuffled_target_control",
            "time_shift_control",
            "subject_identity_control",
        ),
        requires_forecast_eligibility=True,
    ),
    "masked_neural_reconstruction": TaskClaimContract(
        task_id="masked_neural_reconstruction",
        claim_kind=ClaimKind.RECONSTRUCTION,
        required_metric_fields=("test_mse", "best_val_mse"),
        required_baselines=("train_mean", "linear_ridge"),
        required_controls=("random_permutation",),
    ),
    "cross_modal_translation": TaskClaimContract(
        task_id="cross_modal_translation",
        claim_kind=ClaimKind.TRANSLATION,
        required_metric_fields=("test_mse", "best_val_mse"),
        required_baselines=("train_mean", "linear_ridge"),
        required_controls=("random_permutation",),
    ),
    "stimulus_to_fmri_response": TaskClaimContract(
        task_id="stimulus_to_fmri_response",
        claim_kind=ClaimKind.RESPONSE_PREDICTION,
        required_metric_fields=("test_mse", "best_val_mse"),
        required_baselines=("train_mean", "linear_ridge"),
        required_controls=("random_permutation",),
    ),
}


def task_claim_contract(task_id: str) -> TaskClaimContract | None:
    return TASK_CLAIM_CONTRACTS.get(str(task_id).strip())


def collect_task_claim_contracts(
    payload: Any,
) -> tuple[tuple[TaskClaimContract, ...], tuple[str, ...]]:
    task_ids = sorted(_collect_task_ids(payload))
    contracts: list[TaskClaimContract] = []
    unknown: list[str] = []
    for task_id in task_ids:
        contract = task_claim_contract(task_id)
        if contract is None:
            unknown.append(task_id)
        else:
            contracts.append(contract)
    return tuple(contracts), tuple(unknown)


def claim_contract_sha256(contracts: tuple[TaskClaimContract, ...]) -> str:
    return stable_hash(
        [
            contract.to_dict()
            for contract in sorted(contracts, key=lambda row: row.task_id)
        ]
    )


def _collect_task_ids(payload: Any) -> set[str]:
    task_ids: set[str] = set()
    if hasattr(payload, "seed_results"):
        return _collect_task_ids(payload.seed_results)
    if isinstance(payload, Mapping):
        task_id = payload.get("task_id")
        if isinstance(task_id, str) and task_id.strip():
            task_ids.add(task_id.strip())
        tasks = payload.get("tasks")
        if isinstance(tasks, Mapping):
            task_ids.update(str(key).strip() for key in tasks if str(key).strip())
        for value in payload.values():
            task_ids.update(_collect_task_ids(value))
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            task_ids.update(_collect_task_ids(value))
    return task_ids
