from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    name: str
    inputs: tuple[str, ...]
    targets: tuple[str, ...]
    required_splits: tuple[str, ...]
    metrics: tuple[str, ...]
    description: str
    executable_task_ids: tuple[str, ...] = ()


def default_translation_tasks() -> tuple[TaskSpec, ...]:
    held_out = ("subject", "site", "dataset")
    return (
        TaskSpec(
            task_id="stimulus_past_fmri_to_future_fmri",
            name="stimulus + past fMRI -> future fMRI",
            inputs=("stimulus_embedding", "past_fmri"),
            targets=("future_fmri",),
            required_splits=held_out,
            metrics=("mse", "pearsonr", "rollout_stability"),
            description="TRIBE v2 and BrainVista become baselines; NeuroTwin must improve future-state rollout under held-out splits.",
            executable_task_ids=("future_state_forecasting", "stimulus_to_fmri_response"),
        ),
        TaskSpec(
            task_id="eeg_meg_to_shared_latent_state",
            name="EEG/MEG -> shared latent brain state",
            inputs=("eeg", "meg", "sensor_geometry"),
            targets=("shared_latent_state",),
            required_splits=held_out,
            metrics=("retrieval_recall", "alignment_score", "calibration_error"),
            description="Fast electrophysiology is projected into a shared neural-state space without claiming Brain-OF left this untouched.",
            executable_task_ids=("cross_modal_translation",),
        ),
        TaskSpec(
            task_id="fmri_to_eeg_meg_spectral_proxy",
            name="fMRI -> EEG/MEG spectral-state proxy",
            inputs=("fmri", "anatomy_optional"),
            targets=("eeg_spectral_proxy", "meg_spectral_proxy"),
            required_splits=("subject", "site"),
            metrics=("spectral_mse", "pearsonr", "uncertainty_calibration"),
            description="Only valid where paired or tightly aligned fMRI/electrophysiology data exists.",
            executable_task_ids=("cross_modal_translation",),
        ),
        TaskSpec(
            task_id="anatomy_fmri_to_subject_conditioned_state",
            name="anatomy/fMRI -> subject-conditioned latent state",
            inputs=("structural_mri", "connectome_optional", "fmri"),
            targets=("subject_conditioned_latent_state",),
            required_splits=held_out,
            metrics=("few_shot_delta", "heldout_subject_score", "calibration_error"),
            description="Brain Harmony is a key comparator for structure + function tokenization.",
            executable_task_ids=("few_shot_subject_adaptation",),
        ),
        TaskSpec(
            task_id="missing_modality_reconstruction",
            name="missing-modality reconstruction",
            inputs=("any_observed_modality_subset",),
            targets=("withheld_modality",),
            required_splits=held_out,
            metrics=("mse", "pearsonr", "retrieval_recall", "bootstrap_ci"),
            description="Core pretraining and evaluation task for cross-modal neural translation.",
            executable_task_ids=("masked_neural_reconstruction", "cross_modal_translation"),
        ),
        TaskSpec(
            task_id="few_shot_subject_adaptation",
            name="few-shot subject adaptation",
            inputs=("5_to_20_minutes_subject_data", "global_model_state"),
            targets=("heldout_subject_future_state", "heldout_subject_missing_modality"),
            required_splits=("subject",),
            metrics=("adaptation_gain", "negative_transfer_rate", "calibration_error"),
            description="Tests whether personalization improves held-out performance without subject identity leakage.",
            executable_task_ids=("few_shot_subject_adaptation",),
        ),
    )


def paper_to_executable_task_aliases() -> dict[str, tuple[str, ...]]:
    return {task.task_id: task.executable_task_ids for task in default_translation_tasks()}


def executable_task_ids_for(paper_task_id: str) -> tuple[str, ...]:
    aliases = paper_to_executable_task_aliases()
    if paper_task_id in aliases:
        return aliases[paper_task_id]
    return ()


def paper_task_ids_for_executable(executable_task_id: str) -> tuple[str, ...]:
    return tuple(
        task.task_id
        for task in default_translation_tasks()
        if executable_task_id in task.executable_task_ids
    )
