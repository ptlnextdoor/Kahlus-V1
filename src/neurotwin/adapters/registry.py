from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetAdapterSpec:
    dataset_id: str
    display_name: str
    modalities: tuple[str, ...]
    species: str
    access: str
    license_status: str
    alignment_value: str
    best_use: str
    adapter_status: str


def dataset_registry() -> tuple[DatasetAdapterSpec, ...]:
    return (
        DatasetAdapterSpec(
            dataset_id="openneuro_bids",
            display_name="OpenNeuro / BIDS",
            modalities=("fmri", "mri", "eeg", "meg", "behavior", "stimulus"),
            species="human",
            access="public datasets with dataset-specific terms",
            license_status="dataset-specific; inspect before training",
            alignment_value="BIDS metadata and task events make split manifests auditable.",
            best_use="fMRI naturalistic tasks, task fMRI, paired behavioral metadata.",
            adapter_status="planned",
        ),
        DatasetAdapterSpec(
            dataset_id="dandi_nwb",
            display_name="DANDI / NWB",
            modalities=("spikes", "calcium", "behavior", "stimulus"),
            species="animal/human mixed",
            access="public archive with dandiset-specific terms",
            license_status="dandiset-specific",
            alignment_value="NWB gives structured trials, electrodes, units, and behavior where metadata is clean.",
            best_use="population dynamics, spikes/calcium forecasting, behavior decoding.",
            adapter_status="planned",
        ),
        DatasetAdapterSpec(
            dataset_id="moabb_eeg",
            display_name="MOABB",
            modalities=("eeg", "behavior"),
            species="human",
            access="public benchmark fetchers",
            license_status="dataset-specific via MOABB metadata",
            alignment_value="Useful cross-dataset EEG benchmark APIs and reproducible splits.",
            best_use="EEG device/generalization and BCI baselines.",
            adapter_status="planned",
        ),
        DatasetAdapterSpec(
            dataset_id="tuh_eeg",
            display_name="Temple University Hospital EEG Corpus",
            modalities=("eeg", "clinical"),
            species="human",
            access="credentialed access",
            license_status="restricted clinical corpus",
            alignment_value="Large messy clinical EEG distribution for robustness tests.",
            best_use="secondary clinical EEG transfer after Neural Translation works.",
            adapter_status="restricted_adapter",
        ),
        DatasetAdapterSpec(
            dataset_id="hcp_young_adult",
            display_name="Human Connectome Project Young Adult",
            modalities=("fmri", "mri", "dti", "behavior"),
            species="human",
            access="registration required",
            license_status="HCP data-use terms",
            alignment_value="High-quality same-subject anatomy/function/behavior.",
            best_use="subject adaptation and anatomy/fMRI latent-state tests.",
            adapter_status="planned",
        ),
        DatasetAdapterSpec(
            dataset_id="uk_biobank_imaging",
            display_name="UK Biobank Imaging",
            modalities=("fmri", "mri", "dti", "clinical"),
            species="human",
            access="application required",
            license_status="UK Biobank access agreement",
            alignment_value="Large scale and repeat imaging, but access-gated.",
            best_use="scale tests and longitudinal transfer after open-data prototype.",
            adapter_status="restricted_adapter",
        ),
        DatasetAdapterSpec(
            dataset_id="abcd",
            display_name="ABCD Study",
            modalities=("fmri", "mri", "behavior", "clinical"),
            species="human",
            access="application required",
            license_status="NDA/ABCD data-use terms",
            alignment_value="Longitudinal development data with rich behavior.",
            best_use="longitudinal adaptation and non-clinical phenotype transfer.",
            adapter_status="restricted_adapter",
        ),
        DatasetAdapterSpec(
            dataset_id="adni",
            display_name="ADNI",
            modalities=("mri", "pet", "clinical"),
            species="human",
            access="application required",
            license_status="ADNI data-use terms",
            alignment_value="Longitudinal disease progression metadata, but clinical claims remain secondary.",
            best_use="downstream transfer only after benchmark proof.",
            adapter_status="restricted_adapter",
        ),
        DatasetAdapterSpec(
            dataset_id="ibl",
            display_name="International Brain Laboratory",
            modalities=("spikes", "behavior", "stimulus"),
            species="mouse",
            access="public",
            license_status="dataset-specific",
            alignment_value="Brain-wide mouse behavior tasks with strong alignment.",
            best_use="mechanistic population dynamics and behavior-conditioned forecasting.",
            adapter_status="planned",
        ),
        DatasetAdapterSpec(
            dataset_id="allen_brain_observatory",
            display_name="Allen Brain Observatory",
            modalities=("calcium", "spikes", "stimulus", "behavior"),
            species="mouse",
            access="public",
            license_status="Allen terms",
            alignment_value="Stimulus-aligned visual cortex recordings.",
            best_use="stimulus -> population dynamics baseline tasks.",
            adapter_status="planned",
        ),
        DatasetAdapterSpec(
            dataset_id="microns",
            display_name="MICrONS",
            modalities=("connectomics", "calcium", "anatomy"),
            species="mouse",
            access="public with large storage needs",
            license_status="dataset-specific",
            alignment_value="Dense structure/function in mouse visual cortex.",
            best_use="connectome-constrained dynamics experiments, not human twin claims.",
            adapter_status="planned_large",
        ),
        DatasetAdapterSpec(
            dataset_id="neurovault",
            display_name="NeuroVault",
            modalities=("fmri", "statistical_maps"),
            species="human",
            access="public",
            license_status="image-specific terms",
            alignment_value="Useful maps, weaker raw temporal dynamics.",
            best_use="secondary representation probing and atlas alignment.",
            adapter_status="planned",
        ),
        DatasetAdapterSpec(
            dataset_id="paired_eeg_fmri",
            display_name="Paired EEG-fMRI collections",
            modalities=("eeg", "fmri", "behavior", "stimulus"),
            species="human",
            access="dataset-specific",
            license_status="dataset-specific",
            alignment_value="Critical for fMRI -> EEG/MEG spectral proxy and missing-modality tests.",
            best_use="cross-modal translation validation where same-subject pairing exists.",
            adapter_status="survey_required",
        ),
        DatasetAdapterSpec(
            dataset_id="nlb",
            display_name="Neural Latents Benchmark",
            modalities=("spikes", "behavior"),
            species="animal",
            access="public benchmark",
            license_status="dataset-specific",
            alignment_value="Standardized held-out neural/behavior tasks.",
            best_use="spike/population dynamics baselines and acceptance smoke tasks.",
            adapter_status="planned",
        ),
    )
