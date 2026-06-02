from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


CatalogStatus = Callable[[set[str], set[str]], str]


@dataclass(frozen=True)
class BaselineCatalogEntry:
    model_id: str
    display_name: str
    status: str | CatalogStatus
    notes: str
    upstream_reference: str | None = None
    exact_reproduction: bool = False
    uses_upstream_code: bool = False
    uses_upstream_weights: bool = False

    def catalog_row(self, task_ids: set[str], modalities: set[str]) -> dict[str, object]:
        status = self.status(task_ids, modalities) if callable(self.status) else self.status
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "status": status,
            "notes": self.notes,
            "upstream_reference": self.upstream_reference,
            "exact_reproduction": self.exact_reproduction,
            "uses_upstream_code": self.uses_upstream_code,
            "uses_upstream_weights": self.uses_upstream_weights,
        }


def baseline_catalog_rows(task_ids: set[str], modalities: set[str]) -> list[dict[str, object]]:
    return [spec.catalog_row(task_ids, modalities) for spec in BASELINE_CATALOG]


def _tribe_style_catalog_status(task_ids: set[str], modalities: set[str]) -> str:
    return "clean_room_approximation" if "stimulus_to_fmri_response" in task_ids and "fmri" in modalities else "unavailable"


def _brainvista_catalog_status(task_ids: set[str], modalities: set[str]) -> str:
    return "approximation" if "future_state_forecasting" in task_ids and "fmri" in modalities else "unavailable"


def _brain_of_catalog_status(task_ids: set[str], modalities: set[str]) -> str:
    return "approximation" if "masked_neural_reconstruction" in task_ids and len(modalities) >= 2 else "unavailable"


def _brainomni_catalog_status(task_ids: set[str], modalities: set[str]) -> str:
    return "approximation" if modalities & {"eeg", "meg"} else "unavailable"


BASELINE_CATALOG: tuple[BaselineCatalogEntry, ...] = (
    BaselineCatalogEntry(
        model_id="persistence",
        display_name="Persistence",
        status="local_baseline",
        notes="Last-observation or identity-style forecast with shape-safe fallback.",
    ),
    BaselineCatalogEntry(
        model_id="train_mean",
        display_name="Train Mean",
        status="local_baseline",
        notes="Broadcast train-target mean negative baseline.",
    ),
    BaselineCatalogEntry(
        model_id="random_permutation",
        display_name="Random Permutation",
        status="negative_control",
        notes="Seeded permutation of training targets with target-shaped output.",
    ),
    BaselineCatalogEntry(
        model_id="linear_ridge",
        display_name="Linear Ridge",
        status="local_baseline",
        notes="Closed-form sanity baseline on identical prepared windows.",
    ),
    BaselineCatalogEntry(
        model_id="autoregressive_ridge",
        display_name="Autoregressive Ridge",
        status="local_baseline",
        notes="Ridge from previous source timepoint to next target timepoint where sequence shapes allow.",
    ),
    BaselineCatalogEntry(
        model_id="mlp",
        display_name="MLP",
        status="local_baseline",
        notes="Per-timepoint neural-window baseline.",
    ),
    BaselineCatalogEntry(
        model_id="tcn",
        display_name="TCN",
        status="local_baseline",
        notes="Local temporal convolution baseline.",
    ),
    BaselineCatalogEntry(
        model_id="transformer",
        display_name="Transformer",
        status="local_baseline",
        notes="Small local Transformer with shared splits.",
    ),
    BaselineCatalogEntry(
        model_id="ssm_fallback",
        display_name="SSM Fallback",
        status="local_baseline",
        notes="GRU-based SSM fallback until Mamba is pinned.",
    ),
    BaselineCatalogEntry(
        model_id="neurotwin",
        display_name="NeuroTwin",
        status="local_baseline",
        notes="Current NeuroTwin implementation under the same task API.",
    ),
    BaselineCatalogEntry(
        model_id="tribe_style",
        display_name="TRIBE-Style",
        status=_tribe_style_catalog_status,
        notes="NeuroTwin-native stimulus-to-fMRI approximation; not an exact TRIBE v2 reproduction.",
        upstream_reference="TRIBE v2",
        exact_reproduction=False,
        uses_upstream_code=False,
        uses_upstream_weights=False,
    ),
    BaselineCatalogEntry(
        model_id="brainvista_style",
        display_name="BrainVista-Style",
        status=_brainvista_catalog_status,
        notes="Approximate autoregressive fMRI rollout lane; not an exact BrainVista reproduction.",
        upstream_reference="BrainVista",
    ),
    BaselineCatalogEntry(
        model_id="brain_of_style",
        display_name="Brain-OF-Style",
        status=_brain_of_catalog_status,
        notes="Approximate multimodal masked reconstruction lane; not an exact Brain-OF reproduction.",
        upstream_reference="Brain-OF",
    ),
    BaselineCatalogEntry(
        model_id="brainomni_style",
        display_name="BrainOmni-Style",
        status=_brainomni_catalog_status,
        notes="Approximate EEG/MEG tokenizer lane; not an exact BrainOmni reproduction.",
        upstream_reference="BrainOmni",
    ),
    BaselineCatalogEntry(
        model_id="braindecode_wrapper",
        display_name="Braindecode Wrapper",
        status="unavailable",
        notes="Optional EEG wrapper slot; exact use requires installed Braindecode and compatible task protocols.",
        upstream_reference="Braindecode",
    ),
    BaselineCatalogEntry(
        model_id="cebra_wrapper",
        display_name="CEBRA Wrapper",
        status="unavailable",
        notes="Optional neural-behavior embedding wrapper slot; exact use requires installed CEBRA and aligned behavior data.",
        upstream_reference="CEBRA",
    ),
)
