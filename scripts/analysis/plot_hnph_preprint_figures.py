#!/usr/bin/env python3
"""Render the HNPH protocol figure set and paired provenance artifacts.

No data root means protocol-only mode: a hash-bound Sleep-EDF panel is included
only as descriptive single-label transport context. A supplied data root must
contain a fail-closed ``source_qualification.json`` and a local, hash-bound NPZ
example. Raw samples and local paths are never copied into output artifacts.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Mapping

_STYLE_PATH = Path(__file__).resolve().parent / "_kahlus_figure_style.py"
_STYLE_SPEC = importlib.util.spec_from_file_location("kahlus_figure_style", _STYLE_PATH)
if _STYLE_SPEC is None or _STYLE_SPEC.loader is None:
    raise ImportError(f"could not load figure style module from {_STYLE_PATH}")
_STYLE = importlib.util.module_from_spec(_STYLE_SPEC)
_STYLE_SPEC.loader.exec_module(_STYLE)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

BLUE = _STYLE.BLUE
RED = _STYLE.RED
GREEN = _STYLE.GREEN
ORANGE = _STYLE.ORANGE
PURPLE = _STYLE.PURPLE
GRAY = _STYLE.GRAY
INK = _STYLE.INK
WHITE = _STYLE.WHITE
LIGHT = _STYLE.LIGHT
LIGHT_BLUE = _STYLE.LIGHT_BLUE
LIGHT_RED = _STYLE.LIGHT_RED
LIGHT_GREEN = _STYLE.LIGHT_GREEN
LIGHT_ORANGE = _STYLE.LIGHT_ORANGE
LIGHT_PURPLE = _STYLE.LIGHT_PURPLE
apply_kahlus_style = _STYLE.apply_kahlus_style
provenance_footer = _STYLE.provenance_footer
stamp_schematic = _STYLE.stamp_schematic
FIGURE_SCHEMA = "kahlus.hnph.figure_manifest.v1"
QUALIFICATION_SCHEMA = "kahlus.hnph.dod_source_qualification.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DESCRIPTIVE_SOURCE_DIR = (
    REPO_ROOT / "docs" / "paper" / "hnph_preprint" / "figure_sources" / "sleep_edf_descriptive"
)
DESCRIPTIVE_SOURCE_PROVENANCE = DESCRIPTIVE_SOURCE_DIR / "source_provenance.json"


CAPTIONS = {
    "fig1_operational_task": (
        "Operational HNPH task. An oracle current-stage annotation and causal EEG history precede a "
        "guard interval and four future lead bands. The five-way leave-one-rater-out target is scored "
        "against the best eligible nuisance comparator. This is a task schematic, not a performance result."
    ),
    "fig2_leakage_label_contract": (
        "Leakage and label contract. People are split before any fitted transform; input and outcome supports "
        "are disjoint in physical time; and the held-out target rater never contributes to the consensus target."
    ),
    "fig3_study_flow": (
        "Baseline-first study flow. DOD-H source qualification and development gates precede model-family freeze; "
        "DOD-O remains sealed until then. Every stage is UNRUN in this protocol release."
    ),
    "fig4_protocol_outcomes": (
        "Possible protocol outcomes. A supported frontier, bounded or null result, failed calibration or control, "
        "and invalid execution are distinct outcomes. Symbolic bands encode decisions, not observed effects."
    ),
    "figA1_verified_example": (
        "Descriptive Sleep-EDF trace, spectrogram, and single-label hypnogram from record SC4002E0. "
        "The origin package reports 100 Hz sampling and hash-bound PSG/hypnogram sources. This transport "
        "illustration is not independently source-qualified under HNPH v0.4 and cannot enable a repeated-rater "
        "construct-validity or empirical frontier claim."
    ),
    "figA2_lineage": (
        "Raw-to-evidence lineage. Hash-bound local sources feed qualification, person-first manifests, causal "
        "targets, frozen comparisons, and paired JSON/Markdown evidence; raw neural samples remain outside git."
    ),
    "figA3_score_ceiling": (
        "Proper-score decomposition and observation-contraction ceiling. The conditional-information identity is "
        "an oracle statement, while contraction is assumption-dependent; neither is presented as a novel theorem."
    ),
}


@dataclass(frozen=True)
class RawExample:
    signal: np.ndarray
    stages: np.ndarray
    sampling_rate_hz: float
    unit: str
    channel_name: str
    safe_record_id: str
    dataset: str


@dataclass(frozen=True)
class BuildContext:
    protocol: Mapping[str, Any]
    protocol_sha256: str
    qualification: Mapping[str, Any] | None
    qualification_sha256: str | None
    raw_example: RawExample | None
    input_hashes: Mapping[str, str]
    dataset_versions: Mapping[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"{label} could not be read: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _load_descriptive_source(input_hashes: dict[str, str]) -> dict[str, Path]:
    provenance = _load_mapping(DESCRIPTIVE_SOURCE_PROVENANCE, "descriptive figure provenance")
    if provenance.get("schema") != "kahlus.hnph.descriptive_figure_source.v1":
        raise ValueError("descriptive figure provenance schema is invalid")
    if provenance.get("status") != "descriptive_single_label_transport_not_claim_evidence":
        raise ValueError("descriptive figure claim boundary is missing")
    assets = provenance.get("assets")
    if not isinstance(assets, dict):
        raise ValueError("descriptive figure provenance lacks assets")
    paths: dict[str, Path] = {}
    for kind in ("pdf", "png"):
        record = assets.get(kind)
        if not isinstance(record, dict) or not _sha(record.get("sha256")):
            raise ValueError(f"descriptive figure {kind} hash is missing")
        path = DESCRIPTIVE_SOURCE_DIR / str(record.get("file", ""))
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"descriptive figure {kind} does not match its source hash")
        paths[kind] = path
        input_hashes[f"sleep_edf_descriptive_{kind}"] = record["sha256"]
    input_hashes["sleep_edf_descriptive_provenance"] = sha256_file(DESCRIPTIVE_SOURCE_PROVENANCE)
    return paths


def _validate_dataset(dataset_id: str, value: object) -> str:
    if not isinstance(value, dict):
        raise ValueError(f"{dataset_id}: qualification entry must be an object")
    for field in ("version", "license", "source_sha256", "subject_identity_field", "channels", "raters"):
        if not value.get(field):
            raise ValueError(f"{dataset_id}: missing {field}")
    if not _sha(value["source_sha256"]):
        raise ValueError(f"{dataset_id}: source_sha256 must be a lowercase SHA-256")
    channels = value["channels"]
    if not isinstance(channels, list) or not channels:
        raise ValueError(f"{dataset_id}: physical channel metadata are missing")
    for channel in channels:
        if not isinstance(channel, dict) or not channel.get("name") or not channel.get("unit"):
            raise ValueError(f"{dataset_id}: every channel needs name and physical unit")
        rate = channel.get("sampling_rate_hz")
        if not isinstance(rate, (int, float)) or not np.isfinite(rate) or rate <= 0:
            raise ValueError(f"{dataset_id}: every channel needs a positive sampling rate")
    raters = value["raters"]
    if not isinstance(raters, list) or len(raters) < 5:
        raise ValueError(f"{dataset_id}: fewer than five independent source raters")
    rater_ids: set[str] = set()
    for rater in raters:
        if not isinstance(rater, dict) or not isinstance(rater.get("id"), str):
            raise ValueError(f"{dataset_id}: each rater needs a stable ID")
        if rater["id"] in rater_ids or not _sha(rater.get("annotation_sha256")):
            raise ValueError(f"{dataset_id}: rater IDs must be unique and hash-bound")
        rater_ids.add(rater["id"])
    return str(value["version"])


def _validate_qualification(payload: Mapping[str, Any]) -> dict[str, str]:
    if payload.get("schema") != QUALIFICATION_SCHEMA or payload.get("qualified") is not True:
        raise ValueError("source qualification is missing or not qualified")
    if payload.get("external_opened") is not False:
        raise ValueError("external DOD-O data were opened before protocol/model freeze")
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("source qualification lacks dataset records")
    versions = {dataset_id: _validate_dataset(dataset_id, datasets.get(dataset_id)) for dataset_id in ("DOD-H", "DOD-O")}
    loo = payload.get("leave_one_rater_out")
    if not isinstance(loo, list) or not loo:
        raise ValueError("leave-one-rater-out target audit is missing")
    for record in loo:
        if not isinstance(record, dict):
            raise ValueError("leave-one-rater-out audit records must be objects")
        target = record.get("target_rater_id")
        consensus = record.get("consensus_rater_ids")
        if not isinstance(target, str) or not isinstance(consensus, list) or len(set(consensus)) < 3:
            raise ValueError("every target needs at least three independent consensus raters")
        if target in consensus:
            raise ValueError("consensus target contains the held-out rater")
        if not _sha(record.get("target_sha256")):
            raise ValueError("leave-one-rater-out target hash is missing")
    return versions


def _load_raw_example(data_root: Path, qualification: Mapping[str, Any], input_hashes: dict[str, str]) -> RawExample:
    meta = qualification.get("raw_example")
    if not isinstance(meta, dict):
        raise ValueError("qualified source packet lacks raw_example metadata")
    for field in ("dataset", "safe_record_id", "npz", "signal_sha256", "annotation_sha256", "sampling_rate_hz", "unit", "channel_names"):
        if not meta.get(field):
            raise ValueError(f"raw_example missing {field}")
    if meta["dataset"] not in {"DOD-H", "DOD-O"}:
        raise ValueError("raw_example dataset is not a qualified DOD cohort")
    if not _sha(meta["signal_sha256"]) or not _sha(meta["annotation_sha256"]):
        raise ValueError("raw_example hashes are malformed")
    path = (data_root / str(meta["npz"])).resolve()
    if data_root.resolve() not in path.parents or not path.is_file():
        raise ValueError("raw_example NPZ must be a local file beneath --data-root")
    if sha256_file(path) != meta["signal_sha256"]:
        raise ValueError("raw_example signal hash does not match the local NPZ")
    with np.load(path, allow_pickle=False) as values:
        if "signal" not in values or "stages" not in values:
            raise ValueError("raw_example NPZ requires signal and stages arrays")
        signal = np.asarray(values["signal"], dtype=np.float64)
        stages = np.asarray(values["stages"], dtype=np.int64)
    if signal.ndim == 2:
        signal = signal[0]
    if signal.ndim != 1 or signal.size < 128 or not np.isfinite(signal).all():
        raise ValueError("raw_example signal must be a finite one-channel vector")
    if stages.ndim != 1 or stages.size < 2:
        raise ValueError("raw_example stages must be a one-dimensional annotation vector")
    rate = float(meta["sampling_rate_hz"])
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("raw_example sampling rate is invalid")
    input_hashes["raw_example_npz"] = meta["signal_sha256"]
    input_hashes["raw_example_annotations"] = meta["annotation_sha256"]
    return RawExample(
        signal=signal,
        stages=stages,
        sampling_rate_hz=rate,
        unit=str(meta["unit"]),
        channel_name=str(meta["channel_names"][0]),
        safe_record_id=str(meta["safe_record_id"]),
        dataset=str(meta["dataset"]),
    )


def load_context(protocol_path: Path, data_root: Path | None) -> BuildContext:
    protocol = _load_mapping(protocol_path, "protocol")
    if protocol.get("protocol_id") != "kahlus.hnph.phase0.v0.4":
        raise ValueError("figure renderer requires the frozen HNPH v0.4 protocol")
    protocol_hash = sha256_file(protocol_path)
    inputs: dict[str, str] = {
        "protocol": protocol_hash,
        "renderer": sha256_file(Path(__file__).resolve()),
    }
    _load_descriptive_source(inputs)
    qualification = None
    qualification_hash = None
    raw_example = None
    versions: dict[str, str] = {"DOD-H": "unverified", "DOD-O": "unverified"}
    if data_root is not None:
        qualification_path = data_root / "source_qualification.json"
        qualification = _load_mapping(qualification_path, "source qualification")
        qualification_hash = sha256_file(qualification_path)
        versions = _validate_qualification(qualification)
        inputs["source_qualification"] = qualification_hash
        raw_example = _load_raw_example(data_root, qualification, inputs)
        result_path = data_root / "hnph_result.json"
        if result_path.exists():
            result = _load_mapping(result_path, "HNPH result")
            if (
                result.get("claim_eligible") is not True
                or result.get("gate_passed") is not True
                or result.get("protocol_sha256") != protocol_hash
            ):
                raise ValueError("empirical-looking HNPH result lacks claim-eligible hash-bound evidence")
            inputs["claim_eligible_result"] = sha256_file(result_path)
    return BuildContext(protocol, protocol_hash, qualification, qualification_hash, raw_example, inputs, versions)


def apply_style() -> None:
    apply_kahlus_style(dpi=160)
    plt.rcParams.update(
        {
            "font.size": 10.7,
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.7,
            "xtick.labelsize": 10.7,
            "ytick.labelsize": 10.7,
            "legend.fontsize": 10.7,
            "axes.grid": False,
        }
    )


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str = BLUE,
    hatch: str | None = None,
    facecolor: str = WHITE,
    fontsize: float = 10.7,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02",
        facecolor=facecolor,
        edgecolor=color,
        linewidth=1.4,
        hatch=hatch,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color=INK,
        fontsize=fontsize,
        linespacing=1.25,
    )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = GRAY) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=1.2, color=color))


def _stamp(fig: plt.Figure, text: str) -> None:
    stamp_schematic(fig, text)


def _protocol_footer(fig: plt.Figure, ctx: BuildContext) -> None:
    provenance_footer(
        fig,
        f"HNPH v0.4 · protocol_sha256={ctx.protocol_sha256[:12]}… · "
        f"source_qualification={ctx.qualification is not None and 'qualified' or 'unverified'} · "
        "protocol schematic — no empirical frontier claim",
    )


def fig_operational(ctx: BuildContext) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_facecolor(LIGHT)
    ax.set_title(
        "Sleep-transition forecasting is anchored on causal history, not future labels",
        loc="left",
        weight="bold",
        fontsize=12.5,
        color=INK,
    )
    ax.text(3, 91, "A.  CAUSAL OBSERVATION AND PHYSICAL TIME", weight="bold", fontsize=11, color=INK)
    if ctx.raw_example is None:
        _box(
            ax,
            (3, 66),
            29,
            17,
            "EEG HISTORY\nSOURCE UNVERIFIED",
            GRAY,
            "///",
            facecolor=LIGHT,
        )
    else:
        raw = ctx.raw_example
        n = min(raw.signal.size, int(raw.sampling_rate_hz * 20))
        segment = raw.signal[:n]
        scale = np.ptp(segment) or 1.0
        x = np.linspace(5, 30, n)
        y = 74 + 9 * (segment - np.mean(segment)) / scale
        ax.plot(x, y, color=INK, linewidth=0.8)
        ax.text(18, 62, f"{raw.channel_name} [{raw.unit}] · {raw.sampling_rate_hz:g} Hz", ha="center", fontsize=10)
    ax.text(43, 84, "strictly causal history", color=BLUE, weight="bold", fontsize=10)
    ax.plot([39, 96], [74, 74], color=INK, linewidth=1.4)
    ax.add_patch(Rectangle((39, 70), 16, 8, facecolor=LIGHT_BLUE, edgecolor=BLUE, linewidth=1.6))
    ax.text(47, 74, "preceding\n10 min", ha="center", va="center", weight="bold", color=BLUE, fontsize=10)
    ax.add_patch(Rectangle((55, 70), 4, 8, facecolor=LIGHT, hatch="xx", edgecolor=GRAY, linewidth=1.2))
    bands = [(59, 67, "B1"), (67, 78, "B2"), (78, 87, "B3"), (87, 96, "B4")]
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    for (left, right, label), color in zip(bands, colors):
        ax.add_patch(Rectangle((left, 70), right - left, 8, facecolor=WHITE, edgecolor=color, linewidth=1.6))
        ax.text((left + right) / 2, 74, label, ha="center", va="center", weight="bold", fontsize=10)
    ax.text(77.5, 64, "B1 0.5–2 · B2 2–5 primary · B3 5–10 · B4 10–20 min", ha="center", fontsize=9.5, color=GRAY)
    _box(
        ax,
        (59, 53),
        37,
        7,
        "five-way outcome: no event · Wake · NREM · REM · Ambiguous",
        GREEN,
        facecolor=LIGHT_GREEN,
        fontsize=9.5,
    )
    ax.text(68, 48, "$C_t\\;\\cap\\;T_{t,h}=\\varnothing$", ha="center", weight="bold", fontsize=13, color=INK)
    ax.text(68, 44, "history and future target occupy disjoint physical support", ha="center", color=GRAY, fontsize=9.5)
    ax.text(3, 36, "B.  ISOLATING INCREMENTAL EEG INFORMATION", weight="bold", fontsize=11, color=INK)
    _box(
        ax,
        (4, 13),
        28,
        17,
        "nuisance $q_b$\nstate · history\nbout age · elapsed time\ntransitions · quality",
        ORANGE,
        facecolor=LIGHT_ORANGE,
    )
    _box(
        ax,
        (4, 1),
        28,
        9,
        "EEG + nuisance $q_\\theta$\ncausal history + covariates",
        GREEN,
        facecolor=LIGHT_GREEN,
        fontsize=9.5,
    )
    _box(ax, (44, 13), 20, 15, "same five-way\noutcome target", BLUE, facecolor=LIGHT_BLUE)
    _arrow(ax, (32, 21), (44, 21), ORANGE)
    _arrow(ax, (32, 5.5), (44, 17), GREEN)
    _box(ax, (72, 13), 24, 15, "subject-balanced\nlog-score gain (bits)", PURPLE, facecolor=LIGHT_PURPLE)
    _arrow(ax, (64, 20.5), (72, 20.5), GRAY)
    _stamp(fig, "PROTOCOL SCHEMATIC — SOURCE UNVERIFIED · NO PERFORMANCE RESULT")
    _protocol_footer(fig, ctx)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def fig_contract(ctx: BuildContext) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_facecolor(LIGHT)
    ax.set_title(
        "Leakage guards require disjoint physical time and person-first splits",
        loc="left",
        weight="bold",
        fontsize=12.5,
        color=INK,
    )
    ax.text(3, 91, "A.  PHYSICAL-TIME FORECAST ANCHOR", weight="bold", fontsize=11, color=INK)
    ax.add_patch(Rectangle((4, 77), 31, 8, facecolor=LIGHT_BLUE, edgecolor=BLUE, linewidth=1.6))
    ax.add_patch(Rectangle((35, 77), 7, 8, facecolor=LIGHT, hatch="xx", edgecolor=GRAY, linewidth=1.2))
    ax.add_patch(Rectangle((42, 77), 53, 8, facecolor=LIGHT_ORANGE, edgecolor=ORANGE, linewidth=1.6))
    ax.text(19.5, 81, "causal EEG history", ha="center", weight="bold", color=BLUE, fontsize=10)
    ax.text(38.5, 70, "filter\nguard", ha="center", fontsize=9.5, color=GRAY)
    ax.text(68.5, 81, "future target support", ha="center", weight="bold", color=ORANGE, fontsize=10)
    ax.text(50, 88, "$C_t\\;\\cap\\;T_{t,h}=\\varnothing$", ha="center", fontsize=14, weight="bold", color=INK)
    ax.text(50, 65, "complete follow-up required · native 30-second grid", ha="center", color=GRAY, fontsize=9.5)
    ax.text(3, 56, "B.  PERSON FIRST, THEN SPLIT", weight="bold", fontsize=11, color=INK)
    _box(ax, (4, 38), 23, 12, "DOD-H / DOD-O\nSOURCE UNVERIFIED", GRAY, "///", facecolor=LIGHT)
    _box(ax, (36, 38), 19, 12, "canonical\nperson bundles", BLUE, facecolor=LIGHT_BLUE)
    _box(ax, (65, 38), 12, 12, "DOD-H\ndevelopment", ORANGE, "//", facecolor=LIGHT_ORANGE, fontsize=9.5)
    _box(ax, (84, 38), 12, 12, "DOD-O\nSEALED", GREEN, "//", facecolor=LIGHT_GREEN, fontsize=9.5)
    _arrow(ax, (27, 44), (36, 44))
    _arrow(ax, (55, 44), (65, 44))
    _arrow(ax, (77, 44), (84, 44))
    ax.text(50, 32, "all recordings from one person remain in one partition", ha="center", color=GRAY, fontsize=9.5)
    ax.text(3, 24, "C.  ARTIFACT-BOUND CONTRACT", weight="bold", fontsize=11, color=INK)
    _box(
        ax,
        (28, 4),
        44,
        17,
        "subject-safe ID · physical supports\nLOO rater IDs · units/rates\nsource and target hashes",
        BLUE,
        facecolor=LIGHT_BLUE,
        fontsize=10,
    )
    _box(ax, (3, 4), 20, 17, "FORBIDDEN\nfuture / overlap\nIDs / local paths", RED, facecolor=LIGHT_RED)
    _box(ax, (77, 4), 20, 17, "REQUIRED\nafter qualification\nall UNRUN", RED, "//", facecolor=LIGHT_RED)
    _stamp(fig, "PROTOCOL SCHEMATIC — NOT EMPIRICAL EVIDENCE")
    _protocol_footer(fig, ctx)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def fig_study_flow(ctx: BuildContext) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_facecolor(LIGHT)
    ax.set_title(
        "Baseline-first gates must pass before any sealed external evaluation opens",
        loc="left",
        weight="bold",
        fontsize=12.5,
        color=INK,
    )
    ax.text(50, 91, "Protocol v0.4 execution order", ha="center", weight="bold", fontsize=13, color=INK)
    steps = [
        (2, "1", "DOD-H\nqualify source", BLUE),
        (22, "2", "baseline +\npower gates", ORANGE),
        (42, "3", "freeze model\nfamily + seeds", PURPLE),
        (62, "4", "open sealed\nDOD-O", GREEN),
        (82, "5", "JSON + MD\nevidence", BLUE),
    ]
    for x, number, label, color in steps:
        ax.text(
            x + 8,
            79,
            number,
            ha="center",
            va="center",
            fontsize=13,
            color=WHITE,
            weight="bold",
            bbox={"boxstyle": "circle", "facecolor": color, "edgecolor": color, "linewidth": 1.4},
        )
        _box(ax, (x, 46), 16, 25, f"{label}\n\nUNRUN", color, "//", facecolor=WHITE)
    for (x1, _, _, _), (x2, _, _, _) in zip(steps[:-1], steps[1:]):
        _arrow(ax, (x1 + 16, 61.5), (x2, 61.5))
    _box(
        ax,
        (7, 13),
        36,
        19,
        "DOD-H qualification\npeople · raters · physical metadata\nBLOCKED",
        BLUE,
        "//",
        facecolor=LIGHT_BLUE,
        fontsize=10,
    )
    _box(
        ax,
        (57, 13),
        36,
        19,
        "DOD-O seal\nfreeze protocol and model choices\nUNRUN",
        GREEN,
        "//",
        facecolor=LIGHT_GREEN,
        fontsize=10,
    )
    ax.text(50, 7, "Sleep-EDF / CAP remain descriptive single-label transport checks only.", ha="center", color=GRAY, fontsize=9.5)
    _stamp(fig, "ALL STAGES UNRUN — PROTOCOL FLOW ONLY")
    _protocol_footer(fig, ctx)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def fig_outcomes(ctx: BuildContext) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0))
    fig.patch.set_facecolor(WHITE)
    definitions = [
        ("SUPPORTED FRONTIER", GREEN, "contiguous passing prefix", [1, 1, 0, 0]),
        ("BOUNDED / NULL", BLUE, "best baseline ties or wins", [0, 0, 0, 0]),
        ("CALIBRATION / CONTROL FAIL", ORANGE, "scientific result blocked", [1, -1, 1, -1]),
        ("INVALID EXECUTION", RED, "integrity or provenance failure", [-1, -1, -1, -1]),
    ]
    for ax, (title, color, subtitle, states) in zip(axes.flat, definitions):
        ax.set_facecolor(LIGHT)
        ax.set_title(title, color=color, weight="bold", fontsize=11.5)
        ax.set_xlim(0, 4)
        ax.set_ylim(-1.5, 1.6)
        ax.axhline(0, color=INK, linewidth=0.9)
        for index, state in enumerate(states):
            face = WHITE if state == 0 else (LIGHT_GREEN if state > 0 else LIGHT_RED)
            hatch = "//" if state < 0 else None
            ax.add_patch(Rectangle((index + 0.15, -0.35), 0.7, 0.7, facecolor=face, edgecolor=color, hatch=hatch, linewidth=1.6))
        ax.set_xticks(np.arange(4) + 0.5, ["B1", "B2", "B3", "B4"])
        ax.set_yticks([])
        ax.text(2, -1.0, subtitle, ha="center", va="center", fontsize=9.5, color=GRAY)
        ax.spines[["left", "bottom"]].set_visible(False)
    fig.suptitle(
        "Four distinct protocol outcomes — symbolic bands, not observed effects",
        x=0.02,
        ha="left",
        weight="bold",
        fontsize=12.5,
        color=INK,
    )
    _stamp(fig, "SYMBOLIC DECISION BANDS — NO PERFORMANCE CURVE")
    _protocol_footer(fig, ctx)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    return fig


def fig_verified(ctx: BuildContext) -> plt.Figure:
    if ctx.raw_example is None:
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.axis("off")
        ax.set_title("Verified trace / spectrogram / hypnogram example", loc="left", weight="bold")
        _box(ax, (0.12, 0.25), 0.76, 0.45, "BLOCKED — SOURCE QUALIFICATION UNVERIFIED\n\nNo synthetic or decorative neural trace substituted.\nProvide a local hash-bound qualified DOD example to render this panel.", GRAY, "///")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _stamp(fig, "RAW ILLUSTRATION UNAVAILABLE — NOT EVIDENCE")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        return fig
    raw = ctx.raw_example
    duration = raw.signal.size / raw.sampling_rate_hz
    time = np.arange(raw.signal.size) / raw.sampling_rate_hz
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=False, gridspec_kw={"height_ratios": [1, 1.5, 0.7]})
    axes[0].plot(time, raw.signal, color=INK, linewidth=0.6)
    axes[0].set_ylabel(f"{raw.channel_name}\n[{raw.unit}]")
    axes[0].set_title(f"{raw.dataset} · record {raw.safe_record_id} · {raw.sampling_rate_hz:g} Hz", loc="left", weight="bold")
    axes[1].specgram(raw.signal, NFFT=min(256, raw.signal.size // 2), Fs=raw.sampling_rate_hz, noverlap=min(128, raw.signal.size // 4), cmap="gray_r")
    axes[1].set_ylabel("frequency [Hz]")
    axes[1].set_xlabel("time [s]")
    stage_time = np.linspace(0, duration, raw.stages.size, endpoint=False)
    axes[2].step(stage_time, raw.stages, where="post", color=BLUE, linewidth=1.2)
    axes[2].set_ylabel("rater stage")
    axes[2].set_xlabel("time [s]")
    _stamp(fig, "RAW TASK ILLUSTRATION — NO PERFORMANCE RESULT")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_lineage(ctx: BuildContext) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_facecolor(LIGHT)
    ax.set_title(
        "Hash-bound lineage keeps raw neural data out of git while preserving auditability",
        loc="left",
        weight="bold",
        fontsize=12.5,
        color=INK,
    )
    steps = [
        (3, "LOCAL RAW\nnot in git", GRAY),
        (28, "QUALIFY +\nPERSON SPLIT", BLUE),
        (53, "TARGETS +\nBASELINES", ORANGE),
        (78, "GATE\nJSON + MD", PURPLE),
    ]
    for x, label, color in steps:
        _box(ax, (x, 46), 18, 24, label, color, facecolor=WHITE)
    for (x1, _, _), (x2, _, _) in zip(steps[:-1], steps[1:]):
        _arrow(ax, (x1 + 18, 58), (x2, 58))
    ax.text(50, 31, "SHA-256 binds every boundary; emitted artifacts contain subject-safe IDs only", ha="center", weight="bold", fontsize=10.5, color=INK)
    ax.text(50, 19, "runner → gate → JSON + Markdown", ha="center", color=BLUE, fontsize=12)
    _stamp(fig, "PROVENANCE SCHEMATIC — NOT EMPIRICAL EVIDENCE")
    _protocol_footer(fig, ctx)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def fig_score(ctx: BuildContext) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.8))
    fig.patch.set_facecolor(WHITE)
    axes[0].axis("off")
    axes[0].set_facecolor(LIGHT)
    axes[0].set_title("Proper-score estimand", loc="left", weight="bold", fontsize=12, color=INK)
    _box(axes[0], (0.05, 0.56), 0.9, 0.24, "Δ log skill\nNLL nuisance − NLL EEG+nuisance\ndivided by ln 2", BLUE, facecolor=LIGHT_BLUE)
    _box(axes[0], (0.05, 0.22), 0.9, 0.24, "Oracle limit\nI(Y; EEG | nuisance)\nnot a fitted-model equality", GREEN, facecolor=LIGHT_GREEN)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[1].axis("off")
    axes[1].set_facecolor(LIGHT)
    axes[1].set_title("Observation contraction", loc="left", weight="bold", fontsize=12, color=INK)
    _box(axes[1], (0.18, 0.70), 0.64, 0.14, "latent state", PURPLE, facecolor=LIGHT_PURPLE)
    _box(axes[1], (0.18, 0.46), 0.64, 0.14, "observed EEG", BLUE, facecolor=LIGHT_BLUE)
    _box(axes[1], (0.18, 0.22), 0.64, 0.14, "future annotation", GREEN, facecolor=LIGHT_GREEN)
    _arrow(axes[1], (0.50, 0.70), (0.50, 0.60))
    _arrow(axes[1], (0.50, 0.46), (0.50, 0.36))
    axes[1].text(0.5, 0.10, "Bound requires stated channel assumptions\nNo constants estimated", ha="center", color=RED, weight="bold", fontsize=10)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    _stamp(fig, "ASSUMPTION-DEPENDENT · STANDARD CONSEQUENCE · NON-NOVEL")
    _protocol_footer(fig, ctx)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


FIGURES: tuple[tuple[str, str, Callable[[BuildContext], plt.Figure], list[str]], ...] = (
    ("fig1_operational_task", "raw_illustration", fig_operational, ["causal task", "lead bands", "score comparison"]),
    ("fig2_leakage_label_contract", "conceptual", fig_contract, ["split", "firebreak", "rater exclusion"]),
    ("fig3_study_flow", "conceptual", fig_study_flow, ["development", "gates", "sealed external"]),
    ("fig4_protocol_outcomes", "conceptual", fig_outcomes, ["supported", "null", "failed", "invalid"]),
    ("figA1_verified_example", "raw_illustration", fig_verified, ["trace", "spectrogram", "hypnogram"]),
    ("figA2_lineage", "conceptual", fig_lineage, ["source", "manifests", "evidence"]),
    ("figA3_score_ceiling", "conceptual", fig_score, ["proper score", "oracle identity", "data processing"]),
)


def _render_or_copy_descriptive_sleep_edf(pdf_path: Path, png_path: Path, input_hashes: dict[str, str]) -> None:
    """Render figA1 from hash-verified local EDF files; fall back to cached assets."""

    sleep_script = Path(__file__).resolve().parent / "plot_sleep_edf_descriptive_figure.py"
    spec = importlib.util.spec_from_file_location("plot_sleep_edf_descriptive_figure", sleep_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load Sleep-EDF renderer from {sleep_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        fig = module.render_sleep_edf_descriptive_figure()
        metadata = {"Creator": "Kahlus Sleep-EDF descriptive renderer", "CreationDate": None, "ModDate": None}
        fig.savefig(pdf_path, bbox_inches="tight", facecolor=WHITE, transparent=False, metadata=metadata)
        fig.savefig(
            png_path,
            dpi=170,
            bbox_inches="tight",
            facecolor=WHITE,
            transparent=False,
            metadata={"Software": "Kahlus Sleep-EDF descriptive renderer"},
        )
        plt.close(fig)
        _sync_descriptive_source_assets(pdf_path, png_path, input_hashes)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(f"Sleep-EDF descriptive render unavailable ({exc}); using cached source assets.", file=sys.stderr)
        descriptive_assets = _load_descriptive_source(input_hashes)
        shutil.copyfile(descriptive_assets["pdf"], pdf_path)
        shutil.copyfile(descriptive_assets["png"], png_path)


def _sync_descriptive_source_assets(pdf_path: Path, png_path: Path, input_hashes: dict[str, str]) -> None:
    """Mirror freshly rendered descriptive assets back into figure_sources with updated hashes."""

    DESCRIPTIVE_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    target_pdf = DESCRIPTIVE_SOURCE_DIR / "figure_a1_real_sleep_edf_waveform.pdf"
    target_png = DESCRIPTIVE_SOURCE_DIR / "figure_a1_real_sleep_edf_waveform.png"
    shutil.copyfile(pdf_path, target_pdf)
    shutil.copyfile(png_path, target_png)
    provenance = _load_mapping(DESCRIPTIVE_SOURCE_PROVENANCE, "descriptive figure provenance")
    assets = provenance.setdefault("assets", {})
    if not isinstance(assets, dict):
        raise ValueError("descriptive figure provenance lacks assets")
    assets["pdf"] = {"file": target_pdf.name, "sha256": sha256_file(target_pdf)}
    assets["png"] = {"file": target_png.name, "sha256": sha256_file(target_png)}
    derivative = provenance.setdefault("derivative_renderer", {})
    if isinstance(derivative, dict):
        derivative["change_scope"] = (
            "Regenerated from hash-verified local Sleep-EDF EDF files with Kahlus ridge-style "
            "matplotlib renderer; signal window, anchor selection, and stage mapping unchanged."
        )
        derivative["renderer"] = "scripts/analysis/plot_sleep_edf_descriptive_figure.py"
    DESCRIPTIVE_SOURCE_PROVENANCE.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    input_hashes["sleep_edf_descriptive_pdf"] = assets["pdf"]["sha256"]
    input_hashes["sleep_edf_descriptive_png"] = assets["png"]["sha256"]
    input_hashes["sleep_edf_descriptive_provenance"] = sha256_file(DESCRIPTIVE_SOURCE_PROVENANCE)


def _tex(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def render(ctx: BuildContext, out_dir: Path, command: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_records: list[dict[str, Any]] = []
    for name, classification, renderer, panels in FIGURES:
        pdf_path = out_dir / f"{name}.pdf"
        png_path = out_dir / f"{name}.png"
        if name == "figA1_verified_example" and ctx.raw_example is None:
            _render_or_copy_descriptive_sleep_edf(pdf_path, png_path, ctx.input_hashes)
        else:
            fig = renderer(ctx)
            metadata = {"Creator": "Kahlus HNPH figure renderer", "CreationDate": None, "ModDate": None}
            fig.savefig(pdf_path, bbox_inches="tight", facecolor=WHITE, transparent=False, metadata=metadata)
            fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=WHITE, transparent=False, metadata={"Software": "Kahlus HNPH figure renderer"})
            plt.close(fig)
        status = "verified_raw_illustration" if classification == "raw_illustration" and ctx.raw_example is not None else classification
        if name == "figA1_verified_example" and ctx.raw_example is None:
            status = "descriptive_single_label_transport_not_claim_evidence"
        figure_records.append(
            {
                "id": name,
                "classification": classification,
                "status": status,
                "claim_scope": "protocol_only_no_empirical_hnph_frontier_claim",
                "panels": panels,
                "caption": CAPTIONS[name],
                "outputs": {"pdf": sha256_file(pdf_path), "png": sha256_file(png_path)},
            }
        )
    captions_path = out_dir / "figure_captions.tex"
    caption_lines = ["% Generated by plot_hnph_preprint_figures.py; do not edit manually."]
    macro_suffixes = ("One", "Two", "Three", "Four", "Five", "Six", "Seven")
    for suffix, (name, _, _, _) in zip(macro_suffixes, FIGURES):
        caption_lines.append(rf"\newcommand{{\HNPHFigureCaption{suffix}}}{{{_tex(CAPTIONS[name])}}}")
    captions_path.write_text("\n".join(caption_lines) + "\n", encoding="utf-8")
    manifest = {
        "schema": FIGURE_SCHEMA,
        "protocol_id": ctx.protocol["protocol_id"],
        "protocol_sha256": ctx.protocol_sha256,
        "claim_scope": "protocol_only_no_empirical_hnph_frontier_claim",
        "source_qualification_status": "qualified" if ctx.qualification is not None else "unverified",
        "external_test_opened": False,
        "dataset_versions": dict(ctx.dataset_versions),
        "input_hashes": dict(ctx.input_hashes),
        "command": command,
        "figures": figure_records,
        "supporting_outputs": {"figure_captions_tex": sha256_file(captions_path)},
    }
    manifest_path = out_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance_path = out_dir / "FIGURE_PROVENANCE.md"
    lines = [
        "# HNPH Figure Provenance",
        "",
        f"- protocol: `{manifest['protocol_id']}`",
        f"- protocol_sha256: `{manifest['protocol_sha256']}`",
        f"- claim_scope: `{manifest['claim_scope']}`",
        f"- source_qualification_status: `{manifest['source_qualification_status']}`",
        "- external_test_opened: `false`",
        f"- command: `{command}`",
        "",
        "Raw data and local paths are not included. The Sleep-EDF appendix panel is a hash-bound descriptive "
        "source asset, not independently qualified HNPH v0.4 evidence.",
        "",
        "## Dataset versions",
        "",
        *[f"- {name}: `{version}`" for name, version in sorted(ctx.dataset_versions.items())],
        "",
        "## Input hashes",
        "",
        *[f"- {name}: `{digest}`" for name, digest in sorted(ctx.input_hashes.items())],
        "",
        "## Figures",
        "",
    ]
    for record in figure_records:
        lines.extend(
            [
                f"### {record['id']}",
                "",
                f"- classification: `{record['classification']}`",
                f"- status: `{record['status']}`",
                f"- PDF SHA-256: `{record['outputs']['pdf']}`",
                f"- PNG SHA-256: `{record['outputs']['png']}`",
                f"- panels: {', '.join(record['panels'])}",
                f"- caption: {record['caption']}",
                "",
            ]
        )
    provenance_path.write_text("\n".join(lines), encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    apply_style()
    try:
        ctx = load_context(args.protocol.resolve(), args.data_root.resolve() if args.data_root else None)
        command = (
            "python scripts/analysis/plot_hnph_preprint_figures.py "
            "--protocol configs/protocol/hnph_phase0_v0.4.yaml "
            "--out-dir docs/figures/hnph_protocol"
        )
        if args.data_root is not None:
            command += " --data-root <LOCAL_QUALIFIED_DATA_ROOT>"
        render(ctx, args.out_dir.resolve(), command)
    except (OSError, ValueError, KeyError) as exc:
        print(f"HNPH figure generation refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
