#!/usr/bin/env python3
"""Render the hash-bound Sleep-EDF descriptive appendix panel from local EDF files."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_STYLE_PATH = Path(__file__).resolve().parent / "_kahlus_figure_style.py"
_STYLE_SPEC = importlib.util.spec_from_file_location("kahlus_figure_style", _STYLE_PATH)
if _STYLE_SPEC is None or _STYLE_SPEC.loader is None:
    raise ImportError(f"could not load figure style module from {_STYLE_PATH}")
_STYLE = importlib.util.module_from_spec(_STYLE_SPEC)
_STYLE_SPEC.loader.exec_module(_STYLE)

BLUE = _STYLE.BLUE
GREEN = _STYLE.GREEN
GRAY = _STYLE.GRAY
INK = _STYLE.INK
ORANGE = _STYLE.ORANGE
PURPLE = _STYLE.PURPLE
RED = _STYLE.RED
WHITE = _STYLE.WHITE
LIGHT_BLUE = _STYLE.LIGHT_BLUE
LIGHT_ORANGE = _STYLE.LIGHT_ORANGE
apply_kahlus_style = _STYLE.apply_kahlus_style
provenance_footer = _STYLE.provenance_footer
stamp_schematic = _STYLE.stamp_schematic

REPO_ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_PATH = (
    REPO_ROOT / "docs" / "paper" / "hnph_preprint" / "figure_sources" / "sleep_edf_descriptive" / "source_provenance.json"
)
DEFAULT_SOURCE_DIR = Path.home() / "Downloads" / "kahlus_hnph_raw_figure_source"

RK_TO_STATE = {
    "sleep stage w": "Wake",
    "sleep stage 1": "NREM",
    "sleep stage 2": "NREM",
    "sleep stage 3": "NREM",
    "sleep stage 4": "NREM",
    "sleep stage r": "REM",
}


@dataclass(frozen=True)
class SleepEdfDescriptiveRecord:
    issue_time_s: float
    stable_transition_time_s: float
    transition_lead_min: float
    destination: str
    safe_record_id: str
    sampling_rate_hz: float
    psg_sha256: str
    hypnogram_sha256: str


def _load_provenance() -> SleepEdfDescriptiveRecord:
    import json

    payload = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    record = payload["record"]
    return SleepEdfDescriptiveRecord(
        issue_time_s=float(record["issue_time_seconds"]),
        stable_transition_time_s=float(record["stable_transition_time_seconds"]),
        transition_lead_min=float(record["transition_lead_minutes"]),
        destination=str(record["destination"]),
        safe_record_id=str(record["safe_record_id"]),
        sampling_rate_hz=float(record["sampling_rate_hz"]),
        psg_sha256=str(record["psg_sha256_reported_by_origin"]),
        hypnogram_sha256=str(record["hypnogram_sha256_reported_by_origin"]),
    )


def _resolve_source_dir(source_dir: Path | None) -> Path:
    candidates = [
        source_dir,
        DEFAULT_SOURCE_DIR,
        REPO_ROOT / ".context" / "sleep_edf_raw",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        psg = candidate / "SC4002E0-PSG.edf"
        hyp = candidate / "SC4002EC-Hypnogram.edf"
        if psg.is_file() and hyp.is_file():
            return candidate
    raise FileNotFoundError(
        "Sleep-EDF SC4002E0 pair not found; expected SC4002E0-PSG.edf and SC4002EC-Hypnogram.edf"
    )


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_edf(path: Path) -> Any:
    try:
        import edfio
    except ImportError as exc:
        raise RuntimeError("Sleep-EDF descriptive rendering requires edfio: pip install edfio") from exc
    return edfio.read_edf(path, lazy_load_data=False)


def _signal_by_label(edf: Any, label: str) -> np.ndarray:
    for signal in edf.signals:
        if str(signal.label).strip() == label:
            return np.asarray(signal.data, dtype=np.float64)
    raise KeyError(f"EDF signal {label!r} not found")


def _macrostate_timeline(
    annotations: tuple[Any, ...],
    start_s: float,
    end_s: float,
    cadence_s: float = 30.0,
) -> tuple[np.ndarray, list[str]]:
    epoch_count = int(round((end_s - start_s) / cadence_s))
    times = start_s + np.arange(epoch_count) * cadence_s
    states = ["Unknown"] * epoch_count
    for annotation in annotations:
        text = " ".join(str(getattr(annotation, "text", "")).strip().lower().split())
        state = RK_TO_STATE.get(text, "Unknown")
        onset = float(getattr(annotation, "onset"))
        duration = getattr(annotation, "duration")
        duration_s = cadence_s if duration is None else float(duration)
        epoch_start = int(round((onset - start_s) / cadence_s))
        epoch_end = int(round((onset + duration_s - start_s) / cadence_s))
        for index in range(max(0, epoch_start), min(epoch_count, epoch_end)):
            states[index] = state
    return times, states


def render_sleep_edf_descriptive_figure(source_dir: Path | None = None) -> plt.Figure:
    apply_kahlus_style(dpi=160)
    meta = _load_provenance()
    root = _resolve_source_dir(source_dir)
    psg_path = root / "SC4002E0-PSG.edf"
    hyp_path = root / "SC4002EC-Hypnogram.edf"
    if _sha256_file(psg_path) != meta.psg_sha256:
        raise ValueError("PSG SHA-256 does not match source_provenance.json")
    if _sha256_file(hyp_path) != meta.hypnogram_sha256:
        raise ValueError("hypnogram SHA-256 does not match source_provenance.json")

    psg = _read_edf(psg_path)
    hyp = _read_edf(hyp_path)
    fpz = _signal_by_label(psg, "EEG Fpz-Cz")
    pz = _signal_by_label(psg, "EEG Pz-Oz")
    fs = meta.sampling_rate_hz
    issue = meta.issue_time_s
    transition = meta.stable_transition_time_s

    fig = plt.figure(figsize=(8.5, 7.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.35, 0.95], hspace=0.34)

    # Panel A: 30 s raw EEG immediately before issue time.
    ax_a = fig.add_subplot(gs[0, 0])
    rel_start = issue - 30.0
    i0 = int(round(rel_start * fs))
    i1 = int(round(issue * fs))
    t_rel = (np.arange(i1 - i0) / fs) + rel_start - issue
    ax_a.plot(t_rel, fpz[i0:i1], color=INK, lw=0.7, label="EEG Fpz-Cz")
    ax_a.plot(t_rel, pz[i0:i1], color=BLUE, lw=0.7, alpha=0.85, label="EEG Pz-Oz")
    ax_a.axvline(0.0, color=ORANGE, lw=1.4, label="issue time t")
    ax_a.set_xlim(-30, 0)
    ax_a.set_ylabel("EEG (µV)")
    ax_a.set_title("Raw EEG immediately before the issue time", loc="left", weight="bold", fontsize=11.5, color=INK)
    ax_a.set_xlabel("Seconds relative to issue time t")
    ax_a.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax_a.text(-0.02, 1.02, "A", transform=ax_a.transAxes, weight="bold", fontsize=12, color=INK)

    # Panel B: 10-minute causal spectrogram ending at issue time.
    ax_b = fig.add_subplot(gs[1, 0])
    b0 = int(round((issue - 600.0) * fs))
    b1 = int(round(issue * fs))
    segment = fpz[b0:b1]
    spec, freqs, t_spec, im = ax_b.specgram(
        segment,
        NFFT=256,
        Fs=fs,
        noverlap=192,
        cmap="magma",
        scale="dB",
    )
    ax_b.set_ylim(0, 30)
    ax_b.set_xlim(-10, 0)
    ax_b.set_ylabel("Frequency (Hz)")
    ax_b.set_xlabel("Minutes relative to issue time t")
    ax_b.set_title("Ten-minute causal history (Fpz-Cz spectrogram)", loc="left", weight="bold", fontsize=11.5, color=INK)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.025, pad=0.01)
    cbar.set_label("Power (dB)", fontsize=8.5)
    ax_b.text(-0.02, 1.02, "B", transform=ax_b.transAxes, weight="bold", fontsize=12, color=INK)

    # Panel C: macrostate timeline with registered future band.
    ax_c = fig.add_subplot(gs[2, 0])
    window_start = issue - 600.0
    window_end = issue + 600.0
    times, states = _macrostate_timeline(tuple(hyp.annotations), window_start, window_end)
    rel_min = (times - issue) / 60.0
    state_colors = {"Wake": ORANGE, "NREM": BLUE, "REM": PURPLE, "Unknown": GRAY}
    for left, right, state in zip(rel_min[:-1], rel_min[1:], states[:-1]):
        ax_c.axvspan(left, right, color=state_colors.get(state, GRAY), alpha=0.35, linewidth=0)
    ax_c.axvspan(-10, 0, color=LIGHT_BLUE, alpha=0.45, label="causal context")
    ax_c.axvspan(2, 5, color=LIGHT_ORANGE, alpha=0.55, label="primary 2–5 min band")
    ax_c.axvline(0.0, color=INK, lw=1.3, label="issue time t")
    transition_min = (transition - issue) / 60.0
    ax_c.axvline(transition_min, color=GREEN, lw=1.5, label=f"stable transition to {meta.destination}")
    ax_c.text(
        transition_min + 0.15,
        0.55,
        f"stable transition to {meta.destination}\n(+{meta.transition_lead_min:g} min)",
        color=GREEN,
        fontsize=9,
        va="center",
    )
    ax_c.set_xlim(-10, 10)
    ax_c.set_ylim(0, 1)
    ax_c.set_yticks([])
    ax_c.set_xlabel("Minutes relative to issue time t")
    ax_c.set_title("Scored macrostate sequence and registered future band", loc="left", weight="bold", fontsize=11.5, color=INK)
    handles = [
        plt.Line2D([0], [0], color=ORANGE, lw=6, alpha=0.5, label="Wake"),
        plt.Line2D([0], [0], color=BLUE, lw=6, alpha=0.5, label="NREM"),
        plt.Line2D([0], [0], color=PURPLE, lw=6, alpha=0.5, label="REM"),
    ]
    ax_c.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.22))
    ax_c.text(-0.02, 1.08, "C", transform=ax_c.transAxes, weight="bold", fontsize=12, color=INK)

    fig.suptitle(
        f"Sleep-EDF Expanded · record {meta.safe_record_id} · descriptive single-label transport example",
        x=0.01,
        ha="left",
        weight="bold",
        fontsize=12.5,
        color=INK,
        y=0.995,
    )
    stamp_schematic(
        fig,
        "DESCRIPTIVE SINGLE-LABEL TRANSPORT — NOT HNPH v0.4 QUALIFIED EVIDENCE · NO PERFORMANCE RESULT",
    )
    provenance_footer(
        fig,
        f"source: Sleep-EDF Expanded {meta.safe_record_id} · fs={meta.sampling_rate_hz:g} Hz · "
        f"psg_sha256={meta.psg_sha256[:12]}… · issue_time={meta.issue_time_s:g}s · "
        "not independently source-qualified under HNPH v0.4",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return fig


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    fig = render_sleep_edf_descriptive_figure(args.source_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), dpi=170, bbox_inches="tight", facecolor=WHITE)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"wrote {args.out.with_suffix('.png')}")
    print(f"wrote {args.out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
