#!/usr/bin/env python3
"""Build real BNCI2014_001 tensors and a horizon sweep for the ridge sanity study.

Mentor context (Amrith): analyze the *existing* benchmark and explain why the
ridge baseline performs well, instead of adding new datasets.

This script:
  1. Loads cached BNCI2014_001 MATLAB files directly with scipy (no MOABB/MNE
     download needed) from the local MOABB cache.
  2. Reproduces the repo's EEG future-forecasting windowing
     (`_future_windows`: X = signal[start:start+L], Y = signal[start+h:start+h+L]),
     including the historical `forecast_horizon=1` overlapping-target case that
     the repo audit flagged (target shares L-1 of L samples with input).
  3. Writes an `.npz` consumable by `plot_ridge_eeg_diagnostics.py --npz` for the
     benchmark-derived figures, with real channel names and sampling rate.
  4. Runs a forecast-horizon sweep with matched persistence and ridge baselines
     (repo `NumpyRidgeBaseline`, channel-to-channel) so the "why is ridge strong"
     question is answered quantitatively: skill collapses as the target stops
     overlapping the input.

Scope: sanity analysis of an existing public motor-imagery EEG benchmark. No
clinical, diagnostic, or brain-foundation-model claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[2]


def _portable_path(path: Path) -> str:
    """Render a path relative to the repo root or home dir, never a bare local absolute path.

    Committed provenance (ridge_bnci_summary.json) should stay meaningful across
    machines rather than embedding one contributor's home directory.
    """
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        pass
    try:
        return str(Path("~") / resolved.relative_to(Path.home()))
    except ValueError:
        return str(resolved)
sys.path.insert(0, str(ROOT / "src"))


class NumpyRidgeBaseline:
    """Inlined copy of neurotwin.models.baselines.NumpyRidgeBaseline.

    Reproduced verbatim so this analysis script has no torch dependency while
    matching the exact channel-to-channel ridge used by the repo baseline suite
    (max-abs scaling, closed-form solve, alpha on scaled features).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.coef_ = None
        self.x_mean_ = self.x_scale_ = self.y_mean_ = self.y_scale_ = None

    @staticmethod
    def _safe_scale(scale: np.ndarray) -> np.ndarray:
        scale = np.asarray(scale, dtype=np.float64)
        scale[scale < 1e-8] = 1.0
        return scale

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyRidgeBaseline":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.x_mean_ = x.mean(axis=0, keepdims=True)
        self.y_mean_ = y.mean(axis=0, keepdims=True)
        self.x_scale_ = self._safe_scale(np.max(np.abs(x - self.x_mean_), axis=0, keepdims=True))
        self.y_scale_ = self._safe_scale(np.max(np.abs(y - self.y_mean_), axis=0, keepdims=True))
        xs = (x - self.x_mean_) / self.x_scale_
        ys = (y - self.y_mean_) / self.y_scale_
        xtx = np.einsum("ni,nj->ij", xs, xs, optimize=True)
        reg = self.alpha * np.eye(xtx.shape[0], dtype=np.float64)
        rhs = np.einsum("ni,nj->ij", xs, ys, optimize=True)
        try:
            self.coef_ = np.linalg.solve(xtx + reg, rhs)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(xtx + reg, rhs, rcond=None)[0]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        pred = np.einsum("ni,ij->nj", (x - self.x_mean_) / self.x_scale_, self.coef_, optimize=True)
        return pred * self.y_scale_ + self.y_mean_

# BNCI2014_001 (BCI Competition IV-2a) fixed metadata.
# 22 EEG channels (10-20), then 3 EOG channels. fs = 250 Hz.
EEG_CHANNEL_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
    "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]
N_EEG = len(EEG_CHANNEL_NAMES)
DEFAULT_CACHE = Path.home() / ".cache/kahlus/moabb/MNE-bnci-data/~bci/database/001-2014"


def _iter_runs(mat_path: Path):
    """Yield (X[time, 25], fs) for each run struct in a BNCI .mat file."""
    m = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    for run in np.atleast_1d(m["data"]):
        X = np.asarray(getattr(run, "X", None), dtype=np.float64)
        if X.ndim != 2 or X.shape[1] < N_EEG:
            continue
        fs = float(getattr(run, "fs", 250.0))
        yield X, fs


def load_subject_eeg(cache_dir: Path, subject: int) -> tuple[np.ndarray, float]:
    """Concatenate EEG-only signal for a subject's training file A0{n}T.mat.

    Returns signal [time, 22] in microvolts-scale as stored, and fs.
    """
    path = cache_dir / f"A0{subject}T.mat"
    if not path.exists():
        raise FileNotFoundError(f"missing cached BNCI file: {path}")
    chunks = []
    fs_seen = None
    for X, fs in _iter_runs(path):
        fs_seen = fs
        chunks.append(X[:, :N_EEG])
    if not chunks:
        raise RuntimeError(f"no usable runs in {path}")
    return np.concatenate(chunks, axis=0), float(fs_seen or 250.0)


def zscore_train(sig_train: np.ndarray, sig: np.ndarray) -> np.ndarray:
    mu = sig_train.mean(axis=0, keepdims=True)
    sd = sig_train.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (sig - mu) / sd


def future_windows(signal: np.ndarray, L: int, horizon: int, stride: int):
    """Match repo `_future_windows`: X=sig[s:s+L], Y=sig[s+h:s+h+L]."""
    xs, ys = [], []
    last = signal.shape[0] - L - horizon + 1
    for s in range(0, max(0, last), stride):
        xs.append(signal[s : s + L])
        ys.append(signal[s + horizon : s + horizon + L])
    if not xs:
        return (np.empty((0, L, signal.shape[1])), np.empty((0, L, signal.shape[1])))
    return np.stack(xs), np.stack(ys)


def _flatten_time(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(-1, a.shape[-1])


def ridge_metrics(x_tr, y_tr, x_te, y_te, alpha: float):
    model = NumpyRidgeBaseline(alpha=alpha)
    model.fit(_flatten_time(x_tr), _flatten_time(y_tr))
    pred = model.predict(_flatten_time(x_te)).reshape(y_te.shape)
    return _scores(y_te, pred), pred, model.coef_


def persistence_metrics(x_te, y_te):
    # Persistence for horizon>=1: predict the last observed window as the future.
    return _scores(y_te, x_te), x_te


def _scores(y_true, y_pred) -> dict:
    yt = np.asarray(y_true, float).reshape(-1)
    yp = np.asarray(y_pred, float).reshape(-1)
    mse = float(np.mean((yt - yp) ** 2))
    mae = float(np.mean(np.abs(yt - yp)))
    if yt.std() < 1e-12 or yp.std() < 1e-12:
        r = float("nan")
    else:
        r = float(np.corrcoef(yt, yp)[0, 1])
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2)) or 1.0
    return {"mse": mse, "mae": mae, "pearsonr": r, "r2": 1.0 - ss_res / ss_tot}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "ridge_bnci_real")
    ap.add_argument("--train-subjects", type=str, default="1,2,3,4,5,6")
    ap.add_argument("--test-subjects", type=str, default="7,8,9")
    ap.add_argument("--window-length", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=1e-2)
    ap.add_argument("--max-train-windows", type=int, default=4000)
    ap.add_argument("--max-test-windows", type=int, default=1200)
    ap.add_argument("--horizons", type=str, default="1,4,16,64,128,192")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_subj = [int(s) for s in args.train_subjects.split(",") if s.strip()]
    test_subj = [int(s) for s in args.test_subjects.split(",") if s.strip()]
    L = args.window_length

    # Load raw EEG per subject; z-score each subject by its own train file stats
    # (subject-held-out: no cross-subject leakage in normalization).
    rng = np.random.default_rng(0)
    sigs: dict[int, tuple[np.ndarray, float]] = {}
    for s in sorted(set(train_subj + test_subj)):
        sig, fs = load_subject_eeg(args.cache_dir, s)
        sig = zscore_train(sig, sig)
        sigs[s] = (sig, fs)
    fs = sigs[train_subj[0]][1]

    def build(subjects, horizon, cap):
        xs, ys = [], []
        for s in subjects:
            sig, _ = sigs[s]
            x, y = future_windows(sig, L, horizon, args.stride)
            if x.shape[0]:
                xs.append(x)
                ys.append(y)
        X = np.concatenate(xs, axis=0)
        Y = np.concatenate(ys, axis=0)
        if X.shape[0] > cap:
            idx = rng.choice(X.shape[0], size=cap, replace=False)
            idx.sort()
            X, Y = X[idx], Y[idx]
        return X, Y

    # ---- Horizon sweep: the core "why is ridge strong" evidence -------------
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    sweep = []
    for h in horizons:
        x_tr, y_tr = build(train_subj, h, args.max_train_windows)
        x_te, y_te = build(test_subj, h, args.max_test_windows)
        ridge_s, _, _ = ridge_metrics(x_tr, y_tr, x_te, y_te, args.alpha)
        pers_s, _ = persistence_metrics(x_te, y_te)
        overlap = max(0, L - h)
        sweep.append(
            {
                "horizon_samples": h,
                "horizon_ms": 1000.0 * h / fs,
                "input_target_overlap_samples": overlap,
                "overlap_fraction": overlap / L,
                "n_test_windows": int(x_te.shape[0]),
                "ridge": ridge_s,
                "persistence": pers_s,
            }
        )
        print(
            f"h={h:4d} overlap={overlap:3d}/{L}  "
            f"ridge r={ridge_s['pearsonr']:.3f} mse={ridge_s['mse']:.3f}  "
            f"persistence r={pers_s['pearsonr']:.3f} mse={pers_s['mse']:.3f}"
        )

    # ---- Figure tensors: use horizon=1 (the historical overlapping task) ----
    x_tr, y_tr = build(train_subj, 1, args.max_train_windows)
    x_te, y_te = build(test_subj, 1, args.max_test_windows)
    _, y_pred, _ = ridge_metrics(x_tr, y_tr, x_te, y_te, args.alpha)

    npz_path = args.out_dir / "ridge_bnci_tensors.npz"
    np.savez_compressed(
        npz_path,
        x_train=x_tr.astype(np.float32),
        y_train=y_tr.astype(np.float32),
        x_test=x_te.astype(np.float32),
        y_test=y_te.astype(np.float32),
        y_pred_test=y_pred.astype(np.float32),
        sfreq=np.float64(fs),
        channel_names=np.array(EEG_CHANNEL_NAMES),
        dataset=np.array("BNCI2014_001 (BCI IV-2a), MOABB local cache"),
        task_id=np.array("future_state_forecasting_v1_overlap horizon=1"),
        split_manifest=np.array(f"subject_held_out train={train_subj} test={test_subj}"),
        run_label=np.array("ridge sanity study, real cached data"),
    )

    summary = {
        "dataset": "BNCI2014_001 (BCI Competition IV-2a) via MOABB local cache",
        "source_dir": _portable_path(args.cache_dir),
        "sampling_rate_hz": fs,
        "n_eeg_channels": N_EEG,
        "channel_names": EEG_CHANNEL_NAMES,
        "window_length": L,
        "stride": args.stride,
        "ridge_alpha": args.alpha,
        "train_subjects": train_subj,
        "test_subjects": test_subj,
        "normalization": "per-subject z-score (subject-held-out; no cross-subject stats)",
        "horizon_sweep": sweep,
        "figure_npz": _portable_path(npz_path),
        "interpretation": (
            "At horizon=1 the target window overlaps the input window in L-1 of L "
            "samples, so a channel-to-channel linear map (ridge) trivially scores "
            "high. As horizon grows and input/target overlap goes to zero, both "
            "persistence and ridge skill collapse toward the noise floor. This shows "
            "the headline ridge number reflects short-horizon waveform continuity, "
            "not a rich neural-state model."
        ),
        "scope": "Sanity analysis of an existing public benchmark. No clinical claim.",
    }
    (args.out_dir / "ridge_bnci_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {npz_path}")
    print(f"wrote {args.out_dir / 'ridge_bnci_summary.json'}")


if __name__ == "__main__":
    main()
