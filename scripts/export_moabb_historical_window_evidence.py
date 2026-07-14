#!/usr/bin/env python3
"""Export real MOABB windows that explain the historical shifted-window task.

This is a local diagnostic only. It writes a small NPZ containing real public
EEG snippets, ridge and persistence predictions, and a provenance manifest for
the existing plotting script. It does not train a replacement Kahlus model and
does not claim to reproduce the unreleased historical checkpoint.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.adapters.moabb import (  # noqa: E402
    load_balanced_moabb_subject_trials,
    trials_to_event_batches,
    trials_to_recordings,
)
from neurotwin.data.split_manifest import build_split_manifest  # noqa: E402
from neurotwin.eeg_v1.window_evidence import (  # noqa: E402
    WindowEvidenceConfig,
    build_historical_window_evidence,
    write_historical_window_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="BNCI2014_001")
    parser.add_argument("--paradigm", default="LeftRightImagery")
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--max-trials", type=int, default=60, help="Balanced cap across selected subjects; 0 uses all trials")
    parser.add_argument("--context-samples", type=int, default=127)
    parser.add_argument("--forecast-horizon-samples", type=int, default=1)
    parser.add_argument("--stride-samples", type=int, default=127)
    parser.add_argument("--max-train-windows", type=int, default=512)
    parser.add_argument("--max-test-windows", type=int, default=3)
    parser.add_argument("--mne-data", type=Path, help="Optional local MNE cache root; never deleted by this command")
    parser.add_argument("--bnci-data-path", type=Path, help="Optional local BNCI cache root; never deleted by this command")
    args = parser.parse_args()
    if args.mne_data is not None:
        os.environ["MNE_DATA"] = str(args.mne_data)
    if args.bnci_data_path is not None:
        os.environ["MNE_DATASETS_BNCI_PATH"] = str(args.bnci_data_path)
    if args.max_trials < 0:
        parser.error("--max-trials must be non-negative")

    trials = load_balanced_moabb_subject_trials(
        args.dataset,
        subjects=tuple(args.subjects),
        paradigm=args.paradigm,
        max_trials=args.max_trials or None,
    )
    records = trials_to_recordings(trials, dataset_id=args.dataset)
    batches = trials_to_event_batches(trials, dataset_id=args.dataset)
    split = build_split_manifest(records, policy="subject", seed=0)
    export = build_historical_window_evidence(
        batches,
        split,
        config=WindowEvidenceConfig(
            context_samples=args.context_samples,
            forecast_horizon_samples=args.forecast_horizon_samples,
            stride_samples=args.stride_samples,
            max_train_windows=args.max_train_windows,
            max_test_windows=args.max_test_windows,
        ),
    )
    paths = write_historical_window_evidence(export, args.out_dir)
    print(f"window_evidence_npz={paths['npz']}")
    print(f"window_evidence_manifest={paths['manifest']}")
    print(f"sampling_rate_hz={export.manifest['sampling_rate_hz']}")
    print(f"signal_unit={export.manifest['signal_unit']}")
    print(f"shared_samples_per_example={export.manifest['shared_samples_per_example']}")
    print(f"shared_sample_fraction={export.manifest['shared_sample_fraction']}")
    print(f"model_prediction_status={export.manifest['model_prediction_status']}")
    print("claim_eligible=false")
    print("a100_jobs_launched=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
