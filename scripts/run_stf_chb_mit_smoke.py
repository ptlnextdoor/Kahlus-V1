#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.stf.chb_mit import run_chb_mit_public_smoke, write_chb_mit_public_smoke  # noqa: E402
from neurotwin.stf.public_data import CHB_MIT_DATASET_ID  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local CHB-MIT EDF forecasting smoke.")
    parser.add_argument("--dataset", choices=(CHB_MIT_DATASET_ID,), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=6)
    parser.add_argument("--max-samples-per-record", type=int, default=512)
    parser.add_argument("--max-channels", type=int, default=8)
    args = parser.parse_args()

    payload = run_chb_mit_public_smoke(
        args.data_root,
        seed=args.seed,
        max_records=args.max_records,
        max_samples_per_record=args.max_samples_per_record,
        max_channels=args.max_channels,
    )
    paths = write_chb_mit_public_smoke(args.out_dir, payload)
    print(f"branch=stf dataset={payload['dataset']}")
    print(f"public_smoke_passed={payload['public_smoke_passed']}")
    print(f"failure_reasons={payload['failure_reasons']}")
    print("a100_jobs_launched=false")
    print(f"metrics={paths['metrics']}")
    print(f"baseline_table={paths['baseline_table']}")
    return 0 if payload["public_smoke_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
