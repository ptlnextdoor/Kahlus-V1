#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.stf import CHB_MIT_DATASET_ID, CHB_MIT_FILE_BASE_URL, fetch_chb_mit_smoke_subset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a small CHB-MIT subset outside the repo.")
    parser.add_argument("--dataset", choices=(CHB_MIT_DATASET_ID,), required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--patients", type=int, default=2)
    parser.add_argument("--records-per-patient", type=int, default=2)
    parser.add_argument("--base-url", default=CHB_MIT_FILE_BASE_URL)
    args = parser.parse_args()

    manifest = fetch_chb_mit_smoke_subset(
        args.out_root,
        patients=args.patients,
        records_per_patient=args.records_per_patient,
        base_url=args.base_url,
    )
    print(f"branch=stf dataset={manifest['dataset_id']}")
    print(f"data_root={manifest['data_root']}")
    print(f"selected_records={len(manifest['selected_records'])}")
    print(f"selected_seizure_records={len(manifest['selected_seizure_records'])}")
    print("a100_jobs_launched=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
