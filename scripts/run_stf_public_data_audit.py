#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.stf.public_data import (  # noqa: E402
    CHB_MIT_DATASET_ID,
    audit_chb_mit_root,
    write_chb_mit_root_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a local public STF dataset root.")
    parser.add_argument("--dataset", choices=(CHB_MIT_DATASET_ID,), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    audit = audit_chb_mit_root(args.data_root)
    paths = write_chb_mit_root_audit(args.out_dir, audit)
    print(f"branch=stf dataset={audit.dataset_id}")
    print(f"public_data_audit_passed={audit.passed}")
    print(f"record_count={audit.record_count}")
    print(f"seizure_record_count={audit.seizure_record_count}")
    print(f"failure_reasons={list(audit.failure_reasons)}")
    print("a100_jobs_launched=false")
    print(f"audit={paths['audit']}")
    print(f"report={paths['report']}")
    return 0 if audit.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
