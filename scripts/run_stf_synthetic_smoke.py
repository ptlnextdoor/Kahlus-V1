#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.stf import STF_CLAIM_SCOPE, run_stf_synthetic_smoke, write_stf_smoke_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Kahlus-STF synthetic smoke benchmark.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = run_stf_synthetic_smoke(seed=args.seed)
    paths = write_stf_smoke_artifacts(args.out_dir, payload)
    summary = payload["summary"]
    print(f"branch=stf dataset={payload['dataset']} out_dir={Path(args.out_dir).resolve()}")
    print(f"claim_scope={STF_CLAIM_SCOPE}")
    print(f"scientific_claim_allowed={payload['gate']['scientific_claim_allowed']}")
    print(f"failure_reasons={payload['gate']['failure_reasons']}")
    print(f"a100_jobs_launched={summary['a100_jobs_launched']}")
    print(f"evidence_gate={paths['evidence_gate']}")
    print(f"baseline_table={paths['baseline_table']}")
    return 0 if payload["gate"]["scientific_claim_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
