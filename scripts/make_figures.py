#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neurotwin.eval.paper_gate import effective_scientific_claim_allowed_for_run, load_run_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Create lightweight text figure specs from NeuroTwin run metrics.")
    parser.add_argument("run_dir")
    parser.add_argument("--allow-synthetic", action="store_true", help="Allow synthetic-only plumbing runs in generated figure specs.")
    args = parser.parse_args()
    metrics_path = Path(args.run_dir) / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"No metrics.json found in {args.run_dir}")
    run_path = Path(args.run_dir)
    summary = load_run_summary(run_path)
    if isinstance(summary, dict) and summary.get("synthetic_only") and not args.allow_synthetic:
        raise SystemExit(f"{args.run_dir} is synthetic-only; rerun with --allow-synthetic for plumbing figures")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    out = run_path / "figure_summary.txt"
    lines = []
    baseline_suite = metrics.get("baseline_suite") if isinstance(metrics, dict) else None
    if isinstance(baseline_suite, dict):
        aggregate = baseline_suite.get("aggregate", {})
        for row in aggregate.get("aggregate_rank", []):
            lines.append(f"baseline_rank {row['model_id']} mean_rank={row['mean_rank']}")
    if not lines and isinstance(metrics, dict):
        lines = [f"{key}: {value}" for key, value in sorted(metrics.items())]
    if isinstance(summary, dict) and summary.get("real_data_smoke"):
        lines.insert(0, "claim_status: real_data_smoke")
    elif isinstance(summary, dict):
        lines.insert(
            0,
            f"scientific_claim_allowed: {effective_scientific_claim_allowed_for_run(run_path, summary)}",
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
