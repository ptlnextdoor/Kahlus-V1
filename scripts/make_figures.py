#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    from _bootstrap import ensure_src_import_path

    ensure_src_import_path(__file__)
    from neurotwin.eval.paper_gate import load_run_summary, paper_mode_gate_allows_claim_for_run

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
        claim_allowed = bool(summary.get("scientific_claim_allowed"))
        gate_allows = paper_mode_gate_allows_claim_for_run(run_path)
        lines.insert(
            0,
            f"paper_mode_gate_allows_claim: {gate_allows}",
        )
        lines.insert(0, f"scientific_claim_allowed: {claim_allowed}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
