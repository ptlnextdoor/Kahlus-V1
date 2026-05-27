#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create simple markdown tables from NeuroTwin run metrics.")
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--allow-synthetic", action="store_true", help="Allow synthetic-only plumbing runs in generated tables.")
    args = parser.parse_args()
    print("| run | metric | value |")
    print("| --- | --- | --- |")
    for run_dir in args.run_dirs:
        metrics_path = Path(run_dir) / "metrics.json"
        if not metrics_path.exists():
            continue
        summary = _read_summary(Path(run_dir))
        if summary.get("synthetic_only") and not args.allow_synthetic:
            raise SystemExit(f"{run_dir} is synthetic-only; rerun with --allow-synthetic for plumbing tables")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        claim_status = "real_data_smoke" if summary.get("real_data_smoke") else "scientific" if summary.get("scientific_claim_allowed") else "plumbing"
        print(f"| {Path(run_dir).name} | claim_status | {claim_status} |")
        for key, value in _flatten_metrics(metrics):
            print(f"| {Path(run_dir).name} | {key} | {value} |")
    return 0


def _flatten_metrics(payload: object, prefix: str = "") -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = []
    if isinstance(payload, dict):
        for key, value in sorted(payload.items()):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_metrics(value, next_prefix))
    elif isinstance(payload, list):
        if all(isinstance(item, dict) and "model_id" in item for item in payload):
            for item in payload:
                label = str(item.get("model_id"))
                for key, value in sorted(item.items()):
                    if key == "model_id":
                        continue
                    rows.extend(_flatten_metrics(value, f"{prefix}.{label}.{key}"))
        elif not payload:
            rows.append((prefix, "[]"))
    elif isinstance(payload, (int, float, str, bool)) or payload is None:
        rows.append((prefix, payload))
    return rows


def _read_summary(run_dir: Path) -> dict[str, object]:
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
