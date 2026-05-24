#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create simple markdown tables from NeuroTwin run metrics.")
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()
    print("| run | metric | value |")
    print("| --- | --- | --- |")
    for run_dir in args.run_dirs:
        metrics_path = Path(run_dir) / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
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


if __name__ == "__main__":
    raise SystemExit(main())
