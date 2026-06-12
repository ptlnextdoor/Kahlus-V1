#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize one Pair-Operator train config per ablation variant.")
    parser.add_argument("--template", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--include", nargs="*", default=None, help="Optional allowlist of ablation variant names.")
    args = parser.parse_args()

    template = Path(args.template)
    prepared_root = Path(args.prepared_root).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    if not template.exists():
        raise SystemExit(f"template does not exist: {template}")
    if not prepared_root.is_absolute():
        raise SystemExit(f"prepared-root must be absolute: {prepared_root}")
    if not out_dir.is_absolute():
        raise SystemExit(f"out-dir must be absolute: {out_dir}")

    event_manifest = prepared_root / "event_manifest.json"
    split_manifest = prepared_root / "split_manifest.json"
    if not event_manifest.exists():
        raise SystemExit(f"event_manifest does not exist: {event_manifest}")
    if not split_manifest.exists():
        raise SystemExit(f"split_manifest does not exist: {split_manifest}")

    payload = yaml.safe_load(template.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit("template must be a YAML mapping")
    variants = payload.get("ablation_variants")
    if not isinstance(variants, list) or not variants:
        raise SystemExit("template must define non-empty ablation_variants")
    include = set(args.include or [])

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for variant in variants:
        if not isinstance(variant, dict):
            raise SystemExit("ablation variant must be a mapping")
        name = str(variant.get("name", "")).strip()
        if not name or not all(char.isalnum() or char in "._-" for char in name):
            raise SystemExit(f"invalid ablation variant name: {name!r}")
        if include and name not in include:
            continue
        rendered = deepcopy(payload)
        rendered.pop("ablation_variants", None)
        rendered["experiment"] = name
        rendered["run_id"] = name
        if args.seed is not None:
            rendered["seed"] = int(args.seed)
            rendered["experiment"] = f"{name}_seed{args.seed}"
            rendered["run_id"] = f"{name}_seed{args.seed}"
        if args.steps is not None:
            rendered["steps"] = int(args.steps)
        data = rendered.get("data") if isinstance(rendered.get("data"), dict) else {}
        rendered["data"] = data
        data["event_manifest"] = str(event_manifest)
        data["split_manifest"] = str(split_manifest)
        rendered["ablation"] = name
        rendered["model"] = _merge_dicts(rendered.get("model"), variant.get("model"))
        text = yaml.safe_dump(rendered, sort_keys=False)
        if "/path/to/" in text:
            raise SystemExit(f"materialized config for {name} still contains placeholder path")
        path = out_dir / f"{name}.materialized.yaml"
        if args.seed is not None:
            path = out_dir / f"{name}.seed{args.seed}.materialized.yaml"
        path.write_text(text, encoding="utf-8")
        written.append(path)

    if not written:
        raise SystemExit("no ablation configs were written; check --include names")
    for path in written:
        print(path)
    return 0


def _merge_dicts(base: Any, override: Any) -> dict[str, Any]:
    merged = dict(base) if isinstance(base, dict) else {}
    if isinstance(override, dict):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = _merge_dicts(merged[key], value)
            else:
                merged[key] = value
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
