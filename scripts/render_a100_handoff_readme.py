#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from string import Template


def render_handoff_readme(
    template_path: Path,
    output_path: Path,
    *,
    full_sha: str,
    short_sha: str,
    runner_name: str,
    persistent_root_example: str | None = None,
) -> Path:
    runner_tarball = f"{runner_name}.tar.gz"
    rendered = Template(template_path.read_text(encoding="utf-8")).safe_substitute(
        FULL_SHA=full_sha,
        SHORT_SHA=short_sha,
        RUNNER_NAME=runner_name,
        RUNNER_TARBALL=runner_tarball,
        PERSISTENT_ROOT_EXAMPLE=persistent_root_example or f"/raid/scratch/$USER/neurotwin-{short_sha}",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the NeuroTwin A100 handoff README.")
    parser.add_argument("--template", type=Path, default=Path("deploy/a100/README_HANDOFF.md.in"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-sha", required=True)
    parser.add_argument("--short-sha", required=True)
    parser.add_argument("--runner-name", required=True)
    parser.add_argument("--persistent-root-example")
    args = parser.parse_args()

    render_handoff_readme(
        args.template,
        args.output,
        full_sha=args.full_sha,
        short_sha=args.short_sha,
        runner_name=args.runner_name,
        persistent_root_example=args.persistent_root_example,
    )
    print(f"readme={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
