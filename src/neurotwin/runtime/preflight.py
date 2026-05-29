from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any

import torch
import yaml

from neurotwin.config import load_config
from neurotwin.eval.audit import audit_prepared_eval_inputs


@dataclass(frozen=True)
class ClusterPreflightReport:
    passed: bool
    violations: tuple[str, ...]
    warnings: tuple[str, ...]
    config: str
    run_root: str
    event_manifest: str | None
    split_manifest: str | None
    cuda_available: bool
    cuda_device_count: int
    window_count: int | None
    window_counts_by_split: dict[str, int] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClusterMaterializeConfigReport:
    passed: bool
    violations: tuple[str, ...]
    warnings: tuple[str, ...]
    template: str
    prepared_root: str
    out: str
    event_manifest: str
    split_manifest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_cluster_preflight(
    config_path: str | Path,
    run_root: str | Path,
    require_cuda: bool = False,
    require_prepared_windows: bool = False,
    expect_window_count: int | None = None,
    expect_split_windows: dict[str, int] | None = None,
) -> ClusterPreflightReport:
    """Validate cluster launch inputs before an expensive SLURM allocation runs."""

    config_file = Path(config_path)
    config_text = config_file.read_text(encoding="utf-8") if config_file.exists() else ""
    config = load_config(config_file)
    data_config = config.get("data") if isinstance(config.get("data"), dict) else {}
    event_manifest = _config_value(config, data_config, "event_manifest")
    split_manifest = _config_value(config, data_config, "split_manifest")
    run_root_path = Path(run_root).expanduser()
    repo_runs = Path.cwd().resolve() / "runs"
    violations: list[str] = []
    warnings: list[str] = []
    window_count: int | None = None
    window_counts_by_split: dict[str, int] | None = None

    if "/path/to/" in config_text:
        violations.append("config contains placeholder path '/path/to/'")

    event_path = _validate_manifest_path("event_manifest", event_manifest, violations)
    split_path = _validate_manifest_path("split_manifest", split_manifest, violations)

    if not str(run_root).strip():
        violations.append("run_root is required")
    elif not run_root_path.is_absolute():
        violations.append(f"run_root must be absolute: {run_root}")
    elif not run_root_path.exists():
        violations.append(f"run_root does not exist: {run_root_path}")

    if os.environ.get("SLURM_JOB_ID") and not os.environ.get("NEUROTWIN_DATA"):
        violations.append("NEUROTWIN_DATA must be set under SLURM")
    if os.environ.get("SLURM_JOB_ID") and run_root_path.is_absolute():
        resolved_run_root = run_root_path.resolve(strict=False)
        if resolved_run_root == repo_runs or repo_runs in resolved_run_root.parents:
            violations.append("RUN_ROOT must not point under repo-local runs/ under SLURM")

    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()
    if require_cuda and (not cuda_available or cuda_device_count <= 0):
        violations.append("CUDA is required but no CUDA device is available")

    needs_window_audit = require_prepared_windows or expect_window_count is not None or bool(expect_split_windows)
    if needs_window_audit:
        if event_path is not None and split_path is not None and event_path.exists() and split_path.exists():
            audit = audit_prepared_eval_inputs(
                event_path,
                split_path,
                window_length=int(config.get("window_size", config.get("window_length", 8))),
                stride=int(config.get("stride", config.get("window_size", config.get("window_length", 8)))),
                require_windows=True,
            )
            window_count = audit.window_count
            window_counts_by_split = audit.window_counts_by_split
            violations.extend(audit.violations)
            warnings.extend(audit.warnings)
            if expect_window_count is not None and window_count != expect_window_count:
                violations.append(f"expected window_count={expect_window_count}, got {window_count}")
            if expect_split_windows:
                expected = {split_name: int(count) for split_name, count in expect_split_windows.items()}
                actual = {
                    split_name: int(window_counts_by_split.get(split_name, 0))
                    for split_name in ("train", "val", "test")
                    if split_name in expected
                }
                if actual != expected:
                    expected_text = ",".join(
                        f"{split_name}:{expected[split_name]}" for split_name in ("train", "val", "test") if split_name in expected
                    )
                    actual_text = ",".join(
                        f"{split_name}:{actual.get(split_name, 0)}" for split_name in ("train", "val", "test") if split_name in expected
                    )
                    violations.append(f"expected split windows={expected_text}, got {actual_text}")
        else:
            violations.append("prepared window audit requires existing event and split manifests")

    return ClusterPreflightReport(
        passed=not violations,
        violations=tuple(violations),
        warnings=tuple(warnings),
        config=str(config_file),
        run_root=str(run_root_path),
        event_manifest=str(event_path) if event_path is not None else None,
        split_manifest=str(split_path) if split_path is not None else None,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        window_count=window_count,
        window_counts_by_split=window_counts_by_split,
    )


def materialize_cluster_config(
    template_path: str | Path,
    prepared_root: str | Path,
    out_path: str | Path,
    allow_tracked_output: bool = False,
) -> ClusterMaterializeConfigReport:
    """Write a cluster config with absolute prepared-manifest paths."""

    template = Path(template_path)
    prepared = Path(prepared_root).expanduser()
    out = Path(out_path).expanduser()
    violations: list[str] = []
    warnings: list[str] = []

    if not template.exists():
        violations.append(f"template config does not exist: {template}")
    if not prepared.is_absolute():
        violations.append(f"prepared_root must be absolute: {prepared_root}")
    event_manifest = prepared / "event_manifest.json"
    split_manifest = prepared / "split_manifest.json"
    if prepared.is_absolute() and not event_manifest.exists():
        violations.append(f"event_manifest does not exist: {event_manifest}")
    if prepared.is_absolute() and not split_manifest.exists():
        violations.append(f"split_manifest does not exist: {split_manifest}")

    resolved_out = (Path.cwd() / out).resolve(strict=False) if not out.is_absolute() else out.resolve(strict=False)
    tracked_configs = (Path.cwd() / "configs").resolve(strict=False)
    if not allow_tracked_output and (resolved_out == tracked_configs or tracked_configs in resolved_out.parents):
        violations.append("refusing to write generated cluster config under tracked configs/; use outputs/configs/")

    payload: dict[str, Any] = {}
    if template.exists():
        payload = load_config(template)
    if violations:
        return ClusterMaterializeConfigReport(
            passed=False,
            violations=tuple(violations),
            warnings=tuple(warnings),
            template=str(template),
            prepared_root=str(prepared),
            out=str(out),
            event_manifest=str(event_manifest),
            split_manifest=str(split_manifest),
        )

    data_config = payload.get("data")
    if not isinstance(data_config, dict):
        data_config = {}
        payload["data"] = data_config
    data_config["event_manifest"] = str(event_manifest)
    data_config["split_manifest"] = str(split_manifest)
    if "event_manifest" in payload:
        payload["event_manifest"] = str(event_manifest)
    if "split_manifest" in payload:
        payload["split_manifest"] = str(split_manifest)

    rendered = yaml.safe_dump(payload, sort_keys=False)
    if "/path/to/" in rendered:
        violations.append("materialized config still contains placeholder path '/path/to/'")
    if violations:
        return ClusterMaterializeConfigReport(
            passed=False,
            violations=tuple(violations),
            warnings=tuple(warnings),
            template=str(template),
            prepared_root=str(prepared),
            out=str(out),
            event_manifest=str(event_manifest),
            split_manifest=str(split_manifest),
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")
    return ClusterMaterializeConfigReport(
        passed=True,
        violations=(),
        warnings=tuple(warnings),
        template=str(template),
        prepared_root=str(prepared),
        out=str(out),
        event_manifest=str(event_manifest),
        split_manifest=str(split_manifest),
    )


def format_cluster_materialize_config(report: ClusterMaterializeConfigReport) -> str:
    lines = [
        f"materialize_config_passed={report.passed}",
        f"template={report.template}",
        f"prepared_root={report.prepared_root}",
        f"out={report.out}",
        f"event_manifest={report.event_manifest}",
        f"split_manifest={report.split_manifest}",
    ]
    for violation in report.violations:
        lines.append(f"violation={violation}")
    for warning in report.warnings:
        lines.append(f"warning={warning}")
    return "\n".join(lines)


def format_cluster_preflight(report: ClusterPreflightReport) -> str:
    counts = report.window_counts_by_split or {"train": 0, "val": 0, "test": 0}
    lines = [
        f"preflight_passed={report.passed}",
        f"config={report.config}",
        f"cuda_available={report.cuda_available}",
        f"cuda_device_count={report.cuda_device_count}",
        f"event_manifest={report.event_manifest}",
        f"split_manifest={report.split_manifest}",
        f"run_root={report.run_root}",
        f"window_count={report.window_count if report.window_count is not None else 'not_checked'}",
        "window_counts_by_split="
        + ",".join(f"{split_name}:{counts.get(split_name, 0)}" for split_name in ("train", "val", "test")),
    ]
    for violation in report.violations:
        lines.append(f"violation={violation}")
    for warning in report.warnings:
        lines.append(f"warning={warning}")
    return "\n".join(lines)


def _config_value(config: dict[str, Any], data_config: dict[str, Any], key: str) -> Any:
    return config.get(key) or data_config.get(key)


def _validate_manifest_path(name: str, value: Any, violations: list[str]) -> Path | None:
    if value is None or not str(value).strip():
        violations.append(f"{name} is required")
        return None
    path = Path(str(value)).expanduser()
    if "/path/to/" in str(value):
        violations.append(f"{name} contains placeholder path: {value}")
    if not path.is_absolute():
        violations.append(f"{name} must be absolute: {value}")
    elif not path.exists():
        violations.append(f"{name} does not exist: {path}")
    return path
