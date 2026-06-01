from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
import sys
from pathlib import Path

import torch

from neurotwin.repro import create_run_dir


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[DoctorCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.status in {"ok", "warn"} for check in self.checks)


OPTIONAL_MODULES = ("moabb", "mne", "pandas")


def run_doctor(dataset_roots: tuple[str, ...] = ("NEUROTWIN_DATA", "MOABB_DATA", "BIDS_ROOT")) -> DoctorReport:
    checks: list[DoctorCheck] = [
        DoctorCheck("python", "ok", sys.version.split()[0]),
        DoctorCheck("torch", "ok", torch.__version__),
        DoctorCheck("cuda_available", "ok" if torch.cuda.is_available() else "warn", str(torch.cuda.is_available())),
        DoctorCheck("cuda_device_count", "ok" if torch.cuda.device_count() > 0 else "warn", str(torch.cuda.device_count())),
    ]
    for module in OPTIONAL_MODULES:
        status = "ok" if importlib.util.find_spec(module) else "warn"
        detail = "present" if status == "ok" else "missing optional dependency"
        checks.append(DoctorCheck(module, status, detail))
    for env_var in dataset_roots:
        value = os.environ.get(env_var)
        if not value:
            checks.append(DoctorCheck(env_var, "warn", "not set"))
            continue
        path = Path(value)
        checks.append(DoctorCheck(env_var, "ok" if path.exists() else "warn", str(path)))
    runs_status, runs_detail = _check_runs_writable()
    checks.append(DoctorCheck("runs_writable", runs_status, runs_detail))
    return DoctorReport(tuple(checks))


def format_doctor_report(report: DoctorReport) -> str:
    lines = ["# NeuroTwin Doctor", ""]
    for check in report.checks:
        lines.append(f"{check.name}: {check.status} ({check.detail})")
    return "\n".join(lines)


def _is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        return False
    return True


def _check_runs_writable() -> tuple[str, str]:
    try:
        run_dir = create_run_dir("runs", run_id="doctor-check")
    except OSError:
        return "error", "runs/ unavailable and fallback path is not writable"
    detail = "runs/"
    if str(run_dir.parent) != "runs":
        detail = f"writable fallback: {run_dir.parent}"
        return "warn", detail
    return "ok", detail
