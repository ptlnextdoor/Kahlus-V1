#!/usr/bin/env python3
"""Render all Kahlus IEEE figures (house style) and sync to the IEEE package."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
FIGURES = Path(__file__).resolve().parent / "figures"
IEEE_OUT = Path("/Users/aayu/Downloads/Kahlus_v1_IEEE_Full_Paper_Source_Package_unpacked/figures")

SCRIPTS = [
    "Figure1_core_task.py",
    "Figure2_nfc_schematic.py",
    "Figure3_gate_protocol.py",
    "Figure4_mse_bar.py",
    "Figure5_amrith_overlap.py",
]


def run_script(name: str) -> None:
    path = SRC / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    module.main()


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for script in SCRIPTS:
        print(f"rendering {script}...")
        run_script(script)

    if IEEE_OUT.parent.exists():
        IEEE_OUT.mkdir(parents=True, exist_ok=True)
        for path in FIGURES.glob("fig*.*"):
            shutil.copy2(path, IEEE_OUT / path.name)
        print(f"synced figures to {IEEE_OUT}")
    print(f"done -> {FIGURES}")


if __name__ == "__main__":
    main()
