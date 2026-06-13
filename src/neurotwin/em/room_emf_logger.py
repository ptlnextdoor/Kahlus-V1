"""Passive room/device EMF logger (schema + append-to-log).

Records environment readings that the caller supplies — it does NOT access any hardware,
sensor, or device. Storage is a JSONL log via the shared repro helper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neurotwin.em.em_context_schema import RoomEnvironmentLog
from neurotwin.repro import append_jsonl


class RoomEMFLogger:
    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)

    def log(self, reading: RoomEnvironmentLog) -> Path:
        return append_jsonl(self.log_path, reading.to_dict())

    def read_all(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
