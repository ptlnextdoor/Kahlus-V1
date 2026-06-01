from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandResult:
    output: str
    exit_code: int = 0
    error: str | None = None
