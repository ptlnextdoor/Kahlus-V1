from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Raised when an experiment config cannot be loaded or validated."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML config {config_path}: {exc}") from exc
    if payload is None:
        raise ConfigError(f"Config is empty: {config_path}")
    if not isinstance(payload, dict):
        raise ConfigError(f"Config must be a mapping: {config_path}")
    return payload


def require_config_keys(config: dict[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ConfigError(f"Config is missing required keys: {', '.join(missing)}")
