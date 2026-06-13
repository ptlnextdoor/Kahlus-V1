"""Offline geomagnetic context loader.

OFFLINE ONLY. This module never performs network access. It optionally parses a local JSON
file that the caller has already obtained; otherwise it returns a clear ``not_fetched``
status. This keeps Stage 0 dependency-free and side-effect-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def fetch_geomagnetic(local_path: str | Path | None = None) -> dict[str, Any]:
    """Load geomagnetic records from a local file, or report ``not_fetched``.

    Expected local JSON shape (when provided): ``{"records": [...]}`` or a bare list.
    """

    if local_path is None:
        return {
            "status": "not_fetched",
            "reason": "offline_only_no_local_source",
            "network_access": False,
            "records": [],
        }
    path = Path(local_path)
    if not path.exists():
        return {
            "status": "not_fetched",
            "reason": f"local_source_missing:{path}",
            "network_access": False,
            "records": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {
            "status": "error",
            "reason": f"local_source_unreadable:{exc}",
            "network_access": False,
            "records": [],
        }
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        records = []
    return {
        "status": "loaded",
        "source": str(path),
        "network_access": False,
        "records": records,
    }
