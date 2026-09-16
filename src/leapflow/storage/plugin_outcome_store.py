# Copyright (c) Alibaba, Inc. and its affiliates.
"""Profile-scoped audit store for adaptive plugin execution outcomes."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Mapping


class JsonPluginOutcomeStore:
    """Append-only outcome timeline used by lifecycle governance."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def add_outcome(
        self,
        *,
        plugin_id: str,
        tool_name: str,
        ok: bool,
        requirement_id: str = "",
        plan_id: str = "",
        duration_ms: float = 0.0,
        failure_class: str = "",
        side_effect_state: str = "none",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one execution outcome summary."""
        record = {
            "outcome_id": f"out-{uuid.uuid4().hex}",
            "created_at": time.time(),
            "plugin_id": str(plugin_id),
            "tool_name": str(tool_name),
            "ok": bool(ok),
            "requirement_id": str(requirement_id or ""),
            "plan_id": str(plan_id or ""),
            "duration_ms": float(duration_ms or 0.0),
            "failure_class": str(failure_class or ""),
            "side_effect_state": str(side_effect_state or "none"),
            "metadata": dict(metadata or {}),
        }
        payload = self._load_payload()
        payload.setdefault("outcomes", []).append(record)
        self._write_payload(payload)
        return record

    def list_outcomes(self, *, plugin_id: str = "", limit: int = 100) -> list[dict[str, Any]]:
        payload = self._load_payload()
        records = [dict(item) for item in payload.get("outcomes", []) if isinstance(item, Mapping)]
        if plugin_id:
            records = [record for record in records if record.get("plugin_id") == plugin_id]
        records.sort(key=lambda item: float(item.get("created_at") or 0.0), reverse=True)
        return records if limit <= 0 else records[:limit]

    def failure_streak(self, plugin_id: str) -> int:
        """Return consecutive latest failures for a plugin."""
        streak = 0
        for record in self.list_outcomes(plugin_id=plugin_id, limit=0):
            if record.get("ok") is True:
                break
            streak += 1
        return streak

    def _load_payload(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"version": 1, "outcomes": []}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, Mapping) and isinstance(data.get("outcomes"), list):
                return {"version": int(data.get("version") or 1), "outcomes": data["outcomes"]}
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {"version": 1, "outcomes": []}
        return {"version": 1, "outcomes": []}

    def _write_payload(self, payload: Mapping[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


__all__ = ["JsonPluginOutcomeStore"]
