# Copyright (c) Alibaba, Inc. and its affiliates.
"""JSON store for adaptive capability decision history.

The store persists transparent resolver output for user review and dashboard /
slash-command rendering. It stores JSON payloads rather than live Python objects
so schema evolution is additive and older records remain inspectable.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Mapping


class JsonCapabilityPlanStore:
    """Profile-scoped JSON store for capability resolutions and plans."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def add_record(
        self,
        *,
        environment: Mapping[str, Any] | None = None,
        requirements: list[Mapping[str, Any]] | None = None,
        resolutions: list[Mapping[str, Any]] | None = None,
        plan: Mapping[str, Any] | None = None,
        source: str = "runtime",
        record_id: str = "",
        phase: str = "",
        loop_id: str = "",
        mutation: Mapping[str, Any] | None = None,
        registry_version_before: int = 0,
        registry_version_after: int = 0,
        decision_delta: Mapping[str, Any] | None = None,
        observation_ids: list[str] | None = None,
        proposal: Mapping[str, Any] | None = None,
        policy_decision: Mapping[str, Any] | None = None,
        governance_results: list[Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a decision record and return the stored payload."""
        record = {
            "record_id": record_id or f"cap-{uuid.uuid4().hex}",
            "created_at": time.time(),
            "source": str(source or "runtime"),
            "environment": dict(environment or {}),
            "requirements": [dict(r) for r in (requirements or [])],
            "resolutions": [dict(r) for r in (resolutions or [])],
            "plan": dict(plan or {}),
        }
        if phase:
            record["phase"] = str(phase)
        if loop_id:
            record["loop_id"] = str(loop_id)
        if mutation:
            record["mutation"] = dict(mutation)
        if registry_version_before or registry_version_after:
            record["registry_version_before"] = int(registry_version_before)
            record["registry_version_after"] = int(registry_version_after)
        if decision_delta:
            record["decision_delta"] = dict(decision_delta)
        if observation_ids:
            record["observation_ids"] = [str(item) for item in observation_ids]
        if proposal:
            record["proposal"] = dict(proposal)
        if policy_decision:
            record["policy_decision"] = dict(policy_decision)
        if governance_results:
            record["governance_results"] = [dict(item) for item in governance_results]
        if metadata:
            record["metadata"] = dict(metadata)
        payload = self._load_payload()
        payload.setdefault("records", []).append(record)
        self._write_payload(payload)
        return record

    def list_records(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return newest records first."""
        payload = self._load_payload()
        records = [dict(r) for r in payload.get("records", []) if isinstance(r, Mapping)]
        records.sort(key=lambda r: float(r.get("created_at") or 0.0), reverse=True)
        if limit <= 0:
            return records
        return records[:limit]

    def latest(self) -> dict[str, Any] | None:
        """Return the newest record, if any."""
        records = self.list_records(limit=1)
        return records[0] if records else None

    def _load_payload(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"version": 1, "records": []}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, Mapping):
                records = data.get("records")
                if isinstance(records, list):
                    return {"version": int(data.get("version") or 1), "records": records}
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {"version": 1, "records": []}
        return {"version": 1, "records": []}

    def _write_payload(self, payload: Mapping[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
