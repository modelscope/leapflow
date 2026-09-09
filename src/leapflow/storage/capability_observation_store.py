"""Durable store for structured capability observations.

The store is profile-scoped and intentionally stores only structured metadata
needed for adaptive plugin governance. It must not persist user prompt text or
secret-bearing payloads.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

_OBSERVATION_FIELDS = frozenset(
    {
        "error_type",
        "original_tool_name",
        "normalized_tool_name",
        "resolution_status",
        "resolution_confidence",
        "resolution_reason",
        "suggestions",
        "available_tools",
        "recovery_hint",
        "failure_code",
        "capability",
        "tool_name",
        # Declarations the detector needs to rebuild a requirement from a
        # persisted observation. Dropping these silently changed behaviour rather
        # than failing: without ``max_risk_level`` the requirement inherited the
        # domain default of ``external`` -- the *most permissive* ceiling -- so a
        # capability declared ``read_only`` came back from the store able to
        # select mutating tools. Without ``origin`` a world-model intent was
        # indistinguishable from an environment probe.
        "origin",
        "max_risk_level",
        "requested_max_risk_level",
        "intent_id",
        "target_affordance",
        "expected_effect",
    }
)


def _stable_hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _safe_result(result: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in sorted(_OBSERVATION_FIELDS):
        if key not in result:
            continue
        value = result.get(key)
        if isinstance(value, (list, tuple)):
            safe[key] = [str(item) for item in value[:20]]
        elif isinstance(value, dict):
            safe[key] = {str(k): str(v) for k, v in value.items()}
        elif value is not None:
            safe[key] = str(value)
    return safe


class JsonCapabilityObservationStore:
    """Append/merge store for runtime capability-gap observations."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def add_observation(
        self,
        *,
        result: Mapping[str, Any],
        environment: Mapping[str, Any] | None = None,
        source: str = "runtime",
        session_id: str = "",
        turn_id: str = "",
        workspace_root: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Merge one observation by dedup key and return the stored record."""
        safe_result = _safe_result(result)
        now = time.time()
        env = dict(environment or {})
        dedup_key = self._dedup_key(safe_result, env, workspace_root)
        payload = self._load_payload()
        observations = payload.setdefault("observations", [])
        for record in observations:
            if not isinstance(record, dict) or record.get("dedup_key") != dedup_key:
                continue
            record["last_seen_at"] = now
            record["occurrence_count"] = int(record.get("occurrence_count") or 0) + 1
            record["result"] = safe_result
            record["environment"] = env
            # A recurrence reopens a retired record. Without this, marking an
            # observation resolved would silence that gap permanently: dedup would
            # keep merging into the closed record and ``unresolved()`` -- which
            # filters on ``status == "open"`` -- would never surface the
            # regression again.
            if str(record.get("status") or "open") != "open":
                record["status"] = "open"
                record["status_reason"] = f"reopened after recurrence at {now}"
            record["session_id"] = str(session_id or record.get("session_id") or "")
            record["turn_id"] = str(turn_id or record.get("turn_id") or "")
            record["workspace_root"] = str(workspace_root or record.get("workspace_root") or "")
            if metadata:
                record["metadata"] = {**dict(record.get("metadata") or {}), **dict(metadata)}
            self._write_payload(payload)
            return dict(record)

        record = {
            "observation_id": f"obs-{_stable_hash({'dedup': dedup_key, 'created_at': now})}",
            "dedup_key": dedup_key,
            "source": str(source or "runtime"),
            "first_seen_at": now,
            "last_seen_at": now,
            "occurrence_count": 1,
            "result": safe_result,
            "environment": env,
            "session_id": str(session_id or ""),
            "turn_id": str(turn_id or ""),
            "workspace_root": str(workspace_root or ""),
            "metadata": dict(metadata or {}),
        }
        observations.append(record)
        self._write_payload(payload)
        return dict(record)

    def list_observations(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Return observations newest first by last_seen_at."""
        payload = self._load_payload()
        records = [
            dict(item) for item in payload.get("observations", []) if isinstance(item, Mapping)
        ]
        records.sort(key=lambda item: float(item.get("last_seen_at") or 0.0), reverse=True)
        return records if limit <= 0 else records[:limit]

    def unresolved(self, *, min_count: int = 1, limit: int = 50) -> list[dict[str, Any]]:
        """Return unresolved observations meeting the occurrence threshold."""
        records = [
            record
            for record in self.list_observations(limit=0)
            if int(record.get("occurrence_count") or 0) >= min_count
            and str(record.get("status") or "open") == "open"
        ]
        return records if limit <= 0 else records[:limit]

    def mark_status(
        self, observation_id: str, status: str, *, reason: str = ""
    ) -> dict[str, Any] | None:
        """Set an observation status, returning the updated record if found."""
        payload = self._load_payload()
        for record in payload.get("observations", []):
            if not isinstance(record, dict) or record.get("observation_id") != observation_id:
                continue
            record["status"] = str(status or "open")
            if reason:
                record["status_reason"] = str(reason)
            self._write_payload(payload)
            return dict(record)
        return None

    def _dedup_key(
        self,
        result: Mapping[str, Any],
        environment: Mapping[str, Any],
        workspace_root: str,
    ) -> str:
        origin = str(result.get("error_type") or result.get("failure_code") or "runtime")
        subject = str(
            result.get("original_tool_name")
            or result.get("capability")
            or result.get("tool_name")
            or "unknown"
        )
        workspace_id = _stable_hash(str(workspace_root or environment.get("workspace_root") or ""))
        platform_hash = _stable_hash(environment.get("platform_capabilities") or [])
        return f"{origin}:{subject}:{workspace_id}:{platform_hash}"

    def _load_payload(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"version": 1, "observations": []}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, Mapping) and isinstance(data.get("observations"), list):
                return {
                    "version": int(data.get("version") or 1),
                    "observations": data["observations"],
                }
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {"version": 1, "observations": []}
        return {"version": 1, "observations": []}

    def _write_payload(self, payload: Mapping[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


__all__ = ["JsonCapabilityObservationStore"]
