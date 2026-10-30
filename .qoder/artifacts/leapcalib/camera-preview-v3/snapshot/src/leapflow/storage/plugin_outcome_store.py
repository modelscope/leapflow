# Copyright (c) Alibaba, Inc. and its affiliates.
"""Event-sourced audit store for adaptive plugin execution outcomes."""

from __future__ import annotations

import time
import uuid
from typing import Any, Mapping

from leapflow.domain.event_types import EvolutionEventType
from leapflow.domain.evolution_event import EvolutionContext, EvolutionEvent


class EvolutionPluginOutcomeStore:
    """Plugin outcome projection backed by the append-only evolution event stream."""

    def __init__(self, event_store: Any, *, profile_id: str) -> None:
        self._event_store = event_store
        self._profile_id = str(profile_id)

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
        outcome_id = f"out-{uuid.uuid4().hex}"
        created_at = time.time()
        record = {
            "outcome_id": outcome_id,
            "created_at": created_at,
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
        event = EvolutionEvent.create(
            EvolutionEventType.PLUGIN_OUTCOME_RECORDED,
            context=EvolutionContext(
                profile_id=self._profile_id,
                requirement_id=record["requirement_id"],
                plugin_id=record["plugin_id"],
                correlation_id=record["plan_id"] or outcome_id,
            ),
            payload=record,
            producer="plugin.lifecycle_governor",
            privacy_class="profile",
            occurred_at=created_at,
            dedup_key=f"plugin.outcome_recorded:{outcome_id}",
        )
        if not self._event_store.append(event):
            raise RuntimeError(f"duplicate plugin outcome: {outcome_id}")
        return record

    def list_outcomes(self, *, plugin_id: str = "", limit: int = 100) -> list[dict[str, Any]]:
        records = self._event_store.read(
            profile_id=self._profile_id,
            event_type=EvolutionEventType.PLUGIN_OUTCOME_RECORDED,
            limit=5000,
        )
        outcomes = [
            dict(record.event.to_dict()["payload"])
            for record in reversed(records)
            if not plugin_id or record.event.context.plugin_id == plugin_id
        ]
        return outcomes if limit <= 0 else outcomes[:limit]

    def failure_streak(self, plugin_id: str) -> int:
        streak = 0
        for record in self.list_outcomes(plugin_id=plugin_id, limit=0):
            if record.get("ok") is True:
                break
            streak += 1
        return streak


__all__ = ["EvolutionPluginOutcomeStore"]
