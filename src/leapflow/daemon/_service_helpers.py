# Copyright (c) Alibaba, Inc. and its affiliates.
"""Pure utility functions extracted from service.py to keep the orchestrator slim."""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from leapflow.engine import StreamEvent
from leapflow.memory.protocol import MemoryEntry

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ── Stream event helpers ─────────────────────────────────────────────

def normalize_stream_event(event: object) -> StreamEvent:
    """Coerce an arbitrary engine event into a StreamEvent."""
    if isinstance(event, StreamEvent):
        return event
    return StreamEvent(type="chunk", content=str(event), metadata=None)


def memory_entry_to_dict(entry: MemoryEntry) -> dict[str, Any]:
    """Serialize a MemoryEntry to a JSON-friendly dict."""
    return {
        "entry_id": entry.entry_id,
        "kind": entry.kind.value,
        "domain": entry.domain.value,
        "content": entry.content,
        "timestamp": entry.timestamp,
        "score": entry.score,
        "metadata": dict(entry.metadata),
    }


# ── Engine / context metadata ────────────────────────────────────────

def engine_context_metadata(engine: Any | None, settings: Any) -> dict[str, Any]:
    """Return safe context-budget metadata for daemon status and stream events.

    ``llm_model`` rides along because the TUI is a separate process: its status
    bar seeds the model name at startup and can only learn about a change from
    metadata the daemon sends back. Deriving it here means every status/stream
    path reports the model actually in use, including after a mid-turn
    ``config_set``, instead of relying on a one-off change notification.
    """
    context_length = max(0, int(getattr(settings, "llm_context_length", 0) or 0))
    metadata: dict[str, Any] = {
        "llm_context_length": context_length,
        "context_used": 0,
    }
    model = str(getattr(settings, "llm_model", "") or "")
    if model:
        metadata["llm_model"] = model
    if engine is None:
        return metadata
    # Prefer the engine's effective budget over the configured one: an
    # authoritative model capability can cap the configured value, and reporting
    # the config would claim a window compression is not actually using.
    effective = getattr(engine, "active_context_length", 0)
    if isinstance(effective, int) and effective > 0:
        metadata["llm_context_length"] = effective
    metadata["context_used"] = max(0, int(getattr(engine, "context_token_count", 0) or 0))
    snapshot = getattr(engine, "context_budget_snapshot", {})
    if callable(snapshot):
        snapshot = snapshot()
    if isinstance(snapshot, dict) and snapshot:
        safe_snapshot = dict(snapshot)
        if safe_snapshot.get("context_length"):
            metadata["llm_context_length"] = max(1, int(safe_snapshot["context_length"]))
        if safe_snapshot.get("total_tokens") is not None:
            metadata["context_used"] = max(0, int(safe_snapshot["total_tokens"]))
        posture = safe_snapshot.get("context_posture")
        if posture:
            metadata["context_posture"] = str(posture)
        signal = safe_snapshot.get("context_signal")
        if signal:
            metadata["context_signal"] = str(signal)
        guidance = safe_snapshot.get("context_guidance")
        if guidance:
            metadata["context_guidance"] = str(guidance)
        for key in (
            "compression_reason",
            "compression_savings_ratio",
            "compression_saved_tokens",
            "disclosure_level",
            "disclosure_reason",
            "disclosure",
        ):
            if safe_snapshot.get(key) is not None:
                metadata[key] = safe_snapshot[key]
        metadata["context_budget_snapshot"] = safe_snapshot
    return metadata


def engine_runtime_extras(engine: Any | None) -> dict[str, Any]:
    """Return session turn count and cache hit rate for status introspection.

    These live on the engine, not the budget snapshot, and are needed by the
    self-awareness runtime facet. Kept separate from ``engine_context_metadata``
    so they ride only on the (rare) status call, never on every stream chunk.
    The cache hit rate is normalised from the tracker's 0..1 fraction to a
    human-facing percentage.
    """
    extras: dict[str, Any] = {}
    if engine is None:
        return extras
    turn_count = getattr(engine, "_session_turn_count", None)
    if isinstance(turn_count, int) and turn_count >= 0:
        extras["session_turn_count"] = turn_count
    tracker = getattr(engine, "_usage_tracker", None)
    summary_fn = getattr(tracker, "summary", None) if tracker is not None else None
    if callable(summary_fn):
        try:
            rate = getattr(summary_fn(), "cache_hit_rate", None)
        except Exception:  # noqa: BLE001 - telemetry must never fail status
            rate = None
        if isinstance(rate, (int, float)) and rate >= 0:
            extras["cache_hit_rate"] = round(float(rate) * 100, 1)
    return extras


def host_backend_status(ctx: Any | None) -> dict[str, Any]:
    """Inspect daemon host-backend state for status reporting."""
    if ctx is None:
        return {"backend": "none", "started": False, "reason": "runtime_not_initialized"}
    rpc = getattr(ctx, "rpc", None)
    snapshot = getattr(rpc, "status_snapshot", None)
    if callable(snapshot):
        try:
            return dict(snapshot())
        except Exception as exc:
            return {"backend": type(rpc).__name__, "started": False, "last_error": str(exc)}
    return {
        "backend": type(rpc).__name__ if rpc is not None else "none",
        "started": rpc is not None,
        "pid": None,
        "pid_source": "unavailable",
    }


def persisted_session_workspace(engine: Any, session_id: str) -> str:
    """Return the workspace a session was first created in, if persisted."""
    store = getattr(engine, "_conversation_store", None)
    if store is None:
        return ""
    try:
        session = store.get_session(session_id)
    except Exception:
        return ""
    return str(getattr(session, "cwd", "") or "") if session is not None else ""


def checkpoint_open_connection(ctx: Any) -> None:
    """Issue a DuckDB CHECKPOINT before daemon shutdown."""
    holder = getattr(ctx, "_db_holder", None)
    conn = getattr(holder, "_conn", None)
    if conn is None:
        return
    try:
        conn.execute("CHECKPOINT")
    except Exception:
        logger.debug("daemon: DuckDB checkpoint skipped", exc_info=True)


def runtime_source() -> str:
    import leapflow
    return str(getattr(leapflow, "__file__", ""))


def runtime_version() -> str:
    try:
        from leapflow.version import __version__
    except ImportError:
        return "unknown"
    return str(__version__)


# ── Notification wiring ──────────────────────────────────────────────

def install_learn_notifications(ctx: Any, bus: Any) -> None:
    """Wire session learn-progress/completion callbacks to a NotificationBus."""
    def _on_progress(stage: str, current: int, total: int) -> None:
        bus.emit_event(
            "teach.progress",
            phase=stage,
            current=current,
            total=total,
            progress=current / total if total > 0 else 0.0,
        )

    def _on_complete(result: Any) -> None:
        payload: dict[str, Any] = {"phase": "done"}
        if result:
            payload["step_count"] = getattr(result, "step_count", 0)
            payload["duration"] = getattr(result, "duration", 0.0)
            candidates = getattr(result, "candidates", None) or []
            payload["candidate_count"] = len(candidates)
            activated = getattr(result, "activated_skill_names", None) or set()
            payload["activated_skills"] = list(activated)
            new = getattr(result, "new_skills", None) or []
            payload["new_skills"] = list(new)
        bus.emit_event("teach.complete", **payload)

    if ctx.session:
        ctx.session.set_on_learn_progress(_on_progress)
        if hasattr(ctx.session, "set_on_learn_complete"):
            ctx.session.set_on_learn_complete(_on_complete)

        original_on_idle = ctx.session._on_idle_timeout

        def _on_idle_with_notification() -> None:
            bus.emit_event("teach.stopped", reason="idle_timeout")
            original_on_idle()

        ctx.session._on_idle_timeout = _on_idle_with_notification


# ── ProducerServices facade (used by monitor_coordinator) ────────────

class ProducerServices:
    """Facade exposing daemon capabilities to monitor producers (session, etc.)."""

    def __init__(self, service: Any) -> None:
        self._service = service

    async def session_history(self, session_id: str = "") -> dict[str, Any]:
        return await self._service.session_history(session_id=session_id)

    async def analyze_session(
        self,
        messages: list[dict[str, Any]],
        *,
        prior: dict[str, Any] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return await self._service._session_coordinator.analyze_llm(
            self._service._ctx, messages, artifacts=artifacts
        )

    async def should_refresh(self, messages: list[dict[str, Any]]) -> bool:
        return await self._service._session_coordinator.should_refresh(
            self._service._ctx, messages
        )

    async def evolution_projection_aggregate(self) -> dict[str, Any]:
        """Expose only the explicitly aggregate event projection to producers."""
        return await self._service.evolution_projection_aggregate()
