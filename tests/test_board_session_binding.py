# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression tests: LeapBoard must observe the session that opened it.

Root cause of "board opens, status bar shows watch, page stays empty":
conversation state lives on per-session engines from ``SessionRegistry``, but
``SessionCoordinator.get_history`` read ``ctx.engine`` — the *base* engine,
which is only a template and never carries a conversation. The session-analysis
watch therefore analyzed an empty transcript, produced no useful finding, and
the board had nothing to render.

Validates:
- the registry exposes read-only lookup (``get``) and a "current session"
  fallback (``most_recent``) without materializing engines
- history resolves the requested session's engine, then the most recently
  active one, and only falls back to the base engine (in-process mode)
- ``/board`` binds the caller's session id into the watch params
- the session producer forwards that bound id when reading history
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from leapflow.daemon.session_coordinator import SessionCoordinator
from leapflow.daemon.session_registry import SessionRegistry


class _Wm:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = messages

    def as_chat_messages(self) -> list[dict[str, Any]]:
        return list(self._messages)


class _Engine:
    def __init__(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        self._current_session_id = session_id
        self._wm = _Wm(messages)
        self.turn_count = len(messages)
        self.context_token_count = 100 * len(messages)


def _registry(base: Any) -> SessionRegistry:
    return SessionRegistry(
        base_engine=base,
        build_engine=lambda b, sid, wm, root: _Engine(sid, [{"role": "user", "content": f"hi from {sid}"}]),
        build_working_memory=lambda: None,
    )


# ── SessionRegistry read-only lookup ─────────────────────────────────


def test_registry_get_does_not_create_and_most_recent_tracks_activity() -> None:
    base = _Engine("", [])
    registry = _registry(base)

    assert registry.get("absent") is None
    assert registry.most_recent_any_client() is None
    assert registry.active_count() == 0  # lookups must not materialize engines

    first = asyncio.run(registry.acquire("s1", workspace_root="/tmp"))
    second = asyncio.run(registry.acquire("s2", workspace_root="/tmp"))

    assert registry.get("s1") is first
    # s2 was acquired last, so it is the most recent across all clients.
    assert registry.most_recent_any_client() is second
    first.touch()
    assert registry.most_recent_any_client() is first


# ── get_history resolves the right engine ────────────────────────────


def test_history_reads_requested_session_not_base_engine() -> None:
    """The regression: base engine carries no conversation."""
    base = _Engine("", [])  # base is a template: empty transcript
    coordinator = SessionCoordinator()
    registry = _registry(base)
    coordinator._session_registry = registry
    asyncio.run(registry.acquire("tui-a", workspace_root="/tmp"))
    asyncio.run(registry.acquire("tui-b", workspace_root="/tmp"))

    ctx = SimpleNamespace(engine=base, _conversation_store=None)
    history = asyncio.run(coordinator.get_history(ctx, None, session_id="tui-a"))

    assert history["session_id"] == "tui-a"
    assert [m["content"] for m in history["messages"]] == ["hi from tui-a"]


def test_history_falls_back_to_most_recent_session_when_id_absent() -> None:
    base = _Engine("", [])
    coordinator = SessionCoordinator()
    registry = _registry(base)
    coordinator._session_registry = registry
    asyncio.run(registry.acquire("older", workspace_root="/tmp"))
    asyncio.run(registry.acquire("newest", workspace_root="/tmp"))

    ctx = SimpleNamespace(engine=base, _conversation_store=None)
    history = asyncio.run(coordinator.get_history(ctx, None))

    assert history["session_id"] == "newest"


def test_history_uses_base_engine_without_registry() -> None:
    """In-process mode: ctx.engine *is* the conversation engine."""
    engine = _Engine("in-proc", [{"role": "user", "content": "local"}])
    coordinator = SessionCoordinator()
    ctx = SimpleNamespace(engine=engine, _conversation_store=None)

    history = asyncio.run(coordinator.get_history(ctx, None))

    assert history["session_id"] == "in-proc"
    assert [m["content"] for m in history["messages"]] == ["local"]


def test_resolve_session_engine_handles_missing_context() -> None:
    coordinator = SessionCoordinator()
    assert coordinator.resolve_session_engine(None) == (None, "")


# ── /board binds the caller's session into the watch ─────────────────


class _Monitors:
    def __init__(self) -> None:
        self.armed_params: dict[str, Any] = {}
        self.scheduled: list[tuple[str, bool]] = []

    def list_watches(self) -> list[Any]:
        return []

    async def arm_watch(self, spec: Any) -> Any:
        self.armed_params = dict(spec.params or {})
        return SimpleNamespace(watch_id="w-session", name="Session", domain="session")

    def schedule_watch_once(self, watch_id: str, *, force: bool = False) -> None:
        self.scheduled.append((watch_id, force))


@pytest.mark.asyncio
async def test_board_open_binds_caller_session_id_into_watch_params() -> None:
    from leapflow.cli.commands.slash_handlers import command_execute

    monitors = _Monitors()
    ctx = SimpleNamespace(monitors=monitors, settings=None, engine=None)

    payload = await command_execute(ctx, "board", "", session_id="tui-caller")

    assert payload["ok"] is True and payload["view"] == "dashboard"
    assert monitors.armed_params.get("session_id") == "tui-caller"
    assert monitors.scheduled == [("w-session", True)]


# ── producer forwards the bound session id ───────────────────────────


@pytest.mark.asyncio
async def test_session_producer_reads_the_bound_session() -> None:
    from leapflow.monitor.session_producer import SessionAnalysisProducer
    from leapflow.monitor.types import ProducerContext, WatchSpec

    requested: list[str] = []

    class _Services:
        async def session_history(self, session_id: str = "") -> dict[str, Any]:
            requested.append(session_id)
            return {
                "session_id": session_id,
                "turn_count": 3,
                "token_count": 10,
                "messages": [{"role": "user", "content": "q"}],
                "artifacts": [],
            }

        async def analyze_session(self, messages, *, prior=None, artifacts=None):
            return {"story": "analyzed"}

        async def should_refresh(self, messages) -> bool:
            return True

    producer = SessionAnalysisProducer()
    spec = WatchSpec(
        name="Session", domain="session", trigger_expr="2m",
        params={"session_id": "tui-bound"}, watch_id="w1",
    )
    findings = await producer.observe(
        ProducerContext(spec=spec, now=1000.0, run_count=0, last_run_at=0.0,
                        services=_Services(), force=True)
    )

    assert requested == ["tui-bound"]
    assert len(findings) == 1
    assert findings[0].payload["story"] == "analyzed"


@pytest.mark.asyncio
async def test_session_producer_tolerates_legacy_history_signature() -> None:
    """A host predating the session-scoped signature must still work."""
    from leapflow.monitor.session_producer import SessionAnalysisProducer
    from leapflow.monitor.types import ProducerContext, WatchSpec

    class _LegacyServices:
        async def session_history(self) -> dict[str, Any]:  # no session_id param
            return {
                "session_id": "legacy", "turn_count": 1, "token_count": 5,
                "messages": [{"role": "user", "content": "q"}], "artifacts": [],
            }

        async def analyze_session(self, messages, *, prior=None, artifacts=None):
            return {"story": "legacy ok"}

        async def should_refresh(self, messages) -> bool:
            return True

    producer = SessionAnalysisProducer()
    spec = WatchSpec(
        name="Session", domain="session", trigger_expr="2m",
        params={"session_id": "ignored-by-legacy-host"}, watch_id="w2",
    )
    findings = await producer.observe(
        ProducerContext(spec=spec, now=1000.0, run_count=0, last_run_at=0.0,
                        services=_LegacyServices(), force=True)
    )

    assert len(findings) == 1
    assert findings[0].payload["story"] == "legacy ok"
