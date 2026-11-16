# Copyright (c) Alibaba, Inc. and its affiliates.
"""A cancelled or abandoned turn must release everything it owns.

Regression suite for a TUI ``/cancel`` that left the next turn in the same
session on "Thinking…" for 45 minutes. The client closed its socket mid-stream;
``_dispatch_stream`` noticed on the next chunk write and closed ``engine_chat``
from the connection task's Context instead of the pinned Context the turn ran
in. Every ``ContextVar.reset`` in the turn's ``finally`` then raised "created in
a different Context", aborting the cleanup that releases the session lock, the
turn slot, and the ``_active_engines`` entry — and the failure was logged at
DEBUG. The next turn blocked on the session lock forever while heartbeats kept
its spinner alive.
"""
from __future__ import annotations

import asyncio
import contextvars
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from leapflow.daemon._transport import get_transport
from leapflow.daemon.client import DaemonClient
from leapflow.daemon.server import UnixRpcServer
from leapflow.daemon.service import RuntimeLeapService
from leapflow.daemon.session_registry import SessionExecutionContext
from leapflow.engine import StreamEvent


def _short_tempdir() -> str:
    # AF_UNIX socket paths cap near 108 chars; macOS's default TMPDIR is too long.
    return tempfile.gettempdir() if sys.platform == "win32" else "/tmp"


async def _start_server(runtime_dir: Path, service: Any) -> tuple[UnixRpcServer, asyncio.Task[Any]]:
    server = UnixRpcServer(
        service,
        sock_path=get_transport().readiness_path(runtime_dir),
        runtime_dir=runtime_dir,
    )
    task = asyncio.create_task(server.serve_forever())
    readiness = get_transport().readiness_path(runtime_dir)
    for _ in range(50):
        if readiness.exists():
            return server, task
        await asyncio.sleep(0.02)
    task.cancel()
    raise AssertionError("server did not start")


class _StreamingEngine:
    """First turn streams tokens until closed; later turns answer at once."""

    def __init__(self) -> None:
        self._current_session_id = "S"
        self.turns = 0
        self.first_turn_closed = asyncio.Event()
        self.cancel_calls = 0

    def cancel(self) -> None:
        self.cancel_calls += 1

    async def run_stream(self, message: str, *, enable_thinking: bool = False, request_id: str = ""):
        self.turns += 1
        if self.turns == 1:
            try:
                while True:
                    await asyncio.sleep(0.005)
                    yield StreamEvent(type="chunk", content="tok")
            finally:
                self.first_turn_closed.set()
        yield StreamEvent(type="final", content=f"turn{self.turns}-done")


def _service_for(engine: Any, *, max_turns: int = 3) -> tuple[RuntimeLeapService, SessionExecutionContext]:
    session = SessionExecutionContext("S", engine, Path("/tmp"))

    class _Registry:
        async def acquire(self, sid: str, workspace_root: Any = None) -> SessionExecutionContext:
            return session

    class _Context:
        settings = SimpleNamespace(llm_context_length=100)

        def __init__(self) -> None:
            self.engine = engine

        def reload_runtime_config_if_changed(self) -> bool:
            return False

    service = RuntimeLeapService(SimpleNamespace(llm_context_length=100, daemon_max_concurrent_turns=max_turns))
    service._ctx = _Context()
    service._session_registry = _Registry()
    return service, session


def _assert_released(service: RuntimeLeapService, session: SessionExecutionContext) -> None:
    assert not session.lock.locked(), "session lock leaked"
    assert service._active_engines == {}, "engine registration leaked"
    admission = service._turn_admission.snapshot()
    assert admission["active"] == 0 and admission["slots_in_use"] == 0, admission


@pytest.mark.asyncio
async def test_client_disconnect_mid_stream_releases_turn_and_next_turn_runs() -> None:
    """The live incident, end to end over the real socket and ``_dispatch_stream``."""
    engine = _StreamingEngine()
    service, session = _service_for(engine)
    with tempfile.TemporaryDirectory(prefix="lfd-", dir=_short_tempdir()) as root:
        runtime_dir = Path(root) / "runtime"
        server, serve_task = await _start_server(runtime_dir, service)
        client = DaemonClient(get_transport().readiness_path(runtime_dir))
        try:
            async def first_turn() -> None:
                async for event in client.engine_chat("one", session_id="S"):
                    if event.type == "chunk":
                        await asyncio.Event().wait()  # user is reading; then /cancel

            first = asyncio.create_task(first_turn())
            await asyncio.sleep(0.2)
            first.cancel()  # TUI /cancel: dispatch task cancelled, socket closed
            with pytest.raises(asyncio.CancelledError):
                await first

            await asyncio.wait_for(engine.first_turn_closed.wait(), 2.0)
            for _ in range(50):
                if not session.lock.locked() and not service._active_engines:
                    break
                await asyncio.sleep(0.02)
            _assert_released(service, session)

            async def second_turn() -> list[str]:
                return [e.content async for e in client.engine_chat("two", session_id="S") if e.type == "final"]

            assert await asyncio.wait_for(second_turn(), 3.0) == ["turn2-done"]
        finally:
            serve_task.cancel()
            await server.stop()
    _assert_released(service, session)


@pytest.mark.asyncio
async def test_turn_closed_from_a_foreign_context_still_releases_ownership() -> None:
    """Defense in depth: even a finalizer outside the turn's Context frees lock and slot."""
    engine = _StreamingEngine()
    service, session = _service_for(engine)
    stream = service.engine_chat("one", request_id="r1", session_id="S")
    turn_ctx = contextvars.copy_context()
    for _ in range(3):
        await asyncio.create_task(anext(stream), context=turn_ctx)
    assert session.lock.locked() and "r1" in service._active_engines

    await stream.aclose()  # this task's Context, not ``turn_ctx``

    assert engine.first_turn_closed.is_set(), "engine stream must be closed, not abandoned"
    _assert_released(service, session)


@pytest.mark.asyncio
async def test_turn_waiting_on_its_session_says_so() -> None:
    """A turn queued behind its own session's unwinding turn must not look like thinking."""
    engine = _StreamingEngine()
    service, session = _service_for(engine)
    await session.lock.acquire()
    stream = service.engine_chat("two", request_id="r2", session_id="S")
    first = await asyncio.wait_for(anext(stream), 1.0)
    assert first.event_type == "status" and first.metadata.get("session_busy") is True
    await stream.aclose()
    session.lock.release()
    _assert_released(service, session)


@pytest.mark.asyncio
async def test_engine_cancel_with_request_id_targets_only_that_turn() -> None:
    """Concurrent TUIs share leapd: one client's /cancel must not cancel another's turn."""
    mine, theirs, template = _StreamingEngine(), _StreamingEngine(), _StreamingEngine()
    service = RuntimeLeapService(SimpleNamespace(llm_context_length=100))
    service._ctx = SimpleNamespace(engine=template)
    service._active_engines = {"mine": mine, "theirs": theirs}

    assert await service.engine_cancel("mine") is True
    assert (mine.cancel_calls, theirs.cancel_calls) == (1, 0)

    # An already-finished turn is a no-op, never a fallback to someone else's engine.
    assert await service.engine_cancel("gone") is False
    assert (mine.cancel_calls, theirs.cancel_calls, template.cancel_calls) == (1, 0, 0)


@pytest.mark.asyncio
async def test_client_engine_chat_uses_caller_request_id(monkeypatch) -> None:
    """The TUI names its turn so ``engine_cancel(request_id)`` can target it."""
    import json

    from leapflow.daemon import client as client_mod

    sent: list[dict[str, Any]] = []

    async def fake_open(self: DaemonClient) -> tuple[object, object]:
        return object(), object()

    async def fake_send(writer: object, text: str) -> None:
        sent.append(json.loads(text))

    async def fake_read(self: DaemonClient, reader: object) -> dict[str, Any]:
        return {"id": sent[-1]["id"]}

    async def fake_close(writer: object) -> None:
        return None

    monkeypatch.setattr(DaemonClient, "_open", fake_open)
    monkeypatch.setattr(DaemonClient, "_read_payload", fake_read)
    monkeypatch.setattr(client_mod, "_send", fake_send)
    monkeypatch.setattr(client_mod, "_close_writer", fake_close)

    client = DaemonClient(Path("/tmp/leapd-test.sock"))
    _ = [e async for e in client.engine_chat("hi", request_id="turn-42")]
    _ = [e async for e in client.engine_chat("hi")]
    assert sent[0]["id"] == "turn-42"
    assert sent[1]["id"] and sent[1]["id"] != "turn-42"


@pytest.mark.asyncio
async def test_engine_cancel_interrupts_an_in_flight_loop_await() -> None:
    """``cancel()`` must reach the running loop task, not only set a flag it checks later."""
    from leapflow.engine.engine import AgentEngine

    entered = asyncio.Event()

    class _Stub:
        _active_task = None
        _cancel_requested = False

        def _build_root_frame(self, user_text: str, *, enable_thinking: bool = False) -> object:
            return object()

        async def _run_agent_loop(self, frame: object, *, sink: Any) -> None:
            entered.set()
            await asyncio.Event().wait()  # a long LLM call or tool

    stub = _Stub()
    stream = AgentEngine._stream_via_sink(stub, "hi")  # type: ignore[arg-type]
    first = asyncio.create_task(anext(stream))
    await asyncio.wait_for(entered.wait(), 1.0)
    assert stub._active_task is not None

    AgentEngine.cancel(stub)  # type: ignore[arg-type]
    event = await asyncio.wait_for(first, 1.0)
    assert event.type == "error" and event.content == "Cancelled by user."
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert stub._active_task is None
