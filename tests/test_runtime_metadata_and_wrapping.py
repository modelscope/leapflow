# Copyright (c) Alibaba, Inc. and its affiliates.
"""Guards for runtime metadata reporting and long-output rendering.

Two failures that kept coming back, both because a value was read from the wrong
place rather than because the display was wrong:

- The status bar sat at ``0/<limit>`` all session. Conversation state lives on
  per-session engines from ``SessionRegistry``; ``ctx.engine`` is only the
  template they are built from and never accumulates turns, so anything reading
  it reports zero context. The same root cause produced an empty LeapBoard
  earlier, which is why this file guards the *entry point* rather than one call
  site: new metadata code must go through ``_active_engine()``.
- Long answers lost their tail. ``soft_wrap=True`` makes Rich emit one long line
  and defer wrapping to whoever owns the screen; under ``patch_stdout`` that
  renderer clips at the window edge.
"""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

import pytest

from leapflow.daemon.service import RuntimeLeapService
from leapflow.daemon.session_registry import SessionRegistry
from leapflow.engine import StreamEvent

_CONTEXT_LENGTH = 1_000_000
_USED_TOKENS = 48_000


class _BaseEngine:
    """The template engine: it is never handed a conversation."""

    _current_session_id = ""
    context_token_count = 0
    turn_count = 0


class _SessionEngine:
    """A per-session engine, i.e. the one that actually accrues context."""

    def __init__(self, session_id: str, used: int) -> None:
        self._current_session_id = session_id
        self.context_token_count = used
        self.turn_count = 3


def _service_with_session(used: int = _USED_TOKENS) -> tuple[RuntimeLeapService, _BaseEngine]:
    service = RuntimeLeapService(SimpleNamespace())
    base = _BaseEngine()
    registry = SessionRegistry(
        base_engine=base,
        build_engine=lambda b, sid, wm, root: _SessionEngine(sid, used),
        build_working_memory=lambda: None,
    )
    asyncio.run(registry.acquire("s1", workspace_root="/tmp"))
    service._ctx = SimpleNamespace(
        engine=base,
        settings=SimpleNamespace(llm_context_length=_CONTEXT_LENGTH, llm_model="qwen3.8-max"),
    )
    service._session_coordinator._session_registry = registry
    return service, base


# ── Runtime metadata must describe the session, not the template ─────────


def test_active_engine_resolves_the_session_engine() -> None:
    service, base = _service_with_session()

    engine = service._active_engine()

    assert engine is not base, "the base engine carries no conversation"
    assert engine.context_token_count == _USED_TOKENS


def test_stream_metadata_reports_real_context_usage() -> None:
    """The status bar reads this; zero here is what showed as ``0/1M``.

    The producing engine is passed explicitly. There is no fallback to "whichever
    engine looks active": on a daemon serving several TUIs that resolves another
    client's session, and the client adopts the reported id as its own.
    """
    service, _ = _service_with_session()
    session_engine = service._active_engine("s1")

    chunk = service._chunk_from_event(
        StreamEvent(type="content", content="hi"), request_id="r1", engine=session_engine,
    )

    assert chunk.metadata["context_used"] == _USED_TOKENS
    assert chunk.metadata["llm_context_length"] == _CONTEXT_LENGTH
    assert chunk.metadata["session_id"] == "s1"


def test_chunk_metadata_requires_the_producing_engine() -> None:
    """No engine means no session identity may be attached.

    Guarding the shape rather than the value: an optional engine is what allowed
    a foreign session id into a client's metadata.
    """
    import inspect

    service, _ = _service_with_session()
    signature = inspect.signature(service._chunk_from_event)
    assert signature.parameters["engine"].default is inspect.Parameter.empty

    chunk = service._chunk_from_event(
        StreamEvent(type="content", content="hi"), request_id="r1", engine=None,
    )
    assert "session_id" not in chunk.metadata


def test_stream_metadata_prefers_the_engine_that_produced_the_event() -> None:
    """An explicit engine wins, so a concurrent session cannot be misreported."""
    service, _ = _service_with_session()
    other = _SessionEngine("s2", 12_345)

    chunk = service._chunk_from_event(
        StreamEvent(type="content", content="hi"), request_id="r1", engine=other,
    )

    assert chunk.metadata["context_used"] == 12_345
    assert chunk.metadata["session_id"] == "s2"


def test_status_reads_context_from_the_session_engine() -> None:
    """``status()`` must report on the session engine like the stream does.

    Asserted on the engine it resolves plus the metadata builder, rather than by
    calling ``status()``: that needs a full Settings (layout, profile, paths) and
    the value under test here is only which engine gets measured.
    """
    from leapflow.daemon._service_helpers import engine_context_metadata

    service, base = _service_with_session()

    engine = service._active_engine()
    metadata = engine_context_metadata(engine, service._ctx.settings)

    assert engine is not base
    assert metadata["context_used"] == _USED_TOKENS
    assert metadata["llm_context_length"] == _CONTEXT_LENGTH


def test_base_engine_would_have_reported_zero() -> None:
    """Pins why this matters: the old path could only ever report 0."""
    from leapflow.daemon._service_helpers import engine_context_metadata

    service, base = _service_with_session()

    stale = engine_context_metadata(base, service._ctx.settings)

    assert stale["context_used"] == 0


def test_active_engine_tolerates_a_missing_context() -> None:
    service = RuntimeLeapService(SimpleNamespace())
    service._ctx = None

    assert service._active_engine() is None


# ── Link protection: no metadata path may read ctx.engine directly ───────


def test_metadata_paths_do_not_read_ctx_engine_directly() -> None:
    """Regression guard for the class of bug, not one instance of it.

    Each recurrence so far was a fresh ``getattr(ctx, "engine")`` next to a
    metadata assembly. ``_active_engine()`` is the single entry point; anything
    bypassing it silently reports a conversation-free engine, which is invisible
    in review and only shows up as a zeroed status bar or an empty board.
    """
    import inspect

    from leapflow.daemon import service as service_module

    source = inspect.getsource(service_module)
    offenders: list[str] = []
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if "engine_context_metadata(" not in line:
            continue
        # Look at the call and the few lines above it for a direct base-engine read.
        window = "\n".join(lines[max(0, index - 6):index + 1])
        if 'getattr(ctx, "engine"' in window or "getattr(ctx, 'engine'" in window:
            offenders.append(f"line {index + 1}: {line.strip()}")

    assert offenders == [], (
        "runtime metadata must come from _active_engine(); these read the base "
        "engine directly:\n  " + "\n  ".join(offenders)
    )


# ── Long output must wrap, not clip ──────────────────────────────────────


def _console():
    from leapflow.cli.tui_app.console import LeapConsole
    from leapflow.cli.tui_app.theme import _LIGHT, resolve_theme

    console = LeapConsole(resolve_theme(_LIGHT, terminal_bg="#FFFFFF"))
    console._console.file = io.StringIO()
    console._console.width = 100
    return console


def test_console_wraps_rather_than_emitting_one_long_line() -> None:
    """Rich must do the wrapping; patch_stdout's renderer clips instead."""
    console = _console()
    text = "经济增长理论与索洛模型的边际产出递减规律，" * 12

    console.print(text)
    lines = [line for line in console._console.file.getvalue().split("\n") if line.strip()]

    assert len(lines) > 1, "a long paragraph must be wrapped into several lines"
    # Wide (CJK) glyphs count double, so bound on display width, not characters.
    from rich.cells import cell_len

    assert max(cell_len(line) for line in lines) <= 100


@pytest.mark.parametrize("renderer", ["print", "system", "markdown"])
def test_long_text_wraps_across_output_surfaces(renderer: str) -> None:
    """Wrapping is a console-level property, so every surface inherits it."""
    console = _console()
    text = "capital accumulation and productivity growth compound over decades " * 6

    getattr(console, renderer)(text)
    lines = [line for line in console._console.file.getvalue().split("\n") if line.strip()]

    assert len(lines) > 1, f"{renderer}() must wrap long text"


def test_console_is_not_configured_to_defer_wrapping() -> None:
    """Direct guard on the setting, since the symptom is invisible in tests.

    Clipping only happens inside prompt_toolkit's renderer, so a unit test cannot
    observe the truncation itself — it can only pin the configuration that caused
    it.
    """
    console = _console()

    assert console._console.soft_wrap is False
