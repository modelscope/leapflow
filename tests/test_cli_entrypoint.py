from __future__ import annotations

import pytest

from conftest import make_settings


def test_context_constructs_approval_gate_before_initialize(tmp_path) -> None:
    from leapflow.cli.context import Context

    ctx = Context(make_settings(str(tmp_path)), mock_host=True)

    assert hasattr(ctx, "_approval_gate")
    assert hasattr(ctx, "_tui_approval")


@pytest.mark.asyncio
async def test_context_initialize_wires_gateway_approval_gate(tmp_path) -> None:
    from leapflow.cli.context import Context
    import leapflow.tools.gateway_tool as gateway_tool

    ctx = Context(make_settings(str(tmp_path)), mock_host=True)
    await ctx.initialize()
    try:
        assert gateway_tool._approval_gate is ctx._approval_gate
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_context_initialize_degrades_when_primary_db_is_locked(
    tmp_path,
    monkeypatch,
) -> None:
    from pathlib import Path

    from leapflow.cli.context import Context
    from leapflow.storage.duckdb_connect import DatabaseLockedError
    import leapflow.storage.connection as connection_module

    settings = make_settings(str(tmp_path))
    original_connect = connection_module._lock_aware_connect

    def locked_primary_connect(db_path: Path):
        if Path(db_path) == settings.duckdb_path:
            raise DatabaseLockedError(settings.duckdb_path, RuntimeError("locked"))
        return original_connect(db_path)

    monkeypatch.setattr(
        connection_module,
        "_lock_aware_connect",
        locked_primary_connect,
    )

    ctx = Context(settings, mock_host=True)
    await ctx.initialize()
    try:
        assert ctx.storage_volatile is True
        assert ctx.engine is not None
        assert ctx._db_holder.db_path != settings.duckdb_path
    finally:
        await ctx.cleanup()


def test_leap_default_command_initializes_and_runs_interactive(monkeypatch) -> None:
    from leapflow.cli import cli
    import leapflow.cli.commands.interactive as interactive_module

    events: list[str] = []

    class FakeContext:
        def __init__(self, settings, mock_host: bool) -> None:
            self.settings = settings
            self.mock_host = mock_host
            self.initialized = False
            events.append("context")

        async def initialize(self) -> None:
            self.initialized = True
            events.append("initialize")

        async def cleanup(self) -> None:
            events.append("cleanup")

    async def fake_interactive(ctx, *, resume_id=None) -> int:
        assert ctx.initialized is True
        assert resume_id is None
        events.append("interactive")
        return 0

    monkeypatch.setattr(cli, "load_config", lambda: object())
    monkeypatch.setattr(cli, "Context", FakeContext)
    monkeypatch.setattr(interactive_module, "cmd_interactive", fake_interactive)

    assert cli.main([]) == 0
    assert events == ["context", "initialize", "interactive", "cleanup"]
