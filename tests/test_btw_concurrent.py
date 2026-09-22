# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for /btw side-command concurrent bypass in the TUI command queue.

Covers:
- /btw bypasses the serial queue when a main task is active
- Concurrent asyncio.Task is created (not queued)
- Main _active_command is unaffected by the side task
- /btw goes through the normal queue path when idle
- Side task is cleaned up from _side_tasks on completion
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from leapflow.cli.tui_app.app import LeapApp
from leapflow.cli.tui_app.command import TuiCommand, TuiCommandStatus
from leapflow.cli.tui_app.theme import _LIGHT, resolve_theme


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════


class _FakeConsole:
    def __init__(self) -> None:
        self.cards: list[TuiCommand] = []
        self.errors: list[str] = []
        self.systems: list[str] = []
        self.warnings: list[str] = []

    def command_card(self, command: TuiCommand) -> None:
        self.cards.append(command)

    def command_footer(self, command: TuiCommand) -> None:
        self.cards.append(command)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def system(self, message: str) -> None:
        self.systems.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


class _FakeStatus:
    def __init__(self) -> None:
        self.counts: list[tuple[int, int]] = []

    def __call__(self) -> list[tuple[str, str]]:
        return []

    def update_task_counts(self, *, running: int, queued: int) -> None:
        self.counts.append((running, queued))


def _make_app(
    on_input=None,
    *,
    on_control=None,
) -> tuple[LeapApp, _FakeConsole, _FakeStatus]:
    console = _FakeConsole()
    status = _FakeStatus()
    app = LeapApp(
        console=console,
        theme=resolve_theme(_LIGHT, terminal_bg="#FFFFFF"),
        status=status,
        commands=(),
        history_path=Path(tempfile.mkdtemp()) / "tui_history",
        on_input=on_input,
        on_control=on_control,
    )
    return app, console, status


# ════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════


class TestBtwBypassesQueue:
    """Verify /btw does NOT enter _pending_input when a task is active."""

    @pytest.mark.asyncio
    async def test_btw_bypasses_queue_when_task_active(self) -> None:
        """When _active_command is set, /btw must not enter the pending queue."""
        app, console, _ = _make_app(on_input=lambda text: None)

        # Simulate an active command (main task running)
        app._active_command = TuiCommand.create(command_id=99, text="some task").mark_running()

        result = app.submit_text("/btw what is 2+2")

        # Should be immediately marked done (side dispatch), not queued
        assert result.status == TuiCommandStatus.DONE
        # Queue should remain empty — /btw should NOT have entered it
        assert app._pending_input.qsize() == 0
        # Let side task clean up
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_btw_queues_normally_when_no_active_task(self) -> None:
        """When no task is active, /btw should enter the normal queue path."""
        app, console, _ = _make_app(on_input=lambda text: None)

        assert app._active_command is None

        result = app.submit_text("/btw what is 2+2")

        # Should be queued (QUEUED), not immediately dispatched
        assert result.status == TuiCommandStatus.QUEUED
        assert app._pending_input.qsize() == 1


class TestBtwCreatesConcurrentTask:
    """Verify that dispatching /btw creates an asyncio.Task."""

    @pytest.mark.asyncio
    async def test_btw_creates_concurrent_task(self) -> None:
        """submit_text('/btw ...') with active command must create a side task."""
        called = asyncio.Event()

        async def fake_input(text: str) -> None:
            called.set()

        app, console, _ = _make_app(on_input=fake_input)
        app._active_command = TuiCommand.create(command_id=99, text="main task").mark_running()

        result = app.submit_text("/btw hello world")

        assert result.status == TuiCommandStatus.DONE
        # A side task should exist
        assert len(app._side_tasks) == 1

        # Let the side task complete
        await asyncio.sleep(0.05)
        assert called.is_set()

    @pytest.mark.asyncio
    async def test_side_task_cleanup_on_completion(self) -> None:
        """After the side task finishes, it must be removed from _side_tasks."""
        completed = asyncio.Event()

        async def fake_input(text: str) -> None:
            completed.set()

        app, console, _ = _make_app(on_input=fake_input)
        app._active_command = TuiCommand.create(command_id=99, text="main task").mark_running()

        app.submit_text("/btw test question")
        assert len(app._side_tasks) == 1

        # Wait for the side task to complete and its done_callback to fire
        await asyncio.sleep(0.1)
        assert completed.is_set()
        assert len(app._side_tasks) == 0


class TestBtwDoesNotInterfereWithActiveCommand:
    """Verify the main _active_command continues unaffected."""

    @pytest.mark.asyncio
    async def test_btw_does_not_interfere_with_active_command(self) -> None:
        """Dispatching /btw must not modify _active_command or _agent_running."""
        async def fake_input(text: str) -> None:
            await asyncio.sleep(0.01)

        app, console, _ = _make_app(on_input=fake_input)

        # Set up an active command to simulate a running main task
        active = TuiCommand.create(command_id=99, text="main task").mark_running()
        app._active_command = active
        app._agent_running = True

        app.submit_text("/btw side question")

        # Active command must be unchanged
        assert app._active_command is active
        assert app._active_command.id == 99
        assert app._agent_running is True

        # Let the side task finish
        await asyncio.sleep(0.05)
        # Still unchanged
        assert app._active_command is active


class TestIsSideCommand:
    """Verify _is_side_command detection."""

    def test_recognizes_btw(self) -> None:
        app, _, _ = _make_app()
        assert app._is_side_command("/btw hello") is True

    def test_recognizes_btw_without_slash(self) -> None:
        app, _, _ = _make_app()
        # submit_text normalizes before calling; but _is_side_command
        # should handle both forms
        assert app._is_side_command("btw hello") is True

    def test_recognizes_aside_alias(self) -> None:
        app, _, _ = _make_app()
        assert app._is_side_command("/aside hello") is True

    def test_rejects_regular_command(self) -> None:
        app, _, _ = _make_app()
        assert app._is_side_command("/status") is False

    def test_rejects_plain_text(self) -> None:
        app, _, _ = _make_app()
        assert app._is_side_command("hello world") is False

    def test_rejects_empty(self) -> None:
        app, _, _ = _make_app()
        assert app._is_side_command("") is False
        assert app._is_side_command("/") is False


class TestSideTaskErrorHandling:
    """Verify side task error handling."""

    @pytest.mark.asyncio
    async def test_side_task_error_does_not_crash(self) -> None:
        """If the on_input callback raises, the error is caught and logged."""
        async def failing_input(text: str) -> None:
            raise RuntimeError("boom")

        app, console, _ = _make_app(on_input=failing_input)
        app._active_command = TuiCommand.create(command_id=99, text="main task").mark_running()

        app.submit_text("/btw will fail")

        await asyncio.sleep(0.1)
        # Error should be reported to console, not crash the event loop
        assert any("Side question failed" in e for e in console.errors)
        # Task should be cleaned up
        assert len(app._side_tasks) == 0
