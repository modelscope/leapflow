from __future__ import annotations

import asyncio
from contextlib import suppress

import pytest

from leapflow.cli.tui_app.app import LeapApp
from leapflow.cli.tui_app.command import TuiCommand, TuiCommandStatus
from leapflow.cli.tui_app.theme import _LIGHT, resolve_theme


class _FakeConsole:
    def __init__(self) -> None:
        self.cards: list[TuiCommand] = []
        self.errors: list[str] = []
        self.systems: list[str] = []

    def command_card(self, command: TuiCommand) -> None:
        self.cards.append(command)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def system(self, message: str) -> None:
        self.systems.append(message)


class _FakeStatus:
    def __init__(self) -> None:
        self.counts: list[tuple[int, int]] = []

    def __call__(self) -> list[tuple[str, str]]:
        return []

    def update_task_counts(self, *, running: int, queued: int) -> None:
        self.counts.append((running, queued))


async def _wait_for(condition, *, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


def _make_app(on_input=None) -> tuple[LeapApp, _FakeConsole, _FakeStatus]:
    console = _FakeConsole()
    status = _FakeStatus()
    app = LeapApp(
        console=console,
        theme=resolve_theme(_LIGHT, terminal_bg="#FFFFFF"),
        status=status,
        on_input=on_input,
    )
    return app, console, status


def test_submit_text_assigns_ids_and_keeps_input_editable() -> None:
    app, console, status = _make_app()

    first = app.submit_text("first command")
    second = app.submit_text("second command")

    assert first.id == 1
    assert second.id == 2
    assert app._pending_input.qsize() == 2
    assert status.counts[-1] == (0, 2)
    assert [card.status for card in console.cards] == [TuiCommandStatus.QUEUED]
    assert app._input_area.buffer.read_only() is False


def test_submit_text_rejects_empty_commands() -> None:
    app, console, status = _make_app()

    with pytest.raises(ValueError):
        app.submit_text("  \n  ")

    assert app._pending_input.qsize() == 0
    assert console.cards == []
    assert status.counts == []


@pytest.mark.asyncio
async def test_process_loop_marks_failed_commands_and_recovers_counts() -> None:
    async def on_input(text: str) -> None:
        raise RuntimeError(f"boom from {text}")

    app, console, status = _make_app(on_input=on_input)
    app.submit_text("failing command")

    worker = asyncio.create_task(app._process_loop())
    try:
        await _wait_for(
            lambda: len(console.cards) == 2
            and console.cards[-1].status is TuiCommandStatus.FAILED
        )
    finally:
        app._should_exit = True
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker

    assert [(card.id, card.status) for card in console.cards] == [
        (1, TuiCommandStatus.RUNNING),
        (1, TuiCommandStatus.FAILED),
    ]
    assert "RuntimeError: boom from failing command" in console.cards[-1].error
    assert console.errors == ["boom from failing command"]
    assert status.counts[-1] == (0, 0)


def test_failed_command_error_is_single_line_and_truncated() -> None:
    command = TuiCommand.create(command_id=1, text="demo")
    failed = command.mark_failed("line1\n" + "x" * 400)

    assert "\n" not in failed.error
    assert len(failed.error) == 240
    assert failed.error.endswith("…")

@pytest.mark.asyncio
async def test_process_loop_runs_submitted_commands_serially() -> None:
    processed: list[str] = []

    async def on_input(text: str) -> None:
        processed.append(text)
        await asyncio.sleep(0.01)

    app, console, status = _make_app(on_input=on_input)
    app.submit_text("first command")
    app.submit_text("second command")

    worker = asyncio.create_task(app._process_loop())
    try:
        await _wait_for(
            lambda: processed == ["first command", "second command"]
            and len(console.cards) == 5
        )
    finally:
        app._should_exit = True
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker

    rendered = [(card.id, card.status) for card in console.cards]
    assert rendered == [
        (2, TuiCommandStatus.QUEUED),
        (1, TuiCommandStatus.RUNNING),
        (1, TuiCommandStatus.DONE),
        (2, TuiCommandStatus.RUNNING),
        (2, TuiCommandStatus.DONE),
    ]
    assert status.counts[-1] == (0, 0)
