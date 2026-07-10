from __future__ import annotations

import asyncio
from contextlib import suppress

import pytest

import leapflow.cli.tui_app.app as app_module
from prompt_toolkit.auto_suggest import Suggestion
from prompt_toolkit.completion import Completion
from leapflow.cli.tui_app.app import LeapApp
from leapflow.cli.tui_app.console import LeapConsole
from leapflow.cli.tui_app.command import TuiCommand, TuiCommandStatus
from leapflow.cli.tui_app.stream import StreamRenderer
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


def test_tab_accepts_selected_completion() -> None:
    app, _console, _status = _make_app()
    buffer = app._input_area.buffer
    buffer.text = "/he"
    buffer.cursor_position = len(buffer.text)
    completion = Completion("/help", start_position=-len(buffer.text))
    buffer._set_completions([completion])

    app._accept_or_start_completion(buffer)

    assert buffer.text == "/help"
    assert buffer.cursor_position == len("/help")


@pytest.mark.asyncio
async def test_tab_accepts_visible_auto_suggestion() -> None:
    app, _console, _status = _make_app()
    buffer = app._input_area.buffer
    buffer.text = "你能"
    buffer.cursor_position = len(buffer.text)
    buffer.suggestion = Suggestion("帮我找一找 hub 上有啥 skill 么？")

    app._accept_or_start_completion(buffer)

    assert buffer.text == "你能帮我找一找 hub 上有啥 skill 么？"


@pytest.mark.asyncio
async def test_right_arrow_accepts_suggestion_only_at_end() -> None:
    app, _console, _status = _make_app()
    buffer = app._input_area.buffer
    buffer.text = "abc"
    buffer.cursor_position = 1
    buffer.suggestion = Suggestion("def")

    app._move_right_or_accept_suggestion(buffer)

    assert buffer.text == "abc"
    assert buffer.cursor_position == 2

    buffer.cursor_position = len(buffer.text)
    app._move_right_or_accept_suggestion(buffer)

    assert buffer.text == "abcdef"
    assert buffer.cursor_position == len("abcdef")


def test_command_card_keeps_elapsed_in_title_not_body(monkeypatch) -> None:
    class CapturingConsole:
        width = 100

        def __init__(self) -> None:
            self.rendered = []

        def print(self, renderable) -> None:
            self.rendered.append(renderable)

    capture = CapturingConsole()
    leap_console = LeapConsole(resolve_theme(_LIGHT, terminal_bg="#FFFFFF"))
    monkeypatch.setattr(leap_console, "_console", capture)

    command = TuiCommand.create(command_id=1, text="summarize the current project layout")
    command = command.mark_running().mark_done()
    leap_console.command_card(command)

    panel = capture.rendered[0]
    title = getattr(panel.title, "plain", str(panel.title))
    body = getattr(panel.renderable, "plain", str(panel.renderable))
    assert "done" in title
    assert "elapsed:" not in body
    assert "summarize the current project layout" in body


def test_stream_renderer_prints_tool_command_and_success_preview() -> None:
    class CaptureConsole:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def print(self, renderable) -> None:
            self.lines.append(getattr(renderable, "plain", str(renderable)))

        def thinking(self, text: str) -> None:
            pass

        def markdown(self, text: str, *, indent: int = 0, margin_top: int = 0) -> None:
            pass

        def response_label(self, elapsed_s: float, *, tool_count: int = 0) -> None:
            pass

        def newline(self) -> None:
            pass

    console = CaptureConsole()
    renderer = StreamRenderer(console)  # type: ignore[arg-type]
    renderer.start()

    renderer.tool_started("shell_run", metadata={"command": "python -V"})
    renderer.tool_finished("shell_run", metadata={"ok": True, "stdout_preview": "Python 3.13.0"})

    assert any("shell_run" in line and "$ python -V" in line for line in console.lines)
    assert any("shell_run" in line and "Python 3.13.0" in line for line in console.lines)


def test_stream_renderer_prints_tool_failure_exit_and_stderr() -> None:
    class CaptureConsole:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def print(self, renderable) -> None:
            self.lines.append(getattr(renderable, "plain", str(renderable)))

        def thinking(self, text: str) -> None:
            pass

        def markdown(self, text: str, *, indent: int = 0, margin_top: int = 0) -> None:
            pass

        def response_label(self, elapsed_s: float, *, tool_count: int = 0) -> None:
            pass

        def newline(self) -> None:
            pass

    console = CaptureConsole()
    renderer = StreamRenderer(console)  # type: ignore[arg-type]
    renderer.start()

    renderer.tool_started("shell_run", metadata={"command": "false"})
    renderer.tool_finished(
        "shell_run",
        metadata={"ok": False, "exit_code": 1, "stderr_preview": "permission denied"},
    )

    assert any("✗" in line and "exit=1 permission denied" in line for line in console.lines)


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


class _FakeBuffer:
    def __init__(self) -> None:
        self.text = ""

    def insert_text(self, text: str) -> None:
        self.text += text


def test_large_paste_is_compacted_but_submits_full_text() -> None:
    app, _console, _status = _make_app()
    buffer = _FakeBuffer()
    pasted = "line\n" * 2_000

    app._insert_paste_text(buffer, pasted)

    assert pasted not in buffer.text
    assert "pasted block #1" in buffer.text
    assert "full text will be submitted" in buffer.text
    assert buffer.text.isascii()
    command = app.submit_text(f"Please summarize:\n{buffer.text}")

    assert command.text == f"Please summarize:\n{pasted}".strip()
    assert app._paste_blocks == {}


@pytest.mark.asyncio
async def test_buffer_insert_compacts_large_english_paste() -> None:
    app, _console, _status = _make_app()
    pasted = "long english line\n" * 400

    app._input_area.buffer.insert_text(pasted)

    visible = app._input_area.buffer.text
    assert pasted not in visible
    assert visible.startswith("[pasted block #1:")
    assert "full text will be submitted" in visible
    assert len(visible) < 120
    assert app.submit_text(visible).text == pasted.strip()


@pytest.mark.asyncio
async def test_buffer_insert_compacts_large_chinese_paste_with_ascii_marker() -> None:
    app, _console, _status = _make_app()
    pasted = "这是一段用于验证中文大段粘贴不会直接渲染的内容。\n" * 300

    app._input_area.buffer.insert_text(pasted)

    visible = app._input_area.buffer.text
    assert pasted not in visible
    assert "这是一段" not in visible
    assert visible.startswith("[pasted block #1:")
    assert visible.isascii()
    assert len(visible) < 120
    assert app.submit_text(visible).text == pasted.strip()


@pytest.mark.asyncio
async def test_fragmented_chinese_paste_compacts_and_submits_full_text() -> None:
    app, _console, _status = _make_app()
    pasted = "经济活动达到最低点，经济增长理论，索洛增长模型。" * 80

    for index in range(0, len(pasted), 18):
        app._input_area.buffer.insert_text(pasted[index:index + 18])

    visible = app._input_area.buffer.text
    assert pasted not in visible
    assert "经济活动" not in visible
    assert visible.startswith("[pasted block #1:")
    assert visible.isascii()
    assert len(visible) < 120
    assert app.submit_text(visible).text == pasted.strip()


@pytest.mark.asyncio
async def test_fragmented_english_single_line_paste_compacts() -> None:
    app, _console, _status = _make_app()
    pasted = "capital accumulation and productivity growth " * 80

    for index in range(0, len(pasted), 16):
        app._input_area.buffer.insert_text(pasted[index:index + 16])

    visible = app._input_area.buffer.text
    assert pasted not in visible
    assert visible.startswith("[pasted block #1:")
    assert visible.isascii()
    assert app.submit_text(visible).text == pasted.strip()


@pytest.mark.asyncio
async def test_control_character_paste_is_compacted_and_sanitized() -> None:
    app, _console, _status = _make_app()
    pasted = "normal\rtext\x1b[31mred\x1b[0m\x00tail\u202edone"

    app._input_area.buffer.insert_text(pasted)

    visible = app._input_area.buffer.text
    assert visible.startswith("[pasted block #1:")
    assert visible.isascii()
    assert "\x1b" not in visible
    command = app.submit_text(visible)
    assert command.text == "normal\ntextredtaildone"


@pytest.mark.asyncio
async def test_fragmented_paste_window_expiry_keeps_followup_typing_visible(monkeypatch) -> None:
    app, _console, _status = _make_app()
    clock = 100.0

    def fake_monotonic() -> float:
        return clock

    monkeypatch.setattr(app_module.time, "monotonic", fake_monotonic)
    pasted = "fragmented paste " * 80
    for index in range(0, len(pasted), 20):
        app._input_area.buffer.insert_text(pasted[index:index + 20])

    visible = app._input_area.buffer.text
    assert visible.startswith("[pasted block #1:")

    clock = 101.0
    app._input_area.buffer.insert_text(" follow-up")

    assert app._input_area.buffer.text == f"{visible} follow-up"
    assert app.submit_text(app._input_area.buffer.text).text == f"{pasted} follow-up".strip()


def test_small_paste_stays_inline() -> None:
    app, _console, _status = _make_app()
    buffer = _FakeBuffer()
    pasted = "short pasted text"

    app._insert_paste_text(buffer, pasted)

    assert buffer.text == pasted
    assert app._paste_blocks == {}


def test_submit_clears_stale_compacted_paste_when_marker_removed() -> None:
    app, _console, _status = _make_app()
    buffer = _FakeBuffer()
    app._insert_paste_text(buffer, "line\n" * 2_000)

    command = app.submit_text("marker was deleted by user")

    assert command.text == "marker was deleted by user"
    assert app._paste_blocks == {}


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
