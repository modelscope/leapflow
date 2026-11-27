# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression tests for terminal-safe rendering and secret-aware history.

macOS Terminal.app crashed (SIGBUS in its wide-glyph split routine) when the
TUI redraw overwrote part of a CJK glyph still on screen. The guard must make
such overwrites impossible without changing what ends up on screen.
"""
from __future__ import annotations

import asyncio
import re
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from prompt_toolkit import Application
from prompt_toolkit import renderer as renderer_module
from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Point, Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.screen import _CHAR_CACHE, Screen
from prompt_toolkit.output.vt100 import Vt100_Output
from prompt_toolkit.widgets import TextArea
from wcwidth import wcwidth

from leapflow.cli.tui_app import render_guard
from leapflow.cli.tui_app.history import SecretAwareFileHistory, contains_secret
from leapflow.cli.tui_app.render_guard import (
    blank_undrawable_wide_glyphs,
    erase_misaligned_wide_tails,
    frame_unchanged,
    install_render_guard,
)

_CSI = re.compile(r"\x1b\[([?0-9;]*)([ -/]*)([@-~])")
_RIGHT_HALF = object()
_DRAFT = (
    "1. 你帮我将机械臂切换到can模式； 2。 相机节点当然要走leap hw serve，不要自己单独走； "
    "3。host daemon旧配置需要刷新； 4。 任务定义：口香糖盒抓起来，并抬高20cm，"
)


class _TerminalModel:
    """Minimal VT grid that counts writes landing on part of a wide glyph."""

    def __init__(self, columns: int) -> None:
        self.columns = columns
        self.rows: dict[int, list[object]] = {}
        self.x = 0
        self.y = 0
        self.autowrap = True
        self.stats: Counter[str] = Counter()

    def _row(self) -> list[object]:
        return self.rows.setdefault(self.y, [" "] * self.columns)

    def _put(self, ch: str, width: int) -> None:
        row = self._row()
        if self.x + width > self.columns:
            self.stats["overflow"] += 1
            return
        end = self.x + width
        if row[self.x] is _RIGHT_HALF:
            self.stats["split"] += 1
            row[self.x - 1] = " "
        if end < self.columns and row[end] is _RIGHT_HALF:
            self.stats["split"] += 1
            row[end] = " "
        if width == 2 and row[self.x + 1] is not _RIGHT_HALF and wcwidth(str(row[self.x + 1])) == 2:
            self.stats["split"] += 1
        row[self.x] = ch
        if width == 2:
            row[self.x + 1] = _RIGHT_HALF
        self.x = end if self.autowrap else min(self.columns - 1, end)

    def feed(self, data: str) -> None:
        i = 0
        while i < len(data):
            ch = data[i]
            if ch == "\x1b":
                match = _CSI.match(data, i)
                if match is None:
                    i += 2
                    continue
                self._csi(*match.groups())
                i = match.end()
                continue
            if ch == "\r":
                self.x = 0
            elif ch == "\n":
                self.y += 1
            elif ch == "\b":
                self.x = max(0, self.x - 1)
            elif ch >= " ":
                width = max(0, wcwidth(ch))
                if width:
                    if self.autowrap and self.x + width > self.columns:
                        self.x, self.y = 0, self.y + 1
                    self._put(ch, width)
            i += 1

    def _csi(self, params: str, _intermediate: str, final: str) -> None:
        amount = int(params) if params.isdigit() else 1
        if params == "?7l":
            self.autowrap = False
        elif params == "?7h":
            self.autowrap = True
        elif final == "A":
            self.y -= amount
        elif final == "B":
            self.y += amount
        elif final == "C":
            self.x = min(self.columns - 1, self.x + amount)
        elif final == "D":
            self.x = max(0, self.x - amount)
        elif final == "K":
            row = self._row()
            if row[self.x] is _RIGHT_HALF:
                self.stats["split"] += 1
            for col in range(self.x, self.columns):
                row[col] = " "
        elif final == "J":
            for y in [y for y in self.rows if y > self.y]:
                del self.rows[y]
            row = self._row()
            for col in range(self.x, self.columns):
                row[col] = " "

    def text(self, y: int) -> str:
        row = self.rows.get(y, [])
        return "".join("" if cell is _RIGHT_HALF else str(cell) for cell in row).rstrip()


class _Sink:
    encoding = "utf-8"

    def __init__(self) -> None:
        self.chunks: list[str] = []

    def write(self, data: str) -> None:
        self.chunks.append(data)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:  # pragma: no cover - never used by the test output
        raise OSError("no fd")


async def _edit_draft_mid_line(columns: int) -> tuple[_TerminalModel, list[str]]:
    """Paste a CJK draft, then edit inside it like a user fixing a typo."""
    sink = _Sink()
    output = Vt100_Output(sink, lambda: Size(rows=24, columns=columns), term="xterm-256color", enable_cpr=False)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        area = TextArea(multiline=True, wrap_lines=True)
        app: Application[None] = Application(layout=Layout(area), full_screen=False)
        task = asyncio.create_task(app.run_async())
        await asyncio.sleep(0.05)
        pipe.send_text("\x1b[200~" + _DRAFT + "\x1b[201~")
        await asyncio.sleep(0.05)
        for key in ["\x1b[D"] * 25 + ["a", "机", "\x7f", "b", "；", "\x1b[D", "c"] + ["\x7f"] * 3:
            pipe.send_text(key)
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        screen = app.renderer._last_screen
        assert screen is not None
        expected = [
            "".join(screen.data_buffer[y][x].char for x in range(columns)).rstrip()
            for y in range(screen.height)
        ]
        model = _TerminalModel(columns)
        model.feed("".join(sink.chunks))
        app.exit()
        await task
    return model, expected


@pytest.fixture
def unguarded_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    current = renderer_module._output_screen_diff
    monkeypatch.setattr(renderer_module, "_output_screen_diff", getattr(current, "__wrapped__", current))


@pytest.fixture
def guarded_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    current = renderer_module._output_screen_diff
    monkeypatch.setattr(renderer_module, "_output_screen_diff", getattr(current, "__wrapped__", current))
    assert install_render_guard() is True


@pytest.mark.parametrize("guarded", [False, True], ids=["hazard-control", "guarded"])
async def test_unchanged_redraw_emits_cursor_toggles_only_without_guard(unguarded_renderer, guarded: bool) -> None:
    from tests.test_tui_command_queue import _wait_for

    if guarded:
        assert install_render_guard()
    sink = _Sink()
    output = Vt100_Output(sink, lambda: Size(rows=24, columns=80), term="xterm-256color", enable_cpr=False)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        app = Application(layout=Layout(TextArea(text="中文 draft", multiline=True)))
        task = asyncio.create_task(app.run_async())
        try:
            await _wait_for(lambda: app.renderer._last_screen is not None)
            baseline = len(sink.chunks)
            app._redraw()
            emitted = "".join(sink.chunks[baseline:])
            if guarded:
                assert emitted == ""
            else:
                assert "\x1b[?25l" in emitted
                assert "\x1b[?25h" in emitted
        finally:
            if app.is_running:
                app.exit()
            await task


@pytest.mark.parametrize("columns", [40, 61, 80])
async def test_unguarded_redraw_splits_wide_glyphs(unguarded_renderer: None, columns: int) -> None:
    # Documents the hazard: plain prompt_toolkit overwrites halves of CJK glyphs.
    model, _expected = await _edit_draft_mid_line(columns)
    assert model.stats["split"] > 0


@pytest.mark.parametrize("columns", [40, 61, 80])
async def test_guarded_redraw_never_splits_and_matches_screen(guarded_renderer: None, columns: int) -> None:
    model, expected = await _edit_draft_mid_line(columns)

    assert model.stats["split"] == 0
    assert model.stats["overflow"] == 0
    assert [model.text(y) for y in range(len(expected))] == expected


def _row_screen(rows: dict[int, str], width: int) -> Screen:
    screen = Screen()
    empty = _CHAR_CACHE["", ""]
    for y, text in rows.items():
        x = 0
        for ch in text:
            screen.data_buffer[y][x] = _CHAR_CACHE[ch, ""]
            if wcwidth(ch) == 2:
                screen.data_buffer[y][x + 1] = empty
                x += 2
            else:
                x += 1
        screen.height = max(screen.height, y + 1)
    return screen


def test_blank_undrawable_wide_glyphs_handles_last_column_and_covered_halves() -> None:
    screen = _row_screen({0: "abc中", 1: "中文"}, 5)
    screen.data_buffer[1][1] = _CHAR_CACHE["x", ""]  # a float covered the right half

    replaced = blank_undrawable_wide_glyphs(screen, Size(rows=5, columns=4))

    assert replaced == 2
    assert screen.data_buffer[0][3].char == " "  # wide glyph in the last column
    assert screen.data_buffer[1][0].char == " "  # glyph without its right half
    assert screen.data_buffer[1][2].char == "文"  # intact glyph untouched


def test_blank_undrawable_wide_glyphs_blanks_orphaned_right_half() -> None:
    screen = _row_screen({0: "中文"}, 4)
    screen.data_buffer[0][0] = _CHAR_CACHE["x", ""]  # a float covered the left half

    blank_undrawable_wide_glyphs(screen, Size(rows=2, columns=8))

    assert screen.data_buffer[0][1].char == " "
    assert screen.data_buffer[0][2].char == "文"


class _RecordingOutput:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def __getattr__(self, name: str):
        def record(*args: object) -> None:
            self.calls.append((name, args[0] if args else None))

        return record


def test_erase_misaligned_wide_tails_erases_from_first_changed_boundary() -> None:
    previous = _row_screen({0: "ab中文字", 1: "plain row", 2: "中文"}, 20)
    screen = _row_screen({0: "ab中x文字", 1: "plain rows", 2: "中文"}, 20)
    output = _RecordingOutput()

    pos, erased = erase_misaligned_wide_tails(
        output, screen, previous, Point(x=0, y=3), Size(rows=10, columns=20)  # type: ignore[arg-type]
    )

    assert erased == 1  # only the row whose old tail holds wide glyphs
    assert pos == Point(x=4, y=0)
    assert ("cursor_up", 3) in output.calls
    assert ("cursor_forward", 4) in output.calls
    assert ("erase_end_of_line", None) in output.calls
    assert sorted(previous.data_buffer[0]) == [0, 1, 2, 3]  # tail now blank in the model
    assert previous.data_buffer[2][0].char == "中"  # unchanged row untouched


def test_install_is_idempotent_and_refuses_unknown_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    current = renderer_module._output_screen_diff
    original = getattr(current, "__wrapped__", current)
    monkeypatch.setattr(renderer_module, "_output_screen_diff", original)

    assert install_render_guard() is True
    guarded = renderer_module._output_screen_diff
    assert install_render_guard() is True
    assert renderer_module._output_screen_diff is guarded
    assert guarded.__wrapped__ is original

    def unexpected(app, output, screen):  # noqa: ANN001, ANN202
        return None

    monkeypatch.setattr(renderer_module, "_output_screen_diff", unexpected)
    assert install_render_guard() is False
    assert renderer_module._output_screen_diff is unexpected


def test_leap_app_installs_guard_before_first_render(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_tui_command_queue import _make_app

    current = renderer_module._output_screen_diff
    monkeypatch.setattr(renderer_module, "_output_screen_diff", getattr(current, "__wrapped__", current))

    _make_app()

    assert getattr(renderer_module._output_screen_diff, render_guard._GUARD_MARKER, False)


def test_leap_app_refuses_to_render_without_safety_guard(monkeypatch) -> None:
    from leapflow.cli.tui_app import app as app_module
    from tests.test_tui_command_queue import _make_app

    monkeypatch.setattr(app_module, "install_render_guard", lambda: False)
    with pytest.raises(RuntimeError, match="Cannot start the TUI safely"):
        _make_app()


def test_display_clock_holds_and_catches_up_without_changing_measurements() -> None:
    from leapflow.cli.tui_app.display_clock import DisplayClock
    from leapflow.cli.tui_app.status import StatusBar

    state = SimpleNamespace(time=100.0, hold=False)
    clock = DisplayClock(lambda: state.hold, source=lambda: state.time)
    status = StatusBar()
    status._session_start = state.time
    status.use_display_clock(clock.now)
    state.time = 105.0
    before = status()
    state.hold = True
    assert clock.now() == 105.0
    state.time = 165.0
    assert status() == before
    status.update(context_used=100, context_max=1000)
    assert status() != before  # Real metrics must not freeze with the clock.
    assert clock.now() == 105.0
    state.hold = False
    assert clock.now() == 165.0
    assert "1m" in "".join(text for _, text in status())
    state.hold = True
    assert clock.now() == 165.0  # A fresh draft gets a fresh snapshot.


@pytest.mark.parametrize("change", ["text", "style", "cursor", "visibility", "height", "escape"])
def test_frame_comparison_preserves_real_changes(change: str) -> None:
    previous = _row_screen({0: "中文 abc"}, 20)
    screen = _row_screen({0: "中文 abc"}, 20)
    size = Size(rows=24, columns=20)
    current = target = Point(x=7, y=0)
    assert frame_unchanged(screen, previous, target, current, size)
    if change == "text":
        screen.data_buffer[0][5] = _CHAR_CACHE["x", ""]
    elif change == "style":
        screen.data_buffer[0][5] = _CHAR_CACHE["a", "bold"]
    elif change == "cursor":
        target = Point(x=6, y=0)
    elif change == "visibility":
        screen.show_cursor = False
    elif change == "height":
        screen.height += 1
    else:
        screen.zero_width_escapes[0][5] = "\x1b]8;;https://example.com\x1b\\"
    assert not frame_unchanged(screen, previous, target, current, size)


def test_frame_comparison_handles_sparse_defaults_without_mutation() -> None:
    blank = _CHAR_CACHE[" ", "bold"]
    previous, screen = Screen(default_char=blank), Screen(default_char=blank)
    previous.height = screen.height = 1
    previous.data_buffer[0][0] = blank
    assert frame_unchanged(screen, previous, Point(0, 0), Point(0, 0), Size(rows=24, columns=20))
    assert not screen.data_buffer
    assert not screen.zero_width_escapes


@pytest.mark.parametrize("is_done,previous_width,has_previous", [(True, 20, True), (False, 19, True), (False, 20, False)])
def test_guard_does_not_skip_exit_resize_or_initial_render(is_done, previous_width, has_previous) -> None:
    original = Mock(return_value=(Point(0, 0), None))
    previous = _row_screen({0: "abc"}, 20)
    screen = _row_screen({0: "abc"}, 20)
    window = object()
    app = SimpleNamespace(layout=SimpleNamespace(current_window=window))
    screen.set_cursor_position(window, Point(0, 0))
    render_guard._wrap(original)(
        app, _RecordingOutput(), screen, Point(0, 0), None,
        previous if has_previous else None, None, is_done, False,
        {}, {}, Size(rows=24, columns=20), previous_width,
    )
    original.assert_called_once()


@pytest.mark.parametrize("columns", [40, 61, 80, 120])
@pytest.mark.parametrize(
    "running,draft",
    [(False, ""), (False, _DRAFT * 16), (True, _DRAFT * 16), (True, "")],
    ids=["idle-empty", "idle-draft", "running-draft", "running-empty"],
)
async def test_real_tui_is_quiet_without_events_and_keeps_input_responsive(
    tmp_path, guarded_renderer, columns: int, running: bool, draft: str,
) -> None:
    from leapflow.cli.tui_app.app import LeapApp
    from leapflow.cli.tui_app.status import StatusBar
    from leapflow.cli.tui_app.theme import _LIGHT, resolve_theme
    from tests.test_tui_command_queue import _FakeConsole, _wait_for

    size = Size(rows=48, columns=columns)
    sink = _Sink()
    output = Vt100_Output(sink, lambda: size, term="xterm-256color", enable_cpr=False)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        status = StatusBar()
        app = LeapApp(console=_FakeConsole(), theme=resolve_theme(_LIGHT, terminal_bg="#FFFFFF"), status=status, history_path=tmp_path / "history")
        app._agent_running = running
        if running:
            app.spinner_text = "Waiting for tool"
        task = asyncio.create_task(app._app.run_async())
        try:
            await _wait_for(lambda: app._app.renderer._last_screen is not None)
            assert app._app.refresh_interval is None
            # Unbracketed phrase commits. Simulate pauses between phrases without
            # spending human typing time in CI; bulk paste is tested separately.
            for offset in range(0, len(draft), 29):
                app._reset_paste_fragment_window()
                pipe.send_text(draft[offset:offset + 29])
                expected_text = draft[:offset + 29]
                await _wait_for(lambda: app._input_area.buffer.text == expected_text)
            assert not app._paste_store.has_blocks
            await asyncio.sleep(0.05)  # Drain suggestion/completion invalidations.
            baseline = "".join(sink.chunks)
            await asyncio.sleep(1.05)  # More than two of the old refresh ticks.
            assert "".join(sink.chunks) == baseline
            if draft or not running:
                for _ in range(5):
                    app._app._redraw()  # Redundant notifications must emit zero bytes.
                assert "".join(sink.chunks) == baseline
            # With an empty running draft, an explicit redraw can still update
            # elapsed status/spinner text off the input row. There is no timer.
            status.update(context_used=123, context_max=1000)
            app.invalidate()
            await _wait_for(lambda: "".join(sink.chunks) != baseline)
            assert app._input_area.buffer.text == draft
            pipe.send_text("a")
            await _wait_for(lambda: app._input_area.buffer.text == draft + "a")
            pipe.send_text("\x1b[D中\x7f")  # Move and edit inside the long draft.
            await _wait_for(lambda: app._input_area.buffer.cursor_position == len(draft))
            await asyncio.sleep(0.05)
            assert app._input_area.buffer.text == draft + "a"
            model = _TerminalModel(columns)
            model.feed("".join(sink.chunks))
            assert model.stats["split"] == model.stats["overflow"] == 0
            screen = app._app.renderer._last_screen
            assert screen is not None
            expected = ["".join(screen.data_buffer[y][x].char for x in range(columns)).rstrip() for y in range(screen.height)]
            assert [model.text(y) for y in range(len(expected))] == expected
            baseline = "".join(sink.chunks)
            size = Size(rows=12, columns=max(10, columns // 2))
            app._app._redraw()
            assert "".join(sink.chunks) != baseline
            assert app._app.renderer._last_size == size
            assert app._input_area.buffer.text == draft + "a"
            assert app._input_area.buffer.cursor_position == len(draft)
        finally:
            if app._app.is_running:
                app._app.exit()
            await task


async def test_wrapped_cjk_overflow_keeps_cursor_in_input(monkeypatch) -> None:
    from tests.test_tui_command_queue import _make_app, _wait_for

    monkeypatch.setenv("TERM_PROGRAM", "xterm")
    size = Size(rows=24, columns=19)
    output = Vt100_Output(_Sink(), lambda: size, term="xterm-256color", enable_cpr=False)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        app, _, _ = _make_app()
        app.agent_running = True
        task = asyncio.create_task(app._app.run_async())
        try:
            await _wait_for(lambda: app._app.renderer._last_screen is not None)
            # Each full CJK row leaves a cell unused at this odd width. Counting
            # the total cells / width understates the actual wrapped height.
            draft = "a" + "中" * 36
            for phrase in (draft[:17], draft[17:]):
                app._reset_paste_fragment_window()
                pipe.send_text(phrase)
                await _wait_for(lambda: app._input_area.buffer.text.endswith(phrase))
            app._app._redraw()
            assert app._input_area.buffer.text == draft
            screen = app._app.renderer._last_screen
            window = app._input_area.window
            region = screen.visible_windows_to_write_positions[window]
            cursor = screen.get_cursor_position(window)
            assert region.ypos <= cursor.y < region.ypos + region.height
            assert cursor.x == 2  # One final CJK glyph on the newly visible row.
        finally:
            if app._app.is_running:
                app._app.exit()
            await task


@pytest.mark.parametrize("apple_terminal", [False, True], ids=["hazard-control", "apple-terminal"])
async def test_native_only_wide_cell_is_erased_before_rewriting_cursor_row(
    monkeypatch, guarded_renderer, apple_terminal: bool,
) -> None:
    from tests.test_tui_command_queue import _wait_for

    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal" if apple_terminal else "xterm")
    sink = _Sink()
    output = Vt100_Output(sink, lambda: Size(rows=24, columns=40), term="xterm-256color", enable_cpr=False)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        area = TextArea(text="a", multiline=True)
        area.buffer.cursor_position = 1
        app = Application(layout=Layout(area))
        task = asyncio.create_task(app.run_async())
        try:
            await _wait_for(lambda: app.renderer._last_screen is not None)
            terminal = _TerminalModel(40)
            terminal.feed("".join(sink.chunks))
            assert terminal.text(0) == "a" and (terminal.x, terminal.y) == (1, 0)
            # Model a wide native marked glyph absent from the old Screen.
            terminal.feed("中\b\b")
            start = len(sink.chunks)
            pipe.send_text("b")
            await _wait_for(lambda: area.buffer.text == "ab")
            app._redraw()
            terminal.feed("".join(sink.chunks[start:]))
            if apple_terminal:
                assert terminal.stats["split"] == 0
                assert terminal.text(0) == "ab"
            else:
                assert terminal.stats["split"] > 0
        finally:
            if app.is_running:
                app.exit()
            await task


@pytest.mark.parametrize("running", [False, True], ids=["idle", "streaming"])
async def test_apple_terminal_overflow_keeps_cursor_clear_and_submits_full_draft(monkeypatch, running: bool) -> None:
    from tests.test_tui_command_queue import _make_app, _wait_for

    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    sink = _Sink()
    size = Size(rows=16, columns=37)
    output = Vt100_Output(sink, lambda: size, term="xterm-256color", enable_cpr=False)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        app, _, _ = _make_app()
        app.agent_running = running
        task = asyncio.create_task(app._app.run_async())
        try:
            await _wait_for(lambda: app._app.renderer._last_screen is not None)
            window = app._input_area.window
            app._reset_paste_fragment_window()
            parts = ["a中b文"] * 40
            draft = ""
            full_region = None
            for part in parts:
                draft += part
                pipe.send_text(part)
                await _wait_for(lambda: window.content.buffer.text == draft)
                app._app._redraw()
                screen = app._app.renderer._last_screen
                region = screen.visible_windows_to_write_positions[window]
                cursor = screen.get_cursor_position(window)
                assert region.ypos <= cursor.y < region.ypos + region.height
                assert 0 <= cursor.x < region.width == 28  # Nine cells of IME headroom.
                if region.height == app._effective_input_cap(size.rows):
                    if full_region is None:
                        assert app._input_hidden_rows() == 0
                        full_region = region
                    assert (region.ypos, region.height) == (full_region.ypos, full_region.height)
            assert full_region is not None and app._input_hidden_rows() > 0
            assert not app._paste_store.has_blocks
            model = _TerminalModel(size.columns)
            model.feed("".join(sink.chunks))
            assert model.stats["split"] == model.stats["overflow"] == 0
            assert Point(model.x, model.y) == cursor
            pipe.send_text("\x1b[D\x7f中")
            edited = draft[:-2] + "中文"
            await _wait_for(lambda: window.content.buffer.text == edited)
            pipe.send_text("\r")
            await _wait_for(lambda: len(app.queued_commands()) == 1)
            assert app.queued_commands()[0].text == edited
        finally:
            if app._app.is_running:
                app._app.exit()
            await task


async def test_two_tuis_in_separate_workspaces_keep_display_and_draft_isolated(tmp_path, guarded_renderer) -> None:
    from contextlib import ExitStack

    from leapflow.cli.tui_app.app import LeapApp
    from leapflow.cli.tui_app.status import StatusBar
    from leapflow.cli.tui_app.theme import _LIGHT, resolve_theme
    from tests.test_tui_command_queue import _FakeConsole, _wait_for

    apps, tasks, pipes, sinks, statuses = [], [], [], [], []
    with ExitStack() as stack:
        for index in range(2):
            pipe = stack.enter_context(create_pipe_input())
            sink = _Sink()
            output = Vt100_Output(sink, lambda: Size(rows=24, columns=80), term="xterm-256color", enable_cpr=False)
            with create_app_session(input=pipe, output=output):
                status = StatusBar()
                app = LeapApp(console=_FakeConsole(), theme=resolve_theme(_LIGHT, terminal_bg="#FFFFFF"), status=status, history_path=tmp_path / f"workspace-{index}" / "history")
                tasks.append(asyncio.create_task(app._app.run_async()))
            apps.append(app)
            pipes.append(pipe)
            sinks.append(sink)
            statuses.append(status)
        try:
            await _wait_for(lambda: all(app._app.renderer._last_screen is not None for app in apps))
            pipes[0].send_text("first draft 中文")
            await _wait_for(lambda: apps[0]._input_area.buffer.text == "first draft 中文")
            await asyncio.sleep(0.05)
            other_output = "".join(sinks[1].chunks)
            statuses[0].update(context_used=222, context_max=1000)
            baseline = "".join(sinks[0].chunks)
            apps[0].invalidate()
            await _wait_for(lambda: "".join(sinks[0].chunks) != baseline)
            assert "".join(sinks[1].chunks) == other_output
            assert statuses[1].context_used == 0
            assert apps[1]._input_area.buffer.text == ""
            assert apps[0]._display_clock is not apps[1]._display_clock
            pipes[1].send_text("second draft")
            await _wait_for(lambda: apps[1]._input_area.buffer.text == "second draft")
            assert apps[0]._input_area.buffer.text == "first draft 中文"
        finally:
            for app in apps:
                if app._app.is_running:
                    app._app.exit()
            await asyncio.gather(*tasks)


@pytest.mark.parametrize(
    "line",
    [
        "/config llm set --api-key sk-abcdefghijklmnop",
        "/config llm set --api_key=abc123456",
        "/gateway set --app-secret s3cr3tvalue",
        "/login --password hunter22",
        "export OPENAI_API_KEY=sk-abcdefghijklmnop",
    ],
)
def test_contains_secret_detects_credentials(line: str) -> None:
    assert contains_secret(line)


@pytest.mark.parametrize(
    "line",
    [
        "/config llm set --model qwen3-plus",
        "帮我把相机节点切换到 leap hw serve",
        "explain how token budgets work",
        "/help",
    ],
)
def test_contains_secret_keeps_ordinary_input(line: str) -> None:
    assert not contains_secret(line)


def test_secret_aware_history_never_persists_or_recalls_secrets(tmp_path) -> None:
    path = tmp_path / "history"
    path.write_text("\n# old\n+/config llm set --api-key sk-abcdefghijklmnop\n\n# old\n+/status\n", encoding="utf-8")
    history = SecretAwareFileHistory(str(path))

    assert list(history.load_history_strings()) == ["/status"]

    history.append_string("/config llm set --api-key sk-zyxwvutsrqponmlk")
    history.append_string("/help")

    stored = path.read_text(encoding="utf-8")
    assert "sk-zyxwvutsrqponmlk" not in stored
    assert "+/help" in stored
    assert "sk-zyxwvutsrqponmlk" not in "".join(history.get_strings())
