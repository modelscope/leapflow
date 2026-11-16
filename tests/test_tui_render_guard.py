# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression tests for the wide-glyph render guard and secret-aware history.

macOS Terminal.app crashed (SIGBUS in its wide-glyph split routine) when the
TUI redraw overwrote part of a CJK glyph still on screen. The guard must make
such overwrites impossible without changing what ends up on screen.
"""
from __future__ import annotations

import asyncio
import re
from collections import Counter

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
    install_wide_glyph_guard,
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
    assert install_wide_glyph_guard() is True


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

    assert install_wide_glyph_guard() is True
    guarded = renderer_module._output_screen_diff
    assert install_wide_glyph_guard() is True
    assert renderer_module._output_screen_diff is guarded
    assert guarded.__wrapped__ is original

    def unexpected(app, output, screen):  # noqa: ANN001, ANN202
        return None

    monkeypatch.setattr(renderer_module, "_output_screen_diff", unexpected)
    assert install_wide_glyph_guard() is False
    assert renderer_module._output_screen_diff is unexpected


def test_leap_app_installs_guard_before_first_render(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_tui_command_queue import _make_app

    current = renderer_module._output_screen_diff
    monkeypatch.setattr(renderer_module, "_output_screen_diff", getattr(current, "__wrapped__", current))

    _make_app()

    assert getattr(renderer_module._output_screen_diff, render_guard._GUARD_MARKER, False)


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
