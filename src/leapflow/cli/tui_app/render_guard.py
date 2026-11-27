# Copyright (c) Alibaba, Inc. and its affiliates.
"""Terminal-safe screen diffing for prompt_toolkit's renderer.

The guard wrapped around prompt_toolkit's screen diff enforces two
guarantees: no wide-glyph splitting and no output for unchanged frames.

Unchanged frames emit nothing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
prompt_toolkit re-emits cursor hide/show, autowrap and attribute resets on
every render, even when no cell changed. LeapFlow previously requested a
redraw every 0.5 seconds. These cursor changes can interfere with an IME. An
IME composition (pinyin and other input methods) lives only inside the
terminal, so the program cannot see it. Terminal stores the marked text
inline in its line buffer at the cursor. The crash logs show an exception
in ``setMarkedText``: ``TTBufferRemoveObjectsInRange ... Range is out of
bounds``, followed by a NULL dereference during drawing. Earlier exceptions
recurred at the refresh cadence. This implicates program-driven cursor/line
updates under an open composition; the exact native race is not observable
through a pipe-input test. LeapFlow no longer redraws on a timer, and an
identical frame writes no bytes at all. Cosmetic clocks hold during drafts
and idle periods (see ``display_clock``). Real output and approval changes
still render; this guard does not claim to detect terminal-local IME state.

No write lands on part of a wide glyph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
prompt_toolkit repaints a non-fullscreen Application by diffing the new
screen against the previous one cell by cell and overwriting only the cells
that changed. When a draft mixes double-width (CJK) and narrow glyphs, an
edit in the middle of a line shifts every later glyph by one cell, so the
diff overwrites the left or right half of glyphs that are still on screen.

The terminal must then *split* each such glyph (replace it with two blanks
before writing). macOS Terminal.app's split routine miscomputes the tail
length in its UTF-8 line storage and ``memmove``s a negative length off the
heap, killing every Terminal window (SIGBUS in ``_platform_memmove`` called
from its tty output handler). Appending is not inherently safe: overflow
scrolls existing rows, and Terminal's uncommitted IME text can extend beyond
what the previous Screen knows about.

The guard removes the operation altogether. Before prompt_toolkit's own
diff runs, every row whose previous content holds a double-width glyph at or
after its first changed cell is erased from that cell to the end of line
(the first changed cell is always a glyph boundary), and the previous screen
is updated to match. The original diff then appends the new tail into empty
cells, so no write ever lands on part of an existing wide glyph. It also
blanks any wide glyph whose right half cannot be drawn (last column, or
covered by a float), which would otherwise leave the terminal and the
renderer disagreeing about the row.

On Apple Terminal, clear a changed previous cursor row from column zero,
including when the old Screen has no wide cells. Erasing from an interior
column assumes that column is also a glyph boundary in Terminal's untracked
marked text. The input viewport also leaves right-margin composition space
on that terminal. Neither mitigation can observe native IME state.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Callable

from prompt_toolkit.data_structures import Point, Size
from prompt_toolkit.layout.screen import _CHAR_CACHE, Char, Screen
from prompt_toolkit.output import Output

logger = logging.getLogger(__name__)

_GUARD_MARKER = "_leapflow_render_guard"
_EXPECTED_PARAMS = (
    "app",
    "output",
    "screen",
    "current_pos",
    "color_depth",
    "previous_screen",
    "last_style",
    "is_done",
    "full_screen",
    "attrs_for_style_string",
    "style_string_has_style",
    "size",
    "previous_width",
)


def uses_apple_terminal() -> bool:
    """Whether the host terminal advertises Apple's Terminal.app."""
    return os.environ.get("TERM_PROGRAM") == "Apple_Terminal"


def blank_undrawable_wide_glyphs(screen: Screen, size: Size) -> int:
    """Blank wide glyphs that cannot be drawn whole, and orphaned right halves.

    A wide glyph in the last column, or whose right half a float covered,
    and a right-half placeholder whose glyph was covered, would each leave
    the terminal cursor and the renderer's model one cell apart. Returns the
    number of cells replaced.
    """
    width = size.columns
    replaced = 0
    for y, row in screen.data_buffer.items():
        if y >= size.rows:
            continue
        for x, char in sorted(row.items()):
            if x >= width:
                break
            if char.width > 1:
                right = row.get(x + 1)
                if x + 1 < width and right is not None and right.char == "":
                    continue
            elif char.char == "":
                left = row.get(x - 1)
                if left is not None and left.width > 1:
                    continue
            else:
                continue
            row[x] = _CHAR_CACHE[" ", char.style]
            replaced += 1
    return replaced


def _rows_equal(screen: Screen, previous_screen: Screen, y: int, width: int) -> bool:
    new_row = screen.data_buffer.get(y)
    old_row = previous_screen.data_buffer.get(y)
    # Read each Screen's own default without inserting cells into either one.
    new_default_row = screen.data_buffer.default_factory()
    old_default_row = previous_screen.data_buffer.default_factory()
    new_blank = new_default_row.default_factory()
    old_blank = old_default_row.default_factory()
    new_row = new_row if new_row is not None else {}
    old_row = old_row if old_row is not None else {}
    for x in set(new_row) | set(old_row):
        if x >= width:
            continue
        new_char = new_row.get(x, new_blank)
        old_char = old_row.get(x, old_blank)
        if new_char.char != old_char.char or new_char.style != old_char.style:
            return False
    new_escapes = screen.zero_width_escapes.get(y) or {}
    old_escapes = previous_screen.zero_width_escapes.get(y) or {}
    return new_escapes == old_escapes


def frame_unchanged(
    screen: Screen,
    previous_screen: Screen,
    cursor_target: Point,
    current_pos: Point,
    size: Size,
) -> bool:
    """True when the terminal already shows ``screen`` with the cursor in place.

    Such a repaint would only re-emit cursor and attribute toggles, which can
    invalidate IME composition on affected terminals (see the module docstring).
    Uses ``.get`` throughout because the screens' buffers are defaultdicts.
    """
    if screen.show_cursor != previous_screen.show_cursor:
        return False
    if cursor_target != current_pos:
        return False
    if screen.height != previous_screen.height:
        return False  # Preserve space reservation and trimming.
    rows = min(max(screen.height, previous_screen.height), size.rows)
    return all(_rows_equal(screen, previous_screen, y, size.columns) for y in range(rows))


def _first_changed_column(old_row: dict[int, Char], new_row: dict[int, Char], blank: Char) -> int | None:
    columns = set(old_row) | set(new_row)
    changed = [
        x for x in columns
        if old_row.get(x, blank) != new_row.get(x, blank)
    ]
    if not changed:
        return None
    first = min(changed)
    # A glyph boundary in both screens: never start inside a wide glyph.
    if first > 0 and (
        old_row.get(first - 1, blank).width > 1 or new_row.get(first - 1, blank).width > 1
    ):
        first -= 1
    return first


def _move_to(output: Output, current: Point, target: Point) -> Point:
    """Move within rows that are already drawn (no scrolling, no writes)."""
    if target.y > current.y:
        output.cursor_down(target.y - current.y)
    elif target.y < current.y:
        output.cursor_up(current.y - target.y)
    output.write_raw("\r")
    if target.x:
        output.cursor_forward(target.x)
    return target


def erase_misaligned_wide_tails(
    output: Output,
    screen: Screen,
    previous_screen: Screen,
    current_pos: Point,
    size: Size,
    *,
    protect_marked_text: bool = False,
) -> tuple[Point, int]:
    """Erase changed wide tails and, when requested, the changed IME row.

    Mutates ``previous_screen`` so prompt_toolkit's diff sees those cells as
    blank and repaints them by appending. Returns the cursor position and the
    number of rows erased.
    """
    blank = _CHAR_CACHE[" ", ""]
    ime_row = current_pos.y if protect_marked_text and previous_screen.show_cursor else None
    erased = 0
    for y in range(min(previous_screen.height, size.rows)):
        old_row = previous_screen.data_buffer.get(y, {})
        if y == ime_row and not _rows_equal(screen, previous_screen, y, size.columns):
            first = 0
        else:
            if not any(char.width > 1 for char in old_row.values()):
                continue
            new_row = screen.data_buffer.get(y) or {}
            first = _first_changed_column(old_row, new_row, blank)
            if first is None or first >= size.columns:
                continue
            if not any(x >= first and char.width > 1 for x, char in old_row.items()):
                continue
        if erased == 0:
            output.hide_cursor()
            output.reset_attributes()
        current_pos = _move_to(output, current_pos, Point(x=first, y=y))
        output.erase_end_of_line()
        for x in [x for x in old_row if x >= first]:
            del old_row[x]
        erased += 1
    return current_pos, erased


def _wrap(original: Callable[..., tuple[Point, str | None]]) -> Callable[..., tuple[Point, str | None]]:
    def guarded_output_screen_diff(
        app: Any,
        output: Output,
        screen: Screen,
        current_pos: Point,
        color_depth: Any,
        previous_screen: Screen | None,
        last_style: str | None,
        is_done: bool,
        full_screen: bool,
        attrs_for_style_string: Any,
        style_string_has_style: Any,
        size: Size,
        previous_width: int,
    ) -> tuple[Point, str | None]:
        blank_undrawable_wide_glyphs(screen, size)
        if previous_screen is not None and not is_done and previous_width == size.columns:
            cursor_target = screen.get_cursor_position(app.layout.current_window)
            if frame_unchanged(screen, previous_screen, cursor_target, current_pos, size):
                return current_pos, last_style
            current_pos, erased = erase_misaligned_wide_tails(
                output, screen, previous_screen, current_pos, size,
                protect_marked_text=uses_apple_terminal(),
            )
            if erased:
                last_style = None
        return original(
            app,
            output,
            screen,
            current_pos,
            color_depth,
            previous_screen,
            last_style,
            is_done,
            full_screen=full_screen,
            attrs_for_style_string=attrs_for_style_string,
            style_string_has_style=style_string_has_style,
            size=size,
            previous_width=previous_width,
        )

    setattr(guarded_output_screen_diff, _GUARD_MARKER, True)
    guarded_output_screen_diff.__wrapped__ = original  # type: ignore[attr-defined]
    return guarded_output_screen_diff


def install_render_guard() -> bool:
    """Install the guard into prompt_toolkit's renderer (idempotent).

    Returns False, leaving prompt_toolkit untouched, when the renderer's
    internals do not match the expected shape (e.g. a future release).
    """
    try:
        from prompt_toolkit import renderer as renderer_module
    except ImportError:
        return False
    current = getattr(renderer_module, "_output_screen_diff", None)
    if current is None:
        logger.warning("render guard not installed: prompt_toolkit renderer has no screen diff")
        return False
    if getattr(current, _GUARD_MARKER, False):
        return True
    try:
        params = tuple(inspect.signature(current).parameters)
    except (TypeError, ValueError):
        params = ()
    if params != _EXPECTED_PARAMS:
        logger.warning(
            "render guard not installed: unexpected prompt_toolkit screen diff signature %s",
            params,
        )
        return False
    renderer_module._output_screen_diff = _wrap(current)
    return True
