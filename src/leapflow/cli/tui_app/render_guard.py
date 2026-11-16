# Copyright (c) Alibaba, Inc. and its affiliates.
"""Wide-glyph-safe screen diffing for prompt_toolkit's renderer.

prompt_toolkit repaints a non-fullscreen Application by diffing the new
screen against the previous one cell by cell and overwriting only the cells
that changed. When a draft mixes double-width (CJK) and narrow glyphs, an
edit in the middle of a line shifts every later glyph by one cell, so the
diff overwrites the left or right half of glyphs that are still on screen.

The terminal must then *split* each such glyph (replace it with two blanks
before writing). macOS Terminal.app's split routine miscomputes the tail
length in its UTF-8 line storage and ``memmove``s a negative length off the
heap, killing every Terminal window (SIGBUS in ``_platform_memmove`` called
from its tty output handler). Typing or pasting at the end of a draft never
splits a glyph; editing, history recall, and auto-suggest updates do so
hundreds of times per second.

The guard removes the operation altogether. Before prompt_toolkit's own
diff runs, every row whose previous content holds a double-width glyph at or
after its first changed cell is erased from that cell to the end of line
(the first changed cell is always a glyph boundary), and the previous screen
is updated to match. The original diff then appends the new tail into empty
cells, so no write ever lands on part of an existing wide glyph. It also
blanks any wide glyph whose right half cannot be drawn (last column, or
covered by a float), which would otherwise leave the terminal and the
renderer disagreeing about the row.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

from prompt_toolkit.data_structures import Point, Size
from prompt_toolkit.layout.screen import _CHAR_CACHE, Char, Screen
from prompt_toolkit.output import Output

logger = logging.getLogger(__name__)

_GUARD_MARKER = "_leapflow_wide_glyph_guard"
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
) -> tuple[Point, int]:
    """Erase changed row tails that still hold wide glyphs on the terminal.

    Mutates ``previous_screen`` so prompt_toolkit's diff sees those cells as
    blank and repaints them by appending. Returns the cursor position and the
    number of rows erased.
    """
    blank = _CHAR_CACHE[" ", ""]
    erased = 0
    for y in range(min(previous_screen.height, size.rows)):
        old_row = previous_screen.data_buffer.get(y)
        if not old_row or not any(char.width > 1 for char in old_row.values()):
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
            current_pos, erased = erase_misaligned_wide_tails(
                output, screen, previous_screen, current_pos, size
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


def install_wide_glyph_guard() -> bool:
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
        logger.warning("wide-glyph guard not installed: prompt_toolkit renderer has no screen diff")
        return False
    if getattr(current, _GUARD_MARKER, False):
        return True
    try:
        params = tuple(inspect.signature(current).parameters)
    except (TypeError, ValueError):
        params = ()
    if params != _EXPECTED_PARAMS:
        logger.warning(
            "wide-glyph guard not installed: unexpected prompt_toolkit screen diff signature %s",
            params,
        )
        return False
    renderer_module._output_screen_diff = _wrap(current)
    return True
