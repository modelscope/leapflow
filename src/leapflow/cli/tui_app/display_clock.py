# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hold cosmetic TUI clocks during idle periods and draft editing.

Elapsed displays are sampled on real render events, not a repaint timer.
They hold while idle or drafting so redundant invalidations cannot move the
cursor solely to change a timestamp. macOS Terminal.app's marked-text crash
implicates such cursor updates during IME composition (see ``render_guard``).
Turn measurements still use the real clock, and task/status/approval changes
remain visible. Only the cosmetic display time holds and catches up when
execution resumes without a draft.
"""

from __future__ import annotations

import time
from typing import Callable, Optional


class DisplayClock:
    """Monotonic clock whose reading holds while ``hold_while()`` is true."""

    def __init__(
        self,
        hold_while: Callable[[], bool],
        source: Callable[[], float] = time.monotonic,
    ) -> None:
        self._hold_while = hold_while
        self._source = source
        self._held_at: Optional[float] = None

    def now(self) -> float:
        """Return the display time: frozen at the first reading of a hold."""
        if not self._hold_while():
            self._held_at = None
            return self._source()
        if self._held_at is None:
            self._held_at = self._source()
        return self._held_at
