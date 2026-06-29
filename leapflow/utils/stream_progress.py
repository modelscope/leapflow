"""Terminal stream progress writer for LLM chunk callbacks.

Renders LLM streaming output as dim gray text on stdout, giving the user
visual confirmation that the LLM is generating.
"""

from __future__ import annotations

import sys
from typing import TextIO


class StreamProgressWriter:
    """Prints LLM stream deltas in ANSI dim.

    Implements the ChunkCallback protocol: callable with a single str argument.
    Handles line wrapping at max_width and prefixes each new line.

    Usage::

        writer = StreamProgressWriter()
        resp = await llm.achat(messages, stream=True, on_chunk=writer)
        writer.finish()
    """

    _DIM = "\033[2m"
    _RESET = "\033[0m"

    def __init__(
        self,
        *,
        prefix: str = "    │ ",
        max_width: int = 72,
        file: TextIO = sys.stdout,
    ) -> None:
        self._prefix = prefix
        self._max_width = max_width
        self._file = file
        self._col = 0
        self._started = False

    def __call__(self, chunk: str) -> None:
        if not chunk:
            return
        if not self._started:
            self._file.write(self._DIM)
            self._file.write(self._prefix)
            self._col = len(self._prefix)
            self._started = True

        for ch in chunk:
            if ch == "\n":
                self._file.write("\n")
                self._file.write(self._prefix)
                self._col = len(self._prefix)
            else:
                if self._col >= self._max_width:
                    self._file.write("\n")
                    self._file.write(self._prefix)
                    self._col = len(self._prefix)
                self._file.write(ch)
                self._col += 1

        self._file.flush()

    def finish(self) -> None:
        """End the dim output block and reset terminal attributes."""
        if self._started:
            self._file.write(self._RESET)
            self._file.write("\n")
            self._file.flush()
            self._started = False
            self._col = 0
