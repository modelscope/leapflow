# Copyright (c) Alibaba, Inc. and its affiliates.
"""Streaming ``<think>`` tag scrubber — prevents reasoning leakage to users.

The :class:`ThinkScrubber` is a lightweight state machine that processes
streamed text chunks and strips ``<think>...</think>`` blocks.  It handles
tags split across chunk boundaries and conservatively suppresses output when
an opening tag is seen without a matching close.

:class:`ScrubberSink` wraps any :class:`OutputSink` to apply scrubbing
transparently on the chunk/final output path.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, Optional

from leapflow.engine._stream_helpers import OutputSink


# ---------------------------------------------------------------------------
# Tag constants
# ---------------------------------------------------------------------------

_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class _State(enum.Enum):
    """Scrubber FSM states."""

    NORMAL = "normal"
    IN_THINK = "in_think"


class ThinkScrubber:
    """Character-level state machine that strips ``<think>…</think>`` blocks.

    Designed for **streaming** use: call :meth:`scrub` with each successive
    chunk and it returns only the safe-to-display portion.  Cross-chunk tag
    boundaries are handled by an internal look-ahead buffer.

    *Not* thread-safe — instantiate one per turn (which is the normal pattern
    since ``OutputSink`` is per-turn).
    """

    __slots__ = ("_state", "_buf")

    def __init__(self) -> None:
        self._state: _State = _State.NORMAL
        self._buf: str = ""

    # -- public API ----------------------------------------------------------

    def scrub(self, chunk: str) -> str:
        """Process *chunk* and return the cleaned text (may be empty)."""
        if not chunk:
            return ""
        out: list[str] = []
        for ch in chunk:
            emitted = self._feed(ch)
            if emitted:
                out.append(emitted)
        return "".join(out)

    def reset(self) -> None:
        """Reset to initial state — call at the start of each turn."""
        self._state = _State.NORMAL
        self._buf = ""

    def flush(self) -> str:
        """Flush any buffered content at end-of-stream.

        In NORMAL state, pending buffer is emitted (it wasn't a full tag).
        In IN_THINK state, pending buffer is discarded (conservative).
        """
        if self._state is _State.NORMAL and self._buf:
            result = self._buf
            self._buf = ""
            return result
        self._buf = ""
        return ""

    # -- internals -----------------------------------------------------------

    def _feed(self, ch: str) -> str:
        """Feed a single character and return output (empty string = suppress)."""
        if self._state is _State.NORMAL:
            return self._feed_normal(ch)
        return self._feed_in_think(ch)

    def _feed_normal(self, ch: str) -> str:
        """NORMAL state: pass through unless we detect ``<think>``."""
        if self._buf:
            # We are accumulating a potential opening tag.
            candidate = self._buf + ch
            if _OPEN_TAG.startswith(candidate):
                # Still a valid prefix of <think>.
                self._buf = candidate
                if candidate == _OPEN_TAG:
                    # Full match — enter think mode, discard tag.
                    self._buf = ""
                    self._state = _State.IN_THINK
                return ""
            else:
                # Mismatch — flush buffer (it was safe text) + current char.
                flushed = self._buf
                self._buf = ""
                # Current char might itself start a new potential tag.
                if ch == "<":
                    self._buf = ch
                    return flushed
                return flushed + ch
        else:
            if ch == "<":
                # Potential start of <think>.
                self._buf = ch
                return ""
            return ch

    def _feed_in_think(self, ch: str) -> str:
        """IN_THINK state: suppress everything until ``</think>``."""
        if self._buf:
            candidate = self._buf + ch
            if _CLOSE_TAG.startswith(candidate):
                self._buf = candidate
                if candidate == _CLOSE_TAG:
                    # Full match — exit think mode.
                    self._buf = ""
                    self._state = _State.NORMAL
                return ""
            else:
                # Mismatch — discard buffer (inside think block).
                self._buf = ""
                # Current char might start a new </think>.
                if ch == "<":
                    self._buf = ch
                return ""
        else:
            if ch == "<":
                self._buf = ch
            return ""


# ---------------------------------------------------------------------------
# OutputSink wrapper
# ---------------------------------------------------------------------------


class ScrubberSink:
    """Wraps an :class:`OutputSink` to scrub ``<think>`` blocks from streamed text.

    Only ``emit_chunk`` and ``emit_final`` are scrubbed — other event types
    pass through unchanged.  ``emit_thinking`` is *not* scrubbed because its
    content is already intended for the reasoning/thinking display path.
    """

    __slots__ = ("_inner", "_scrubber")

    def __init__(self, inner: OutputSink) -> None:
        self._inner = inner
        self._scrubber = ThinkScrubber()

    # -- property ------------------------------------------------------------

    @property
    def supports_streaming(self) -> bool:  # noqa: D102
        return self._inner.supports_streaming

    # -- scrubbed paths ------------------------------------------------------

    async def emit_chunk(self, chunk: str) -> None:
        """Scrub thinking content from text chunk before forwarding."""
        cleaned = self._scrubber.scrub(chunk)
        if cleaned:
            await self._inner.emit_chunk(cleaned)

    async def emit_final(self, content: str) -> None:
        """Scrub any residual thinking content from the final response."""
        # Use a fresh single-pass scrubber for the final assembled text so
        # it is independently correct even if the streaming scrubber was not
        # used on the same content.
        final_scrubber = ThinkScrubber()
        cleaned = final_scrubber.scrub(content) + final_scrubber.flush()
        await self._inner.emit_final(cleaned)

    # -- pass-through paths --------------------------------------------------

    async def emit_thinking(self, content: str) -> None:  # noqa: D102
        await self._inner.emit_thinking(content)

    async def emit_tool_start(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:  # noqa: D102
        await self._inner.emit_tool_start(name, metadata=metadata)

    async def emit_tool_complete(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:  # noqa: D102
        await self._inner.emit_tool_complete(name, metadata=metadata)

    async def emit_error(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:  # noqa: D102
        await self._inner.emit_error(content, metadata=metadata)

    async def close(self) -> None:
        """Flush any pending buffer and close the inner sink."""
        # Flush residual buffered text from the streaming scrubber.
        residual = self._scrubber.flush()
        if residual:
            await self._inner.emit_chunk(residual)
        await self._inner.close()
