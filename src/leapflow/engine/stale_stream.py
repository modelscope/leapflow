"""Stale stream detection and partial recovery for LLM streaming.

Wraps an async stream iterator with an idle timeout so that hung connections
are detected promptly instead of blocking the agent loop indefinitely.

Inspired by hermes interruptible_streaming_api_call / stale detector, but
implemented as a pure async wrapper (no threads, no client pool rebuilds).
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, List, Optional

logger = logging.getLogger(__name__)


class StaleStreamError(Exception):
    """Raised when no data arrives within the stale timeout."""

    def __init__(self, timeout_s: float, partial_text: str = "") -> None:
        self.timeout_s = timeout_s
        self.partial_text = partial_text
        super().__init__(f"Stream stale: no data for {timeout_s:.0f}s")


async def stale_guarded_stream(
    stream: AsyncIterator[str],
    *,
    timeout_s: float = 180.0,
    min_timeout_s: float = 60.0,
) -> AsyncIterator[str]:
    """Wrap an async text stream with idle-timeout detection.

    Yields chunks from *stream*. If no chunk arrives within *timeout_s*
    seconds, raises :class:`StaleStreamError` with any partial text
    accumulated so far — enabling the caller to attempt recovery.

    Args:
        stream: Upstream async iterator of text chunks.
        timeout_s: Seconds of silence before declaring stream stale.
        min_timeout_s: Floor for timeout (prevents misconfigured zero).

    Yields:
        str: Text chunks from the upstream.

    Raises:
        StaleStreamError: When the stream goes idle beyond the timeout.
    """
    effective_timeout = max(timeout_s, min_timeout_s)
    partial_parts: List[str] = []
    it = stream.__aiter__()

    while True:
        try:
            chunk = await asyncio.wait_for(
                it.__anext__(),
                timeout=effective_timeout,
            )
            partial_parts.append(chunk)
            yield chunk
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            partial = "".join(partial_parts)
            logger.warning(
                "stale_stream: no data for %.0fs, partial=%d chars",
                effective_timeout, len(partial),
            )
            raise StaleStreamError(effective_timeout, partial_text=partial)


def build_continuation_prompt(
    partial_content: str,
    *,
    dropped_tool_names: Optional[List[str]] = None,
) -> str:
    """Build a user-role continuation prompt after truncation or stale stream.

    Mirrors hermes _get_continuation_prompt: tells the model its response
    was cut off and asks it to continue from where it left off.
    """
    parts = ["Your previous response was cut off. Please continue exactly where you left off."]
    if partial_content:
        tail = partial_content[-200:].strip()
        parts.append(f'Your last output ended with: "...{tail}"')
    if dropped_tool_names:
        parts.append(
            f"Note: tool calls for [{', '.join(dropped_tool_names)}] were lost. "
            "Please break them into smaller individual calls."
        )
    return "\n".join(parts)
