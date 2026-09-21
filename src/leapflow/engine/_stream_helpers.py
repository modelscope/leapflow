# Copyright (c) Alibaba, Inc. and its affiliates.
"""Data classes for engine streaming, output sink abstractions, and task contracts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
)

from leapflow.engine.context.context_disclosure import PromptAssemblyPlan


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """Typed event emitted during streaming execution.

    Event types (extensible via Literal union):
    - chunk: intermediate token fragment, safe to display immediately.
    - final: assembled complete response (full content).
    - tool_start: tool execution beginning (content = tool name).
    - tool_complete: tool execution finished (content = brief result).
    - thinking: reasoning/thinking phase indicator.
    - status: lifecycle status update.
    - approval_request: human approval request from a daemon-side action.
    - approval_response: human approval resolution notification.
    - error: error notification.
    """

    type: Literal[
        "chunk",
        "final",
        "tool_start",
        "tool_complete",
        "thinking",
        "status",
        "error",
        "approval_request",
        "approval_response",
    ]
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class _PromptAssembly:
    """Resolved prompt pieces for a unified-loop turn.

    *system* is the **stable** system prompt (identity + capabilities +
    tool catalog + guidelines).  It should be byte-identical across turns
    when disclosure level and tool set have not changed — maximising
    DeepSeek automatic prefix cache hits.

    *volatile_context* holds per-turn dynamic content (memory, knowledge,
    semantic focus, session summary) that must still reach the model but
    must **not** be part of the cacheable system-prompt prefix.  The loop
    injects it as a separate system message placed after *system* and
    before *prior_turns*.
    """

    system: str
    plan: PromptAssemblyPlan
    prior_turns: List[Dict[str, Any]]
    volatile_context: str = ""


@dataclass(frozen=True)
class TaskContract:
    """Stable per-turn task contract that survives compression and retrieval drift."""

    task_id: str
    original_request: str
    workspace_root: str
    allowed_roots: tuple[str, ...]
    research_protocol: tuple[str, ...] = ()

    def render(self) -> str:
        """Render the contract as a compact system block."""
        lines = [
            "## Task Contract",
            f"- Task ID: {self.task_id}",
            f"- Original user request: {self.original_request}",
            f"- Workspace root: {self.workspace_root}",
            f"- Allowed roots: {', '.join(self.allowed_roots)}",
            (
                "- Treat relative project paths as relative to the workspace root; never infer `.` "
                "as the project root when a workspace root is provided."
            ),
            (
                "- Workspace boundary is enforced by tools: do not read, search, edit, or run "
                "commands against paths outside the allowed roots unless the user explicitly "
                "requests an external path and the tool/approval policy permits it."
            ),
            (
                "- LeapFlow workspace config is optional at `<workspace>/.leapflow/config.yaml`; "
                "runtime config is loaded from `~/.leapflow/config/user.yaml` and "
                "`~/.leapflow/profiles/<profile>/config/*.yaml`."
            ),
            (
                "- Preserve this task contract across summarization, compression, "
                "tool loops, and memory retrieval."
            ),
        ]
        if self.research_protocol:
            lines.append("- Research protocol:")
            lines.extend(f"  - {item}" for item in self.research_protocol)
        return "\n".join(lines)


# ── OutputSink abstraction ──────────────────────────────────────────────


@runtime_checkable
class OutputSink(Protocol):
    """Abstraction over output delivery — buffer vs stream.

    Captures every place where the two loop variants (``_run_agent_loop``
    returning ``str`` and the former ``_unified_tool_loop_stream`` yielding
    ``StreamEvent``) diverge in how they surface output.
    """

    @property
    def supports_streaming(self) -> bool:
        """Whether this sink can receive real-time token chunks."""
        ...

    async def emit_chunk(self, chunk: str) -> None:
        """Real-time text token fragment (streaming only)."""
        ...

    async def emit_thinking(self, content: str) -> None:
        """LLM reasoning/thinking phase content."""
        ...

    async def emit_tool_start(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Tool execution starting."""
        ...

    async def emit_tool_complete(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Tool execution finished."""
        ...

    async def emit_error(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Error notification (unrecoverable failure)."""
        ...

    async def emit_final(self, content: str) -> None:
        """Complete assembled response."""
        ...

    async def close(self) -> None:
        """Signal that no more events will be emitted."""
        ...


class BufferSink:
    """Collects output silently — used by the non-streaming ``run()`` path.

    All emit methods are no-ops because the unified loop already returns
    the final text via its normal return value.  ``BufferSink`` exists
    solely so the unified loop can call ``sink.emit_xxx()`` without
    checking the delivery mode at every callsite.
    """

    @property
    def supports_streaming(self) -> bool:
        return False

    async def emit_chunk(self, chunk: str) -> None:
        pass

    async def emit_thinking(self, content: str) -> None:
        pass

    async def emit_tool_start(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        pass

    async def emit_tool_complete(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        pass

    async def emit_error(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        pass

    async def emit_final(self, content: str) -> None:
        pass

    async def close(self) -> None:
        pass


class StreamSink:
    """Pushes ``StreamEvent`` objects to an asyncio queue for streaming.

    Bridges push-based emission from the unified loop to the pull-based
    ``async for event in engine.run_stream(...)`` pattern.  The loop task
    calls ``emit_*`` methods; the consumer iterates over this sink via
    ``__aiter__``.
    """

    _SENTINEL: Any = None  # end-of-stream marker

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Optional[StreamEvent]] = asyncio.Queue()

    @property
    def supports_streaming(self) -> bool:
        return True

    async def emit_chunk(self, chunk: str) -> None:
        await self._queue.put(StreamEvent(type="chunk", content=chunk))

    async def emit_thinking(self, content: str) -> None:
        await self._queue.put(StreamEvent(type="thinking", content=content))

    async def emit_tool_start(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        await self._queue.put(
            StreamEvent(type="tool_start", content=name, metadata=metadata)
        )

    async def emit_tool_complete(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        await self._queue.put(
            StreamEvent(type="tool_complete", content=name, metadata=metadata)
        )

    async def emit_error(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        await self._queue.put(
            StreamEvent(type="error", content=content, metadata=metadata)
        )

    async def emit_final(self, content: str) -> None:
        await self._queue.put(StreamEvent(type="final", content=content))

    async def close(self) -> None:
        """Signal end-of-stream so the consumer stops iterating."""
        await self._queue.put(self._SENTINEL)

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self

    async def __anext__(self) -> StreamEvent:
        event = await self._queue.get()
        if event is self._SENTINEL:
            raise StopAsyncIteration
        return event
