"""Abstract LLM interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

MessageDict = Dict[str, Any]
ChunkCallback = Optional[Callable[[str], None]]


@dataclass
class ToolCallInfo:
    """A single native tool call extracted from an LLM response."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMChatResponse:
    """Normalized chat result shared across providers."""

    content: str
    role: str = "assistant"
    usage: Dict[str, int] = field(default_factory=dict)
    model: str | None = None
    finish_reason: str | None = None
    thinking_content: str | None = None
    tool_calls: List[ToolCallInfo] = field(default_factory=list)


class LLMProvider(ABC):
    """Minimal async-first LLM interface (DIP: engine depends on this)."""

    @abstractmethod
    async def achat(
        self,
        messages: List[MessageDict],
        *,
        stream: bool = True,
        enable_thinking: bool = False,
        on_chunk: ChunkCallback = None,
        **kwargs: Any,
    ) -> LLMChatResponse:
        """Return a full assistant completion (non-streaming or collapsed stream).

        When ``stream=True`` and ``on_chunk`` is provided, each content delta
        is passed to the callback during accumulation (for progress display).
        """

    @abstractmethod
    async def achat_stream(
        self,
        messages: List[MessageDict],
        *,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Yield incremental text deltas (best-effort; provider-dependent)."""
