# Copyright (c) Alibaba, Inc. and its affiliates.
"""Side question fiber — lightweight, read-only LLM interaction.

A ``/btw`` invocation creates an ephemeral conversation fiber that:

- shares the parent engine's LLM provider (preserving prefix cache warmth),
- reuses the parent's static system prompt prefix (maximising cache hits),
- never writes into the parent session's conversation store,
- uses CORE disclosure level (minimal tools, read-only),
- emits usage attribution back to the parent session via EventBus.

Design mirrors ``subagent.py``'s isolation philosophy but is much lighter:
no tool loop, no child session, no working memory — just a one-shot
question/answer against the same provider.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict

if TYPE_CHECKING:
    from leapflow.engine.engine import AgentEngine

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

_DEFAULT_MAX_TOKENS = 2048
_SIDE_QUESTION_MODEL_KWARGS: Dict[str, Any] = {
    # No tool calling — read-only answer
    "tools": None,
    "tool_choice": None,
}


@dataclass(frozen=True)
class SideQuestionConfig:
    """Configuration for a side question fiber."""

    question: str
    parent_session_id: str
    max_tokens: int = _DEFAULT_MAX_TOKENS
    disclosure_level: str = "CORE"
    fiber_id: str = field(default_factory=lambda: f"btw-{uuid.uuid4().hex[:12]}")


# ── Lifecycle events (frozen; safe to pass across asyncio tasks) ──────


@dataclass(frozen=True)
class SideQuestionStarted:
    """Emitted when a side question fiber begins."""

    parent_session_id: str
    fiber_id: str
    question_preview: str
    timestamp: float = field(default_factory=time.time)

    @property
    def event_type(self) -> str:
        return "side_question.started"

    def to_payload(self) -> Dict[str, Any]:
        return {
            "parent_session_id": self.parent_session_id,
            "fiber_id": self.fiber_id,
            "question_preview": self.question_preview,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class SideQuestionCompleted:
    """Emitted when a side question fiber completes."""

    parent_session_id: str
    fiber_id: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    duration_s: float
    timestamp: float = field(default_factory=time.time)

    @property
    def event_type(self) -> str:
        return "side_question.completed"

    def to_payload(self) -> Dict[str, Any]:
        return {
            "parent_session_id": self.parent_session_id,
            "fiber_id": self.fiber_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "duration_s": self.duration_s,
            "timestamp": self.timestamp,
        }


# ── Fiber implementation ──────────────────────────────────────────────


class SideQuestionFiber:
    """Ephemeral one-shot LLM fiber for ``/btw`` side questions.

    The fiber borrows the parent engine's LLM provider and static system
    prompt but maintains complete conversation isolation: no messages are
    written to the parent session's store, and no tool calls are made.
    """

    def __init__(self, engine: "AgentEngine", config: SideQuestionConfig) -> None:
        self._engine = engine
        self._config = config
        self._started_at: float = 0.0
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._cached_tokens: int = 0

    # ── Public API ────────────────────────────────────────────────────

    async def run_stream(self) -> AsyncIterator[str]:
        """Execute the side question and yield response text chunks.

        Yields:
            Incremental text deltas from the LLM provider.
        """
        self._started_at = time.monotonic()
        self._emit_started()

        messages = self._build_messages()
        provider = self._engine._llm

        try:
            response = await provider.achat(
                messages,
                stream=True,
                max_tokens=self._config.max_tokens,
                on_chunk=None,
                enable_thinking=False,
                **_SIDE_QUESTION_MODEL_KWARGS,
            )
            content = str(response.content or "")
            # Extract usage from collapsed-stream response
            usage = getattr(response, "usage", None) or {}
            if isinstance(usage, dict):
                self._prompt_tokens = int(usage.get("prompt_tokens", 0))
                self._completion_tokens = int(usage.get("completion_tokens", 0))
                self._cached_tokens = int(usage.get("cached_tokens", 0))
            elif hasattr(usage, "prompt_tokens"):
                self._prompt_tokens = int(getattr(usage, "prompt_tokens", 0))
                self._completion_tokens = int(getattr(usage, "completion_tokens", 0))
                self._cached_tokens = int(getattr(usage, "cached_tokens", 0))

            # Yield the full content as a single chunk (collapsed stream)
            if content:
                yield content
        except Exception as exc:
            logger.warning(
                "Side question fiber %s failed: %s",
                self._config.fiber_id,
                exc,
                exc_info=True,
            )
            yield f"Side question failed: {exc}"
        finally:
            self._emit_completed()
            self._attribute_usage()

    async def run(self) -> str:
        """Execute the side question and return the full response text."""
        chunks: list[str] = []
        async for chunk in self.run_stream():
            chunks.append(chunk)
        return "".join(chunks)

    # ── Internal ──────────────────────────────────────────────────────

    def _build_messages(self) -> list[Dict[str, Any]]:
        """Build the minimal message list for the side question.

        Reuses the parent engine's last system prompt for prefix cache
        warmth.  Falls back to a minimal system instruction when no
        prompt has been assembled yet (first turn).
        """
        system_prompt = self._engine._last_system_prompt
        if not system_prompt:
            system_prompt = (
                "You are a helpful assistant.  Answer the user's question "
                "concisely and accurately."
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._config.question},
        ]

    def _emit_started(self) -> None:
        """Emit a SideQuestionStarted event on the engine's EventBus."""
        event_bus = self._engine._event_bus
        if event_bus is None:
            return
        event = SideQuestionStarted(
            parent_session_id=self._config.parent_session_id,
            fiber_id=self._config.fiber_id,
            question_preview=self._config.question[:200],
        )
        try:
            import asyncio
            asyncio.create_task(
                event_bus.handle_event(event.event_type, event.to_payload()),
                name=f"btw-started:{self._config.fiber_id}",
            )
        except (RuntimeError, AttributeError):
            logger.debug("Could not emit side_question.started event")

    def _emit_completed(self) -> None:
        """Emit a SideQuestionCompleted event on the engine's EventBus."""
        event_bus = self._engine._event_bus
        if event_bus is None:
            return
        elapsed = time.monotonic() - self._started_at if self._started_at else 0.0
        event = SideQuestionCompleted(
            parent_session_id=self._config.parent_session_id,
            fiber_id=self._config.fiber_id,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            cached_tokens=self._cached_tokens,
            duration_s=round(elapsed, 3),
        )
        try:
            import asyncio
            asyncio.create_task(
                event_bus.handle_event(event.event_type, event.to_payload()),
                name=f"btw-completed:{self._config.fiber_id}",
            )
        except (RuntimeError, AttributeError):
            logger.debug("Could not emit side_question.completed event")

    def _attribute_usage(self) -> None:
        """Attribute token usage to the parent engine's usage tracker.

        This ensures that side question costs appear in the session's
        ``/usage`` report and status bar, attributed to the parent session.
        """
        tracker = getattr(self._engine, "_usage_tracker", None)
        if tracker is None:
            return
        try:
            tracker.record_side_question(
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
                cached_tokens=self._cached_tokens,
            )
        except (AttributeError, TypeError):
            # Tracker may not yet have the record_side_question method;
            # degrade silently — the EventBus event is the durable record.
            logger.debug(
                "Usage tracker does not support record_side_question; "
                "usage attributed via EventBus only"
            )
