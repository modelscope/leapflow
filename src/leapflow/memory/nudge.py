# Copyright (c) Alibaba, Inc. and its affiliates.
"""Periodic memory review nudge policy.

Provides a lightweight policy that decides *when* the agent should pause and
review recent conversation turns for knowledge, preferences, or decisions
worth persisting to long-term memory.  The nudge itself is advisory — it
emits a ``MemoryNudgeTriggered`` event via EventBus; downstream listeners
(e.g. a memory provider or the prompt assembler) decide whether to act.

Design:
- Turn-interval gating prevents nudging on every turn.
- Idle-time gating ensures the user is not actively waiting.
- A per-session cap prevents notification fatigue.
- The nudge prompt is a self-contained snippet that an LLM can use to
  identify memorable topics without requiring tool calls.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ── Event ────────────────────────────────────────────────────────────────

_NUDGE_CATEGORIES = (
    "user preferences or workflow habits",
    "project-specific conventions or architectural decisions",
    "reusable problem-solving patterns or lessons learned",
    "corrections to previously held assumptions",
    "environment or tooling configuration worth remembering",
)


@dataclass(frozen=True)
class MemoryNudgeTriggered:
    """Fired when the nudge policy determines a memory review is due.

    Listeners should treat this as a *suggestion* — the agent may choose
    to skip the review if the current context is unsuitable.
    """

    session_id: str
    turn_count: int
    suggested_topics: tuple[str, ...] = ()
    timestamp: float = field(default_factory=time.time)


# ── Policy ───────────────────────────────────────────────────────────────


class MemoryNudgePolicy:
    """Periodic memory review strategy.

    Parameters
    ----------
    interval_turns:
        Minimum number of turns between consecutive nudges.
    min_idle_seconds:
        Agent must have been idle for at least this long before a nudge
        fires — avoids interrupting an active exchange.
    max_nudges_per_session:
        Hard cap on total nudges within one session to prevent fatigue.
    """

    def __init__(
        self,
        interval_turns: int = 10,
        min_idle_seconds: float = 30.0,
        max_nudges_per_session: int = 5,
    ) -> None:
        if interval_turns < 1:
            raise ValueError("interval_turns must be >= 1")
        if min_idle_seconds < 0:
            raise ValueError("min_idle_seconds must be >= 0")
        if max_nudges_per_session < 0:
            raise ValueError("max_nudges_per_session must be >= 0")

        self._interval_turns = interval_turns
        self._min_idle_seconds = min_idle_seconds
        self._max_nudges_per_session = max_nudges_per_session

        # Mutable counters — reset per session via ``reset()``.
        self._nudge_count: int = 0
        self._last_nudge_turn: int = 0

    # ── Public API ────────────────────────────────────────────────────

    @property
    def nudge_count(self) -> int:
        """Number of nudges already fired in this session."""
        return self._nudge_count

    def should_nudge(self, turn_count: int, idle_seconds: float) -> bool:
        """Return *True* when all gating conditions are satisfied.

        Conditions (all must hold):
        1. Session cap not reached.
        2. Enough turns elapsed since the last nudge (or session start).
        3. Agent has been idle long enough.
        """
        if self._nudge_count >= self._max_nudges_per_session:
            return False
        if turn_count - self._last_nudge_turn < self._interval_turns:
            return False
        if idle_seconds < self._min_idle_seconds:
            return False
        return True

    def record_nudge(self, turn_count: int | None = None) -> None:
        """Mark that a nudge was emitted.

        ``turn_count`` anchors the interval for the *next* nudge.  When
        omitted the counter simply increments (useful in tests).
        """
        self._nudge_count += 1
        if turn_count is not None:
            self._last_nudge_turn = turn_count

    def reset(self) -> None:
        """Reset counters — call at session start."""
        self._nudge_count = 0
        self._last_nudge_turn = 0

    def build_nudge_prompt(self, recent_turns: List[Dict[str, Any]]) -> str:
        """Construct a review prompt the LLM can use to identify memorables.

        The prompt is model-agnostic and does not require tool calls — it
        asks the model to introspect over the supplied turn summaries and
        list anything worth persisting.
        """
        if not recent_turns:
            return ""

        # Build a compact digest of recent conversation turns.
        digest_lines: list[str] = []
        for idx, turn in enumerate(recent_turns[-20:], start=1):
            role = turn.get("role", "unknown")
            content = str(turn.get("content") or "")[:300]
            if content:
                digest_lines.append(f"  [{idx}] {role}: {content}")

        if not digest_lines:
            return ""

        categories = "\n".join(f"  - {c}" for c in _NUDGE_CATEGORIES)
        digest = "\n".join(digest_lines)

        return (
            "## Memory Review Nudge\n"
            "Review the recent conversation excerpt below and identify any "
            "information worth saving to long-term memory.\n\n"
            "Look for:\n"
            f"{categories}\n\n"
            "Recent conversation:\n"
            f"{digest}\n\n"
            "For each item worth remembering, state the topic and a concise "
            "summary (one sentence). If nothing qualifies, reply with "
            '"No new memories identified."'
        )

    def extract_topics(self, recent_turns: List[Dict[str, Any]]) -> list[str]:
        """Heuristically extract candidate topics from recent turns.

        This is a lightweight, non-LLM extraction used to populate the
        ``suggested_topics`` field of :class:`MemoryNudgeTriggered`.  It
        looks for tool names and user-message keywords that hint at
        memorable content.
        """
        topics: list[str] = []
        seen: set[str] = set()
        for turn in recent_turns[-20:]:
            # Tool calls often indicate actionable context.
            for tc in turn.get("tool_calls") or []:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                if name and name not in seen:
                    seen.add(name)
                    topics.append(f"tool:{name}")
            # Short user messages are likely commands; longer ones may carry
            # preferences or decisions.
            if turn.get("role") == "user":
                content = str(turn.get("content") or "")
                if len(content) > 80 and "preference" not in seen:
                    seen.add("preference")
                    topics.append("user_context")
        return topics[:10]
