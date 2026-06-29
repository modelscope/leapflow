"""Silent period classification for gaps between user actions.

Classifies temporal gaps as normal pauses, AI generation waits,
user idle, loading, or unknown — enabling the segment agent to
annotate rather than blindly split on silence.

OCP: new tool patterns are added via register(), not code changes.
SRP: classifies only — does not modify actions or segments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Optional, Pattern

from leapflow.signal_fusion.types import SilentPeriodClass


@dataclass
class GapContext:
    """Contextual information about a gap between actions."""

    current_app_bundle: str = ""
    current_app_url: str = ""
    last_action_type: str = ""
    frame_change_ratio: float = 0.0
    has_loading_indicator: bool = False


class WaitPeriodClassifier:
    """Classify silent periods during action extraction.

    Designed to be extended with new tool patterns without modifying
    classification logic. Default patterns cover common AI tools;
    register() adds new ones at runtime.
    """

    _DEFAULT_PATTERNS: ClassVar[Dict[str, str]] = {
        "chat.deepseek.com": "deepseek",
        "claude.ai": "claude",
        "chat.openai.com": "chatgpt",
        "copilot.microsoft.com": "copilot",
        "gemini.google.com": "gemini",
        "poe.com": "poe",
        "perplexity.ai": "perplexity",
    }

    _SUBMIT_ACTIONS: ClassVar[frozenset] = frozenset({
        "type", "paste", "submit", "click", "shortcut",
    })

    def __init__(
        self,
        *,
        ai_wait_threshold: float = 3.0,
        idle_threshold: float = 30.0,
        loading_threshold: float = 2.0,
    ) -> None:
        self._ai_wait_threshold = ai_wait_threshold
        self._idle_threshold = idle_threshold
        self._loading_threshold = loading_threshold
        self._tool_patterns: Dict[str, str] = dict(self._DEFAULT_PATTERNS)
        self._compiled_pattern: Optional[Pattern[str]] = None
        self._recompile()

    def register(self, url_pattern: str, tool_name: str) -> None:
        """Register a new AI tool URL pattern."""
        self._tool_patterns[url_pattern] = tool_name
        self._recompile()

    def classify(self, gap_duration: float, context: GapContext) -> SilentPeriodClass:
        """Classify a gap between two consecutive actions."""
        if gap_duration < self._loading_threshold:
            return SilentPeriodClass.NORMAL_PAUSE

        if self._is_ai_tool_context(context) and gap_duration >= self._ai_wait_threshold:
            if context.last_action_type in self._SUBMIT_ACTIONS:
                return SilentPeriodClass.AI_GENERATING

        if context.has_loading_indicator and gap_duration >= self._loading_threshold:
            return SilentPeriodClass.LOADING

        if gap_duration >= self._idle_threshold:
            return SilentPeriodClass.USER_IDLE

        if context.frame_change_ratio > 0.0 and gap_duration >= self._loading_threshold:
            return SilentPeriodClass.LOADING

        return SilentPeriodClass.UNKNOWN_WAIT

    def _is_ai_tool_context(self, context: GapContext) -> bool:
        """Check if the current context is an AI tool."""
        if not context.current_app_url:
            return False
        return bool(self._compiled_pattern and self._compiled_pattern.search(context.current_app_url))

    def _recompile(self) -> None:
        if self._tool_patterns:
            escaped = [re.escape(p) for p in self._tool_patterns]
            self._compiled_pattern = re.compile("|".join(escaped), re.IGNORECASE)
        else:
            self._compiled_pattern = None
