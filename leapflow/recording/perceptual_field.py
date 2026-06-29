"""Perceptual Field Engine — context-aware perception control within apps.

Architecture:
    ContextExtractor (meta-perception) → FieldPolicy (rules) → PerceptualFieldFilter (gate)
    ConsentNotifier (non-blocking feedback) — queues notifications for display at prompt

The meta-perception layer extracts WHICH context the user is in (window title, URL, etc.)
without recording content. The policy engine then decides HOW DEEPLY to perceive that context.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple

from leapflow.domain.events import SystemEvent
from leapflow.domain.perception import (
    ContextIdentifier,
    FieldRule,
    PerceptionLevel,
    sort_rules,
)
from leapflow.recording.attention import (
    AttentionFilter,
    FilterResult,
    FilterVerdict,
    RecordingContext,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Context Extraction — Strategy Pattern
# ═══════════════════════════════════════════════════════════════════════


class ContextStrategy(ABC):
    """Extracts a context identifier from event signals for a category of apps."""

    @abstractmethod
    def extract(
        self, app_bundle_id: str, window_title: str, event: SystemEvent,
    ) -> Optional[ContextIdentifier]:
        """Attempt to extract context. Returns None if this strategy doesn't apply."""


class WindowTitleStrategy(ContextStrategy):
    """Default strategy: use the full window title as context value.

    Works for most apps where the window title carries semantic context:
    WeChat/DingTalk (group name), IDE (file name), Finder (directory), etc.
    """

    def extract(
        self, app_bundle_id: str, window_title: str, event: SystemEvent,
    ) -> Optional[ContextIdentifier]:
        if not window_title:
            return None
        return ContextIdentifier(
            app_bundle_id=app_bundle_id,
            context_type="window_title",
            context_value=window_title,
        )


class BrowserUrlStrategy(ContextStrategy):
    """Browser strategy: extract domain from window title.

    Typical browser title formats:
        "Page Title - Domain.com - Google Chrome"
        "Page Title - Browser Name"
        "(Incognito) Page Title - Google Chrome"
    """

    _INCOGNITO_MARKERS = (
        "(Incognito)", "(无痕)", "(Private Browsing)", "(隐私浏览)",
        "(InPrivate)",
    )

    def extract(
        self, app_bundle_id: str, window_title: str, event: SystemEvent,
    ) -> Optional[ContextIdentifier]:
        if not window_title:
            return None

        for marker in self._INCOGNITO_MARKERS:
            if marker in window_title:
                return ContextIdentifier(
                    app_bundle_id=app_bundle_id,
                    context_type="window_title",
                    context_value=f"({marker.strip('()')})",
                )

        return ContextIdentifier(
            app_bundle_id=app_bundle_id,
            context_type="window_title",
            context_value=window_title,
        )


class TerminalStrategy(ContextStrategy):
    """Terminal strategy: extract working directory or command context from title.

    Common formats:
        Terminal.app: "user@host: ~/path — zsh"
        iTerm2: "~/path (zsh)"
        Generic: "~/work/project"
    """

    _PATH_PATTERN = re.compile(r"[~\/][\w./-]+")

    def extract(
        self, app_bundle_id: str, window_title: str, event: SystemEvent,
    ) -> Optional[ContextIdentifier]:
        if not window_title:
            return None

        m = self._PATH_PATTERN.search(window_title)
        context_value = m.group(0) if m else window_title
        return ContextIdentifier(
            app_bundle_id=app_bundle_id,
            context_type="window_title",
            context_value=context_value,
        )


# Strategy registry: (app_pattern, strategy) — first match wins
_DEFAULT_STRATEGIES: List[Tuple[str, ContextStrategy]] = [
    ("com.google.Chrome*", BrowserUrlStrategy()),
    ("com.apple.Safari*", BrowserUrlStrategy()),
    ("org.mozilla.firefox*", BrowserUrlStrategy()),
    ("com.microsoft.edgemac*", BrowserUrlStrategy()),
    ("company.thebrowser.Browser*", BrowserUrlStrategy()),
    ("com.apple.Terminal*", TerminalStrategy()),
    ("com.googlecode.iterm2*", TerminalStrategy()),
    ("io.warp.warpterm*", TerminalStrategy()),
    ("dev.warp.Warp*", TerminalStrategy()),
    ("com.mitchellh.ghostty*", TerminalStrategy()),
]


class ContextExtractor:
    """Meta-perception layer: extracts in-app context from event signals.

    Uses a strategy pattern dispatched by app bundle ID. Falls back to
    WindowTitleStrategy for unregistered apps.
    """

    def __init__(
        self,
        strategies: Optional[List[Tuple[str, ContextStrategy]]] = None,
    ) -> None:
        self._strategies = strategies or list(_DEFAULT_STRATEGIES)
        self._fallback = WindowTitleStrategy()
        self._current: Optional[ContextIdentifier] = None

    @property
    def current_context(self) -> Optional[ContextIdentifier]:
        return self._current

    def extract(self, event: SystemEvent, context: RecordingContext) -> Optional[ContextIdentifier]:
        """Extract or return cached context identifier for the current event.

        Only re-extracts on context-changing events (app focus, window title change).
        Other events reuse the cached value.
        """
        if self._is_context_changing(event):
            app_bundle_id = self._resolve_app(event, context)
            window_title = self._resolve_window_title(event, context)
            if app_bundle_id:
                self._current = self._dispatch(app_bundle_id, window_title, event)
        return self._current

    def reset(self) -> None:
        self._current = None

    def _is_context_changing(self, event: SystemEvent) -> bool:
        if event.event_type == "app.focus_change":
            return True
        if event.event_type == "ui.action":
            return bool(event.payload.get("window_title"))
        return False

    def _resolve_app(self, event: SystemEvent, context: RecordingContext) -> str:
        if event.event_type == "app.focus_change":
            return str(event.payload.get("bundle_id", event.source))
        return str(event.payload.get("app_bundle_id", "")) or context.current_focused_app

    def _resolve_window_title(self, event: SystemEvent, context: RecordingContext) -> str:
        wt = event.payload.get("window_title", "")
        if wt:
            return str(wt)
        return context.last_window_title

    def _dispatch(
        self, app_bundle_id: str, window_title: str, event: SystemEvent,
    ) -> Optional[ContextIdentifier]:
        for pattern, strategy in self._strategies:
            if fnmatch(app_bundle_id.lower(), pattern.lower()):
                result = strategy.extract(app_bundle_id, window_title, event)
                if result is not None:
                    return result
                break
        return self._fallback.extract(app_bundle_id, window_title, event)


# ═══════════════════════════════════════════════════════════════════════
# Field Policy — Rules Engine
# ═══════════════════════════════════════════════════════════════════════


class FieldPolicy:
    """Evaluates perception level for a given context based on priority-ordered rules.

    Rule evaluation: first-match-wins, sorted by priority descending.
    Unmatched contexts fall through to default_level.
    """

    def __init__(
        self,
        *,
        rules: Sequence[FieldRule] = (),
        default_level: PerceptionLevel = PerceptionLevel.FULL,
    ) -> None:
        self._rules: list[FieldRule] = sort_rules(rules)
        self._default_level = default_level

    @property
    def default_level(self) -> PerceptionLevel:
        return self._default_level

    def evaluate(self, ctx: ContextIdentifier) -> PerceptionLevel:
        """Determine the perception level for a context."""
        rule = self.find_matching_rule(ctx)
        if rule is not None:
            return rule.level
        return self._default_level

    def find_matching_rule(self, ctx: ContextIdentifier) -> Optional[FieldRule]:
        """Find the first rule matching the given context (or None)."""
        for rule in self._rules:
            if rule.matches(ctx):
                return rule
        return None

    def add_rule(self, rule: FieldRule) -> None:
        """Dynamically add a rule, maintaining priority sort."""
        self._rules.append(rule)
        self._rules = sort_rules(self._rules)

    def get_all_rules(self) -> list[FieldRule]:
        return list(self._rules)


# ═══════════════════════════════════════════════════════════════════════
# Perceptual Field Filter — AttentionFilter implementation
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class _ContextStats:
    """Tracks observed interactions per context (for reporting / learning signals)."""
    count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0


class PerceptualFieldFilter(AttentionFilter):
    """Attention filter providing per-context perception control within apps.

    Evaluation:
        1. Extract (or reuse cached) ContextIdentifier via ContextExtractor
        2. Evaluate PerceptionLevel via FieldPolicy
        3. Map to FilterResult:
            FULL → ACCEPT
            STRUCTURAL → ANNOTATE_NOISE (recorder strips text, preserves structure)
            OPAQUE → ANNOTATE_NOISE (recorder strips to timestamp+app skeleton)
            DENY → REJECT
    """

    def __init__(
        self,
        extractor: ContextExtractor,
        policy: FieldPolicy,
        *,
        consent_callback: "Optional[Any]" = None,
    ) -> None:
        self._extractor = extractor
        self._policy = policy
        self._observed: Dict[ContextIdentifier, _ContextStats] = {}
        self._consent_callback = consent_callback

    @property
    def extractor(self) -> ContextExtractor:
        return self._extractor

    @property
    def policy(self) -> FieldPolicy:
        return self._policy

    def set_consent_callback(self, callback: "Optional[Any]") -> None:
        """Set a callback invoked on first observation of each new context."""
        self._consent_callback = callback

    def evaluate(self, event: SystemEvent, context: RecordingContext) -> FilterResult:
        ctx_id = self._extractor.extract(event, context)
        if ctx_id is None:
            return FilterResult(FilterVerdict.ACCEPT)

        # Update behavioral tracking on RecordingContext
        context.update_context(ctx_id, event.timestamp)
        if self._is_substantive_action(event):
            context.record_context_interaction(ctx_id)

        is_new_context = ctx_id not in self._observed
        self._track_observation(ctx_id, event.timestamp)
        level = self._policy.evaluate(ctx_id)

        if is_new_context and self._consent_callback is not None:
            rule = self._policy.find_matching_rule(ctx_id)
            rule_source = rule.source if rule else ""
            self._consent_callback(ctx_id, level, rule_source)

        return self._level_to_result(level, ctx_id)

    @staticmethod
    def _is_substantive_action(event: SystemEvent) -> bool:
        """Determine if this event represents a substantive user interaction."""
        if event.event_type != "ui.action":
            return False
        sub = event.payload.get("sub_type", "") or event.payload.get("action", "")
        return sub in ("click", "type", "shortcut", "drag")

    def get_observed_contexts(self) -> Dict[ContextIdentifier, Dict[str, Any]]:
        """Return all observed contexts and their stats (for perception summary)."""
        result: Dict[ContextIdentifier, Dict[str, Any]] = {}
        for ctx_id, stats in self._observed.items():
            rule = self._policy.find_matching_rule(ctx_id)
            result[ctx_id] = {
                "count": stats.count,
                "first_seen": stats.first_seen,
                "last_seen": stats.last_seen,
                "rule_source": rule.source if rule else "",
            }
        return result

    def reset(self) -> None:
        """Reset observation state for a new session."""
        self._observed.clear()
        self._extractor.reset()

    def _track_observation(self, ctx_id: ContextIdentifier, timestamp: float) -> None:
        if ctx_id not in self._observed:
            self._observed[ctx_id] = _ContextStats(
                count=1, first_seen=timestamp, last_seen=timestamp,
            )
        else:
            stats = self._observed[ctx_id]
            stats.count += 1
            stats.last_seen = timestamp

    @staticmethod
    def _level_to_result(level: PerceptionLevel, ctx_id: ContextIdentifier) -> FilterResult:
        if level == PerceptionLevel.FULL:
            return FilterResult(FilterVerdict.ACCEPT)
        elif level == PerceptionLevel.STRUCTURAL:
            return FilterResult(
                FilterVerdict.ANNOTATE_NOISE,
                reason=f"perception_field:structural:{ctx_id.context_value}",
                confidence=1.0,
            )
        elif level == PerceptionLevel.OPAQUE:
            return FilterResult(
                FilterVerdict.ANNOTATE_NOISE,
                reason=f"perception_field:opaque:{ctx_id.app_bundle_id}",
                confidence=1.0,
            )
        else:
            return FilterResult(FilterVerdict.REJECT)


# ═══════════════════════════════════════════════════════════════════════
# Consent Notifier — non-blocking feedback for context transitions
# ═══════════════════════════════════════════════════════════════════════


class ConsentNotifier:
    """Non-blocking notification queue for perceptual field context transitions.

    Queues messages when the user enters a new context that is not fully recorded.
    Messages are flushed to stderr before the next user prompt, avoiding interruption.

    Suppression rules:
        - FULL contexts: no notification needed
        - Builtin rules: security floor, user doesn't need to know
        - Rate limit: max one notification per _MIN_INTERVAL_SEC
        - Dedup: same context only notified once per session
    """

    _MIN_INTERVAL_SEC = 5.0

    def __init__(self) -> None:
        self._pending: List[str] = []
        self._last_notify_time: float = 0.0
        self._notified_keys: Set[str] = set()

    def maybe_notify(
        self,
        ctx_id: "ContextIdentifier",
        level: "PerceptionLevel",
        rule_source: str,
    ) -> None:
        """Queue a notification if the context transition warrants one."""
        if level == PerceptionLevel.FULL:
            return
        if rule_source == "builtin":
            return

        key = f"{ctx_id.app_bundle_id}::{ctx_id.context_value}"
        if key in self._notified_keys:
            return

        now = time.time()
        if now - self._last_notify_time < self._MIN_INTERVAL_SEC:
            return

        self._notified_keys.add(key)
        self._last_notify_time = now

        app_short = ctx_id.app_bundle_id.rsplit(".", 1)[-1]
        self._pending.append(
            f"  ℹ New context: {app_short}/{ctx_id.context_value} "
            f"(recording as {level.value.upper()})"
        )

    def flush(self) -> None:
        """Write all pending notifications to stderr (dimmed). Call before prompt."""
        if not self._pending:
            return
        for msg in self._pending:
            sys.stderr.write(f"\033[2m{msg}\033[0m\n")
        sys.stderr.flush()
        self._pending.clear()

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def reset(self) -> None:
        """Clear all state for a new session."""
        self._pending.clear()
        self._notified_keys.clear()
        self._last_notify_time = 0.0
