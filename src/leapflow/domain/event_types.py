"""Canonical event type constants — single source of truth for all event types.

Every module that emits, subscribes, or matches event types MUST import
from here. Using string literals elsewhere is a bug waiting to happen
(cf. the app.focus_change vs app.activated mismatch in copilot/context.py).

Naming convention: past-tense domain verbs, dot-separated hierarchy.
"""

from __future__ import annotations


class NormalizedEventType:
    """Normalized event types emitted by EventNormalizer / EventBus.

    These are the canonical strings that flow through the entire system
    after raw platform events have been normalized.
    """

    # File system
    FS_CHANGE = "fs.change"

    # Clipboard
    CLIPBOARD_CHANGE = "clipboard.change"

    # Application focus
    APP_FOCUS_CHANGE = "app.focus_change"

    # UI interaction
    UI_ACTION = "ui.action"

    # Context (window title / URL change without focus change)
    CONTEXT_CHANGE = "context.change"

    # Intent signal from external integrations
    INTENT_SIGNAL = "intent.signal"

    # Unmapped / internal
    INTERNAL_UNMAPPED = "internal.unmapped"


class UIActionSubType:
    """Sub-types for UI_ACTION events (payload["sub_type"])."""

    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    SHORTCUT = "shortcut"
    DRAG = "drag"
    TYPE = "type"
    KEYBOARD = "keyboard"
    SCROLL = "scroll"


class ImplicitFeedbackType:
    """Event types emitted by ImplicitFeedbackObserver."""

    PREFIX = "implicit_feedback"

    INACTIVITY = f"{PREFIX}.inactivity"
    UNDO_STORM = f"{PREFIX}.undo_storm"
    APP_THRASHING = f"{PREFIX}.app_thrashing"
    RETRY_FAILURE = f"{PREFIX}.retry_failure"


class LearningEventType:
    """Event types for the learning pipeline."""

    PATTERN_DISCOVERED = "learning.pattern_discovered"
    SKILL_CREATED = "learning.skill_created"
    SKILL_PROMOTED = "learning.skill_promoted"
    SKILL_DEMOTED = "learning.skill_demoted"
    SKILL_REGRESSION = "learning.skill_regression"
    CURIOSITY_ALERT = "learning.curiosity_alert"
    PROACTIVE_REVIEW = "learning.proactive_review"
    COLD_START_PROMPT = "learning.cold_start_prompt"


class CLIEventType:
    """Synthetic events from CLI interaction layer."""

    INTERACTION = "cli.interaction"


# All undo-capable shortcuts (cross-platform)
UNDO_SHORTCUTS: frozenset[str] = frozenset({
    "cmd+z", "ctrl+z", "cmd+shift+z", "ctrl+shift+z",
    "ctrl+y",
})
