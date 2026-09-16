# Copyright (c) Alibaba, Inc. and its affiliates.
"""Session-level semantic focus state for prompt context assembly.

The focus plane is deliberately separate from progressive tool disclosure. PCD
answers "which capabilities are visible this turn"; this module answers "what is
the user's current task target" and "which recent events are only control-plane
state". Keeping those concerns separate prevents runtime configuration changes
from stealing the task focus in later turns.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)


class ContextPlane(str, Enum):
    """Semantic plane for context events stored during a session."""

    TASK_SEMANTIC = "task_semantic"
    TOOL_EVIDENCE = "tool_evidence"
    CONTROL_PLANE = "control_plane"
    RUNTIME_DIAGNOSTIC = "runtime_diagnostic"


@dataclass(frozen=True)
class FocusEntity:
    """A user-visible entity that can become the task focus."""

    entity_id: str
    kind: str
    canonical_name: str
    plane: ContextPlane = ContextPlane.TASK_SEMANTIC
    aliases: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    salience: float = 1.0
    first_turn: int = 0
    last_task_turn: int = 0
    last_mentioned_turn: int = 0

    def with_observation(
        self,
        *,
        turn_id: int,
        evidence_refs: Iterable[str] = (),
        salience_boost: float = 0.2,
    ) -> "FocusEntity":
        """Return an updated entity after another task observation."""
        refs = tuple(dict.fromkeys((*self.evidence_refs, *evidence_refs)))
        return replace(
            self,
            evidence_refs=refs,
            salience=min(2.0, self.salience + salience_boost),
            last_task_turn=max(self.last_task_turn, turn_id),
            last_mentioned_turn=max(self.last_mentioned_turn, turn_id),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "kind": self.kind,
            "canonical_name": self.canonical_name,
            "plane": self.plane.value,
            "aliases": list(self.aliases),
            "evidence_refs": list(self.evidence_refs),
            "salience": self.salience,
            "first_turn": self.first_turn,
            "last_task_turn": self.last_task_turn,
            "last_mentioned_turn": self.last_mentioned_turn,
        }


@dataclass(frozen=True)
class ControlEvent:
    """A runtime/control-plane event that must not replace task focus."""

    action: str
    key: str
    value: str
    tool_name: str
    turn_id: int
    user_visible_summary: str
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "key": self.key,
            "value": self.value,
            "tool_name": self.tool_name,
            "turn_id": self.turn_id,
            "user_visible_summary": self.user_visible_summary,
            "ts": self.ts,
        }


@dataclass(frozen=True)
class ReferenceResolution:
    """Result of resolving a deictic user reference against focus state."""

    target_kind: str
    target_id: str
    target_name: str
    plane: ContextPlane
    confidence: float
    reason: str
    needs_clarification: bool = False

    @classmethod
    def unresolved(cls, reason: str, *, target_kind: str = "") -> "ReferenceResolution":
        return cls(
            target_kind=target_kind,
            target_id="",
            target_name="",
            plane=ContextPlane.TASK_SEMANTIC,
            confidence=0.0,
            reason=reason,
            needs_clarification=False,
        )

    @classmethod
    def ambiguous(cls, reason: str, *, target_kind: str = "") -> "ReferenceResolution":
        return cls(
            target_kind=target_kind,
            target_id="",
            target_name="",
            plane=ContextPlane.TASK_SEMANTIC,
            confidence=0.0,
            reason=reason,
            needs_clarification=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "target_name": self.target_name,
            "plane": self.plane.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "needs_clarification": self.needs_clarification,
        }


# ── Deprecated legacy tool name lists ─────────────────────────────────
# Prefer declaring x_leapflow.plane in tool schema metadata instead.

_LEGACY_CONTROL_TOOLS: frozenset[str] = frozenset(
    {"config_get", "config_set", "config_list"}
)  # deprecated: declare x_leapflow.plane = "control" on the tool schema

_LEGACY_TASK_TOOLS: frozenset[str] = frozenset(
    {"file_read", "web_fetch", "code_search", "text_search", "memory_search"}
)  # deprecated: declare x_leapflow.plane = "task" on the tool schema


def _tool_plane(tool_name: str, tool_schema: dict[str, Any] | None = None) -> str:
    """Determine semantic plane from tool metadata, falling back to legacy list."""
    if tool_schema:
        x_leapflow = tool_schema.get("x_leapflow", {})
        if isinstance(x_leapflow, dict):
            plane = x_leapflow.get("plane", "")
            if plane:
                return str(plane)
    # Deprecated fallback
    if tool_name in _LEGACY_CONTROL_TOOLS:
        logger.debug(
            "Tool %s has no x_leapflow.plane declaration, using legacy classification",
            tool_name,
        )
        return "control"
    if tool_name in _LEGACY_TASK_TOOLS:
        logger.debug(
            "Tool %s has no x_leapflow.plane declaration, using legacy classification",
            tool_name,
        )
        return "task"
    return "unknown"


# ── Entity kind pattern registry ──────────────────────────────────────


@dataclass(frozen=True)
class KindPattern:
    """URL or name pattern to entity kind mapping."""

    pattern: str
    kind: str
    match_type: str = "substring"  # "substring" | "suffix" | "glob"


_KIND_PATTERNS: list[KindPattern] = [
    KindPattern(pattern="arxiv.org/", kind="paper"),
    KindPattern(pattern=".pdf", kind="document", match_type="suffix"),
    KindPattern(pattern=".md", kind="document", match_type="suffix"),
    KindPattern(pattern=".docx", kind="document", match_type="suffix"),
]


# ── Kind synonym groups ───────────────────────────────────────────────


_KIND_GROUPS: dict[str, frozenset[str]] = {
    "document": frozenset({"paper", "document", "file", "report"}),
    "code": frozenset({"code", "script", "module", "package"}),
}


class SessionFocusState:
    """Mutable session focus state used by the engine's prompt assembly."""

    def __init__(self, *, max_focus_items: int = 12, max_control_events: int = 12) -> None:
        self.active_focus: FocusEntity | None = None
        self.focus_stack: list[FocusEntity] = []
        self.entity_registry: dict[str, FocusEntity] = {}
        self.recent_control_events: list[ControlEvent] = []
        self.max_focus_items = max(1, max_focus_items)
        self.max_control_events = max(1, max_control_events)

    def record_focus(self, entity: FocusEntity) -> FocusEntity:
        """Promote a task entity to active focus."""
        if entity.plane == ContextPlane.CONTROL_PLANE:
            raise ValueError("control-plane entities cannot become task focus")
        existing = self.entity_registry.get(entity.entity_id)
        if existing is not None:
            entity = existing.with_observation(
                turn_id=max(entity.last_task_turn, entity.last_mentioned_turn),
                evidence_refs=entity.evidence_refs,
            )
        self.entity_registry[entity.entity_id] = entity
        self.focus_stack = [item for item in self.focus_stack if item.entity_id != entity.entity_id]
        self.focus_stack.append(entity)
        self.focus_stack = self.focus_stack[-self.max_focus_items:]
        self.active_focus = entity
        return entity

    def record_control_event(self, event: ControlEvent) -> None:
        """Record a control-plane event without changing active task focus."""
        self.recent_control_events.append(event)
        self.recent_control_events = self.recent_control_events[-self.max_control_events:]

    def record_tool_result(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None,
        result: Any,
        *,
        turn_id: int,
        tool_schema: dict[str, Any] | None = None,
    ) -> None:
        """Update focus ledgers from a completed tool result."""
        name = _canonical_tool_name(tool_name)
        args = dict(arguments or {})
        plane = _tool_plane(name, tool_schema)
        if plane == "control":
            event = control_event_from_tool(name, args, result, turn_id=turn_id)
            if event is not None:
                self.record_control_event(event)
            return
        if plane == "task":
            entity = focus_entity_from_tool(name, args, result, turn_id=turn_id)
            if entity is not None:
                self.record_focus(entity)

    def latest_control_event(self, *, key: str = "") -> ControlEvent | None:
        """Return the newest control event, optionally scoped to one config key."""
        for event in reversed(self.recent_control_events):
            if not key or event.key == key:
                return event
        return None

    def task_focus_candidates(self, *, kind: str = "") -> list[FocusEntity]:
        """Return active task candidates ordered by recency and salience."""
        candidates = [
            item for item in self.focus_stack
            if item.plane != ContextPlane.CONTROL_PLANE and _kind_matches(item.kind, kind)
        ]
        candidates.sort(key=lambda item: (item.last_task_turn, item.salience), reverse=True)
        return candidates

    def render_prompt_context(self, resolution: ReferenceResolution | None = None) -> str:
        """Render a compact prompt block describing task focus and control events."""
        if self.active_focus is None and not self.recent_control_events and not resolution:
            return ""
        lines = ["## Semantic Focus Plane"]
        if self.active_focus is not None:
            focus = self.active_focus
            lines.append("Current task focus:")
            lines.append(f"- Target: {focus.canonical_name}")
            lines.append(f"- Type: {focus.kind}")
            if focus.evidence_refs:
                lines.append(f"- Evidence refs: {', '.join(focus.evidence_refs[:4])}")
        else:
            lines.append("Current task focus: (none established)")
        if resolution and resolution.target_id:
            lines.append("Resolved user reference:")
            lines.append(
                f"- {resolution.target_kind or 'target'} -> {resolution.target_name} "
                f"({resolution.plane.value}, confidence={resolution.confidence:.2f})"
            )
        elif resolution and resolution.needs_clarification:
            lines.append(f"Reference ambiguity: {resolution.reason}")
        if self.recent_control_events:
            lines.append("Recent control-plane events (runtime settings, not task targets unless explicitly requested):")
            for event in self.recent_control_events[-3:]:
                lines.append(f"- {event.user_visible_summary}")
        lines.append(
            "Use Current task focus for deictic task references such as 'the above paper'; "
            "use control-plane events only when the user explicitly asks about runtime settings."
        )
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        return {
            "active_focus": self.active_focus.to_dict() if self.active_focus else None,
            "focus_stack": [item.to_dict() for item in self.focus_stack],
            "recent_control_events": [event.to_dict() for event in self.recent_control_events],
        }


def _canonical_tool_name(tool_name: str) -> str:
    return str(tool_name or "").removeprefix("gp_")


def _stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{prefix}:{digest}"


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _result_mapping(result: Any) -> Mapping[str, Any]:
    return result if isinstance(result, Mapping) else {}


def control_event_from_tool(
    tool_name: str,
    arguments: Mapping[str, Any],
    result: Any,
    *,
    turn_id: int,
) -> ControlEvent | None:
    """Build a control event from a config tool result."""
    payload = _result_mapping(result)
    key = _string_value(payload.get("key") or arguments.get("key"))
    if not key and tool_name != "config_list":
        return None
    if tool_name == "config_set":
        value = _string_value(payload.get("value")) if "value" in payload else ""
    else:
        value = _string_value(payload.get("value") if "value" in payload else arguments.get("value"))
    action = "list" if tool_name == "config_list" else ("set" if tool_name == "config_set" else "get")
    if key:
        summary = f"{key} -> {value}" if value else f"{key} {action}"
    else:
        summary = "config catalog listed"
    return ControlEvent(
        action=action,
        key=key,
        value=value,
        tool_name=tool_name,
        turn_id=turn_id,
        user_visible_summary=summary,
    )


def focus_entity_from_tool(
    tool_name: str,
    arguments: Mapping[str, Any],
    result: Any,
    *,
    turn_id: int,
) -> FocusEntity | None:
    """Infer a task focus entity from structured evidence-producing tools."""
    payload = _result_mapping(result)
    name = _entity_name(tool_name, arguments, payload)
    if not name:
        return None
    kind = _entity_kind(tool_name, arguments, payload)
    ref = _evidence_ref(tool_name, arguments, payload)
    entity_id = _stable_id(kind, name.lower())
    return FocusEntity(
        entity_id=entity_id,
        kind=kind,
        canonical_name=name[:200],
        plane=ContextPlane.TASK_SEMANTIC,
        aliases=tuple(_aliases(name)),
        evidence_refs=(ref,) if ref else (),
        salience=1.0,
        first_turn=turn_id,
        last_task_turn=turn_id,
        last_mentioned_turn=turn_id,
    )


def _entity_name(tool_name: str, arguments: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    for key in ("title", "name", "document_title"):
        value = _string_value(payload.get(key))
        if value:
            return value
    for key in ("url", "path", "file_path", "query"):
        value = _string_value(arguments.get(key) or payload.get(key))
        if value:
            return value
    content = _string_value(payload.get("content") or payload.get("text") or payload.get("summary"))
    if content:
        return _first_meaningful_line(content)
    if isinstance(payload, Mapping) and payload:
        return _first_meaningful_line(json.dumps(payload, ensure_ascii=False, default=str))
    return _string_value(arguments.get("query"))


def _entity_kind(tool_name: str, arguments: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    """Infer entity kind from URL/name using the pattern registry."""
    source = " ".join(
        _string_value(value)
        for value in (
            arguments.get("url"), arguments.get("path"), arguments.get("file_path"), payload.get("url"),
        )
        if value
    ).lower()
    for rule in _KIND_PATTERNS:
        if rule.match_type == "substring" and rule.pattern in source:
            return rule.kind
        if rule.match_type == "suffix" and source.endswith(rule.pattern):
            return rule.kind
    # Fallback heuristics by tool name
    if tool_name == "file_read":
        return "file"
    if tool_name in {"code_search", "text_search"}:
        return "search_result"
    return "document"


def _evidence_ref(tool_name: str, arguments: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    for key in ("url", "path", "file_path", "query"):
        value = _string_value(arguments.get(key) or payload.get(key))
        if value:
            return f"{tool_name}:{value[:120]}"
    return tool_name


def _first_meaningful_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip(" #\t")
        if stripped:
            return stripped[:200]
    return ""


def _aliases(name: str) -> list[str]:
    aliases = [name]
    compact = re.sub(r"\s+", " ", name).strip()
    if compact and compact != name:
        aliases.append(compact)
    return list(dict.fromkeys(aliases))[:4]


def _kind_matches(actual: str, requested: str) -> bool:
    """Check if two kinds are in the same semantic group."""
    if not requested:
        return True
    if actual == requested:
        return True
    for group in _KIND_GROUPS.values():
        if actual in group and requested in group:
            return True
    return False


__all__ = [
    "ContextPlane",
    "ControlEvent",
    "FocusEntity",
    "KindPattern",
    "ReferenceResolution",
    "SessionFocusState",
    "control_event_from_tool",
    "focus_entity_from_tool",
]
