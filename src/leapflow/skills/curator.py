# Copyright (c) Alibaba, Inc. and its affiliates.
"""Skill curation layer — automatic lifecycle management for skills.

Implements a three-state lifecycle (ACTIVE → STALE → ARCHIVED) with
automatic transitions based on activity, manual overrides, and pin
protection. Designed to integrate with EventBus and SkillIndex.

Hermes-inspired curator adapted for LeapFlow's EventBus + Protocol
architecture.
"""

from __future__ import annotations

import enum
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from leapflow.llm.base import LLMProvider

logger = logging.getLogger(__name__)


# ── Consolidation types ──


class ConsolidationAction(str, enum.Enum):
    """Possible actions when two skills overlap significantly."""

    MERGE = "merge"                  # Combine into a single skill
    KEEP_SEPARATE = "keep_separate"  # Intentional overlap, keep both
    DEPRECATE_ONE = "deprecate_one"  # One skill supersedes the other


@dataclass(frozen=True)
class ConsolidationSuggestion:
    """LLM-generated suggestion for consolidating overlapping skills."""

    skill_a: str
    skill_b: str
    action: ConsolidationAction
    reason: str
    confidence: float
    merged_name: str = ""
    merged_description: str = ""


# ── Curation state machine ──

class CurationState(str, enum.Enum):
    """Three-state lifecycle for skill curation."""

    ACTIVE = "active"       # Skill is actively used and available for matching
    STALE = "stale"         # Exceeded stale_after_days without usage
    ARCHIVED = "archived"   # Exceeded archive_after_days without usage; excluded from matching


@dataclass
class SkillCurationEntry:
    """Persistent curation metadata for a single skill."""

    skill_name: str
    state: CurationState = CurationState.ACTIVE
    pinned: bool = False
    last_activity_at: Optional[float] = None  # epoch seconds
    created_at: float = field(default_factory=time.time)
    archive_reason: Optional[str] = None


@dataclass(frozen=True)
class CurationTransition:
    """Record of a single automatic state transition."""

    skill_name: str
    from_state: CurationState
    to_state: CurationState
    reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class CurationReport:
    """Summary of current curation state and recent transitions."""

    total: int
    active: int
    stale: int
    archived: int
    pinned: int
    transitions: list[CurationTransition] = field(default_factory=list)


# ── Store protocol (DIP) ──

@runtime_checkable
class SkillCurationStore(Protocol):
    """Protocol for persistent curation state storage."""

    def load_all(self) -> list[SkillCurationEntry]: ...

    def load(self, skill_name: str) -> Optional[SkillCurationEntry]: ...

    def save(self, entry: SkillCurationEntry) -> None: ...

    def delete(self, skill_name: str) -> bool: ...


# ── SkillCurator ──

_DEFAULT_STALE_DAYS = 14
_DEFAULT_ARCHIVE_DAYS = 30
_MIN_SWEEP_INTERVAL_S = 300.0  # Throttle: at most one sweep per 5 min


class SkillCurator:
    """Manages skill lifecycle through automatic and manual curation.

    Lifecycle transitions:
    - ACTIVE → STALE: no activity for stale_after_days
    - STALE → ARCHIVED: no activity for archive_after_days (from creation/last activity)
    - STALE → ACTIVE: activity recorded while stale (auto-reactivation)
    - ARCHIVED → ACTIVE: manual reactivate() call only

    Pinned skills are exempt from all automatic transitions.
    """

    def __init__(
        self,
        store: SkillCurationStore,
        *,
        event_bus: Optional[Any] = None,
        llm_provider: Optional[LLMProvider] = None,
        stale_after_days: int = _DEFAULT_STALE_DAYS,
        archive_after_days: int = _DEFAULT_ARCHIVE_DAYS,
    ) -> None:
        self._store = store
        self._event_bus = event_bus
        self._llm_provider: Optional[LLMProvider] = llm_provider
        self._stale_after_days = stale_after_days
        self._archive_after_days = archive_after_days
        self._last_sweep_time: float = 0.0
        # In-memory cache for fast lookups (lazily populated)
        self._cache: Optional[Dict[str, SkillCurationEntry]] = None

    def set_llm(self, provider: LLMProvider) -> None:
        """Inject or replace the LLM provider (back-reference pattern)."""
        self._llm_provider = provider

    # ── Cache management ──

    def _ensure_cache(self) -> Dict[str, SkillCurationEntry]:
        if self._cache is None:
            entries = self._store.load_all()
            self._cache = {e.skill_name: e for e in entries}
        return self._cache

    def _invalidate_cache(self) -> None:
        self._cache = None

    # ── Activity recording ──

    def record_activity(self, skill_name: str) -> None:
        """Record skill usage; auto-reactivate if stale."""
        cache = self._ensure_cache()
        entry = cache.get(skill_name)
        now = time.time()

        if entry is None:
            # First time seeing this skill — register as ACTIVE
            entry = SkillCurationEntry(
                skill_name=skill_name,
                state=CurationState.ACTIVE,
                last_activity_at=now,
                created_at=now,
            )
            cache[skill_name] = entry
            self._store.save(entry)
            logger.debug("curator.new_skill name=%s", skill_name)
            return

        old_state = entry.state
        entry.last_activity_at = now

        # Auto-reactivate stale skills on usage
        if entry.state == CurationState.STALE:
            entry.state = CurationState.ACTIVE
            entry.archive_reason = None
            self._emit_transition(
                skill_name, old_state, CurationState.ACTIVE, "activity detected"
            )
            logger.info("curator.reactivated name=%s", skill_name)

        self._store.save(entry)

    # ── Automatic transitions ──

    def apply_automatic_transitions(self) -> CurationReport:
        """Apply time-based lifecycle transitions to all skills.

        Throttled: returns a no-op report if called within MIN_SWEEP_INTERVAL_S.
        """
        now = time.time()
        if now - self._last_sweep_time < _MIN_SWEEP_INTERVAL_S:
            return self.get_curation_report()

        self._last_sweep_time = now
        cache = self._ensure_cache()
        transitions: list[CurationTransition] = []
        stale_threshold = now - (self._stale_after_days * 86400)
        archive_threshold = now - (self._archive_after_days * 86400)

        for entry in list(cache.values()):
            if entry.pinned:
                continue

            last_active = entry.last_activity_at or entry.created_at
            old_state = entry.state

            if entry.state == CurationState.ACTIVE and last_active < stale_threshold:
                entry.state = CurationState.STALE
                entry.archive_reason = None
                transition = CurationTransition(
                    skill_name=entry.skill_name,
                    from_state=old_state,
                    to_state=CurationState.STALE,
                    reason=f"inactive for >{self._stale_after_days} days",
                )
                transitions.append(transition)
                self._store.save(entry)
                self._emit_transition(
                    entry.skill_name, old_state, CurationState.STALE,
                    transition.reason,
                )
                logger.info(
                    "curator.transition name=%s %s→%s",
                    entry.skill_name, old_state.value, CurationState.STALE.value,
                )

            elif entry.state == CurationState.STALE and last_active < archive_threshold:
                entry.state = CurationState.ARCHIVED
                entry.archive_reason = (
                    f"inactive for >{self._archive_after_days} days (auto)"
                )
                transition = CurationTransition(
                    skill_name=entry.skill_name,
                    from_state=old_state,
                    to_state=CurationState.ARCHIVED,
                    reason=entry.archive_reason,
                )
                transitions.append(transition)
                self._store.save(entry)
                self._emit_transition(
                    entry.skill_name, old_state, CurationState.ARCHIVED,
                    transition.reason,
                )
                logger.info(
                    "curator.transition name=%s %s→%s",
                    entry.skill_name, old_state.value, CurationState.ARCHIVED.value,
                )

        return self._build_report(transitions)

    # ── Manual operations ──

    def archive(self, skill_name: str, reason: str = "") -> None:
        """Manually archive a skill."""
        cache = self._ensure_cache()
        entry = cache.get(skill_name)
        if entry is None:
            raise KeyError(f"Skill '{skill_name}' has no curation entry")
        old_state = entry.state
        entry.state = CurationState.ARCHIVED
        entry.archive_reason = reason or "manual archive"
        self._store.save(entry)
        self._emit_transition(skill_name, old_state, CurationState.ARCHIVED, entry.archive_reason)
        logger.info("curator.archived name=%s reason=%s", skill_name, entry.archive_reason)

    def reactivate(self, skill_name: str) -> None:
        """Manually reactivate an archived or stale skill."""
        cache = self._ensure_cache()
        entry = cache.get(skill_name)
        if entry is None:
            raise KeyError(f"Skill '{skill_name}' has no curation entry")
        old_state = entry.state
        entry.state = CurationState.ACTIVE
        entry.archive_reason = None
        entry.last_activity_at = time.time()
        self._store.save(entry)
        self._emit_transition(skill_name, old_state, CurationState.ACTIVE, "manual reactivation")
        logger.info("curator.reactivated name=%s", skill_name)

    def pin(self, skill_name: str) -> None:
        """Pin a skill — exempt from automatic transitions."""
        cache = self._ensure_cache()
        entry = cache.get(skill_name)
        if entry is None:
            # Auto-register on pin
            entry = SkillCurationEntry(
                skill_name=skill_name,
                state=CurationState.ACTIVE,
                pinned=True,
                last_activity_at=time.time(),
                created_at=time.time(),
            )
            cache[skill_name] = entry
        else:
            entry.pinned = True
        self._store.save(entry)
        logger.info("curator.pinned name=%s", skill_name)

    def unpin(self, skill_name: str) -> None:
        """Remove pin protection from a skill."""
        cache = self._ensure_cache()
        entry = cache.get(skill_name)
        if entry is None:
            raise KeyError(f"Skill '{skill_name}' has no curation entry")
        entry.pinned = False
        self._store.save(entry)
        logger.info("curator.unpinned name=%s", skill_name)

    # ── Queries ──

    def get_state(self, skill_name: str) -> CurationState:
        """Get curation state for a skill. Returns ACTIVE for unknown skills."""
        cache = self._ensure_cache()
        entry = cache.get(skill_name)
        return entry.state if entry is not None else CurationState.ACTIVE

    def get_entry(self, skill_name: str) -> Optional[SkillCurationEntry]:
        """Get full curation entry for a skill."""
        cache = self._ensure_cache()
        return cache.get(skill_name)

    def list_by_state(self, state: CurationState) -> list[SkillCurationEntry]:
        """List all skills in a given curation state."""
        cache = self._ensure_cache()
        return [e for e in cache.values() if e.state == state]

    def get_archived_names(self) -> set[str]:
        """Return the set of archived skill names (for SkillIndex filtering)."""
        cache = self._ensure_cache()
        return {e.skill_name for e in cache.values() if e.state == CurationState.ARCHIVED}

    def get_stale_names(self) -> set[str]:
        """Return the set of stale skill names (for SkillIndex priority lowering)."""
        cache = self._ensure_cache()
        return {e.skill_name for e in cache.values() if e.state == CurationState.STALE}

    def get_curation_report(self) -> CurationReport:
        """Generate a current-state curation report."""
        return self._build_report([])

    # ── LLM-powered consolidation ──

    async def consolidate(
        self,
        skill_entries: List[Any],
        *,
        llm_provider: Optional[LLMProvider] = None,
    ) -> List[ConsolidationSuggestion]:
        """Detect overlapping skills and suggest merges via LLM.

        Collects active skills from *skill_entries* (SkillEntry instances),
        groups them by category, and asks the LLM to identify significant
        overlaps.  Returns a list of :class:`ConsolidationSuggestion`
        instances for user review — no automatic mutations are performed.

        Args:
            skill_entries: Active SkillEntry instances from SkillIndex.
            llm_provider: Override the instance-level LLM provider for
                          this single call.

        Returns:
            Suggestions list (empty when no LLM is available or no
            overlaps are detected).
        """
        provider = llm_provider or self._llm_provider
        if provider is None:
            logger.warning(
                "curator.consolidate: no LLM provider available; "
                "returning empty suggestions"
            )
            return []

        # Group skills by category for efficient comparison
        by_category: Dict[str, List[Any]] = {}
        for entry in skill_entries:
            cat = getattr(entry, "category", "") or "uncategorized"
            by_category.setdefault(cat, []).append(entry)

        suggestions: List[ConsolidationSuggestion] = []

        for category, group in by_category.items():
            if len(group) < 2:
                continue
            prompt = self._build_consolidation_prompt(category, group)
            try:
                response = await provider.achat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are a skill-catalog analyst. "
                                "Respond ONLY with the JSON array described "
                                "in the user message. No markdown fences, "
                                "no commentary."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    stream=False,
                )
                parsed = self._parse_consolidation_response(response.content)
                suggestions.extend(parsed)
            except Exception as exc:
                logger.warning(
                    "curator.consolidate: LLM call failed for "
                    "category=%s error=%s",
                    category, exc,
                )

        logger.info(
            "curator.consolidate: %d suggestion(s) across %d categories",
            len(suggestions), len(by_category),
        )
        return suggestions

    @staticmethod
    def _build_consolidation_prompt(
        category: str, entries: List[Any],
    ) -> str:
        """Format skill entries into a structured LLM prompt."""
        skill_lines: List[str] = []
        for entry in entries:
            name = getattr(entry, "name", "unknown")
            desc = getattr(entry, "description", "")[:200]
            tags = ", ".join(getattr(entry, "tags", ()) or ())
            triggers = ", ".join(getattr(entry, "triggers", ()) or ())
            parts = [f"  name: {name}", f"  description: {desc}"]
            if tags:
                parts.append(f"  tags: {tags}")
            if triggers:
                parts.append(f"  triggers: {triggers}")
            skill_lines.append("\n".join(parts))

        skills_block = "\n---\n".join(skill_lines)

        return (
            f"Category: {category}\n"
            f"Skills ({len(entries)}):\n"
            f"{skills_block}\n\n"
            "Analyze these skills for significant overlap. "
            "For each overlapping pair, produce a JSON object with:\n"
            '  "skill_a": <name>,\n'
            '  "skill_b": <name>,\n'
            '  "action": "merge" | "keep_separate" | "deprecate_one",\n'
            '  "reason": <concise explanation>,\n'
            '  "confidence": <0.0-1.0>,\n'
            '  "merged_name": <suggested name if merge>,\n'
            '  "merged_description": <suggested description if merge>\n\n'
            "Return a JSON array of these objects. "
            "If no overlaps exist, return [].\n"
            "Do NOT wrap in markdown code fences."
        )

    @staticmethod
    def _parse_consolidation_response(
        raw: str,
    ) -> List[ConsolidationSuggestion]:
        """Defensively parse LLM JSON into ConsolidationSuggestion list."""
        # Strip markdown code fences if present
        text = raw.strip()
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(
                "curator.consolidate: malformed JSON from LLM: %s", exc
            )
            return []

        if not isinstance(data, list):
            logger.warning(
                "curator.consolidate: expected JSON array, got %s",
                type(data).__name__,
            )
            return []

        action_map = {a.value: a for a in ConsolidationAction}
        suggestions: List[ConsolidationSuggestion] = []

        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                action_str = str(item.get("action", "")).lower()
                action = action_map.get(action_str)
                if action is None:
                    logger.debug(
                        "curator.consolidate: unknown action '%s', skipping",
                        action_str,
                    )
                    continue
                suggestion = ConsolidationSuggestion(
                    skill_a=str(item.get("skill_a", "")),
                    skill_b=str(item.get("skill_b", "")),
                    action=action,
                    reason=str(item.get("reason", "")),
                    confidence=float(item.get("confidence", 0.5)),
                    merged_name=str(item.get("merged_name", "")),
                    merged_description=str(item.get("merged_description", "")),
                )
                suggestions.append(suggestion)
            except (ValueError, TypeError) as exc:
                logger.debug(
                    "curator.consolidate: skipping malformed entry: %s", exc
                )
                continue

        return suggestions

    # ── Internal helpers ──

    def _build_report(self, transitions: list[CurationTransition]) -> CurationReport:
        cache = self._ensure_cache()
        entries = list(cache.values())
        return CurationReport(
            total=len(entries),
            active=sum(1 for e in entries if e.state == CurationState.ACTIVE),
            stale=sum(1 for e in entries if e.state == CurationState.STALE),
            archived=sum(1 for e in entries if e.state == CurationState.ARCHIVED),
            pinned=sum(1 for e in entries if e.pinned),
            transitions=transitions,
        )

    def _emit_transition(
        self,
        skill_name: str,
        from_state: CurationState,
        to_state: CurationState,
        reason: str,
    ) -> None:
        """Emit a SkillCurationChanged event on the EventBus (fire-and-forget)."""
        if self._event_bus is None:
            return
        payload = {
            "skill_name": skill_name,
            "from_state": from_state.value,
            "to_state": to_state.value,
            "reason": reason,
            "timestamp": time.time(),
        }
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._event_bus.handle_event("skill.curation_changed", payload),
                name=f"curator-transition:{skill_name}",
            )
        except RuntimeError:
            # No running event loop — skip emission
            logger.debug("curator: no event loop for transition event")


__all__ = [
    "ConsolidationAction",
    "ConsolidationSuggestion",
    "CurationState",
    "CurationReport",
    "CurationTransition",
    "SkillCurationEntry",
    "SkillCurationStore",
    "SkillCurator",
]
