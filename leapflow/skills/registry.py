"""Parameterized skill registry with validation, metadata, and trigger matching."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Sequence

from leapflow.domain.skill_types import SkillMetadata, SkillParameter  # noqa: F401

if TYPE_CHECKING:
    from leapflow.utils.resilience import ResiliencePolicy
    from leapflow.skills.conditions import ConditionChecker

logger = logging.getLogger(__name__)

SkillFn = Callable[..., Awaitable[Any]]

# ── Type coercion map ──

_TYPE_COERCERS: Dict[str, Callable[[Any], Any]] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": lambda v: v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes"),
    "dict": lambda v: v if isinstance(v, dict) else {},
    "list": lambda v: v if isinstance(v, list) else [v] if v is not None else [],
    "path": str,
}


@dataclass
class SkillResult:
    """Standardized skill execution result."""

    ok: bool
    output: Any = None
    error: Optional[str] = None
    duration_s: float = 0.0


@dataclass
class Skill:
    """A parameterized, metadata-rich skill definition.

    Backward compatible: existing skills with only (name, description, run) still work.
    """

    name: str
    description: str
    run: SkillFn
    parameters: List[SkillParameter] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    instructions: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)  # natural language trigger phrases
    metadata: SkillMetadata = field(default_factory=SkillMetadata)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Skill):
            return self.name == other.name
        return NotImplemented


@dataclass(frozen=True)
class TriggerMatch:
    """Immutable result of a trigger-phrase match against a registered skill."""

    skill: Skill
    score: float           # 0..1 token overlap
    matched_trigger: str   # 命中的具体 trigger 文本


class SkillRegistry:
    """Registry supporting parameterized invocation, trigger matching, and lifecycle."""

    def __init__(
        self,
        *,
        conditions: Optional["ConditionChecker"] = None,
        default_timeout: float = 60.0,
    ) -> None:
        self._skills: Dict[str, Skill] = {}
        self._conditions = conditions
        self._default_timeout = default_timeout
        self._prediction_loop: Optional[Any] = None

    @property
    def prediction_loop(self) -> Optional[Any]:
        """Read-only access to the injected PredictionLoop (or None)."""
        return self._prediction_loop

    def set_prediction_loop(self, prediction_loop: Any) -> None:
        """Inject the PredictionLoop for world-model prediction hooks."""
        self._prediction_loop = prediction_loop

    # ═══ Basic CRUD (backward compatible) ═══

    def register(self, skill: Skill) -> None:
        """Register a skill. Overwrites if name already exists."""
        self._skills[skill.name] = skill
        logger.debug("skill_registry.register name=%s", skill.name)

    def unregister(self, name: str) -> bool:
        """Remove a skill by name. Returns True if it existed."""
        return self._skills.pop(name, None) is not None

    def get(self, name: str) -> Optional[Skill]:
        """Retrieve a skill by exact name."""
        return self._skills.get(name)

    def names(self) -> List[str]:
        """Sorted list of registered skill names."""
        return sorted(self._skills.keys())

    # ═══ Description (for LLM planning) ═══

    def describe(self) -> str:
        """Basic catalog (backward compatible)."""
        lines = [f"- {s.name}: {s.description}" for s in self._skills.values()]
        return "\n".join(sorted(lines))

    def describe_with_params(self) -> str:
        """Rich catalog with parameter declarations for LLM planning.

        Format per skill:
            skill_name(param1: type = default, param2: type [required]): description
            preconditions: [...]
            triggers: [...]
        """
        sections: List[str] = []
        for s in sorted(self._skills.values(), key=lambda x: x.name):
            # Build parameter signature
            params_parts: List[str] = []
            for p in s.parameters:
                part = f"{p.name}: {p.type}"
                if p.required:
                    part += " [required]"
                elif p.default is not None:
                    part += f" = {p.default!r}"
                params_parts.append(part)
            sig = ", ".join(params_parts)
            line = f"- {s.name}({sig}): {s.description}"
            sections.append(line)

            if s.preconditions:
                sections.append(f"  preconditions: {s.preconditions}")
            if s.triggers:
                sections.append(f"  triggers: {s.triggers}")
        return "\n".join(sections)

    # ═══ Parameterized invocation ═══

    async def invoke(
        self, name: str, context: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> SkillResult:
        """Validated invocation with resilience, preconditions, and postconditions.

        Steps:
            1. Look up skill by name
            2. Validate required params are provided
            3. Type coercion (lenient: allows string auto-conversion)
            4. Fill defaults for missing optional params
            5. Check preconditions (if configured)
            6. Execute skill.run(**validated_kwargs) with timeout
            7. Verify postconditions (if configured)
            8. Time execution, wrap in SkillResult
        """
        skill = self._skills.get(name)
        if skill is None:
            return SkillResult(ok=False, error=f"unknown_skill: {name}")

        # Validate and coerce declared parameters
        try:
            validated = self._validate_params(skill, kwargs)
        except _ParamError as e:
            return SkillResult(ok=False, error=str(e))

        # Merge context if provided
        if context:
            validated.setdefault("context", context)

        # Precondition check
        if self._conditions:
            pre = await self._conditions.check_preconditions(skill, validated)
            if not pre.passed:
                return SkillResult(ok=False, error=f"precondition_failed: {pre.reason}")

        # Core execution closure for prediction loop wrapping
        async def _execute_core() -> Any:
            from leapflow.utils.resilience import ResiliencePolicy, execute_with_resilience
            policy = ResiliencePolicy(timeout_s=self._timeout_for(skill))
            return await execute_with_resilience(
                lambda: skill.run(**validated), policy
            )

        # Execute with optional prediction loop
        t0 = time.perf_counter()
        prediction_outcome = None
        try:
            if self._prediction_loop is not None:
                user_goal = str(kwargs.get("user_goal", "") or "")
                if user_goal and hasattr(self._prediction_loop, "set_goal"):
                    self._prediction_loop.set_goal(user_goal)
                action_desc = f"skill:{name}"
                output, prediction_outcome = await self._prediction_loop.wrap_execution(
                    action_desc, _execute_core,
                )
            else:
                output = await _execute_core()
            elapsed = time.perf_counter() - t0
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - t0
            return SkillResult(
                ok=False,
                error=f"timeout after {self._timeout_for(skill):.0f}s",
                duration_s=elapsed,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.warning("skill_registry.invoke error name=%s err=%s", name, exc)
            return SkillResult(ok=False, error=str(exc), duration_s=elapsed)

        result = SkillResult(ok=True, output=output, duration_s=elapsed)

        # Postcondition verification
        if self._conditions:
            post = await self._conditions.verify_postconditions(skill, result, validated)
            if not post.passed:
                logger.warning(
                    "skill_registry.postcondition_failed name=%s reason=%s",
                    name, post.reason,
                )

        return result

    _PER_INSTRUCTION_BUDGET = 60.0

    def _timeout_for(self, skill: Skill) -> float:
        """Derive timeout from skill type and complexity.

        Tool-use skills (with instructions) run a ReAct loop per instruction,
        each involving multiple LLM round-trips. A flat 60s is insufficient.
        """
        if skill.instructions:
            return max(
                self._default_timeout,
                len(skill.instructions) * self._PER_INSTRUCTION_BUDGET,
            )
        if skill.metadata.source == "builtin":
            return self._default_timeout * 2.0
        return self._default_timeout

    def _validate_params(self, skill: Skill, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and coerce parameters against skill.parameters declarations.

        - Required params must be present
        - Known params are type-coerced (lenient)
        - Unknown params pass through untouched (for backward compat with **kwargs)
        """
        if not skill.parameters:
            # No declared parameters — pass everything through
            return dict(kwargs)

        declared = {p.name: p for p in skill.parameters}
        result: Dict[str, Any] = {}

        # Check required params
        for p in skill.parameters:
            if p.required and p.name not in kwargs:
                raise _ParamError(
                    f"missing required parameter '{p.name}' for skill '{skill.name}'"
                )

        # Validate and coerce declared params
        for p in skill.parameters:
            if p.name in kwargs:
                result[p.name] = self._coerce(p, kwargs[p.name])
            elif p.default is not None:
                result[p.name] = p.default

        # Pass through undeclared params (backward compat)
        for k, v in kwargs.items():
            if k not in declared:
                result[k] = v

        return result

    @staticmethod
    def _coerce(param: SkillParameter, value: Any) -> Any:
        """Attempt type coercion; return value as-is on failure (lenient)."""
        if value is None:
            return value
        coercer = _TYPE_COERCERS.get(param.type)
        if coercer is None:
            return value
        try:
            return coercer(value)
        except (ValueError, TypeError):
            # Lenient: return as-is rather than failing
            return value

    # ═══ Trigger phrase matching ═══

    def find_matches(self, phrase: str, threshold: float = 0.3) -> List[TriggerMatch]:
        """Find skills matching a natural language trigger phrase.

        Uses token overlap scoring (simple but effective for MVP).
        Returns ``TriggerMatch`` entries with score > threshold, sorted by
        descending score.  For a given skill with multiple triggers, only the
        highest-scoring trigger is kept.
        """
        phrase_tokens = _tokenize(phrase)
        if not phrase_tokens:
            return []

        best_by_skill: Dict[str, TriggerMatch] = {}
        for skill in self._skills.values():
            if not skill.triggers:
                continue
            best_score = 0.0
            best_trigger = ""
            for trigger in skill.triggers:
                trigger_tokens = _tokenize(trigger)
                if not trigger_tokens:
                    continue
                score = _token_overlap(phrase_tokens, trigger_tokens)
                if score > best_score:
                    best_score = score
                    best_trigger = trigger
            if best_score > threshold:
                best_by_skill[skill.name] = TriggerMatch(
                    skill=skill, score=best_score, matched_trigger=best_trigger,
                )

        return sorted(best_by_skill.values(), key=lambda m: m.score, reverse=True)

    def find_by_trigger(self, phrase: str, threshold: float = 0.3) -> List[Skill]:
        """Find skills matching a natural language trigger phrase.

        Thin wrapper around :meth:`find_matches` for backward compatibility.
        Returns skills with score > threshold, sorted by descending score.
        """
        return [m.skill for m in self.find_matches(phrase, threshold)]

    # ═══ Batch operations ═══

    def list_all(self) -> List[Skill]:
        """All registered skills, sorted by name."""
        return sorted(self._skills.values(), key=lambda s: s.name)

    def register_batch(self, skills: Sequence[Skill]) -> None:
        """Register multiple skills at once."""
        for skill in skills:
            self._skills[skill.name] = skill

    @property
    def count(self) -> int:
        """Number of registered skills."""
        return len(self._skills)


# ── Private helpers ──


class _ParamError(Exception):
    """Internal: parameter validation failure."""


def _tokenize(text: str) -> set[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric, filter short tokens."""
    import re
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    return {t for t in tokens if len(t) >= 1}


def _token_overlap(a: set[str], b: set[str]) -> float:
    """Jaccard-like overlap: |intersection| / min(|a|, |b|) for asymmetric matching."""
    if not a or not b:
        return 0.0
    intersection = a & b
    # Use min-denominator for better recall on short trigger phrases
    return len(intersection) / min(len(a), len(b))
