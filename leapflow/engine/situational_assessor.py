"""Pre-execution situational assessment for skill execution.

Evaluates whether the current environment satisfies a skill's execution
conditions before committing to the full ReAct loop. Uses lightweight
local probes + a single LLM call to produce a structured verdict.

Architecture:
    SituationalAssessor (Protocol) — DIP: callers depend on abstraction
    LLMSituationalAssessor         — concrete: one-shot LLM evaluation
    _probe_environment             — gathers path-based env snapshot
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from leapflow.skills.registry import Skill

logger = logging.getLogger(__name__)


class AssessmentVerdict(Enum):
    READY = "ready"
    ALREADY_DONE = "already_done"
    BLOCKED = "blocked"
    RISKY = "risky"


@dataclass
class Assessment:
    """Result of a pre-execution situational assessment."""

    verdict: AssessmentVerdict
    reason: str
    suggestions: List[str] = field(default_factory=list)


@runtime_checkable
class SituationalAssessor(Protocol):
    """Protocol for pre-execution assessment (DIP)."""

    async def assess(
        self,
        user_goal: str,
        skill: Skill,
        params: Dict[str, Any],
    ) -> Assessment: ...


_SYSTEM_PROMPT = (
    "You are a pre-execution situation assessor. "
    "Given the user's goal, a matched skill, and the current environment state, "
    "determine whether execution should proceed.\n\n"
    "Verdicts:\n"
    '- "ready": environment is suitable, goal is not yet achieved\n'
    '- "already_done": the goal appears already accomplished\n'
    '- "blocked": a critical precondition is unmet (e.g. path missing)\n'
    '- "risky": can proceed but there are notable concerns\n\n'
    "Return ONLY a JSON object: "
    '{"verdict": "...", "reason": "brief explanation", "suggestions": ["actionable fix"]}\n'
    "Keep reason under 50 words. Suggestions only for blocked/risky."
)


class LLMSituationalAssessor:
    """Concrete assessor using a single LLM call over environment probes."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def assess(
        self,
        user_goal: str,
        skill: Skill,
        params: Dict[str, Any],
    ) -> Assessment:
        snapshot = _probe_environment(params)

        from leapflow.llm.message_builder import build_system_message, build_user_message_text
        from leapflow.utils.stream_progress import StreamProgressWriter

        params_display = {k: v for k, v in params.items() if k != "user_goal"}
        preconditions = skill.preconditions or []

        user_msg = (
            f"Goal: {user_goal}\n"
            f"Skill: {skill.name} — {skill.description}\n"
            f"Preconditions: {preconditions}\n"
            f"Parameters: {json.dumps(params_display, ensure_ascii=False, default=str)}\n"
            f"Environment snapshot:\n{json.dumps(snapshot, ensure_ascii=False, default=str)}"
        )

        writer = StreamProgressWriter(prefix="  │ ")
        try:
            resp = await self._llm.achat(
                [
                    build_system_message(_SYSTEM_PROMPT),
                    build_user_message_text(user_msg),
                ],
                stream=True,
                enable_thinking=False,
                max_tokens=200,
                on_chunk=writer,
            )
        except Exception as e:
            logger.warning("situational_assessor.llm_failed: %s", e)
            return Assessment(verdict=AssessmentVerdict.READY, reason="assessment unavailable (fail-open)")
        finally:
            writer.finish()

        return _parse_assessment(resp.content or "")


def _probe_environment(params: Dict[str, Any]) -> Dict[str, Any]:
    """Gather local environment state from path-like parameters.

    Scans params for values that look like filesystem paths, checks
    existence/contents. Pure local I/O, no subprocess.
    """
    snapshot: Dict[str, Any] = {}

    for key, value in params.items():
        if key == "user_goal":
            continue
        if not isinstance(value, str):
            continue
        if "/" not in value and "~" not in value:
            continue

        try:
            path = Path(value).expanduser().resolve()
        except (ValueError, OSError):
            continue

        if path.exists():
            if path.is_dir():
                try:
                    entries = sorted(e.name for e in path.iterdir())
                    snapshot[key] = {
                        "path": str(path),
                        "exists": True,
                        "is_dir": True,
                        "count": len(entries),
                        "entries": entries[:20],
                    }
                except PermissionError:
                    snapshot[key] = {"path": str(path), "exists": True, "is_dir": True, "readable": False}
            else:
                snapshot[key] = {
                    "path": str(path),
                    "exists": True,
                    "is_dir": False,
                    "size_bytes": path.stat().st_size,
                }
        else:
            snapshot[key] = {"path": str(path), "exists": False}

    return snapshot


def _parse_assessment(raw: str) -> Assessment:
    """Parse LLM JSON response into Assessment. Fail-open on parse error."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return Assessment(verdict=AssessmentVerdict.READY, reason="unparseable response (fail-open)")

    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return Assessment(verdict=AssessmentVerdict.READY, reason="unparseable response (fail-open)")

    verdict_str = data.get("verdict", "ready").lower().strip()
    try:
        verdict = AssessmentVerdict(verdict_str)
    except ValueError:
        verdict = AssessmentVerdict.READY

    reason = data.get("reason", "")
    suggestions = data.get("suggestions", [])
    if isinstance(suggestions, str):
        suggestions = [suggestions] if suggestions else []

    return Assessment(verdict=verdict, reason=reason, suggestions=suggestions)
