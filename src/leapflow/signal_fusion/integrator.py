"""Cross-scale bidirectional integration.

Bottom-up: enriches Segments with statistical summaries from AtomicActions.
Top-down: uses episode-level intent to assign semantic roles to actions,
          via LLM batch inference (not hardcoded rules).

SRP: integrates across scales — does not fuse, segment, or infer intent.
DIP: depends on LLMProvider abstraction for top-down role assignment.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Tuple

from leapflow.signal_fusion.types import (
    AtomicAction,
    EnrichedEpisode,
    FusionMode,
    Segment,
)

if TYPE_CHECKING:
    from leapflow.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# Schema-validated semantic role vocabulary. Any role outside this set is
# downgraded to "none" with a warning so downstream consumers always see a
# bounded set of values.
_VALID_ROLES: FrozenSet[str] = frozenset(
    {"trigger", "response", "effect", "tool", "reference", "source", "sink", "none"}
)

# Free-form semantic-role labels emitted by the LLM may be longer than the
# schema-validated set above (e.g., "prepare_input", "submit_query"). The
# prompt explicitly enumerates these task-level roles, so we accept any
# non-empty string but still validate against "none" sentinel handling.
_ROLE_FALLBACK = "none"

_ROLE_ASSIGNMENT_PROMPT = """\
Given the episode intent and action sequence below, assign a semantic role
to each action. Roles describe the action's purpose within the workflow
(e.g., "prepare_input", "submit_query", "wait_response", "copy_result",
"paste_result", "navigate", "verify", "configure", "cleanup").

Episode intent: {intent}
Workflow type: {workflow_type}

Actions:
{actions_text}

Respond with a JSON array of role strings, one per action. Use "none" if
the role is unclear. Example: ["prepare_input", "submit_query", "none"]
"""


class CrossScaleIntegrator:
    """Bidirectional cross-scale consistency enforcement.

    Bottom-up is pure computation (statistics aggregation).
    Top-down uses a single LLM call for semantic role assignment,
    replacing v4's hardcoded intent-to-role mapping with a
    generalizable LLM-driven approach.
    """

    def __init__(self, llm: Optional["LLMProvider"] = None) -> None:
        self._llm = llm

    async def integrate(
        self,
        actions: List[AtomicAction],
        segments: List[Segment],
        episodes: List[EnrichedEpisode],
    ) -> List[EnrichedEpisode]:
        self._bottom_up_enrich(segments, actions)

        if episodes and episodes[0].intent and self._llm:
            await self._top_down_assign_roles(actions, episodes[0])

        return episodes

    # ── Bottom-up: statistics enrichment ──

    @staticmethod
    def _bottom_up_enrich(
        segments: List[Segment], actions: List[AtomicAction]
    ) -> None:
        """Aggregate action-level statistics into segment.stats."""
        for seg in segments:
            if not seg.actions:
                continue

            confidences = [a.confidence for a in seg.actions]
            mode_counts: Dict[str, int] = {}
            for a in seg.actions:
                mode_counts[a.fusion_mode.value] = mode_counts.get(a.fusion_mode.value, 0) + 1

            dominant_mode = max(mode_counts, key=mode_counts.get) if mode_counts else FusionMode.MINIMAL.value

            seg.stats = {
                "avg_confidence": sum(confidences) / len(confidences),
                "min_confidence": min(confidences),
                "action_count": len(seg.actions),
                "dominant_fusion_mode": dominant_mode,
                "fusion_mode_distribution": mode_counts,
                "wait_period_count": len(seg.wait_periods),
                "total_wait_duration": sum(w.duration for w in seg.wait_periods),
            }

    # ── Top-down: LLM semantic role assignment ──

    async def _top_down_assign_roles(
        self,
        actions: List[AtomicAction],
        episode: EnrichedEpisode,
    ) -> None:
        """Use LLM to assign semantic roles to all actions in one call."""
        if not actions or not self._llm:
            return

        actions_text = "\n".join(
            f"{i+1}. {a.action}({a.target}) [{a.fusion_mode.value}] "
            f"app={a.app_bundle}"
            for i, a in enumerate(actions[:50])
        )

        prompt = _ROLE_ASSIGNMENT_PROMPT.format(
            intent=episode.intent,
            workflow_type=episode.metadata.get("workflow_type", "unknown"),
            actions_text=actions_text,
        )

        try:
            t0 = time.perf_counter()
            response = await self._llm.complete(prompt)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            roles, parsed_ok = self._parse_roles(response, len(actions))
            logger.info(
                "[integrator] llm_call elapsed_ms=%.0f parsed_ok=%s roles_count=%d",
                elapsed_ms, parsed_ok, len(actions),
            )

            if not parsed_ok:
                # Parsing fell back to all-"none"; downstream tags help
                # observability surfaces (e.g., FusionQuality consumers) detect
                # a degraded top-down enrichment pass without inspecting logs.
                logger.info(
                    "Top-down role assignment produced fallback roles for %d actions",
                    len(actions),
                )

            for action, role in zip(actions, roles):
                if role and role != _ROLE_FALLBACK:
                    action.semantic_role = role
                    action.tags.add("episode_contextualized")

        except Exception:
            logger.warning(
                "LLM role assignment failed, skipping top-down enrichment",
                exc_info=True,
            )

    @staticmethod
    def _parse_roles(response: str, expected_count: int) -> Tuple[List[str], bool]:
        """Parse LLM response into role list.

        Performs a tolerant JSON extraction (the LLM may wrap the array in
        prose) and validates each role against the configured vocabulary.
        Invalid or unparseable entries are coerced to ``"none"``.

        Parameters
        ----------
        response:
            Raw LLM completion text.
        expected_count:
            Number of actions the prompt covered; the result is padded or
            truncated to this length.

        Returns
        -------
        Tuple[List[str], bool]
            ``(roles_list, parsed_ok)``. When ``parsed_ok`` is ``False`` the
            list is the all-``"none"`` fallback, signalling callers that the
            JSON could not be decoded as a list. ``parsed_ok=True`` is
            returned even when *some* entries were coerced (logged via
            warning), because the overall structure was valid.
        """
        text = response.strip()

        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            text = text[start : end + 1]

        try:
            roles = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            preview = response.strip().replace("\n", " ")[:200]
            logger.warning(
                "Failed to JSON-decode LLM role response (%s): %s",
                exc.__class__.__name__,
                preview,
            )
            return [_ROLE_FALLBACK] * expected_count, False

        if not isinstance(roles, list):
            preview = response.strip().replace("\n", " ")[:200]
            logger.warning(
                "LLM role response is not a JSON list (got %s): %s",
                type(roles).__name__,
                preview,
            )
            return [_ROLE_FALLBACK] * expected_count, False

        result: List[str] = []
        invalid_seen = 0
        for raw in roles:
            if not raw:
                result.append(_ROLE_FALLBACK)
                continue
            value = str(raw).strip()
            # Schema validation: accept the controlled vocabulary plus the
            # task-level role labels enumerated in the prompt (any non-empty
            # snake_case identifier). Reject anything that looks like
            # free-form prose or contains structural punctuation.
            if value in _VALID_ROLES or _is_acceptable_task_role(value):
                result.append(value)
            else:
                invalid_seen += 1
                result.append(_ROLE_FALLBACK)

        if invalid_seen:
            logger.warning(
                "Discarded %d invalid role label(s) from LLM response",
                invalid_seen,
            )

        while len(result) < expected_count:
            result.append(_ROLE_FALLBACK)
        return result[:expected_count], True


def _is_acceptable_task_role(value: str) -> bool:
    """Accept snake_case identifiers up to a sane length as task roles."""
    if not value or len(value) > 64:
        return False
    return all(c.isalnum() or c == "_" for c in value)
