"""Skill similarity scoring — heuristic fast-filter and optional LLM refinement.

Two-phase architecture:
    Phase 1 (HeuristicSimilarityScorer): O(n) scan using Levenshtein, Jaccard,
        and token Jaccard — zero LLM cost, filters obvious matches/mismatches.
    Phase 2 (LLMSimilarityScorer): async LLM evaluation on the narrow candidate
        set that survived Phase 1 — resolves semantic ambiguity.

Follows DIP: callers depend on the SkillSimilarityScorer Protocol.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from leapflow.domain.skill_types import DistillationCandidate
from leapflow.storage.skill_library import StoredSkill

logger = logging.getLogger(__name__)


# ── Data model ──


@dataclass(frozen=True)
class SimilarityResult:
    """Similarity assessment between a candidate and a stored skill."""

    stored_skill_id: str
    stored_skill_title: str
    overall_score: float
    action_sequence_score: float
    app_set_score: float
    goal_text_score: float
    llm_score: Optional[float] = None
    llm_rationale: str = ""


# ── Protocol ──


class SkillSimilarityScorer(Protocol):
    """Pluggable similarity interface (DIP)."""

    def score(
        self, candidate: DistillationCandidate, stored: StoredSkill
    ) -> SimilarityResult: ...

    def find_similar(
        self,
        candidate: DistillationCandidate,
        skills: Sequence[StoredSkill],
        *,
        threshold: float = 0.3,
    ) -> List[SimilarityResult]: ...


# ── Heuristic scorer ──


class HeuristicSimilarityScorer:
    """Three-channel composite scorer: action sequence + app set + goal text."""

    def __init__(
        self, *, weights: Tuple[float, float, float] = (0.45, 0.20, 0.35)
    ) -> None:
        self._w_action, self._w_app, self._w_goal = weights

    def score(
        self, candidate: DistillationCandidate, stored: StoredSkill
    ) -> SimilarityResult:
        candidate_actions = _extract_action_tokens(candidate.steps)
        stored_actions = stored.action_names or _extract_action_tokens(stored.steps)
        action_score = _action_sequence_similarity(
            candidate_actions, stored_actions
        )
        app_score = _jaccard(
            set(_candidate_apps(candidate)), set(stored.app_sequence)
        )
        goal_score = max(
            _token_jaccard(_build_goal_text(candidate), _build_stored_text(stored)),
            _normalized_title_similarity(candidate.title, stored.title),
        )
        overall = (
            self._w_action * action_score
            + self._w_app * app_score
            + self._w_goal * goal_score
        )
        return SimilarityResult(
            stored_skill_id=stored.skill_id,
            stored_skill_title=stored.title,
            overall_score=overall,
            action_sequence_score=action_score,
            app_set_score=app_score,
            goal_text_score=goal_score,
        )

    def find_similar(
        self,
        candidate: DistillationCandidate,
        skills: Sequence[StoredSkill],
        *,
        threshold: float = 0.3,
    ) -> List[SimilarityResult]:
        results = [
            r
            for s in skills
            if (r := self.score(candidate, s)).overall_score >= threshold
        ]
        results.sort(key=lambda r: r.overall_score, reverse=True)
        return results


# ── LLM-enhanced scorer ──


class LLMSimilarityScorer(HeuristicSimilarityScorer):
    """Extends heuristic scorer with async LLM semantic refinement."""

    def __init__(self, llm: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._llm = llm

    async def refine(
        self,
        candidate: DistillationCandidate,
        matches: List[SimilarityResult],
    ) -> List[SimilarityResult]:
        """Re-score heuristic matches with LLM semantic evaluation."""
        refined: List[SimilarityResult] = []
        for match in matches:
            llm_result = await self._llm_compare(candidate, match)
            blended = 0.35 * match.overall_score + 0.65 * llm_result["score"]
            refined.append(
                SimilarityResult(
                    stored_skill_id=match.stored_skill_id,
                    stored_skill_title=match.stored_skill_title,
                    overall_score=blended,
                    action_sequence_score=match.action_sequence_score,
                    app_set_score=match.app_set_score,
                    goal_text_score=match.goal_text_score,
                    llm_score=llm_result["score"],
                    llm_rationale=llm_result["rationale"],
                )
            )
        refined.sort(key=lambda r: r.overall_score, reverse=True)
        return refined

    async def _llm_compare(
        self, candidate: DistillationCandidate, match: SimilarityResult
    ) -> Dict[str, Any]:
        from leapflow.llm.message_builder import (
            build_system_message,
            build_user_message_text,
        )

        prompt = (
            "Compare these two desktop automation skills and assess similarity.\n\n"
            f"Skill A (existing): {match.stored_skill_title}\n"
            f"  ID: {match.stored_skill_id}\n\n"
            f"Skill B (new observation): {candidate.title}\n"
            f"  Steps: {candidate.steps}\n"
            f"  Triggers: {candidate.trigger_phrases}\n\n"
            "Evaluate:\n"
            "1. Do they serve the same user goal?\n"
            "2. Is Skill B a variant/extension of Skill A, or a different skill?\n"
            "3. What specific differences exist?\n\n"
            'Return JSON: {"score": 0.0-1.0, "relationship": '
            '"same|variant|different", "rationale": "...", '
            '"new_elements": ["steps/triggers only in B"]}'
        )
        try:
            resp = await self._llm.achat(
                [
                    build_system_message(
                        "You are a skill similarity analyzer. Return ONLY JSON."
                    ),
                    build_user_message_text(prompt),
                ],
                stream=True,
                enable_thinking=False,
            )
            return self._parse_response(resp.content or "")
        except Exception:
            logger.debug("LLM similarity compare failed", exc_info=True)
            return {"score": match.overall_score, "rationale": ""}

    @staticmethod
    def _parse_response(raw: str) -> Dict[str, Any]:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return {"score": 0.5, "rationale": ""}
        try:
            data = json.loads(raw[start : end + 1])
            return {
                "score": float(data.get("score", 0.5)),
                "rationale": str(data.get("rationale", "")),
            }
        except (json.JSONDecodeError, ValueError):
            return {"score": 0.5, "rationale": ""}


# ── String similarity primitives ──


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """Standard Levenshtein edit distance on token sequences."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


def _action_sequence_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    """Normalized Levenshtein similarity in [0, 1]."""
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - _levenshtein(a, b) / max_len


def _jaccard(a: set, b: set) -> float:
    """Jaccard index of two sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _tokenize(text: str) -> set[str]:
    """Word-level tokenization supporting Latin and CJK."""
    return set(re.findall(r"[\w]+|[一-鿿]+", text.lower()))


def _token_jaccard(text_a: str, text_b: str) -> float:
    """Token-level Jaccard similarity between two text strings."""
    return _jaccard(_tokenize(text_a), _tokenize(text_b))


_STOPWORDS = frozenset({
    "the", "a", "an", "to", "from", "via", "with", "and", "or",
    "in", "for", "of", "on", "by", "at", "is", "it", "do", "use",
})


def _normalized_title_similarity(a: str, b: str) -> float:
    """Order-insensitive, stopword-free title comparison.

    Extracts content tokens (ignoring stopwords and punctuation),
    then computes overlap ratio against the larger set.
    """
    tokens_a = {t for t in re.split(r"[^a-z0-9一-鿿]+", a.lower()) if t and t not in _STOPWORDS}
    tokens_b = {t for t in re.split(r"[^a-z0-9一-鿿]+", b.lower()) if t and t not in _STOPWORDS}
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    return len(intersection) / max(len(tokens_a), len(tokens_b))


# ── Text builders ──


def _candidate_apps(c: DistillationCandidate) -> List[str]:
    """Extract app identifiers from candidate pre_conditions."""
    apps: List[str] = []
    for cond in c.pre_conditions:
        if cond.endswith(" available"):
            apps.append(cond[: -len(" available")])
    return apps


def _extract_action_tokens(steps: List[str]) -> List[str]:
    """Extract a normalized action token per step for sequence comparison."""
    return [s.split()[0].lower() if s.strip() else "" for s in steps]


def _build_goal_text(c: DistillationCandidate) -> str:
    """Combine candidate fields into a single text for token comparison."""
    parts = [c.title] + list(c.trigger_phrases) + list(c.steps)
    return " ".join(parts)


def _build_stored_text(s: StoredSkill) -> str:
    """Combine stored skill fields into a single text for token comparison."""
    parts = [s.title] + list(s.trigger_phrases) + list(s.steps)
    return " ".join(parts)
