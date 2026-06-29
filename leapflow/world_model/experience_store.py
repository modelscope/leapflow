"""Experience storage backed by SemanticMemoryProvider.

Stores (state, action, prediction, actual_effect, δ) tuples as a dedicated
memory kind, reusing DuckDB persistence, decay scoring, and keyword search.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from leapflow.memory.providers.semantic import SemanticMemoryProvider, MemoryHit

logger = logging.getLogger(__name__)

MEMORY_KIND = "prediction_experience"


@dataclass(frozen=True)
class ExperienceTuple:
    """A single prediction-observation experience record."""

    experience_id: str
    action_description: str
    app_context: str
    predicted_effect: str
    actual_effect: str
    delta: float
    curiosity_score: float
    pre_state_summary: str
    post_state_summary: str
    timestamp: float
    replay_count: int = 0
    # OPD (On-Policy Distillation) fields — teacher grading results
    advantage: float = 0.0
    is_forking: bool = False
    grade_label: str = ""

    def to_content(self) -> str:
        """Serialize to a structured text format for LLM and keyword search."""
        base = (
            f"[ACTION] {self.action_description}\n"
            f"[APP] {self.app_context}\n"
            f"[PREDICTED] {self.predicted_effect}\n"
            f"[ACTUAL] {self.actual_effect}\n"
            f"[DELTA] {self.delta:.2f}\n"
            f"[TIMESTAMP] {time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(self.timestamp))}"
        )
        if self.grade_label:
            base += f"\n[GRADE] {self.grade_label} (advantage={self.advantage:.2f})"
        return base

    @classmethod
    def from_memory_hit(cls, hit: MemoryHit) -> ExperienceTuple:
        """Reconstruct from a MemoryHit retrieved via search."""
        meta = hit.metadata
        lines = {
            _parse_tag(line): _parse_value(line)
            for line in hit.content.split("\n")
            if line.startswith("[")
        }
        return cls(
            experience_id=hit.memory_id,
            action_description=lines.get("ACTION", ""),
            app_context=lines.get("APP", meta.get("app", "")),
            predicted_effect=lines.get("PREDICTED", ""),
            actual_effect=lines.get("ACTUAL", ""),
            delta=float(meta.get("delta", lines.get("DELTA", "0"))),
            curiosity_score=float(meta.get("curiosity", "0")),
            pre_state_summary=meta.get("pre_state", ""),
            post_state_summary=meta.get("post_state", ""),
            timestamp=hit.created_at,
            replay_count=int(meta.get("replay_count", "0")),
            advantage=float(meta.get("advantage", "0")),
            is_forking=meta.get("is_forking", False) in (True, "true", "True"),
            grade_label=str(meta.get("grade_label", "")),
        )


class ExperienceStore:
    """Domain-specific facade over SemanticMemoryProvider for prediction experiences.

    Optional semantic_rerank: if an EmbeddingProvider is supplied, keyword
    retrieval results are re-scored with cosine similarity for higher recall.
    """

    def __init__(
        self,
        lt_memory: SemanticMemoryProvider,
        *,
        embedding_provider: Any = None,
        semantic_weight: float = 0.4,
    ) -> None:
        self._lt = lt_memory
        self._session_start: float = time.time()
        self._embedder = embedding_provider
        self._semantic_weight = max(0.0, min(1.0, semantic_weight))

    def mark_session_start(self) -> None:
        """Mark the on-policy boundary for the current session."""
        self._session_start = time.time()

    @property
    def session_start(self) -> float:
        return self._session_start

    def store(
        self,
        action_description: str,
        app_context: str,
        predicted_effect: str,
        actual_effect: str,
        delta: float,
        pre_state_summary: str = "",
        post_state_summary: str = "",
        curiosity_score: float = 0.0,
        advantage: float = 0.0,
        is_forking: bool = False,
        grade_label: str = "",
    ) -> str:
        """Store a prediction experience. Returns experience_id."""
        exp = ExperienceTuple(
            experience_id=str(uuid.uuid4()),
            action_description=action_description,
            app_context=app_context,
            predicted_effect=predicted_effect,
            actual_effect=actual_effect,
            delta=delta,
            curiosity_score=curiosity_score,
            pre_state_summary=pre_state_summary,
            post_state_summary=post_state_summary,
            timestamp=time.time(),
            advantage=advantage,
            is_forking=is_forking,
            grade_label=grade_label,
        )
        metadata: Dict[str, Any] = {
            "delta": delta,
            "app": app_context,
            "action": action_description[:100],
            "curiosity": curiosity_score,
            "pre_state": pre_state_summary[:200],
            "post_state": post_state_summary[:200],
            "replay_count": 0,
            "advantage": advantage,
            "is_forking": is_forking,
            "grade_label": grade_label,
        }
        return self._lt.insert_raw(
            kind=MEMORY_KIND,
            content=exp.to_content(),
            metadata=metadata,
            memory_id=exp.experience_id,
        )

    def retrieve_similar(
        self,
        action_desc: str,
        app_context: str,
        *,
        limit: int = 5,
        on_policy_boost: float = 1.5,
        advantage_floor: Optional[float] = None,
    ) -> List[ExperienceTuple]:
        """Find experiences similar to the given action/app context.

        OPD enhancements:
        - *on_policy_boost*: multiplicative recency weight for experiences
          created in the current session (on-policy data is prioritised).
        - *advantage_floor*: if set, exclude experiences with advantage below
          this value (filters out graded-harmful experiences).
        - *semantic rerank*: when an EmbeddingProvider is available, blends
          keyword-based relevance with cosine similarity for better recall.
        """
        keywords = _extract_keywords(action_desc, app_context)
        if not keywords:
            return []
        needs_oversample = (
            on_policy_boost > 1.0
            or advantage_floor is not None
            or self._embedder is not None
        )
        fetch_limit = limit * 3 if needs_oversample else limit
        hits = self._lt.search_keywords(
            keywords, kinds=[MEMORY_KIND], limit=fetch_limit,
        )
        exps = [ExperienceTuple.from_memory_hit(h) for h in hits]

        if advantage_floor is not None:
            exps = [e for e in exps if e.advantage >= advantage_floor]

        if self._embedder is not None and exps:
            exps = self._semantic_rerank(
                action_desc, app_context, exps
            )

        if on_policy_boost > 1.0 and exps:
            exps = _rank_on_policy(exps, self._session_start, on_policy_boost)

        return exps[:limit]

    def _semantic_rerank(
        self,
        action_desc: str,
        app_context: str,
        exps: List[ExperienceTuple],
    ) -> List[ExperienceTuple]:
        """Re-rank experiences by blending keyword position with cosine similarity."""
        try:
            from leapflow.world_model.embedding import cosine_similarity

            query_text = f"{action_desc} {app_context}"
            query_vec = self._embedder.embed(query_text)
            if not query_vec:
                return exps

            w_sem = self._semantic_weight
            w_kw = 1.0 - w_sem
            scored: list = []
            for idx, exp in enumerate(exps):
                kw_score = 1.0 / (1.0 + idx * 0.3)
                exp_vec = self._embedder.embed(exp.to_content())
                sem_score = cosine_similarity(query_vec, exp_vec)
                combined = w_kw * kw_score + w_sem * sem_score
                scored.append((combined, exp))

            scored.sort(key=lambda p: p[0], reverse=True)
            return [exp for _, exp in scored]
        except Exception:
            logger.debug("semantic_rerank failed, using keyword order", exc_info=True)
            return exps

    def retrieve_high_delta(
        self,
        *,
        delta_min: float = 0.5,
        limit: int = 20,
    ) -> List[ExperienceTuple]:
        """Retrieve experiences with high prediction error for off-policy replay."""
        try:
            hits = self._lt.query_by_kind(MEMORY_KIND, limit=limit * 3)
        except Exception:
            return []

        results: List[ExperienceTuple] = []
        for hit in hits:
            if float(hit.metadata.get("delta", 0)) >= delta_min:
                results.append(ExperienceTuple.from_memory_hit(hit))
                if len(results) >= limit:
                    break
        return results

    def count(self) -> int:
        """Total number of stored experiences."""
        try:
            return self._lt.count_by_kind(MEMORY_KIND)
        except Exception:
            return 0

    def increment_replay_count(self, experience_id: str) -> None:
        """Increment the replay count for an experience."""
        try:
            hit = self._lt.get_by_id(experience_id)
            if hit and hit.kind == MEMORY_KIND:
                meta = dict(hit.metadata)
                meta["replay_count"] = int(meta.get("replay_count", 0)) + 1
                self._lt.update_metadata(experience_id, meta)
        except Exception:
            logger.debug("Failed to increment replay count", exc_info=True)

    def update_curiosity_score(self, experience_id: str, curiosity: float) -> None:
        """Write the curiosity score back to an existing experience."""
        try:
            hit = self._lt.get_by_id(experience_id)
            if hit and hit.kind == MEMORY_KIND:
                meta = dict(hit.metadata)
                meta["curiosity"] = curiosity
                self._lt.update_metadata(experience_id, meta)
        except Exception:
            logger.debug("Failed to update curiosity_score", exc_info=True)

    def update_advantage(
        self,
        experience_id: str,
        advantage: float,
        is_forking: bool = False,
        grade_label: str = "",
    ) -> None:
        """Write OPD teacher grading results back to an existing experience."""
        try:
            hit = self._lt.get_by_id(experience_id)
            if hit and hit.kind == MEMORY_KIND:
                meta = dict(hit.metadata)
                meta["advantage"] = advantage
                meta["is_forking"] = is_forking
                meta["grade_label"] = grade_label
                self._lt.update_metadata(experience_id, meta)
        except Exception:
            logger.debug("Failed to update advantage", exc_info=True)


def _extract_keywords(action_desc: str, app_context: str) -> List[str]:
    """Extract search keywords from action description and app context."""
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", f"{action_desc} {app_context}".lower())
    return [t for t in tokens if len(t) >= 2][:8]


def _parse_tag(line: str) -> str:
    """Extract tag name from '[TAG] value' format."""
    end = line.find("]")
    if end > 1:
        return line[1:end]
    return ""


def _parse_value(line: str) -> str:
    """Extract value from '[TAG] value' format."""
    end = line.find("]")
    if end > 0 and end + 2 <= len(line):
        return line[end + 2:]
    return ""


def _rank_on_policy(
    exps: List[ExperienceTuple],
    session_start: float,
    boost: float,
) -> List[ExperienceTuple]:
    """Re-rank experiences, blending original relevance with OPD signals.

    The original list order (from keyword search) encodes relevance.
    We multiply a position-decaying relevance score with on-policy boost
    and advantage, so relevant experiences aren't displaced by
    irrelevant-but-high-advantage ones.
    """
    def _sort_key(pair: tuple) -> float:
        idx, e = pair
        relevance = 1.0 / (1.0 + idx * 0.3)
        advantage_factor = 1.0 + max(0.0, e.advantage) * 0.5
        on_policy_factor = boost if e.timestamp >= session_start else 1.0
        return relevance * advantage_factor * on_policy_factor

    ranked = sorted(enumerate(exps), key=_sort_key, reverse=True)
    return [e for _, e in ranked]
