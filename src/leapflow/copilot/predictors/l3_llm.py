"""L3 LLM Reasoning Predictor — deep inference with RAG context.

Uses a large-language model to generate action predictions for complex
contexts that simpler layers (L0–L2) cannot handle.  Includes a
complexity gate that avoids expensive LLM calls for trivial contexts.

Thread-safety: Relies on the underlying LLMClient being async-safe.
The predictor holds no mutable state beyond configuration.

Resource control: Respects a 3-second timeout and gracefully degrades
(returns empty list + logs warning) on any parse or network failure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Protocol

from leapflow.copilot.types import (
    ContextState,
    FeedbackSignal,
    PredictionCandidate,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# LLM Client Protocol
# ────────────────────────────────────────────────────────────────────────────


class LLMClient(Protocol):
    """Abstraction over LLM inference backends.

    Implementations may target OpenAI, Anthropic, local models, etc.
    The ``complete`` method accepts a prompt string and returns the raw
    text response.
    """

    async def complete(self, prompt: str) -> str:
        """Send a prompt and return the completion text."""
        ...


# ────────────────────────────────────────────────────────────────────────────
# RAG Provider Protocol (optional — reuses L2's search interface)
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RAGHit:
    """A single RAG retrieval result for prompt augmentation."""

    content: str
    experience_id: str = ""
    relevance_score: float = 0.0


class RAGProvider(Protocol):
    """Retrieval-Augmented Generation context provider.

    Returns relevant historical experiences for prompt enrichment.
    """

    async def retrieve(self, query: str, *, limit: int = 3) -> List[RAGHit]:
        """Retrieve relevant context for the given query."""
        ...


# ────────────────────────────────────────────────────────────────────────────
# L3 Predictor
# ────────────────────────────────────────────────────────────────────────────


class L3LLMPredictor:
    """L3 LLM Reasoning Predictor — deep inference for complex contexts.

    Only triggers when the heuristic context complexity exceeds a configurable
    threshold, preventing unnecessary token expenditure on simple patterns.

    The prediction pipeline:
      1. Evaluate context complexity (gate).
      2. Retrieve relevant experiences via RAG (if provider available).
      3. Build a structured prompt with context + RAG hits.
      4. Call LLM for JSON-formatted predictions.
      5. Parse response into PredictionCandidate list.
      6. Graceful degradation on any failure.

    Lifecycle:
      - Constructed once at startup with LLM client and optional RAG provider.
      - ``predict`` is called when the engine schedules L3 (< 3000ms budget).
      - ``on_feedback`` is a no-op (LLM model is updated externally).

    Usage::

        llm = MyOpenAIClient(...)
        rag = MyRAGStore(...)
        predictor = L3LLMPredictor(llm, rag_provider=rag)
        candidates = await predictor.predict(context)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        rag_provider: Optional[RAGProvider] = None,
        complexity_threshold: float = 0.5,
    ) -> None:
        self._llm = llm_client
        self._rag = rag_provider
        self._threshold = complexity_threshold

    # ── PredictorLayer Protocol ────────────────────────────────────────────

    @property
    def layer_id(self) -> str:
        return "L3"

    @property
    def priority(self) -> int:
        return 3

    @property
    def timeout_ms(self) -> int:
        return 3000

    async def predict(self, context: ContextState) -> List[PredictionCandidate]:
        """Generate LLM-based predictions for complex contexts."""
        # Complexity gate
        complexity = self._context_complexity(context)
        if complexity < self._threshold:
            return []

        # RAG retrieval (optional)
        rag_context = ""
        if self._rag is not None:
            rag_context = await self._retrieve_rag(context)

        # Build prompt and call LLM
        prompt = self._build_prompt(context, rag_context)
        try:
            response = await self._llm.complete(prompt)
        except Exception as exc:
            logger.warning("L3 LLM call failed: %s", exc)
            return []

        # Parse response
        return self._parse_response(response, context)

    async def on_feedback(self, signal: FeedbackSignal) -> None:
        """No-op — LLM fine-tuning is handled externally by EvolutionLoop."""
        pass

    # ── Internal ───────────────────────────────────────────────────────────

    def _context_complexity(self, ctx: ContextState) -> float:
        """Heuristic complexity score: cross-app diversity + sequence length.

        Formula: unique_apps / 3.0 + len(action_ring) / 10.0, capped at 1.0.
        """
        unique_apps = len(
            set(a.split(":")[1] for a in ctx.action_ring if ":" in a)
        )
        return min(unique_apps / 3.0 + len(ctx.action_ring) / 10.0, 1.0)

    async def _retrieve_rag(self, context: ContextState) -> str:
        """Retrieve relevant experiences for prompt augmentation."""
        query = f"{context.app_bundle} {' '.join(context.action_ring[-5:])}"
        try:
            hits = await self._rag.retrieve(query, limit=3)  # type: ignore[union-attr]
            return "\n".join(h.content for h in hits)
        except Exception as exc:
            logger.warning("L3 RAG retrieval failed: %s", exc)
            return ""

    def _build_prompt(self, context: ContextState, rag_context: str) -> str:
        """Construct a structured prompt for the LLM."""
        lines = [
            "You are a workflow prediction assistant. Based on the user's current",
            "operational context, predict the most likely next action(s).",
            "",
            "## Current Context",
            f"- Application: {context.app_bundle}",
            f"- Window: {context.window_title}",
            f"- Recent actions: {' → '.join(context.action_ring[-5:])}",
            f"- Time bucket: {context.time_bucket}",
        ]

        if rag_context:
            lines.extend([
                "",
                "## Similar Historical Experiences",
                rag_context,
            ])

        lines.extend([
            "",
            "## Instructions",
            "Return a JSON array of predicted next actions. Each element should have:",
            '  {"action": "<description>", "confidence": <0.0-1.0>, "reasoning": "<why>"}',
            "",
            "Return at most 3 predictions. Only include predictions with confidence > 0.3.",
            "Return ONLY the JSON array, no additional text.",
        ])

        return "\n".join(lines)

    def _parse_response(
        self, response: str, context: ContextState
    ) -> List[PredictionCandidate]:
        """Parse LLM JSON response into PredictionCandidate list.

        Gracefully returns empty list on parse failure.
        """
        # Try to extract JSON array from response
        text = response.strip()

        # Strategy 1: Find JSON array boundaries directly (most robust)
        arr_start = text.find("[")
        arr_end = text.rfind("]")
        if arr_start != -1 and arr_end > arr_start:
            text = text[arr_start: arr_end + 1]
        else:
            # Strategy 2: Extract from markdown code block (```json ... ```)
            import re
            md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if md_match:
                text = md_match.group(1).strip()

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("L3 response parse failed: %s — response: %.200s", exc, response)
            return []

        if not isinstance(data, list):
            logger.warning("L3 response is not a JSON array: %.200s", response)
            return []

        candidates: List[PredictionCandidate] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            action = item.get("action", "")
            confidence = item.get("confidence", 0.0)
            reasoning = item.get("reasoning", "")

            if not action or not isinstance(confidence, (int, float)):
                continue
            if confidence <= 0.3:
                continue

            candidates.append(
                PredictionCandidate(
                    action_description=str(action),
                    confidence=min(float(confidence), 0.99),
                    source_layer="L3",
                    context_hash=context.context_hash,
                    display_delay_ms=1000,
                    reasoning=str(reasoning),
                )
            )

        return candidates[:3]  # Cap at 3 predictions
