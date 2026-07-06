"""Pattern miner — discovers recurring operation patterns from event history.

LLM-Native design: uses simple frequency statistics to identify candidate
sequences, then delegates to LLM for semantic pattern recognition and
skill candidate generation.

Implements EventConsumer protocol for integration with EventBus.
Discovery results are delivered via an async callback (on_candidates),
enabling downstream consumers (ActiveLearningObserver, CLI) to act on
discovered patterns without tight coupling.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from leapflow.memory.providers.episodic import EpisodicMemoryProvider
    from leapflow.domain.events import SystemEvent

logger = logging.getLogger(__name__)

CandidateCallback = Callable[[List["SkillCandidate"]], Union[None, Awaitable[None]]]


@dataclass(frozen=True, slots=True)
class SkillCandidate:
    """A potential skill discovered by PatternMiner."""

    title: str
    description: str
    trigger_phrases: list[str]
    steps: list[str]
    frequency: int  # How many times observed
    confidence: float  # LLM's confidence this is a meaningful pattern
    source_events: list[str] = field(default_factory=list)  # Event IDs that formed this pattern


class LLMClient(Protocol):
    """Minimal LLM interface needed by PatternMiner.

    Compatible with leapflow.llm.base.LLMProvider via structural subtyping:
    callers use ``achat(messages, stream=False)`` and extract ``.content``.
    """

    async def achat(
        self,
        messages: list[dict],
        *,
        stream: bool = ...,
        enable_thinking: bool = ...,
        **kwargs: Any,
    ) -> Any: ...


class PatternMiner:
    """Discovers recurring operation patterns from EpisodicMemory.

    Three-step discovery cycle:
    1. Query recent events from EpisodicMemory
    2. Find repeated subsequences using frequency counting
    3. Ask LLM to judge if sequences represent meaningful, abstractable patterns

    Implements EventConsumer protocol — receives batched events from EventBus
    and accumulates them for periodic discovery runs.
    """

    def __init__(
        self,
        memory: "EpisodicMemoryProvider",
        llm: LLMClient,
        *,
        min_frequency: int = 5,
        min_sequence_length: int = 2,
        max_sequence_length: int = 8,
        max_candidates_per_run: int = 3,
        discovery_cooldown_s: float = 3600.0,
        max_events_per_query: int = 500,
        on_candidates: Optional[CandidateCallback] = None,
    ) -> None:
        self._memory = memory
        self._llm = llm
        self._min_frequency = min_frequency
        self._min_seq_len = min_sequence_length
        self._max_seq_len = max_sequence_length
        self._max_candidates = max_candidates_per_run
        self._cooldown_s = discovery_cooldown_s
        self._max_events = max_events_per_query
        self._on_candidates = on_candidates

        self._last_discovery_ts: float = 0.0
        self._event_accumulator: list[Dict[str, Any]] = []
        self._enabled: bool = True

    # ── EventConsumer Protocol ──

    @property
    def consumer_id(self) -> str:
        return "pattern_miner"

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def on_events_batch(self, events: list["SystemEvent"]) -> None:
        """Accumulate events for next discovery cycle."""
        for event in events:
            self._event_accumulator.append({
                "type": event.event_type,
                "source": event.source,
                "ts": event.timestamp,
                "payload": event.payload,
            })

        now = time.time()
        if (
            now - self._last_discovery_ts >= self._cooldown_s
            and len(self._event_accumulator) >= self._min_frequency * self._min_seq_len
        ):
            candidates = await self.discover()
            if candidates:
                await self._deliver_candidates(candidates)

    # ── Core Discovery ──

    async def discover(self) -> list[SkillCandidate]:
        """Run one discovery cycle: query → cluster → LLM judge → candidates."""
        self._last_discovery_ts = time.time()

        # Step 1: Get recent events
        events = self._get_recent_events()
        if len(events) < self._min_frequency * self._min_seq_len:
            logger.debug("PatternMiner: insufficient events (%d), skipping", len(events))
            return []

        # Step 2: Find frequent subsequences
        frequent_sequences = self._find_frequent_sequences(events)
        if not frequent_sequences:
            logger.debug("PatternMiner: no frequent sequences found")
            return []

        logger.info(
            "PatternMiner: found %d frequent sequences, evaluating top %d",
            len(frequent_sequences),
            min(len(frequent_sequences), self._max_candidates),
        )

        # Step 3: LLM evaluation
        candidates = await self._evaluate_with_llm(
            frequent_sequences[: self._max_candidates * 2]
        )

        if candidates:
            self._event_accumulator.clear()
        return candidates[: self._max_candidates]

    def _get_recent_events(self) -> list[Dict[str, Any]]:
        """Get events from accumulator + memory."""
        # Prefer accumulator (recent, in-memory)
        if self._event_accumulator:
            return self._event_accumulator[-self._max_events :]

        try:
            recent = self._memory.recent(limit=self._max_events)
            return [
                {
                    "type": getattr(r, "event_type", ""),
                    "source": getattr(r, "path", "") or "",
                    "ts": getattr(r, "created_at", 0),
                    "payload": getattr(r, "metadata", {}),
                }
                for r in recent
            ]
        except Exception:
            logger.warning("PatternMiner: failed to query EpisodicMemory", exc_info=True)
            return []

    def _find_frequent_sequences(
        self, events: list[Dict[str, Any]]
    ) -> list[tuple[tuple[str, ...], int]]:
        """Find event type subsequences that appear >= min_frequency times.

        Uses sliding window over event types to extract N-grams,
        then counts frequency. Simple and effective for pattern discovery.
        """
        # Extract event type sequence
        type_sequence = [e.get("type", "unknown") for e in events]

        # Count N-grams for each valid length
        ngram_counts: Counter[tuple[str, ...]] = Counter()

        for length in range(self._min_seq_len, self._max_seq_len + 1):
            for i in range(len(type_sequence) - length + 1):
                ngram = tuple(type_sequence[i : i + length])
                # Skip if all same type (boring pattern)
                if len(set(ngram)) > 1:
                    ngram_counts[ngram] += 1

        # Filter by minimum frequency and sort by frequency desc
        frequent = [
            (seq, count)
            for seq, count in ngram_counts.items()
            if count >= self._min_frequency
        ]
        frequent.sort(key=lambda x: (-x[1], -len(x[0])))

        # Deduplicate: remove subsequences of longer sequences
        deduplicated = self._deduplicate_sequences(frequent)

        return deduplicated

    def _deduplicate_sequences(
        self, sequences: list[tuple[tuple[str, ...], int]]
    ) -> list[tuple[tuple[str, ...], int]]:
        """Remove sequences that are strict subsequences of higher-frequency longer ones."""
        result: list[tuple[tuple[str, ...], int]] = []
        seen_supersequences: set[tuple[str, ...]] = set()

        for seq, count in sequences:
            # Check if this is a subsequence of any already-selected sequence
            is_sub = False
            for super_seq in seen_supersequences:
                if len(seq) < len(super_seq) and self._is_subsequence(seq, super_seq):
                    is_sub = True
                    break
            if not is_sub:
                result.append((seq, count))
                seen_supersequences.add(seq)

        return result

    @staticmethod
    def _is_subsequence(short: tuple[str, ...], long: tuple[str, ...]) -> bool:
        """Check if short is a contiguous subsequence of long."""
        short_str = "|".join(short)
        long_str = "|".join(long)
        return short_str in long_str

    async def _evaluate_with_llm(
        self, sequences: list[tuple[tuple[str, ...], int]]
    ) -> list[SkillCandidate]:
        """Ask LLM to evaluate if sequences represent meaningful, automatable patterns."""
        if not sequences:
            return []

        # Format sequences for LLM
        seq_descriptions = []
        for i, (seq, count) in enumerate(sequences, 1):
            seq_descriptions.append(
                f"{i}. Sequence: {' → '.join(seq)} (observed {count} times)"
            )

        prompt = (
            "You are analyzing user operation patterns on a desktop computer.\n"
            "The following event sequences were observed repeatedly:\n\n"
            + "\n".join(seq_descriptions)
            + "\n\n"
            "For each sequence, determine:\n"
            "1. Is this a meaningful, automatable workflow pattern? (yes/no)\n"
            "2. If yes, what would be a good skill name?\n"
            "3. What trigger phrase would a user say to invoke this?\n"
            "4. Brief step-by-step description of what the skill does.\n"
            "5. Confidence (0.0-1.0) that this is a real recurring user intent.\n\n"
            "Respond in JSON array format:\n"
            '[{"meaningful": true/false, "title": "...", "trigger": "...", '
            '"steps": ["step1", "step2"], "confidence": 0.8}]\n'
            "Only include meaningful patterns."
        )

        try:
            resp = await self._llm.achat(
                [
                    {"role": "system", "content": "You are a pattern analysis expert. Respond only in valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                enable_thinking=False,
            )
            content = getattr(resp, "content", None) or str(resp)
            candidates = self._parse_llm_response(content, sequences)
            return candidates
        except Exception:
            logger.warning("PatternMiner: LLM evaluation failed", exc_info=True)
            return []

    def _parse_llm_response(
        self, response: str, sequences: list[tuple[tuple[str, ...], int]]
    ) -> list[SkillCandidate]:
        """Parse LLM JSON response into SkillCandidates."""
        import json

        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("PatternMiner: failed to parse LLM JSON response")
            return []

        if not isinstance(items, list):
            items = [items]

        max_frequency = max((s[1] for s in sequences), default=self._min_frequency)

        candidates: list[SkillCandidate] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if not item.get("meaningful", False):
                continue

            candidates.append(
                SkillCandidate(
                    title=item.get("title", f"Pattern {len(candidates)+1}"),
                    description=f"Auto-discovered pattern: {item.get('title', '')}",
                    trigger_phrases=[item["trigger"]] if item.get("trigger") else [],
                    steps=item.get("steps", []),
                    frequency=max_frequency,
                    confidence=float(item.get("confidence", 0.5)),
                )
            )

        return candidates

    # ── Delivery ──

    async def _deliver_candidates(self, candidates: list[SkillCandidate]) -> None:
        """Deliver discovered candidates to the registered callback."""
        if not self._on_candidates or not candidates:
            return
        try:
            result = self._on_candidates(candidates)
            if result is not None:
                await result
        except Exception:
            logger.warning("PatternMiner: on_candidates callback failed", exc_info=True)

    def set_on_candidates(self, callback: Optional[CandidateCallback]) -> None:
        """Late-bind the candidate delivery callback."""
        self._on_candidates = callback

    def set_min_frequency(self, value: int) -> None:
        """Adjust min_frequency at runtime (used by ColdStartManager)."""
        self._min_frequency = max(2, value)

    # ── Control ──

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def pending_events(self) -> int:
        """Number of accumulated events awaiting next discovery cycle."""
        return len(self._event_accumulator)

    @property
    def last_discovery_ts(self) -> float:
        """Timestamp of last discovery run."""
        return self._last_discovery_ts
