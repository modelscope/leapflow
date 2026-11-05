# Copyright (c) Alibaba, Inc. and its affiliates.
"""What the teacher distilled, projected from the append-only evolution log.

This is the C1 channel: the cheapest way the system adapts to a changed environment.
Three of the four adaptation actions change nothing except what the acting agent knows,
so a statement like "the send control is now labelled Dispatch and lives in the toolbar"
lets the next session succeed with no code written, no approval, and no trust rebuilt.

The single source of truth is ``evolution_events``: the durable teacher worker commits a
``TEACHER_VERDICT_RECORDED`` fact per verdict, and this store *projects* those facts into
the in-memory read model the acting agent consults. There is no second durable store and
no write-back path; the projection is rebuilt from events on demand.

**Retirement is designed in, not bolted on.** An assertion about a world that keeps
changing is only true for a while, and stale knowledge does not merely go unused -- it
actively misleads, because the acting agent has no way to tell a current fact from one
that expired three upgrades ago. So an entry leaves in exactly three ways:

* **Superseded** -- a newer verdict about the same capability replaces the older one.
  One live entry per capability, because the teacher's latest conclusion is its
  conclusion; keeping the history live would present the agent with a capability's
  contradictory past as though every version were current.
* **Expired** -- entries have a bounded lifetime, configurable rather than fixed.
  Unbounded accumulation would eventually dominate the context it was meant to improve.
* **Retracted** -- an explicit ``KNOWLEDGE_RETRACTED`` fact, used when a capability is
  observed working again and the knowledge describing its failure is therefore obsolete.

What is deliberately *not* a retirement rule: an environment fingerprint that no longer
matches the current one. It is recorded and disclosed, never used to filter. Whether an
OS point release invalidates "the send control is labelled Dispatch" is a judgement about
meaning, and the storage layer guessing it would be a hard rule with no ability to
generalise. The mismatch is surfaced so the reader -- a language model -- can weigh it.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from leapflow.domain.event_types import EvolutionEventType
from leapflow.domain.evolution_event import EvolutionContext, EvolutionEvent, EvolutionEventRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistilledKnowledge:
    """One thing the teacher concluded is true about the environment."""

    capability: str
    knowledge: str
    action: str = ""
    verdict_id: str = ""
    confidence: float = 0.0
    rationale: str = ""
    target: str = ""
    #: The environment this was learned in. Disclosed to the reader, never used to
    #: filter: see the module docstring.
    environment_id: str = ""
    created_at: float = 0.0


class EvolutionDistilledKnowledgeStore:
    """In-memory knowledge read model derived only from evolution events.

    Hydration and refresh are cold-path operations. ``live`` and
    ``rebind_preferences`` only read the in-memory snapshot, so adding this
    always-on context channel does not add a DuckDB query to every turn.
    """

    def __init__(self, event_store: Any, *, profile_id: str, ttl_seconds: float = 0.0) -> None:
        self._event_store = event_store
        self._profile_id = str(profile_id)
        self._ttl = max(0.0, float(ttl_seconds))
        self._entries: dict[str, DistilledKnowledge] = {}
        self._last_sequence = 0
        self._lock = threading.RLock()

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    def refresh(self) -> int:
        """Apply newly committed facts and return the latest consumed sequence."""
        with self._lock:
            cursor = self._last_sequence
            while True:
                records = self._event_store.read(
                    profile_id=self._profile_id,
                    after_sequence=cursor,
                    limit=5000,
                )
                if not records:
                    break
                for record in records:
                    self._apply(record)
                cursor = records[-1].sequence
                if len(records) < 5000:
                    break
            self._last_sequence = cursor
            return cursor

    def live(self, *, now: float | None = None) -> tuple[DistilledKnowledge, ...]:
        """Return the current projected knowledge without performing I/O.

        Expiry is applied on read rather than by a sweep, so a stale entry cannot be
        disclosed just because no write happened to trigger a cleanup.
        """
        with self._lock:
            entries = list(self._entries.values())
        if self._ttl > 0.0:
            cutoff = (time.time() if now is None else now) - self._ttl
            entries = [entry for entry in entries if entry.created_at >= cutoff]
        return tuple(sorted(entries, key=lambda entry: entry.created_at, reverse=True))

    def for_capability(self, capability: str) -> DistilledKnowledge | None:
        name = str(capability or "").strip()
        return next((entry for entry in self.live() if entry.capability == name), None)

    def rebind_preferences(self) -> tuple[tuple[str, str], ...]:
        """``(capability, preferred provider)`` for every live ``rebind`` entry.

        Only ``rebind``, because only that action names a provider that should be chosen.
        An ``escalate`` target names what a *person* must do and an ``absorb`` has no
        target at all, so admitting them would turn an instruction to a human into a
        selection preference.
        """
        return tuple(
            (entry.capability, entry.target)
            for entry in self.live()
            if entry.action == "rebind" and entry.target
        )

    def count(self) -> int:
        return len(self.live())

    def retract(self, capability: str, *, reason: str = "") -> bool:
        """Record retirement as a fact, then update the local read model.

        The one retirement neither supersession nor expiry covers: no newer verdict is
        coming precisely because there is no longer anything wrong.
        """
        name = str(capability or "").strip()
        if not name:
            return False
        with self._lock:
            current = self._entries.get(name)
        if current is None:
            return False
        event = EvolutionEvent.create(
            EvolutionEventType.KNOWLEDGE_RETRACTED,
            context=EvolutionContext(
                profile_id=self._profile_id,
                decision_id=current.verdict_id,
                correlation_id=current.verdict_id,
            ),
            payload={"capability": name, "reason": str(reason or "")},
            producer="evolution.knowledge_projection",
            privacy_class="profile",
            dedup_key=f"knowledge.retracted:{name}:{current.verdict_id}",
        )
        inserted = bool(self._event_store.append(event))
        self.refresh()
        return inserted

    def _apply(self, record: EvolutionEventRecord) -> None:
        event = record.event
        payload = event.to_dict()["payload"]
        if event.event_type == EvolutionEventType.TEACHER_VERDICT_RECORDED:
            # Every verdict carries mandatory knowledge, and all four actions are
            # disclosed to the student: an ``escalate`` tells it a person must act, an
            # ``acquire`` tells it nothing installed serves the capability, and the two
            # cheap verdicts describe the environment. ``rebind_preferences`` narrows to
            # rebind on read, so recording all four here loses nothing.
            entry = DistilledKnowledge(
                capability=str(payload.get("capability") or ""),
                knowledge=str(payload.get("knowledge") or ""),
                action=str(payload.get("action") or ""),
                verdict_id=str(payload.get("verdict_id") or event.context.decision_id),
                confidence=float(payload.get("confidence") or 0.0),
                rationale=str(payload.get("rationale") or ""),
                target=str(payload.get("target") or ""),
                environment_id=str(payload.get("environment_id") or ""),
                created_at=float(payload.get("created_at") or event.occurred_at),
            )
            if entry.capability and entry.knowledge:
                # Supersession, not append: the teacher's latest conclusion about a
                # capability is its conclusion, keyed by capability.
                self._entries[entry.capability] = entry
        elif event.event_type == EvolutionEventType.KNOWLEDGE_RETRACTED:
            self._entries.pop(str(payload.get("capability") or ""), None)


__all__ = [
    "DistilledKnowledge",
    "EvolutionDistilledKnowledgeStore",
]
