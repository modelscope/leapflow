# Copyright (c) Alibaba, Inc. and its affiliates.
"""Durable home for what the teacher distilled, and the rules that retire it.

This is the C1 channel: the cheapest way the system adapts to a changed environment.
Three of the four adaptation actions change nothing except what the acting agent knows,
so a statement like "the send control is now labelled Dispatch and lives in the toolbar"
lets the next session succeed with no code written, no approval, and no trust rebuilt.

**Retirement is designed in, not bolted on.** An assertion about a world that keeps
changing is only true for a while, and stale knowledge does not merely go unused -- it
actively misleads, because the acting agent has no way to tell a current fact from one
that expired three upgrades ago. Telling it "the send control is labelled Dispatch" after
the control was renamed again is worse than telling it nothing. So an entry leaves in
exactly three ways:

* **Superseded** -- a newer verdict about the same capability replaces the older one.
  One live entry per capability, because the teacher's latest conclusion is its
  conclusion; keeping the history live would present the agent with a capability's
  contradictory past as though every version were current.
* **Expired** -- entries have a bounded lifetime, configurable rather than fixed.
  Unbounded accumulation would eventually dominate the context it was meant to improve.
* **Retracted** -- an explicit call, used when a capability is observed working again and
  the knowledge describing its failure is therefore obsolete.

What is deliberately *not* a retirement rule: an environment fingerprint that no longer
matches the current one. It is recorded and disclosed, never used to filter. Whether an
OS point release invalidates "the send control is labelled Dispatch" is a judgement about
meaning, and the storage layer guessing it would be a hard rule with no ability to
generalise -- precisely the kind that looks safe and quietly discards good knowledge. The
mismatch is surfaced so the reader can weigh it; the reader is a language model, and this
is the sort of thing it is better at than a predicate.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: Fields persisted for one entry. An explicit allow-list, matching the observation
#: store's convention -- and carrying its scar: a field absent from that store's list was
#: silently dropped, so a fact reached the teacher with ``None`` where an identity should
#: have been and nothing raised. Adding a field to the record means adding it here.
#:
#: Bound to ``DistilledKnowledge.to_dict()`` *exactly*, not loosely, and a test asserts
#: it. A field the dataclass has and this set lacks is the historical silent drop; an
#: entry here with no matching field is the mirror image -- it claims to persist something
#: that never existed, which is how a whitelist stops being trustworthy. ``"environment"``
#: was exactly that: a leftover from considering whether to store the whole fingerprint.
_ENTRY_FIELDS: frozenset[str] = frozenset(
    {
        "capability",
        "knowledge",
        "action",
        "verdict_id",
        "confidence",
        "rationale",
        "target",
        "environment_id",
        "created_at",
    }
)


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

    @classmethod
    def from_verdict(
        cls, verdict: Any, *, environment: Mapping[str, Any] | None = None
    ) -> DistilledKnowledge:
        """Project an ``AdaptationVerdict`` into a storable entry.

        Takes the verdict duck-typed rather than imported, so storage does not depend on
        the domain module that depends on it.
        """
        env = dict(environment or {})
        return cls(
            capability=str(getattr(verdict, "capability", "") or ""),
            knowledge=str(getattr(verdict, "knowledge", "") or ""),
            action=str(getattr(verdict, "action", "") or ""),
            verdict_id=str(getattr(verdict, "verdict_id", "") or ""),
            confidence=float(getattr(verdict, "confidence", 0.0) or 0.0),
            rationale=str(getattr(verdict, "rationale", "") or ""),
            target=str(getattr(verdict, "target", "") or ""),
            environment_id=str(env.get("fingerprint_id") or ""),
            created_at=float(getattr(verdict, "created_at", 0.0) or time.time()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "knowledge": self.knowledge,
            "action": self.action,
            "verdict_id": self.verdict_id,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "target": self.target,
            "environment_id": self.environment_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DistilledKnowledge:
        return cls(
            capability=str(payload.get("capability") or ""),
            knowledge=str(payload.get("knowledge") or ""),
            action=str(payload.get("action") or ""),
            verdict_id=str(payload.get("verdict_id") or ""),
            confidence=float(payload.get("confidence") or 0.0),
            rationale=str(payload.get("rationale") or ""),
            target=str(payload.get("target") or ""),
            environment_id=str(payload.get("environment_id") or ""),
            created_at=float(payload.get("created_at") or 0.0),
        )


class JsonDistilledKnowledgeStore:
    """Profile-scoped store for distilled knowledge, keyed by capability.

    JSON rather than the semantic memory provider, for a reason that decided the design:
    the reader needs *every* live entry, and semantic memory answers keyword queries. A
    fact the agent needs is not necessarily a fact whose words appear in the request --
    "the send control is now Dispatch" is exactly what a request saying "reply to Ana"
    needs and would never retrieve. Complete enumeration is the requirement, so the store
    that offers it is the right one.

    Reads are hot (every turn that discloses knowledge) and writes are cold (once per
    session, at grading time), which is why an entry cache is kept and invalidated on
    write rather than re-reading the file per turn.
    """

    def __init__(self, path: Path, *, ttl_seconds: float = 0.0) -> None:
        self._path = Path(path)
        self._ttl = max(0.0, float(ttl_seconds))
        self._cache: tuple[DistilledKnowledge, ...] | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    # ── writes (cold path) ────────────────────────────────────────────────────

    def record(
        self, verdict: Any, *, environment: Mapping[str, Any] | None = None
    ) -> DistilledKnowledge | None:
        """Store one verdict's knowledge, superseding any earlier entry for it.

        Returns the stored entry, or ``None`` when the verdict carries nothing usable.
        Refusing silently here would be wrong in the other direction: a verdict without
        knowledge is a defect in the parser, which already rejects that shape, so
        reaching this point means something upstream changed.
        """
        entry = DistilledKnowledge.from_verdict(verdict, environment=environment)
        if not entry.capability or not entry.knowledge:
            logger.debug(
                "distilled_knowledge: refused entry without capability or knowledge (%r)",
                entry.verdict_id,
            )
            return None
        # Supersession, not append: the teacher's latest conclusion about a capability is
        # its conclusion, and keeping the older one live would show the agent a
        # capability's contradictory past as though every version were current.
        kept = [e for e in self._load() if e.capability != entry.capability]
        kept.append(entry)
        self._write(kept)
        return entry

    def record_all(
        self, verdicts: Any, *, environment: Mapping[str, Any] | None = None
    ) -> tuple[DistilledKnowledge, ...]:
        """Store a batch, one write for the lot.

        Later verdicts about the same capability win, matching ``record``'s supersession
        within the batch as well as across batches.
        """
        incoming: dict[str, DistilledKnowledge] = {}
        for verdict in verdicts or ():
            entry = DistilledKnowledge.from_verdict(verdict, environment=environment)
            if entry.capability and entry.knowledge:
                incoming[entry.capability] = entry
        if not incoming:
            return ()
        kept = [e for e in self._load() if e.capability not in incoming]
        kept.extend(incoming.values())
        self._write(kept)
        return tuple(incoming.values())

    def retract(self, capability: str, *, reason: str = "") -> bool:
        """Drop the entry for one capability. Used when its knowledge is obsolete.

        The third retirement path, and the only one a caller drives: a capability
        observed working again makes knowledge describing its failure misleading, and
        nothing about supersession or expiry would remove it -- no newer verdict is
        coming precisely because there is no longer anything wrong.
        """
        name = str(capability or "").strip()
        if not name:
            return False
        entries = self._load()
        kept = [e for e in entries if e.capability != name]
        if len(kept) == len(entries):
            return False
        logger.debug(
            "distilled_knowledge: retracted %s (%s)", name, reason or "no reason given"
        )
        self._write(kept)
        return True

    # ── reads (hot path) ──────────────────────────────────────────────────────

    def live(self, *, now: float | None = None) -> tuple[DistilledKnowledge, ...]:
        """Every entry still considered true, newest first.

        Expiry is applied on read rather than by a sweep, so a stale entry cannot be
        disclosed just because no write happened to trigger a cleanup.
        """
        entries = self._load()
        if self._ttl > 0.0:
            cutoff = (time.time() if now is None else now) - self._ttl
            entries = [e for e in entries if e.created_at >= cutoff]
        return tuple(sorted(entries, key=lambda e: e.created_at, reverse=True))

    def for_capability(self, capability: str) -> DistilledKnowledge | None:
        """The live entry for one capability, if any."""
        name = str(capability or "").strip()
        return next((e for e in self.live() if e.capability == name), None)

    def rebind_preferences(self) -> tuple[tuple[str, str], ...]:
        """``(capability, preferred provider)`` for every live ``rebind`` entry.

        Only ``rebind``, because only that action names a provider that should be chosen.
        An ``escalate`` target names what a *person* must do and an ``absorb`` has no
        target at all, so admitting them would turn an instruction to a human into a
        selection preference.

        Expiry and retraction apply, so a preference stops being read when the knowledge
        behind it stops being true -- there is nothing to unlearn.
        """
        return tuple(
            (entry.capability, entry.target)
            for entry in self.live()
            if entry.action == "rebind" and entry.target
        )

    def count(self) -> int:
        return len(self.live())

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> list[DistilledKnowledge]:
        if self._cache is not None:
            return list(self._cache)
        entries: list[DistilledKnowledge] = []
        if self._path.exists():
            try:
                payload = json.loads(self._path.read_text(encoding="utf-8") or "{}")
                for raw in payload.get("entries", []) or ():
                    if isinstance(raw, Mapping):
                        entries.append(DistilledKnowledge.from_dict(raw))
            except (OSError, ValueError, TypeError):
                # A corrupt file must not break a turn. Distilled knowledge is an
                # improvement to context, so its absence degrades quality rather than
                # correctness -- exactly the case for starting empty over raising.
                logger.debug(
                    "distilled_knowledge: unreadable store at %s", self._path, exc_info=True
                )
                entries = []
        self._cache = tuple(entries)
        return list(entries)

    def _write(self, entries: list[DistilledKnowledge]) -> None:
        self._cache = tuple(entries)
        payload = {
            "version": 1,
            "entries": [
                {k: v for k, v in e.to_dict().items() if k in _ENTRY_FIELDS}
                for e in entries
            ],
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self._path)
        except OSError:
            logger.debug(
                "distilled_knowledge: could not persist to %s", self._path, exc_info=True
            )


__all__ = [
    "DistilledKnowledge",
    "JsonDistilledKnowledgeStore",
]
