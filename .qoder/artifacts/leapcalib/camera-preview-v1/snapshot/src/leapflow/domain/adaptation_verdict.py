# Copyright (c) Alibaba, Inc. and its affiliates.
"""What the teacher concluded an environment change warrants.

The world model is asked what *action* the evidence supports, never who is at fault.
Those are not the same question and conflating them suppresses the case that most needs
answering: when an application upgrades, the incumbent implementation was not written
wrongly -- it was right for the old version -- yet a new adapter may still be the only
way forward. A prompt that asks "is the implementation wrong?" gets "no" and nothing
happens.

So the answer space is the set of things the system can actually do about a change,
ordered by cost:

* ``absorb``   the retry or semantic-addressing layer already handles it; the capability
                set does not change. Cheapest, and the correct answer most of the time.
* ``rebind``   another installed plugin already covers the new environment; name it.
* ``acquire``  nothing covers it, so a new implementation is warranted. **The only
                verdict that leads to code being written**, and therefore the only one
                that becomes an :class:`EvolutionIntent`.
* ``escalate`` it needs a human -- a scope, a credential, a decision the agent cannot
                make for itself.

Every verdict carries ``knowledge``, and that is mandatory rather than optional. A
verdict without it teaches the student nothing, so the teacher would have done no useful
work even when its judgement was correct. Distilling what the environment now looks like
is the cheapest way to adapt and the reason this type exists at all: three of the four
verdicts change nothing except what the student knows.

A verdict is a *hypothesis with a recommendation*. It carries no authorisation: an
``acquire`` still passes validation, approval, sandboxing and trust exactly as an
``unknown_tool`` signal does, and an ``escalate`` produces a message rather than an
action.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from leapflow.domain.evolution_intent import (
    EvolutionIntent,
    RiskLevel,
    is_capability_name,
)

#: The action space. Not an open string: a verdict outside this set has no consumer, and
#: silently ignoring one would look identical to the teacher having nothing to say.
AdaptationAction = Literal["absorb", "rebind", "acquire", "escalate"]

ADAPTATION_ACTIONS: frozenset[str] = frozenset({"absorb", "rebind", "acquire", "escalate"})

#: The only verdict that results in code being written.
ACQUIRE: str = "acquire"


@dataclass(frozen=True)
class AdaptationVerdict:
    """One teacher conclusion about one capability, with what the student should know."""

    verdict_id: str
    action: AdaptationAction
    capability: str
    #: What the student should know as a result. Rendered into its context, so it must
    #: read as a statement about the world rather than an instruction to the framework.
    knowledge: str
    rationale: str = ""
    confidence: float = 0.0
    #: For ``rebind``, the plugin or tool that should serve this capability instead.
    #: For ``escalate``, what the human has to do. Empty otherwise.
    target: str = ""
    #: Only consulted for ``acquire``, and still clamped downstream by the trusted
    #: caller: a model cannot widen the ceiling of what it asks to have built.
    max_risk_level: RiskLevel = "read_only"
    expected_effect: str = ""
    target_affordance: str = ""
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    created_at: float = 0.0

    @classmethod
    def create(
        cls,
        action: str,
        capability: str,
        knowledge: str,
        *,
        rationale: str = "",
        confidence: float = 0.0,
        target: str = "",
        max_risk_level: RiskLevel = "read_only",
        expected_effect: str = "",
        target_affordance: str = "",
        evidence_ids: Any = None,
        verdict_id: str = "",
        created_at: float | None = None,
    ) -> AdaptationVerdict:
        """Build a normalised verdict, refusing the shapes that cannot be acted on.

        Rejects rather than repairs. A verdict outside the action space, without a
        capability, or without knowledge has no consumer -- and quietly dropping it
        downstream would be indistinguishable from the teacher having said nothing,
        which is exactly the failure mode that made a whole pipeline look idle while it
        was in fact producing on every session.
        """
        normalized_action = str(action or "").strip().lower()
        if normalized_action not in ADAPTATION_ACTIONS:
            raise ValueError(
                f"action must be one of {sorted(ADAPTATION_ACTIONS)}, got {action!r}"
            )
        normalized_capability = str(capability or "").strip()
        if not is_capability_name(normalized_capability):
            raise ValueError(
                "capability must be a short dotted name such as 'chat.reply', "
                f"got {capability!r}"
            )
        distilled = str(knowledge or "").strip()
        if not distilled:
            raise ValueError(
                "knowledge is required: a verdict that teaches the student nothing "
                "leaves the teacher with no effect even when its judgement is right"
            )
        return cls(
            verdict_id=verdict_id or f"adv-{uuid.uuid4().hex}",
            action=normalized_action,  # type: ignore[arg-type]
            capability=normalized_capability,
            knowledge=distilled,
            rationale=str(rationale or ""),
            confidence=max(0.0, min(1.0, float(confidence))),
            target=str(target or "").strip(),
            max_risk_level=max_risk_level,
            expected_effect=str(expected_effect or ""),
            target_affordance=str(target_affordance or ""),
            evidence_ids=tuple(str(item) for item in (evidence_ids or ()) if str(item)),
            created_at=time.time() if created_at is None else float(created_at),
        )

    @property
    def writes_code(self) -> bool:
        """Whether acting on this verdict would generate an implementation."""
        return self.action == ACQUIRE

    def to_intent(self) -> EvolutionIntent | None:
        """Derive the acquisition intent, or ``None`` for the other three verdicts.

        Derivation rather than a parallel field, so an ``EvolutionIntent`` can only ever
        exist because a verdict asked for one. Keeping the two independent is how "I
        recommend doing X" and "I want a new capability" get mixed into one object, and
        then a recommendation to *rebind* silently queues an acquisition.

        The requested risk level is carried through **unclamped**. Clamping here as well
        looked safer and destroyed the audit trail: the single clamp point downstream
        records the original request only when it differs from what was granted, so
        pre-clamping made the two equal and an approver could no longer see that the
        model had asked for more than it got. One clamp, one place.
        """
        if not self.writes_code:
            return None
        return EvolutionIntent.create(
            self.capability,
            # ``knowledge`` states what is true about the environment, which is exactly
            # what an intent's hypothesis is for. ``rationale`` answers a different
            # question -- why acquire rather than something cheaper -- and belongs in
            # its own field, where the approver reads it.
            self.knowledge,
            confidence=self.confidence,
            target_affordance=self.target_affordance,
            rationale=self.rationale,
            expected_effect=self.expected_effect,
            max_risk_level=self.max_risk_level,
            evidence_ids=self.evidence_ids,
            # Identity comes *from the verdict*, so the derivation is pure. Letting
            # ``EvolutionIntent.create`` mint its own id made this property return a
            # different intent on every read: an immutable domain object whose derived
            # value changed each time it was looked at, which no test would notice
            # until two reads were compared. It also makes an intent traceable back to
            # the verdict that asked for it.
            intent_id=f"wmi-{self.verdict_id.removeprefix('adv-')}",
            created_at=self.created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict_id": self.verdict_id,
            "action": self.action,
            "capability": self.capability,
            "knowledge": self.knowledge,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "target": self.target,
            "writes_code": self.writes_code,
            "max_risk_level": str(self.max_risk_level),
            "expected_effect": self.expected_effect,
            "target_affordance": self.target_affordance,
            "evidence_ids": list(self.evidence_ids),
            "created_at": self.created_at,
        }


__all__ = [
    "ACQUIRE",
    "ADAPTATION_ACTIONS",
    "AdaptationAction",
    "AdaptationVerdict",
]
