"""Contracts for the capability-evolution lifecycle.

``AdaptiveEvolutionPolicy`` and ``LifecycleGovernor`` are the trust, probation and
quarantine machinery. They were written against one concrete backing store, which
left them reachable only from whatever fills that store -- in practice, nothing in
production. These Protocols state what each component actually requires, so either
can be driven by any store that satisfies the contract, including one fed by the
live ``plugin_propose -> plugin_generate -> plugin_install`` chain.

Two distinct vocabularies meet here, and conflating them is the mistake to avoid:

* ``domain.plugin_proposal.ProposalStatus`` -- ``draft | review | approved |
  rejected`` -- is a **review** state: should a human accept this proposal?
* ``storage.capability_proposal_queue.ProposalStatus`` -- ``PENDING | GENERATED |
  APPROVED | INSTALLED | PROBATION | VERIFIED | REJECTED | FAILED | QUARANTINED``
  -- is an **acquisition lifecycle** state: where is this capability in its
  journey from hypothesis to trusted?

They are not duplicates and must not be merged into one field. A proposal that a
human has ``approved`` may still be anywhere in its lifecycle. The lifecycle store
below owns the second vocabulary.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class EvolutionProposalView(Protocol):
    """What ``AdaptiveEvolutionPolicy`` reads when deciding the next action.

    A read-only projection: the policy inspects lifecycle status and declared risk
    and returns a decision. It never writes, so any object exposing these
    attributes can be evaluated -- including a view backed by a live plugin
    proposal rather than the default queue item.
    """

    proposal_id: str
    status: str
    requirements: tuple[Mapping[str, Any], ...]
    risk: Mapping[str, Any]


@runtime_checkable
class EvolutionLifecycleStore(Protocol):
    """Where ``LifecycleGovernor`` records lifecycle transitions.

    The governor reads nothing back during a transition; it writes the new status
    plus the evidence for it (trust state, outcome, install result). Implementations
    must treat an unknown ``proposal_id`` as a no-op rather than raising, because a
    governance write must never fail the turn that produced the outcome.
    """

    def get(self, proposal_id: str) -> Any | None:
        """Return the stored record, or ``None`` when it is unknown."""
        ...

    def update(
        self,
        proposal_id: str,
        *,
        status: str | None = None,
        policy_decision: Mapping[str, Any] | None = None,
        install_result: Mapping[str, Any] | None = None,
        test_results: Sequence[Mapping[str, Any]] | None = None,
        trust_state: Mapping[str, Any] | None = None,
    ) -> Any:
        """Apply a lifecycle transition and return the updated record."""
        ...


@runtime_checkable
class OutcomeStore(Protocol):
    """Where ``LifecycleGovernor`` records per-execution outcomes.

    ``failure_streak`` is the consecutive-failure count the governor compares
    against its quarantine threshold, so an implementation must reset it on
    success.
    """

    def add_outcome(self, **kwargs: Any) -> Mapping[str, Any]:
        """Record one outcome and return the stored record."""
        ...

    def failure_streak(self, plugin_id: str) -> int:
        """Consecutive failures for the plugin, reset by any success."""
        ...


__all__ = ["EvolutionLifecycleStore", "EvolutionProposalView", "OutcomeStore"]
