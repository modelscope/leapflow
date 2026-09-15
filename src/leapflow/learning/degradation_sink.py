# Copyright (c) Alibaba, Inc. and its affiliates.
"""Wire plugin health into capability-scoped evidence.

``LifecycleGovernor`` deliberately holds no registry: it knows a plugin failed, not what
that plugin was *for*. The teacher needs the opposite -- which **capability** is degraded,
because a capability is what a rival could be built for and what knowledge can be attached
to. This module is that translation, and it is the reason the governor takes a sink rather
than reaching for the registry itself.

It exists because the chain it completes was inert. ``self.lifecycle_governor`` was never
assigned anywhere in production, so the sweep resolved it to ``None``, ``record_outcome``
was never called, and the degradation evidence that the teacher prompt, the challenger
identity, and the proposal path were all built to consume was never produced. Every unit
test passed, because every unit test constructed the governor itself.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)


def declared_capabilities_by_plugin(registry: Any) -> dict[str, tuple[str, ...]]:
    """Map plugin id to the capabilities its live tools declare.

    Delegates to the resolver's own candidate builder rather than walking the registry
    again. That builder applies two filters this translation must not lose: first-wins
    name arbitration (a shadowed tool is not live) and handler presence (an unbound tool
    is not callable). Re-implementing the walk would let a plugin be degraded under a
    capability it does not actually serve in this process.

    Declaration only -- nothing is inferred from a tool's name, which is the rule that
    keeps a capability a thing a plugin *claims* rather than a thing a substring guessed.
    """
    from leapflow.plugins.capability_resolver import candidates_from_registry

    declared: dict[str, set[str]] = {}
    for candidate in candidates_from_registry(registry):
        if candidate.provides_capabilities:
            declared.setdefault(candidate.plugin_id, set()).update(
                candidate.provides_capabilities
            )
    return {plugin_id: tuple(sorted(names)) for plugin_id, names in declared.items()}


def build_degradation_sink(
    *,
    intake: Any,
    registry_provider: Callable[[], Any],
    knowledge_store: Any = None,
    environment_provider: Callable[[], Mapping[str, Any]] | None = None,
) -> Callable[..., None]:
    """Return the sink that turns plugin health into capability evidence.

    Both directions are handled, because a health signal that only fires one way has no
    way back:

    * **A non-zero streak** writes a ``capability_degraded`` observation per declared
      capability, so the teacher learns that something is failing *while still serving* --
      the state between healthy and quarantined, which had no expression before.
    * **A zero streak** retracts any distilled knowledge for those capabilities. This is
      the one retirement neither supersession nor expiry covers: no newer verdict is
      coming precisely because there is no longer anything wrong, so knowledge describing
      the failure would otherwise outlive the failure and mislead every later session.

    The registry is resolved through a callable rather than captured, because plugins are
    installed and reloaded at runtime and a snapshot taken at wiring time would report a
    capability set that has since changed.
    """

    def sink(
        *,
        plugin_id: str,
        failure_streak: int,
        trust_level: str,
        failure_class: str = "",
    ) -> None:
        try:
            declared = declared_capabilities_by_plugin(registry_provider())
        except Exception:  # noqa: BLE001 - governance reporting is advisory
            logger.debug("degradation_sink: registry unavailable", exc_info=True)
            return
        capabilities = declared.get(str(plugin_id), ())
        if not capabilities:
            # A plugin that declares no capability cannot be degraded *as* one. Nothing
            # to report, and inventing a name from the plugin id would put a fabricated
            # capability in front of the teacher.
            return

        if int(failure_streak) <= 0:
            _retire(knowledge_store, capabilities, plugin_id)
            return

        environment: Mapping[str, Any] = {}
        if environment_provider is not None:
            try:
                environment = environment_provider() or {}
            except Exception:  # noqa: BLE001 - a fingerprint is context, not a gate
                environment = {}
        for capability in capabilities:
            try:
                intake.observe_result(
                    {
                        "error_type": "capability_degraded",
                        "capability": capability,
                        "plugin_id": str(plugin_id),
                        "failure_streak": int(failure_streak),
                        "trust_level": str(trust_level),
                        "failure_class": str(failure_class or ""),
                    },
                    environment=environment,
                )
            except Exception:  # noqa: BLE001 - one capability must not stop the rest
                logger.debug(
                    "degradation_sink: could not record %s", capability, exc_info=True
                )

    return sink


def build_proposal_sink(*, queue: Any) -> Callable[[Any], str]:
    """Return the sink that turns an accepted acquisition into a queued proposal.

    The last hop of the acquisition chain, and it was missing: the driver derived an
    ``EvolutionIntent`` from an ``acquire`` verdict, turned it into a requirement, and
    stopped. Nothing enqueued it, so resolution reported the capability unmet forever and
    the teacher's most expensive verdict -- the only one that leads to code -- had no
    effect at all.

    Queueing is not acting. The queue is read by the evolution dashboard and by the
    ``self_management`` tools, both of which pass through approval before anything is
    generated, so this hop makes the proposal *visible and actionable* rather than
    executed. That separation is why the sink can be wired by default while generation
    stays governed.
    """

    def sink(proposal: Any) -> str:
        requirement = _requirement_from(proposal)
        if requirement is None:
            # Without a capability the queue has nothing to deduplicate on and resolution
            # has nothing to satisfy, so the item could never be closed.
            logger.debug(
                "proposal_sink: refused proposal without a capability (%r)",
                getattr(proposal, "proposal_id", ""),
            )
            return ""
        evidence = tuple(getattr(proposal, "evidence", ()) or ())
        metadata = dict(getattr(evidence[0], "metadata", {})) if evidence else {}
        try:
            item = queue.enqueue(
                requirements=(requirement,),
                source="world_model",
                risk={"max_risk_level": requirement.max_risk_level},
                metadata={
                    "plugin_id": str(getattr(proposal, "plugin_id", "")),
                    "capability_summary": str(getattr(proposal, "capability_summary", "")),
                    # Carried so a reviewer can see what a challenger is challenging, and
                    # so a rival stays distinguishable from a gap fill for the same
                    # capability.
                    "replaces": str(metadata.get("replaces", "")),
                },
            )
        except Exception:  # noqa: BLE001 - queueing must not fail the session
            logger.debug("proposal_sink: could not enqueue", exc_info=True)
            return ""
        return str(getattr(item, "proposal_id", ""))

    return sink


def _requirement_from(proposal: Any) -> Any:
    """Rebuild the requirement the queue keys on, from the proposal's own evidence.

    ``max_risk_level`` is passed explicitly because the domain default is ``external`` --
    the most permissive value there is. Omitting it would let a proposal that was clamped
    to ``read_only`` enter the queue asking for everything, which is the opposite of what
    the clamp exists for.

    ``requirement_id`` is derived from the capability rather than minted fresh, because the
    queue deduplicates on a hash of the requirement payload. A new uuid on every rebuild
    defeated that silently: the same capability enqueued a new proposal every session, so a
    reviewer would face a growing pile of identical items and the health of the queue would
    measure how long the process had been running.
    """
    from leapflow.domain.capability_requirement import CapabilityRequirement

    evidence = tuple(getattr(proposal, "evidence", ()) or ())
    metadata = dict(getattr(evidence[0], "metadata", {})) if evidence else {}
    capability = str(metadata.get("capability") or "").strip()
    if not capability:
        return None
    return CapabilityRequirement.create(
        capability,
        "world_model",
        evidence=str(getattr(proposal, "capability_summary", "") or capability),
        max_risk_level=str(getattr(proposal, "risk_level", "read_only")),
        requirement_id=f"req-wm-{capability}",
    )


def build_alternatives_provider(
    *, registry_provider: Callable[[], Any], affordances_provider: Callable[[], Any] | None = None
) -> Callable[[str, str], tuple[dict[str, Any], ...]]:
    """Return a reader for the *other* providers of a capability, and whether each fits.

    Without this the teacher is asked to choose between two actions whose definitions are
    exactly the fact it was never given:

        rebind  -- "another installed capability already covers the new environment"
        acquire -- "nothing installed covers this"

    It was shown a flat list of global capability *names* and nothing about how many
    providers a capability has or whether any of them can run here. Measured on a real
    model: ``rebind`` on 3 of 3 trials of a unit whose candidate set had one entry, then
    three different answers in three trials once the catalogue stopped implying that
    everything listed fits. That is what choosing without the deciding fact looks like.

    Admissibility comes from the same declaration the resolver scores on
    (``requires_environment_affordances``), so the teacher and the selection layer cannot
    disagree about what is available. Reported, never enforced: the teacher may still
    answer ``acquire`` when an alternative exists but is a poor fit, which is a judgement
    only it can make.
    """

    def alternatives(capability: str, incumbent: str = "") -> tuple[dict[str, Any], ...]:
        from leapflow.plugins.capability_resolver import candidates_from_registry

        try:
            present = frozenset(str(a) for a in (affordances_provider() or ())) if affordances_provider else frozenset()
        except Exception:  # noqa: BLE001 - unknown affordances must not hide alternatives
            present = frozenset()
        try:
            candidates = candidates_from_registry(registry_provider())
        except Exception:  # noqa: BLE001 - context, never a gate
            logger.debug("alternatives: registry unavailable", exc_info=True)
            return ()
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            if capability not in candidate.provides_capabilities:
                continue
            if incumbent and candidate.plugin_id == incumbent:
                continue
            required = frozenset(candidate.requires_environment_affordances)
            rows.append(
                {
                    "plugin_id": candidate.plugin_id,
                    "tool_name": candidate.tool_name,
                    # Unknown affordances read as "fits": claiming a candidate does not fit
                    # because the environment could not be described would push every
                    # verdict toward acquire, which is the expensive direction.
                    "fits_here": (not required) or (not present) or required <= present,
                    "requires": tuple(sorted(required)),
                }
            )
        return tuple(rows)

    return alternatives


def _retire(knowledge_store: Any, capabilities: tuple[str, ...], plugin_id: str) -> None:
    """Drop knowledge about capabilities that are working again."""
    if knowledge_store is None:
        return
    for capability in capabilities:
        try:
            knowledge_store.retract(
                capability, reason=f"{plugin_id} succeeded; streak reset"
            )
        except Exception:  # noqa: BLE001 - retirement is advisory
            logger.debug(
                "degradation_sink: could not retract %s", capability, exc_info=True
            )


__all__ = [
    "build_alternatives_provider",
    "build_degradation_sink",
    "build_proposal_sink",
    "declared_capabilities_by_plugin",
]
