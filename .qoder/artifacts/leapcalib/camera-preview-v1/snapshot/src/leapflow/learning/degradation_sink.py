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


def build_live_capability_resolver(
    *,
    registry_provider: Callable[[], Any],
    environment_provider: Callable[[], Any],
) -> Callable[[Any, Mapping[str, Any]], dict[str, Any]]:
    """Build the mandatory resolution-before-acquisition gate.

    The result is evidence, not authority. A resolved capability becomes a
    durable no-op; only an unmet requirement may be turned into a proposal.
    Resolution failure is reported explicitly so callers can fail closed.
    """
    from leapflow.plugins.capability_resolver import (
        CapabilityResolver,
        DeclaredMatchScorer,
        EnvironmentAffordanceScorer,
        EnvironmentFitScorer,
        ResolverContext,
        RiskCostScorer,
        candidates_from_registry,
    )

    resolver = CapabilityResolver(
        scorers=(
            DeclaredMatchScorer(),
            EnvironmentFitScorer(),
            EnvironmentAffordanceScorer(),
            RiskCostScorer(),
        )
    )

    def resolve(intent: Any, job: Mapping[str, Any]) -> dict[str, Any]:
        try:
            registry = registry_provider()
            registry.assemble()
            environment = environment_provider()
            if environment is None:
                return {
                    "resolved": False,
                    "satisfied": False,
                    "reason": "environment_unavailable",
                    "environment": {},
                }
            requirement = intent.to_requirement()
            decision = resolver.resolve_one(
                requirement,
                candidates_from_registry(registry),
                ResolverContext(environment=environment),
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            logger.warning("live capability resolution failed", exc_info=True)
            return {
                "resolved": False,
                "satisfied": False,
                "reason": "live_resolution_failed",
                "environment": {},
            }
        selected = decision.selected
        return {
            "resolved": True,
            "satisfied": selected is not None,
            "reason": decision.reason,
            "selected_plugin_id": (
                selected.candidate.plugin_id if selected is not None else ""
            ),
            "selected_tool_name": (
                selected.candidate.tool_name if selected is not None else ""
            ),
            "candidate_count": len(decision.candidates),
            "environment": {
                **environment.to_dict(),
                "workspace_id": str(job.get("workspace_id") or ""),
                "session_id": str(job.get("session_id") or ""),
            },
        }

    return resolve


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
    "build_degradation_sink",
    "build_live_capability_resolver",
    "declared_capabilities_by_plugin",
]
