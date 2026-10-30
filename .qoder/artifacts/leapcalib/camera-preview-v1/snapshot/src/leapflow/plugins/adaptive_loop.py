# Copyright (c) Alibaba, Inc. and its affiliates.
"""Adaptive plugin closed-loop orchestration primitives.

This module is an application service above the plugin registry. It connects
capability requirements, environment evidence, resolver output, plan persistence,
and approval-gated plugin lifecycle actions without adding intent routing to the
engine loop.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.domain.environment_fingerprint import EnvironmentFingerprint
from leapflow.plugins.capability_plan import CapabilityPlan
from leapflow.plugins.capability_resolver import (
    CapabilityCandidate,
    CapabilityResolution,
    CapabilityResolver,
    CandidateScore,
    ResolverContext,
    candidates_from_registry,
)

logger = logging.getLogger(__name__)

CandidateFilter = Callable[[CapabilityCandidate], bool]


@dataclass(frozen=True)
class AdaptiveLoopMutation:
    """One optional registry mutation to apply between two decisions."""

    action: str = "none"
    plugin_id: str = ""
    code: str = ""
    proposal_id: str = ""
    version_label: str = ""
    delete_source: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "plugin_id": self.plugin_id,
            "proposal_id": self.proposal_id,
            "version_label": self.version_label,
            "delete_source": self.delete_source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AdaptiveLoopRequest:
    """Inputs for one adaptive decision or closed-loop mutation run."""

    environment: EnvironmentFingerprint
    requirements: tuple[CapabilityRequirement, ...]
    source: str = "runtime"
    loop_id: str = ""
    mutation: AdaptiveLoopMutation = field(default_factory=AdaptiveLoopMutation)
    candidate_filter: CandidateFilter | None = None

    @property
    def resolved_loop_id(self) -> str:
        return self.loop_id or f"loop-{uuid.uuid4().hex}"


@dataclass(frozen=True)
class AdaptiveDecision:
    """One persisted resolver decision within a loop."""

    phase: str
    registry_version: int
    candidates: tuple[CapabilityCandidate, ...]
    resolutions: tuple[CapabilityResolution, ...]
    plan: CapabilityPlan
    record: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "registry_version": self.registry_version,
            "candidate_count": len(self.candidates),
            "resolutions": [resolution.to_dict() for resolution in self.resolutions],
            "plan": self.plan.to_dict(),
            "record": dict(self.record),
        }


@dataclass(frozen=True)
class AdaptiveLoopResult:
    """Outcome of an adaptive closed-loop run."""

    loop_id: str
    before: AdaptiveDecision
    after: AdaptiveDecision | None = None
    mutation: AdaptiveLoopMutation = field(default_factory=AdaptiveLoopMutation)
    mutation_result: Mapping[str, Any] = field(default_factory=dict)
    registry_version_before: int = 0
    registry_version_after: int = 0
    selected_delta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        if self.mutation.action == "none":
            return True
        return bool(self.mutation_result.get("ok", False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "loop_id": self.loop_id,
            "mutation": self.mutation.to_dict(),
            "mutation_result": dict(self.mutation_result),
            "registry_version_before": self.registry_version_before,
            "registry_version_after": self.registry_version_after,
            "selected_delta": dict(self.selected_delta),
            "before": self.before.to_dict(),
            "after": self.after.to_dict() if self.after is not None else None,
        }


@runtime_checkable
class PluginLifecycleActor(Protocol):
    """Approval-gated plugin lifecycle operations used by the loop."""

    async def install(
        self,
        *,
        plugin_id: str,
        code: str,
        proposal_id: str = "",
        version_label: str = "",
    ) -> Mapping[str, Any]: ...

    async def disable(self, *, plugin_id: str) -> Mapping[str, Any]: ...

    async def remove(self, *, plugin_id: str, delete_source: bool = True) -> Mapping[str, Any]: ...


class SelfManagementLifecycleActor:
    """Lifecycle actor that delegates to the existing self-management plugin."""

    def __init__(self, self_management_plugin: Any) -> None:
        self._plugin = self_management_plugin

    @classmethod
    def from_registry(cls, registry: Any) -> "SelfManagementLifecycleActor":
        plugin = registry.get_plugin("self_management")
        if plugin is None:
            raise RuntimeError("self_management plugin is not registered")
        return cls(plugin)

    async def install(
        self,
        *,
        plugin_id: str,
        code: str,
        proposal_id: str = "",
        version_label: str = "",
    ) -> Mapping[str, Any]:
        return await self._plugin._plugin_install_handler(
            plugin_id=plugin_id,
            code=code,
            proposal_id=proposal_id,
            version_label=version_label,
        )

    async def disable(self, *, plugin_id: str) -> Mapping[str, Any]:
        return await self._plugin._plugin_disable_handler(plugin_id=plugin_id)

    async def remove(self, *, plugin_id: str, delete_source: bool = True) -> Mapping[str, Any]:
        return await self._plugin._plugin_remove_handler(
            plugin_id=plugin_id,
            delete_source=delete_source,
        )


class CapabilityDecisionRecorder:
    """Persist adaptive decisions with additive closed-loop metadata."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def record(
        self,
        *,
        loop_id: str,
        phase: str,
        source: str,
        environment: EnvironmentFingerprint,
        requirements: Sequence[CapabilityRequirement],
        resolutions: Sequence[CapabilityResolution],
        plan: CapabilityPlan,
        mutation: Mapping[str, Any] | None = None,
        registry_version_before: int = 0,
        registry_version_after: int = 0,
        decision_delta: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return self._store.add_record(
            environment=environment.to_dict(),
            requirements=[requirement.to_dict() for requirement in requirements],
            resolutions=[resolution.to_dict() for resolution in resolutions],
            plan=plan.to_dict(),
            source=source,
            record_id=f"{loop_id}:{phase}",
            phase=phase,
            loop_id=loop_id,
            mutation=dict(mutation or {}),
            registry_version_before=registry_version_before,
            registry_version_after=registry_version_after,
            decision_delta=dict(decision_delta or {}),
        )



def live_learning_signals() -> tuple[Any, Any]:
    """The live trust ledger and usage tracker, or ``(None, None)``.

    Both live on the process-global advisor, which is absent in-process and in most
    tests. Absence is degradation, not failure: ``TrustScorer`` and
    ``ReliabilityScorer`` then report "unavailable" and score 0, which is what they
    did for every production selection before this was wired -- meaning both adaptive
    signals contributed nothing and selection fell to the static scorers with an
    alphabetical tie-break.

    Shared with the evolution board's ``_trust_and_usage`` rather than duplicated:
    two accessors would let the board and the resolver disagree about what trust a
    plugin has.
    """
    try:
        from leapflow.learning.plugin_advisor import get_default_advisor

        advisor = get_default_advisor()
    except Exception:  # noqa: BLE001 - no advisor is a degraded signal, not a failure
        return None, None
    if advisor is None:
        return None, None
    return getattr(advisor, "_trust_ledger", None), getattr(advisor, "_usage_tracker", None)


def _configured_policy(
    trust_ledger: Any, usage_tracker: Any, settings: Any = None
) -> Any:
    """Activate the configured selection policy, or fall back to the built-in default.

    This loop is the policy's owner: it is the component that selects, and the only
    one holding the live services a policy is allowed to read. Activation caches the
    instance process-wide so a learning policy accumulates across turns and the
    cold-path sweep reports back to the *same* object.

    ``settings`` is pushed in by the caller rather than read here, because
    ``get_settings()`` is a boot snapshot with no refresh path: reading it would pin
    the policy to whatever configuration existed at process start and make
    ``selection.policy`` a setting that reports itself as hot-reloadable and never
    changes anything.

    Returns ``None`` on any failure, which leaves ``CapabilityResolver`` to construct
    ``GreedyPolicy`` itself. Selection must keep working when configuration is absent
    or wrong -- a misconfigured strategy is a reason to log and use the default, never
    a reason to stop choosing tools.
    """
    try:
        from leapflow.plugins.selection_policy import PolicyDeps
        from leapflow.plugins.selection_policy_registry import get_selection_policy_registry

        return get_selection_policy_registry().activate(
            PolicyDeps(trust_ledger=trust_ledger, usage_tracker=usage_tracker),
            settings=settings,
        )
    except Exception:  # noqa: BLE001 - configuration must not break tool selection
        logger.debug("adaptive_loop: falling back to the default selection policy", exc_info=True)
        return None

class AdaptivePluginLoop:
    """Resolve capability plans before and after approval-gated registry mutations."""

    def __init__(
        self,
        *,
        registry: Any,
        plan_store: Any,
        lifecycle_actor: PluginLifecycleActor | None = None,
        resolver: CapabilityResolver | None = None,
        trust_ledger: Any = None,
        usage_tracker: Any = None,
        settings: Any = None,
        distilled_preferences: Any = None,
    ) -> None:
        self._registry = registry
        self._lifecycle_actor = lifecycle_actor
        self._trust_ledger = trust_ledger
        self._usage_tracker = usage_tracker
        # Channel C2. Read per resolution rather than captured, because the teacher's
        # recommendation is superseded and retracted between sessions and a snapshot would
        # keep preferring a provider the knowledge behind it no longer endorses.
        self._distilled_preferences = distilled_preferences
        self._resolver = resolver or CapabilityResolver(
            policy=_configured_policy(trust_ledger, usage_tracker, settings)
        )
        self._recorder = CapabilityDecisionRecorder(plan_store)

    def plan_next_action(
        self,
        proposal: Any,
        policy: Any,
        *,
        trust_level: Any = None,
        usage: Mapping[str, Any] | None = None,
        sandbox_validated: bool = False,
        rollback_available: bool = False,
    ) -> Any:
        """Delegate structured proposal state to an adaptive evolution policy."""
        return policy.decide(
            proposal,
            trust_level=trust_level if trust_level is not None else "DRAFT",
            usage=usage or {},
            sandbox_validated=sandbox_validated,
            rollback_available=rollback_available,
        )

    def _read_preferences(self) -> tuple[tuple[str, str], ...]:
        """The teacher's rebind recommendations, or nothing.

        Contained and empty on failure: a missing preference costs a better choice, while
        raising here would cost the resolution itself.
        """
        if self._distilled_preferences is None:
            return ()
        try:
            return tuple(self._distilled_preferences() or ())
        except Exception:  # noqa: BLE001 - a preference is evidence, never a gate
            logger.debug("adaptive_loop: distilled preferences unavailable", exc_info=True)
            return ()

    def resolve_once(
        self,
        request: AdaptiveLoopRequest,
        *,
        loop_id: str | None = None,
        phase: str = "resolve",
        mutation: Mapping[str, Any] | None = None,
        registry_version_before: int = 0,
        registry_version_after: int = 0,
        decision_delta: Mapping[str, Any] | None = None,
    ) -> AdaptiveDecision:
        """Resolve and persist one decision without mutating the registry."""
        resolved_loop_id = loop_id or request.resolved_loop_id
        self._registry.assemble()
        candidates = tuple(candidates_from_registry(self._registry))
        if request.candidate_filter is not None:
            candidates = tuple(
                candidate for candidate in candidates if request.candidate_filter(candidate)
            )
        context = ResolverContext(
            environment=request.environment,
            trust_ledger=self._trust_ledger,
            usage_tracker=self._usage_tracker,
            # Channel C2: the provider the teacher named in a ``rebind``. A preference
            # added to a score, never an exclusion -- an inadmissible candidate still
            # loses to the affordance scorer no matter what was recommended.
            distilled_preferences=self._read_preferences(),
        )
        resolutions = self._resolver.resolve_all(request.requirements, candidates, context)
        plan = CapabilityPlan.from_scores(
            _selected_scores(resolutions), plan_id=f"plan-{resolved_loop_id}-{phase}"
        )
        record = self._recorder.record(
            loop_id=resolved_loop_id,
            phase=phase,
            source=request.source,
            environment=request.environment,
            requirements=request.requirements,
            resolutions=resolutions,
            plan=plan,
            mutation=mutation,
            registry_version_before=registry_version_before,
            registry_version_after=registry_version_after,
            decision_delta=decision_delta,
        )
        return AdaptiveDecision(
            phase=phase,
            registry_version=self._registry.version,
            candidates=candidates,
            resolutions=resolutions,
            plan=plan,
            record=record,
        )

    def unmet_requirements(
        self,
        requirements: Sequence[CapabilityRequirement],
        environment: EnvironmentFingerprint,
        *,
        candidate_filter: CandidateFilter | None = None,
        scorers: Any = None,
        authorising_origins: Sequence[str] | None = None,
    ) -> tuple[CapabilityRequirement, ...]:
        """Resolution-first gap gate: return only requirements the live registry
        cannot already satisfy.

        ``authorising_origins`` restricts which requirement *origins* may drive an
        acquisition. Empty or ``None`` means unrestricted, which is the shipped
        behaviour. Setting it to ``("world_model",)`` is the enforceable form of
        "self-evolution is first-driven by the world model": a requirement from any
        other origin is still resolved and still reported by the caller, but is
        excluded from the gap set, so it cannot reach the proposal queue. The filter
        is applied *before* resolution so an unauthorised origin cannot even consume
        resolver work.

        Shipped adaptive evolution has no short-circuit between "a requirement
        exists" and "propose a plugin", so it can generate and install a
        capability the live catalog already provides. This method resolves each
        requirement against the current registry first; a requirement whose best
        candidate is eligible is *not* a gap and is excluded from the result. Only
        the returned (unmet) requirements should enter the proposal queue.

        Read with respect to the capability set: it performs no install, remove,
        or publish. It does call ``assemble()``, which is idempotent and is the
        same precondition ``candidates_from_registry`` already requires; on a
        registry that has never been assembled this performs the one-time index
        and version bump any first read triggers, so a caller that needs a
        strictly untouched version counter should assemble beforehand. Callers
        that resolve against a task environment pass the environment-aware
        ``scorers`` (e.g. including ``EnvironmentAffordanceScorer``); ``None`` uses
        the resolver's default scorers.
        """
        self._registry.assemble()
        # Authority filter first: an origin that may not drive acquisition should not
        # consume resolver work, and must not appear in the gap set at all.
        authorised = tuple(requirements)
        if authorising_origins:
            from leapflow.learning.outcome_governance_feed import filter_authorised

            authorised = filter_authorised(authorised, authorising_origins)
            if not authorised:
                return ()
        candidates = tuple(candidates_from_registry(self._registry))
        if candidate_filter is not None:
            candidates = tuple(c for c in candidates if candidate_filter(c))
        resolver = CapabilityResolver(scorers) if scorers is not None else self._resolver
        context = ResolverContext(
            environment=environment,
            trust_ledger=self._trust_ledger,
            usage_tracker=self._usage_tracker,
        )
        resolutions = resolver.resolve_all(authorised, candidates, context)
        return tuple(r.requirement for r in resolutions if r.unmet)

    async def run(self, request: AdaptiveLoopRequest) -> AdaptiveLoopResult:
        """Resolve, optionally mutate the registry, and resolve again."""
        loop_id = request.resolved_loop_id
        registry_before = int(getattr(self._registry, "version", 0))
        before = self.resolve_once(
            request,
            loop_id=loop_id,
            phase="before",
            registry_version_before=registry_before,
            registry_version_after=registry_before,
        )
        mutation = request.mutation
        if mutation.action == "none":
            return AdaptiveLoopResult(
                loop_id=loop_id,
                before=before,
                registry_version_before=registry_before,
                registry_version_after=registry_before,
            )
        if self._lifecycle_actor is None:
            raise RuntimeError("lifecycle_actor is required for registry mutation")

        mutation_result = await self._apply_mutation(mutation)
        registry_after = int(getattr(self._registry, "version", 0))
        after = self.resolve_once(
            request,
            loop_id=loop_id,
            phase=f"after_{mutation.action}",
            mutation=mutation.to_dict(),
            registry_version_before=registry_before,
            registry_version_after=registry_after,
        )
        delta = _selected_delta(before.resolutions, after.resolutions)
        return AdaptiveLoopResult(
            loop_id=loop_id,
            before=before,
            after=after,
            mutation=mutation,
            mutation_result=mutation_result,
            registry_version_before=registry_before,
            registry_version_after=registry_after,
            selected_delta=delta,
        )

    async def _apply_mutation(self, mutation: AdaptiveLoopMutation) -> Mapping[str, Any]:
        if mutation.action == "install":
            return await self._lifecycle_actor.install(
                plugin_id=mutation.plugin_id,
                code=mutation.code,
                proposal_id=mutation.proposal_id,
                version_label=mutation.version_label,
            )
        if mutation.action == "disable":
            return await self._lifecycle_actor.disable(plugin_id=mutation.plugin_id)
        if mutation.action == "remove":
            return await self._lifecycle_actor.remove(
                plugin_id=mutation.plugin_id,
                delete_source=mutation.delete_source,
            )
        return {"ok": False, "error": f"Unsupported mutation action: {mutation.action}"}


def _selected_scores(resolutions: Sequence[CapabilityResolution]) -> tuple[CandidateScore, ...]:
    return tuple(
        resolution.selected for resolution in resolutions if resolution.selected is not None
    )


def _selected_map(resolutions: Sequence[CapabilityResolution]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for resolution in resolutions:
        if resolution.selected is None:
            continue
        selected[resolution.requirement.capability] = resolution.selected.candidate.tool_name
    return selected


def _selected_delta(
    before: Sequence[CapabilityResolution],
    after: Sequence[CapabilityResolution],
) -> dict[str, Any]:
    before_map = _selected_map(before)
    after_map = _selected_map(after)
    changed = {
        capability: {
            "before": before_map.get(capability, ""),
            "after": after_map.get(capability, ""),
        }
        for capability in sorted(set(before_map) | set(after_map))
        if before_map.get(capability) != after_map.get(capability)
    }
    return {
        "before": before_map,
        "after": after_map,
        "changed": changed,
        "added": {key: after_map[key] for key in sorted(after_map.keys() - before_map.keys())},
        "removed": {key: before_map[key] for key in sorted(before_map.keys() - after_map.keys())},
    }


__all__ = [
    "AdaptiveDecision",
    "AdaptiveLoopMutation",
    "AdaptiveLoopRequest",
    "AdaptiveLoopResult",
    "AdaptivePluginLoop",
    "CapabilityDecisionRecorder",
    "PluginLifecycleActor",
    "SelfManagementLifecycleActor",
]
