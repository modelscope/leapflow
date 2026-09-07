"""Capability orchestration plan derived from selected plugin candidates.

The plan is intentionally declarative. It describes dependency order and risk
metadata for UI/PCD consumption; actual tool execution remains owned by the
existing engine loop and ToolExecutionPipeline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from typing import Any, Sequence

from leapflow.plugins.capability_resolver import CapabilityCandidate, CandidateScore


def _execution_policy_for(candidate: CapabilityCandidate) -> str:
    """Derive a coarse execution policy from declared risk metadata."""
    if candidate.risk_level == "read_only" and not candidate.mutates_state:
        return "read_only"
    if candidate.risk_level == "external":
        return "external_side_effect"
    if candidate.requires_approval:
        return "mutating_once"
    return "mutating_idempotent"


@dataclass(frozen=True)
class MissingCapabilityDependency:
    """A step requires an abstract capability no selected step provides."""

    step_id: str
    capability: str

    def to_dict(self) -> dict[str, str]:
        return {"step_id": self.step_id, "capability": self.capability}


@dataclass(frozen=True)
class CapabilityPlanStep:
    """One declarative step in a capability plan."""

    step_id: str
    plugin_id: str
    tool_name: str
    provides_capabilities: tuple[str, ...] = field(default_factory=tuple)
    requires_capabilities: tuple[str, ...] = field(default_factory=tuple)
    execution_policy: str = "read_only"
    requires_approval: bool = False
    reason: str = ""

    @classmethod
    def from_candidate(
        cls,
        candidate: CapabilityCandidate,
        *,
        reason: str = "",
    ) -> "CapabilityPlanStep":
        """Build a plan step from a selected candidate."""
        return cls(
            step_id=f"{candidate.plugin_id}:{candidate.tool_name}",
            plugin_id=candidate.plugin_id,
            tool_name=candidate.tool_name,
            provides_capabilities=candidate.provides_capabilities,
            requires_capabilities=candidate.requires_capabilities,
            execution_policy=_execution_policy_for(candidate),
            requires_approval=candidate.requires_approval,
            reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "plugin_id": self.plugin_id,
            "tool_name": self.tool_name,
            "provides_capabilities": list(self.provides_capabilities),
            "requires_capabilities": list(self.requires_capabilities),
            "execution_policy": self.execution_policy,
            "requires_approval": self.requires_approval,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CapabilityPlan:
    """Topologically ordered plan plus diagnostics."""

    plan_id: str
    steps: tuple[CapabilityPlanStep, ...]
    missing_dependencies: tuple[MissingCapabilityDependency, ...] = field(default_factory=tuple)
    cycle_detected: bool = False

    @classmethod
    def from_candidates(
        cls,
        candidates: Sequence[CapabilityCandidate],
        *,
        plan_id: str = "",
    ) -> "CapabilityPlan":
        """Create a dependency-ordered plan from selected candidates."""
        steps = tuple(CapabilityPlanStep.from_candidate(c) for c in candidates)
        return cls._order_steps(steps, plan_id=plan_id)

    @classmethod
    def from_scores(
        cls,
        scores: Sequence[CandidateScore],
        *,
        plan_id: str = "",
    ) -> "CapabilityPlan":
        """Create a plan from selected CandidateScore objects."""
        steps = tuple(
            CapabilityPlanStep.from_candidate(
                s.candidate,
                reason=f"resolver_score={s.total_score:.3f}",
            )
            for s in scores
        )
        return cls._order_steps(steps, plan_id=plan_id)

    @classmethod
    def _order_steps(
        cls,
        steps: tuple[CapabilityPlanStep, ...],
        *,
        plan_id: str = "",
    ) -> "CapabilityPlan":
        provider_by_capability: dict[str, str] = {}
        for step in steps:
            for capability in step.provides_capabilities:
                provider_by_capability.setdefault(capability, step.step_id)

        graph: dict[str, set[str]] = {step.step_id: set() for step in steps}
        missing: list[MissingCapabilityDependency] = []
        for step in steps:
            for capability in step.requires_capabilities:
                provider = provider_by_capability.get(capability)
                if provider and provider != step.step_id:
                    graph[step.step_id].add(provider)
                elif not provider:
                    missing.append(MissingCapabilityDependency(step.step_id, capability))

        step_by_id = {step.step_id: step for step in steps}
        cycle = False
        try:
            ordered_ids = tuple(TopologicalSorter(graph).static_order())
        except CycleError:
            cycle = True
            ordered_ids = tuple(step.step_id for step in steps)
        ordered_steps = tuple(step_by_id[step_id] for step_id in ordered_ids)
        return cls(
            plan_id=plan_id or f"plan-{uuid.uuid4().hex}",
            steps=ordered_steps,
            missing_dependencies=tuple(missing),
            cycle_detected=cycle,
        )

    @property
    def executable(self) -> bool:
        """Return whether the plan has no missing deps and no dependency cycle."""
        return not self.missing_dependencies and not self.cycle_detected

    @property
    def is_actionable(self) -> bool:
        """Return whether the plan is executable AND has at least one step.

        ``executable`` is vacuously true for an empty plan (no missing deps, no
        cycle), which reads as success on the capability board when in fact
        nothing was selected. ``is_actionable`` is the honest signal: a plan that
        can actually do something. New surfaces should bind to this; ``executable``
        is retained unchanged for backward compatibility.
        """
        return self.executable and bool(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "executable": self.executable,
            "is_actionable": self.is_actionable,
            "cycle_detected": self.cycle_detected,
            "missing_dependencies": [m.to_dict() for m in self.missing_dependencies],
            "steps": [s.to_dict() for s in self.steps],
        }
