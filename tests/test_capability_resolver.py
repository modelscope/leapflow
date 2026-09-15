# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for deterministic adaptive capability resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pytest

from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.domain.environment_fingerprint import EnvironmentFingerprint
from leapflow.domain.platform import Capability, PlatformID, PlatformManifest
from leapflow.learning.plugin_stats import PluginUsageTracker
from leapflow.learning.plugin_trust import PluginTrustLedger
from leapflow.plugins.capability_resolver import (
    CapabilityCandidate,
    CapabilityResolver,
    CandidateScore,
    ResolverContext,
    candidates_from_registry,
)
from leapflow.plugins.protocol import ToolMetadata
from leapflow.plugins.registry import ToolPluginRegistry


def _env(*caps: Capability) -> EnvironmentFingerprint:
    return EnvironmentFingerprint.from_platform_manifest(
        PlatformManifest(PlatformID.DARWIN_15, "15.0", frozenset(caps))
    )


def _req(capability: str, **kwargs: Any) -> CapabilityRequirement:
    return CapabilityRequirement.create(
        capability,
        "explicit_request",
        requirement_id=f"req-{capability}",
        **kwargs,
    )


def _candidate(
    plugin_id: str,
    tool_name: str,
    *,
    provides: tuple[str, ...],
    requires_platform: tuple[str, ...] = (),
    risk_level: str = "read_only",
    requires_approval: bool = False,
) -> CapabilityCandidate:
    return CapabilityCandidate(
        plugin_id=plugin_id,
        tool_name=tool_name,
        provides_capabilities=provides,
        requires_platform_capabilities=requires_platform,
        risk_level=risk_level,
        requires_approval=requires_approval,
    )


@dataclass
class _TieArbiter:
    chosen: str

    def choose(
        self,
        requirement: CapabilityRequirement,
        tied: Sequence[CandidateScore],
        context: ResolverContext,
    ) -> str | None:
        return self.chosen


async def _handler(**kwargs: Any) -> dict[str, Any]:
    return {"ok": True}


@dataclass
class _Plugin:
    plugin_id: str
    tools: list[ToolMetadata]
    category: str = "test"
    dependencies: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.dependencies is None:
            self.dependencies = []

    def bind_runtime(self, **deps: Any) -> None:
        return None


def test_resolver_selects_highest_scoring_declared_candidate() -> None:
    req = _req("json.pretty")
    env = _env(Capability.FILE_OPS)
    ledger = PluginTrustLedger(candidate_at=1, verified_at=2, production_at=3)
    for _ in range(3):
        ledger.record_success("stable")
    tracker = PluginUsageTracker()
    tracker._get_reverse_index = lambda: {"stable_tool": "stable", "draft_tool": "draft"}
    for _ in range(5):
        tracker.record("stable_tool", True, 5.0)
    tracker.record("draft_tool", False, 5.0)

    candidates = (
        _candidate("draft", "draft_tool", provides=("json.pretty",)),
        _candidate("stable", "stable_tool", provides=("json.pretty",)),
    )
    resolution = CapabilityResolver().resolve_one(
        req,
        candidates,
        ResolverContext(environment=env, trust_ledger=ledger, usage_tracker=tracker),
    )

    assert resolution.selected is not None
    assert resolution.selected.candidate.plugin_id == "stable"
    assert resolution.selected.total_score > resolution.candidates[-1].total_score
    assert resolution.unmet is False


def test_environment_missing_capability_excludes_candidate() -> None:
    req = _req("shell.run")
    candidate = _candidate(
        "shell_plugin",
        "shell_run",
        provides=("shell.run",),
        requires_platform=("shell.exec",),
    )

    resolution = CapabilityResolver().resolve_one(
        req,
        (candidate,),
        ResolverContext(environment=_env(Capability.FILE_OPS)),
    )

    assert resolution.selected is None
    assert resolution.unmet is True
    assert "missing platform capabilities: shell.exec" in resolution.candidates[0].exclusion_reasons


def test_risk_limit_excludes_candidate() -> None:
    req = _req("send.message", max_risk_level="read_only")
    candidate = _candidate(
        "gateway_plugin",
        "gateway_send",
        provides=("send.message",),
        risk_level="external",
    )

    resolution = CapabilityResolver().resolve_one(
        req,
        (candidate,),
        ResolverContext(environment=_env(Capability.FILE_OPS)),
    )

    assert resolution.selected is None
    assert "exceeds max" in resolution.candidates[0].exclusion_reasons[-1]


def test_tie_can_be_resolved_by_optional_arbiter() -> None:
    """The arbiter now belongs to the greedy policy, which is where ties exist.

    A tie-break is a property of greedy scoring: under a sampling policy two
    candidates never tie, because each draw is continuous. Leaving the hook on the
    resolver would have made every future policy inherit something meaningless to it.
    """
    from leapflow.plugins._builtin_policies import GreedyPolicy

    req = _req("json.pretty")
    candidates = (
        _candidate("a", "tool_a", provides=("json.pretty",)),
        _candidate("b", "tool_b", provides=("json.pretty",)),
    )

    resolution = CapabilityResolver(
        policy=GreedyPolicy(arbiter=_TieArbiter("tool_b"))
    ).resolve_one(
        req,
        candidates,
        ResolverContext(environment=_env(Capability.FILE_OPS)),
    )

    assert resolution.selected is not None
    assert resolution.selected.candidate.tool_name == "tool_b"
    assert resolution.arbitration_used is True
    assert resolution.policy_id == "greedy"
    # An arbitrated tie is still the argmax, so it is not an exploration.
    assert resolution.explored is False


def test_no_arbiter_tie_uses_stable_order() -> None:
    req = _req("json.pretty")
    candidates = (
        _candidate("b", "tool_b", provides=("json.pretty",)),
        _candidate("a", "tool_a", provides=("json.pretty",)),
    )

    resolution = CapabilityResolver().resolve_one(
        req,
        candidates,
        ResolverContext(environment=_env(Capability.FILE_OPS)),
    )

    assert resolution.selected is not None
    assert resolution.selected.candidate.tool_name == "tool_a"
    assert resolution.arbitration_used is False


def test_candidates_from_registry_uses_live_conflict_resolved_owners() -> None:
    reg = ToolPluginRegistry()
    tool_a = ToolMetadata(
        name="shared",
        description="from a",
        parameters_schema={"type": "object", "properties": {}},
        handler=_handler,
        provides_capabilities=("json.pretty",),
    )
    tool_b = ToolMetadata(
        name="shared",
        description="from b",
        parameters_schema={"type": "object", "properties": {}},
        handler=_handler,
        provides_capabilities=("json.pretty",),
    )
    reg.register(_Plugin("plugin_a", [tool_a]))
    reg.register(_Plugin("plugin_b", [tool_b]))
    reg.assemble()

    candidates = candidates_from_registry(reg)

    assert [(c.plugin_id, c.tool_name) for c in candidates] == [("plugin_a", "shared")]
    assert len(reg.conflicts) == 1


def test_score_breakdown_sums_to_total_score() -> None:
    req = _req("json.pretty")
    candidate = _candidate("p", "pretty", provides=("json.pretty",))

    resolution = CapabilityResolver().resolve_one(
        req,
        (candidate,),
        ResolverContext(environment=_env(Capability.FILE_OPS)),
    )

    assert resolution.selected is not None
    expected = round(sum(c.weighted_score for c in resolution.selected.components), 6)
    assert resolution.selected.total_score == pytest.approx(expected)
