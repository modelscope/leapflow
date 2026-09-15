# Copyright (c) Alibaba, Inc. and its affiliates.
"""P1: the selection policy seam.

The load-bearing test here is equivalence. Introducing a seam must change no
behaviour, so ``GreedyPolicy`` is checked against the rule it replaced -- highest
weighted score, ties broken by a stable sort on ``(plugin_id, tool_name)`` -- rather
than described as similar to it. Everything else guards a property that only matters
once a *second* policy exists, which is exactly when it is too late to add the guard.
"""

from __future__ import annotations

import pytest

from types import SimpleNamespace

from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.domain.environment_fingerprint import EnvironmentFingerprint
from leapflow.domain.platform import Capability, PlatformID, PlatformManifest
from leapflow.plugins._builtin_policies import GreedyPolicy, GreedyPolicyPlugin
from leapflow.plugins.capability_resolver import (
    CapabilityCandidate,
    CapabilityResolver,
    ResolverContext,
)
from leapflow.plugins.selection_policy import (
    PolicyDeps,
    RewardSignal,
    SelectionOutcome,
    SelectionPolicy,
    SelectionPolicyPlugin,
)
from leapflow.plugins.selection_policy_registry import (
    DEFAULT_POLICY_ID,
    SelectionPolicyRegistry,
    get_selection_policy_registry,
    reset_selection_policy_registry,
)


def _req(capability: str = "json.pretty") -> CapabilityRequirement:
    return CapabilityRequirement.create(capability, "unknown_tool")


def _env(*caps: Capability) -> EnvironmentFingerprint:
    return EnvironmentFingerprint.from_platform_manifest(
        PlatformManifest(PlatformID.DARWIN_15, "15.0", frozenset(caps))
    )


def _candidate(plugin_id: str, tool_name: str, **kw) -> CapabilityCandidate:
    return CapabilityCandidate(
        plugin_id=plugin_id,
        tool_name=tool_name,
        provides_capabilities=kw.pop("provides", ("json.pretty",)),
        requires_platform_capabilities=kw.pop("requires", ()),
        risk_level=kw.pop("risk_level", "read_only"),
        **kw,
    )


def _settings(**kw) -> SimpleNamespace:
    """A minimal live-settings stand-in.

    The registry reads configuration off an object rather than the global snapshot,
    so a test supplies its own instead of mutating process state.
    """
    return SimpleNamespace(
        selection_policy=kw.get("policy", "greedy"),
        selection_prior_strength=kw.get("prior_strength", 2.0),
        selection_exploration=kw.get("exploration", 1.0),
        selection_buckets=kw.get("buckets", ""),
    )


def _ctx() -> ResolverContext:
    return ResolverContext(environment=_env(Capability.FILE_OPS))


@pytest.fixture(autouse=True)
def _isolated_registry():
    """The registry is a process singleton; a test must not inherit another's policy."""
    reset_selection_policy_registry()
    yield
    reset_selection_policy_registry()


# ── equivalence with the behaviour the seam replaced ──────────────────────────


def test_greedy_picks_the_highest_score():
    candidates = (
        _candidate("a", "tool_a", risk_level="high"),
        _candidate("b", "tool_b", risk_level="read_only"),
    )
    resolution = CapabilityResolver().resolve_one(_req(), candidates, _ctx())

    assert resolution.selected is not None
    assert resolution.selected.candidate.tool_name == "tool_b"
    assert resolution.policy_id == "greedy"


def test_greedy_breaks_a_tie_by_the_previous_stable_rule():
    """``(plugin_id, tool_name)`` ascending -- the exact rule the resolver used.

    Order of the input must not matter: a tie resolved by arrival order would make
    selection depend on registry iteration, which changes as plugins load.
    """
    forward = (_candidate("a", "tool_a"), _candidate("b", "tool_b"))
    reverse = tuple(reversed(forward))

    for candidates in (forward, reverse):
        resolution = CapabilityResolver().resolve_one(_req(), candidates, _ctx())
        assert resolution.selected is not None
        assert resolution.selected.candidate.plugin_id == "a"


def test_greedy_never_reports_an_exploration():
    """``explored`` is what distinguishes deliberate exploration from a scoring bug."""
    resolution = CapabilityResolver().resolve_one(
        _req(), (_candidate("a", "tool_a"),), _ctx()
    )
    assert resolution.explored is False


def test_a_resolution_records_which_policy_chose():
    """Without this the decision history stops being auditable once policies vary."""
    resolution = CapabilityResolver().resolve_one(
        _req(), (_candidate("a", "tool_a"),), _ctx()
    )
    assert resolution.policy_id == "greedy"
    assert "greedy" in resolution.reason
    assert resolution.to_dict()["policy_id"] == "greedy"


# ── the safety boundary ───────────────────────────────────────────────────────


class _PicksLast:
    """A policy that would take the worst candidate, to prove it cannot reach one."""

    policy_id = "picks_last"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def select(self, requirement, eligible, context) -> SelectionOutcome:
        self.seen = [c.candidate.tool_name for c in eligible]
        return SelectionOutcome(selected=eligible[-1], policy_id=self.policy_id, explored=True)

    def observe(self, requirement, chosen_tool, reward) -> None:
        return None


def test_an_excluded_candidate_is_never_offered_to_a_policy():
    """Exploration must not reach a tool a hard constraint refused.

    Safety is not a term to be traded off: a candidate excluded for a missing
    platform affordance is filtered out *before* the policy is consulted, so no
    policy -- however adventurous -- can select it.
    """
    policy = _PicksLast()
    candidates = (
        _candidate("a", "tool_a"),
        _candidate("b", "needs_vision", requires=(Capability.SCREEN_CAPTURE.value,)),
    )

    resolution = CapabilityResolver(policy=policy).resolve_one(_req(), candidates, _ctx())

    assert "needs_vision" not in policy.seen, "an inadmissible candidate reached the policy"
    assert resolution.selected is not None
    assert resolution.selected.candidate.tool_name == "tool_a"


def test_a_policy_is_not_consulted_when_nothing_is_admissible():
    """No admissible candidate is a resolver verdict, not a choice to delegate."""
    policy = _PicksLast()
    resolution = CapabilityResolver(policy=policy).resolve_one(
        _req(), (_candidate("b", "needs_vision", requires=(Capability.SCREEN_CAPTURE.value,)),), _ctx()
    )

    assert policy.seen == []
    assert resolution.unmet is True
    assert resolution.policy_id == ""


def test_a_policy_that_explores_says_so():
    resolution = CapabilityResolver(policy=_PicksLast()).resolve_one(
        _req(), (_candidate("a", "tool_a"), _candidate("b", "tool_b")), _ctx()
    )
    assert resolution.explored is True
    assert resolution.policy_id == "picks_last"


# ── registry ──────────────────────────────────────────────────────────────────


class _Fake:
    policy_id = "fake"

    def select(self, requirement, eligible, context) -> SelectionOutcome:
        return SelectionOutcome(selected=eligible[0], policy_id=self.policy_id)

    def observe(self, requirement, chosen_tool, reward) -> None:
        return None


class _FakePlugin:
    def __init__(self, policy_id: str = "fake", name: str = "Fake") -> None:
        self._id = policy_id
        self._name = name
        self.built_with: PolicyDeps | None = None

    @property
    def policy_id(self) -> str:
        return self._id

    @property
    def display_name(self) -> str:
        return self._name

    def create(self, params, deps) -> SelectionPolicy:
        self.built_with = deps
        return _Fake()


def test_the_builtin_greedy_plugin_satisfies_both_protocols():
    assert isinstance(GreedyPolicyPlugin(), SelectionPolicyPlugin)
    assert isinstance(GreedyPolicy(), SelectionPolicy)


def test_greedy_is_registered_and_is_the_default():
    registry = get_selection_policy_registry()
    assert DEFAULT_POLICY_ID in registry.available()
    assert registry.activate().policy_id == DEFAULT_POLICY_ID


def test_first_registration_of_an_id_wins_and_the_collision_is_recorded():
    """A silently replaced policy would change how the framework chooses its own tools.

    Non-fatal on purpose, like tool-name arbitration: one colliding package must not
    stop every other policy from registering.
    """
    registry = SelectionPolicyRegistry()
    assert registry.register(_FakePlugin(name="incumbent")) is True
    assert registry.register(_FakePlugin(name="challenger")) is False

    assert [r["kept"] for r in registry.rejected] == ["incumbent"]
    assert [r["rejected"] for r in registry.rejected] == ["challenger"]


def test_an_unknown_configured_policy_falls_back_to_the_default():
    """A misconfiguration is a reason to log, never a reason to stop choosing tools."""
    registry = SelectionPolicyRegistry()
    from leapflow.plugins._builtin_policies import register_builtin_policies

    register_builtin_policies(registry)

    policy = registry.create_from_config({"selection_policy": "does_not_exist"})

    assert policy is not None
    assert policy.policy_id == DEFAULT_POLICY_ID


def test_a_policy_that_fails_to_build_does_not_break_selection():
    class _Broken(_FakePlugin):
        def create(self, params, deps):
            raise RuntimeError("boom")

    registry = SelectionPolicyRegistry()
    registry.register(_Broken("broken"))

    assert registry.create("broken") is None


def test_deps_reach_the_plugin_that_builds_the_policy():
    registry = SelectionPolicyRegistry()
    plugin = _FakePlugin()
    registry.register(plugin)
    ledger = object()

    registry.create("fake", {}, PolicyDeps(trust_ledger=ledger))

    assert plugin.built_with is not None
    assert plugin.built_with.trust_ledger is ledger


# ── instance ownership: the trap that makes a learning policy never learn ─────


def test_activation_caches_so_a_stateful_policy_accumulates():
    """A fresh instance per call would hand each observation to a throwaway object.

    Silent, because every individual call looks correct -- which is why the cache
    lives in the registry rather than at each call site.
    """
    registry = get_selection_policy_registry()
    first = registry.activate()
    assert first is not None
    assert registry.activate() is first
    assert registry.current() is first


def test_current_never_creates_a_policy():
    """A reporter must not install a dependency-less policy ahead of the real owner.

    ``current()`` answering ``None`` is correct: nothing has selected yet, so there
    is no decision to report on.
    """
    registry = get_selection_policy_registry()
    assert registry.current() is None

    registry.activate(PolicyDeps(trust_ledger=object()))
    assert registry.current() is not None


def test_the_registered_set_is_discoverable_for_the_config_hint():
    """``selection.policy`` accepts an id, so the ids must be findable.

    Read by the config catalog rather than hardcoded there: a policy registered by a
    third-party package through the entry point group has to appear too, and a literal
    enumeration in the catalog would silently omit it.
    """
    registry = get_selection_policy_registry()
    described = {d.policy_id: d.display_name for d in registry.describe()}

    assert set(described) == set(registry.available())
    assert all(described.values()), "every policy needs a human-readable name"

    from leapflow.config_service import _registered_selection_policies

    hint = _registered_selection_policies()
    for policy_id in described:
        assert policy_id in hint, hint


# ── reward semantics ─────────────────────────────────────────────────────────


def test_an_abstaining_reward_is_not_a_failure():
    """``None`` means "no information". Folding it into failure would quarantine
    healthy plugins for a reporting omission, since a successful call whose handler
    declared no effect is the normal state for tools predating the convention.
    """
    assert RewardSignal(value=None).informative is False
    assert RewardSignal(value=0.0).informative is True
    assert RewardSignal(value=1.0).informative is True


def test_greedy_accepts_an_observation_and_ignores_it():
    """A no-op rather than an omission, so the feedback edge is uniform.

    Adding a learning policy must be a new file plus a config value, not a change to
    the call sites that report outcomes.
    """
    policy = GreedyPolicy()
    policy.observe(_req(), "tool_a", RewardSignal(value=1.0))
    policy.observe(_req(), "tool_a", RewardSignal(value=None))


def test_a_failing_arbiter_does_not_fail_selection():
    """The arbiter is advisory; an LLM tie-break that raises must not lose the turn."""

    class _Boom:
        def choose(self, requirement, tied, context):
            raise RuntimeError("nope")

    resolution = CapabilityResolver(policy=GreedyPolicy(arbiter=_Boom())).resolve_one(
        _req(), (_candidate("a", "tool_a"), _candidate("b", "tool_b")), _ctx()
    )

    assert resolution.selected is not None
    assert resolution.selected.candidate.plugin_id == "a"   # stable fallback
    assert resolution.arbitration_used is False


# ── the feedback edge, exercised through the real sweep ───────────────────────


class _Recording:
    """Records what the sweep reports, to prove the edge is live rather than dead."""

    policy_id = "recording"

    def __init__(self) -> None:
        self.observations: list[tuple[str, float | None, str]] = []

    def select(self, requirement, eligible, context) -> SelectionOutcome:
        return SelectionOutcome(selected=eligible[0], policy_id=self.policy_id)

    def observe(self, requirement, chosen_tool, reward) -> None:
        self.observations.append((chosen_tool, reward.value, reward.source))


class _RecordingPlugin:
    def __init__(self, policy: _Recording) -> None:
        self._policy = policy

    @property
    def policy_id(self) -> str:
        return "recording"

    @property
    def display_name(self) -> str:
        return "Recording"

    def create(self, params, deps) -> SelectionPolicy:
        return self._policy


def test_the_sweep_reports_every_verdict_to_the_active_policy(monkeypatch):
    """The edge that was missing: verdicts fed trust but never the chooser.

    Driven through the real ``CoevolutionSweep`` and the real verifier rather than a
    stub, because the value of this edge is that it is *wired* -- a hand-built call
    would pass while production dropped every reward.
    """
    import asyncio

    from leapflow.evolution.sweep import CoevolutionSweep
    from leapflow.plugins import selection_policy_registry as reg

    recorder = _Recording()
    registry = SelectionPolicyRegistry()
    registry.register(_RecordingPlugin(recorder))
    monkeypatch.setattr(reg, "_registry", registry)
    monkeypatch.setattr(reg, "_settings_config", lambda: {"selection_policy": "recording"})
    assert registry.activate() is recorder

    req = CapabilityRequirement.create(
        "file.write", "unknown_tool", metadata={"expected_effect": "wrote the bytes"}
    )
    asyncio.run(CoevolutionSweep().run(verifications=[
        (req, {"ok": True, "observed_effect": "wrote the bytes to disk"}, "good"),
        (req, {"ok": False, "error": "boom"}, "bad"),
        (req, {"ok": True}, "silent"),
    ]))

    by_tool = {tool: value for tool, value, _ in recorder.observations}
    assert by_tool["good"] == 1.0, "a verified effect must arrive as a positive reward"
    assert by_tool["bad"] == 0.0, "a failure must arrive as a negative reward"
    assert by_tool["silent"] is None, "an unverifiable outcome must abstain, not refute"


def test_a_reporter_without_an_active_policy_is_silent(monkeypatch):
    """No active policy means nothing selected, so there is nothing to report.

    Must not raise, and must not install a dependency-less policy as a side effect.
    """
    import asyncio

    from leapflow.evolution.sweep import CoevolutionSweep
    from leapflow.plugins import selection_policy_registry as reg

    registry = SelectionPolicyRegistry()
    monkeypatch.setattr(reg, "_registry", registry)

    req = CapabilityRequirement.create("file.write", "unknown_tool")
    asyncio.run(CoevolutionSweep().run(
        verifications=[(req, {"ok": True}, "tool")]
    ))

    assert registry.current() is None


# ── configuration actually taking effect ──────────────────────────────────────


def test_a_switched_policy_takes_effect_without_a_restart():
    """``selection.policy`` presents itself as hot-reloadable, so it must be.

    The live value is *pushed* in, because ``get_settings()`` is a boot snapshot with
    no refresh path anywhere in the process -- a policy read from it would be pinned
    to whatever configuration existed at startup while ``leap config`` reported the
    change as applied.
    """
    registry = SelectionPolicyRegistry()
    registry.register(_FakePlugin("fake"))
    from leapflow.plugins._builtin_policies import register_builtin_policies

    register_builtin_policies(registry)

    assert registry.activate(settings=_settings(policy="greedy")).policy_id == "greedy"
    switched = registry.activate(settings=_settings(policy="fake"))

    assert switched is not None
    assert switched.policy_id == "fake", "a config change must reach the next selection"
    assert registry.current() is switched


def test_switching_back_and_forth_is_stable():
    registry = SelectionPolicyRegistry()
    registry.register(_FakePlugin("fake"))
    from leapflow.plugins._builtin_policies import register_builtin_policies

    register_builtin_policies(registry)

    ids = [registry.activate(settings=_settings(policy=pid)).policy_id for pid in ("greedy", "fake", "greedy")]
    assert ids == ["greedy", "fake", "greedy"]


def test_an_unknown_id_does_not_thrash_the_active_policy():
    """A misconfiguration must not rebuild the policy on every single activation.

    The fallback answers ``greedy`` while the requested id stays unknown, so a naive
    "rebuild when the ids differ" check would re-create and re-log forever -- and a
    stateful policy would lose its state on every capability observation.
    """
    registry = SelectionPolicyRegistry()
    from leapflow.plugins._builtin_policies import register_builtin_policies

    register_builtin_policies(registry)

    first = registry.activate(settings=_settings(policy="nope"))
    assert first is not None
    assert first.policy_id == "greedy"
    assert registry.activate(settings=_settings(policy="nope")) is first, "an unknown id must not rebuild"


def test_the_loop_pushes_the_configured_policy_through_to_selection(monkeypatch):
    """End to end: the id the engine holds is the policy the resolver ends up using."""
    from leapflow.plugins import selection_policy_registry as reg
    from leapflow.plugins.adaptive_loop import AdaptivePluginLoop

    registry = SelectionPolicyRegistry()
    registry.register(_FakePlugin("fake"))
    monkeypatch.setattr(reg, "_registry", registry)

    class _NoStore:
        def append(self, *a, **k) -> None:
            return None

    loop = AdaptivePluginLoop(
        registry=None, plan_store=_NoStore(), settings=_settings(policy="fake")
    )

    assert registry.current() is not None
    assert registry.current().policy_id == "fake"
    assert loop is not None


def test_the_two_adaptive_scorers_contribute_nothing_without_their_inputs():
    """Measured, not assumed: in the default resolver both learning signals are 0.

    ``AdaptivePluginLoop`` is constructed in the engine without a trust ledger or a
    usage tracker, so ``TrustScorer`` reports "trust ledger unavailable" and
    ``ReliabilityScorer`` reports "usage tracker unavailable" -- both scoring 0.0 for
    every candidate. Selection is therefore decided entirely by the three static
    scorers, and any tie falls to alphabetical order.

    This is recorded as a test because it is the precondition for a learning policy:
    a bandit placed here would have no differentiating signal to learn from until
    those dependencies are wired. Wiring them changes which tool gets selected, so it
    is a deliberate decision rather than something to slip into a seam that is
    supposed to change no behaviour.
    """
    resolution = CapabilityResolver().resolve_one(
        _req(), (_candidate("a", "tool_a"), _candidate("b", "tool_b")), _ctx()
    )

    by_scorer = {
        component.to_dict()["scorer"]: component.to_dict()
        for score in resolution.candidates
        for component in score.components
    }
    assert by_scorer["trust"]["score"] == 0.0
    assert "unavailable" in by_scorer["trust"]["reason"]
    assert by_scorer["reliability"]["score"] == 0.0
    assert "unavailable" in by_scorer["reliability"]["reason"]

    totals = {s.candidate.tool_name: s.total_score for s in resolution.candidates}
    assert len(set(totals.values())) == 1, "both candidates tie on the static scorers alone"


# ── generality: no per-policy hard rules in the shared machinery ───────────────


def test_the_settings_translation_names_no_policy():
    """One shared params dict, so a new policy needs no entry in the translation."""
    from leapflow.plugins.selection_policy_registry import policy_params_from_settings

    assert policy_params_from_settings(_settings()) == {}
    for policy_id in ("greedy", "bucketed", "thompson", "ucb1"):
        assert policy_id not in policy_params_from_settings(_settings())


def test_a_third_party_policy_can_configure_itself_through_the_open_dict():
    """A policy from the entry point group cannot add typed settings, so the open dict
    is its only configuration path.
    """
    from leapflow.plugins.selection_policy_registry import policy_params_from_settings

    settings = _settings()
    settings.policy_params = {"custom_knob": "on"}
    assert policy_params_from_settings(settings)["custom_knob"] == "on"


def test_greedy_is_the_only_shipped_policy():
    """The learning policies were removed after measurement, not shipped and forgotten.

    Measured before removal: the policy was consulted 0 times in production, every
    capability had exactly one candidate, and reward bound only self-acquired plugins.
    The seam stays because re-adding a policy is one file; the policies went because
    they optimised a decision that is not being made.
    """
    registry = get_selection_policy_registry()
    assert registry.available() == ["greedy"]
