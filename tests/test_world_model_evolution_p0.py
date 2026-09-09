"""World-model-first evolution foundation (P0).

Covers the four P0 changes:

* ``RequirementOrigin`` accepts ``world_model``.
* ``CapabilityGapDetector`` turns *declared* non-unknown-tool evidence into
  requirements. Before this, ``CapabilityEvidenceClassifier`` could admit an
  evidence kind into the durable store while ``requirements()`` silently dropped
  it -- a half-wired seam.
* ``EvolutionIntent`` is the world model's proposal contract and travels the
  existing observation path.
* ``PluginTrustLedger.is_frozen`` + ``FrozenExclusionScorer`` make a frozen
  plugin ineligible at selection, independently of governance.
"""

from __future__ import annotations

from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.domain.evolution_intent import (
    WORLD_MODEL_INTENT,
    WORLD_MODEL_ORIGIN,
    EvolutionIntent,
)
from leapflow.learning.capability_gap_detector import CapabilityGapDetector
from leapflow.learning.capability_observation import (
    CapabilityEvidenceClassifier,
    CapabilityObservationBuffer,
)
from leapflow.learning.plugin_trust import PluginTrustLedger, PluginTrustLevel

_UNKNOWN = {
    "error_type": "unknown_tool",
    "original_tool_name": "list_dir",
    "suggestions": ["file_list"],
    "recovery_hint": "use file_list",
}


# ── the unknown_tool path must be unchanged ───────────────────────────────────


def test_unknown_tool_requirement_is_unchanged():
    reqs = CapabilityGapDetector().requirements_from_tool_results([_UNKNOWN])
    assert len(reqs) == 1
    req = reqs[0]
    assert req.capability == "list_dir"
    assert req.origin == "unknown_tool"
    assert req.requirement_id == "req-unknown-tool-list_dir"
    assert req.evidence == "Runtime attempted unknown tool 'list_dir'."
    assert dict(req.metadata)["original_tool_name"] == "list_dir"
    assert dict(req.metadata)["occurrences"] == "1"


def test_min_count_still_filters_unknown_tool():
    detector = CapabilityGapDetector()
    assert detector.requirements_from_tool_results([_UNKNOWN], min_count=2) == ()
    assert len(detector.requirements_from_tool_results([_UNKNOWN, _UNKNOWN], min_count=2)) == 1


# ── declared evidence now produces requirements (the defect fix) ──────────────


def test_declared_evidence_becomes_a_requirement():
    result = {
        "error_type": "interface_drift",
        "capability": "chat.reply",
        "origin": "environment_probe",
        "recovery_hint": "send_button is gone",
        "failure_code": "interface_names_missing",
        "max_risk_level": "read_only",
    }
    reqs = CapabilityGapDetector().requirements_from_tool_results([result])
    assert len(reqs) == 1
    req = reqs[0]
    assert req.capability == "chat.reply"
    assert req.origin == "environment_probe"
    assert req.max_risk_level == "read_only"
    assert req.evidence == "send_button is gone"
    meta = dict(req.metadata)
    assert meta["evidence_kind"] == "interface_drift"
    assert meta["failure_code"] == "interface_names_missing"


def test_declared_evidence_without_capability_is_ignored():
    """Never infer a capability from text: no declaration, no requirement."""
    result = {"error_type": "interface_drift", "recovery_hint": "something changed"}
    assert CapabilityGapDetector().requirements_from_tool_results([result]) == ()


def test_unrecognised_origin_falls_back_and_cannot_be_smuggled():
    result = {
        "error_type": "interface_drift",
        "capability": "chat.reply",
        "origin": "totally_made_up",
    }
    reqs = CapabilityGapDetector().requirements_from_tool_results([result])
    assert reqs[0].origin == "environment_probe"


def test_declared_evidence_buckets_by_kind_and_capability():
    a = {"error_type": "interface_drift", "capability": "chat.reply"}
    b = {"error_type": "affordance_removed", "capability": "chat.reply"}
    c = {"error_type": "interface_drift", "capability": "chat.send"}
    reqs = CapabilityGapDetector().requirements_from_tool_results([a, b, c, a])
    # 3 distinct (kind, capability) pairs; the repeat merges into its bucket.
    assert len(reqs) == 3
    drift_reply = [r for r in reqs if dict(r.metadata)["evidence_kind"] == "interface_drift"
                   and r.capability == "chat.reply"]
    assert dict(drift_reply[0].metadata)["occurrences"] == "2"


def test_mixed_evidence_yields_both_kinds():
    declared = {"error_type": "interface_drift", "capability": "chat.reply"}
    reqs = CapabilityGapDetector().requirements_from_tool_results([_UNKNOWN, declared])
    origins = {r.origin for r in reqs}
    assert origins == {"unknown_tool", "environment_probe"}


# ── EvolutionIntent ───────────────────────────────────────────────────────────


def test_intent_defaults_to_the_lowest_risk_ceiling():
    """A model-authored proposal must not inherit the permissive domain default."""
    intent = EvolutionIntent.create("chat.reply", "v3 dispatch path is unserved")
    assert intent.max_risk_level == "read_only"
    assert CapabilityRequirement.create("x", "world_model").max_risk_level == "external"


def test_intent_requires_capability_and_hypothesis():
    for bad in [("", "h"), ("c", "")]:
        try:
            EvolutionIntent.create(*bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_intent_confidence_is_clamped():
    assert EvolutionIntent.create("c", "h", confidence=5.0).confidence == 1.0
    assert EvolutionIntent.create("c", "h", confidence=-2.0).confidence == 0.0


def test_intent_to_requirement_carries_world_model_origin():
    intent = EvolutionIntent.create(
        "chat.reply", "the v3 dispatch button is unserved",
        confidence=0.7, target_affordance="app.chat.v3",
        expected_effect="message appears in thread", evidence_ids=["exp-1", "exp-2"],
    )
    req = intent.to_requirement()
    assert req.origin == WORLD_MODEL_ORIGIN == "world_model"
    assert req.capability == "chat.reply"
    assert req.requirement_id == f"req-wm-{intent.intent_id}"
    meta = dict(req.metadata)
    assert meta["evidence_kind"] == WORLD_MODEL_INTENT
    assert meta["target_affordance"] == "app.chat.v3"
    assert meta["evidence_ids"] == "exp-1,exp-2"


def test_intent_flows_through_the_governed_observation_path():
    """The whole point: an intent is governed by the same machinery, not a new one."""
    intent = EvolutionIntent.create("chat.reply", "v3 unserved", confidence=0.6)
    result = intent.to_observation_result()

    # Default classifier must NOT admit it (shipped behaviour unchanged).
    assert CapabilityObservationBuffer().add_result(result) is False

    # Opt in, and it reaches the buffer and becomes a requirement.
    buffer = CapabilityObservationBuffer(
        classifier=CapabilityEvidenceClassifier.from_kinds([WORLD_MODEL_INTENT])
    )
    assert buffer.add_result(result) is True
    reqs = buffer.requirements()
    assert len(reqs) == 1
    assert reqs[0].origin == "world_model"
    assert reqs[0].capability == "chat.reply"
    assert reqs[0].max_risk_level == "read_only"


def test_intent_round_trips_through_dict():
    intent = EvolutionIntent.create(
        "chat.reply", "h", confidence=0.5, target_affordance="app.chat.v3",
        evidence_ids=["e1"], required_platform_capabilities=["ui.automation"],
    )
    restored = EvolutionIntent.from_dict(intent.to_dict())
    assert restored.to_dict() == intent.to_dict()


# ── the risk ceiling is an authorisation, so an intent may only narrow it ──────


def test_intent_cannot_widen_its_own_risk_ceiling():
    """A model-authored intent asking for `external` must not get it.

    `max_risk_level` is a ceiling: a larger value permits selecting riskier
    tools. Letting the authoring model choose it would make the intent an
    authorisation rather than a hypothesis.
    """
    greedy = EvolutionIntent.create("chat.reply", "h", max_risk_level="external")
    assert greedy.max_risk_level == "external"          # the request is preserved...
    req = greedy.to_requirement()                       # ...but not granted
    assert req.max_risk_level == "read_only"
    assert dict(req.metadata)["requested_max_risk_level"] == "external"
    assert greedy.to_observation_result()["max_risk_level"] == "read_only"


def test_intent_may_narrow_below_the_caller_ceiling():
    intent = EvolutionIntent.create("chat.reply", "h", max_risk_level="read_only")
    req = intent.to_requirement(risk_ceiling="external")
    assert req.max_risk_level == "read_only"            # stricter of the two wins
    assert "requested_max_risk_level" not in dict(req.metadata)


def test_trusted_caller_may_raise_the_ceiling_deliberately():
    intent = EvolutionIntent.create("chat.reply", "h", max_risk_level="medium")
    assert intent.to_requirement(risk_ceiling="high").max_risk_level == "medium"
    assert intent.to_requirement(risk_ceiling="read_only").max_risk_level == "read_only"


def test_unknown_risk_level_is_treated_as_most_permissive_and_clamped():
    """A typo must not read as safe and slip past the clamp."""
    intent = EvolutionIntent.create("chat.reply", "h", max_risk_level="totally_safe_promise")
    assert intent.to_requirement().max_risk_level == "read_only"


def test_clamped_intent_still_flows_through_the_governed_path():
    greedy = EvolutionIntent.create("chat.reply", "h", max_risk_level="external")
    buffer = CapabilityObservationBuffer(
        classifier=CapabilityEvidenceClassifier.from_kinds([WORLD_MODEL_INTENT])
    )
    assert buffer.add_result(greedy.to_observation_result()) is True
    reqs = buffer.requirements()
    # The requirement that reaches resolution carries the clamped ceiling.
    assert reqs[0].max_risk_level == "read_only"


# ── LF-9: frozen plugins must be ineligible, not merely low-scored ────────────


def test_is_frozen_distinguishes_frozen_from_merely_draft():
    ledger = PluginTrustLedger()
    assert ledger.is_frozen("fresh") is False
    assert ledger.level("fresh") == PluginTrustLevel.DRAFT   # DRAFT but not frozen
    ledger.record_failure("broken", hard=True)
    assert ledger.is_frozen("broken") is True
    assert ledger.level("broken") == PluginTrustLevel.DRAFT   # same level, different meaning


def test_frozen_survives_ledger_serialization():
    ledger = PluginTrustLedger()
    ledger.record_failure("broken", hard=True)
    restored = PluginTrustLedger.load_state(ledger.to_state())
    assert restored.is_frozen("broken") is True


def test_frozen_exclusion_scorer_excludes_only_frozen():
    from leapflow.plugins.capability_resolver import (
        CapabilityCandidate,
        FrozenExclusionScorer,
        ResolverContext,
    )
    from leapflow.domain.environment_fingerprint import EnvironmentFingerprint

    ledger = PluginTrustLedger()
    ledger.record_failure("broken", hard=True)
    context = ResolverContext(
        environment=EnvironmentFingerprint(
            platform_id="linux_gnome", os_version="x", platform_capabilities=(), workspace_root="/w"
        ),
        trust_ledger=ledger,
    )
    req = CapabilityRequirement.create("chat.reply", "task_contract")
    scorer = FrozenExclusionScorer()

    frozen = CapabilityCandidate(plugin_id="broken", tool_name="broken_tool")
    healthy = CapabilityCandidate(plugin_id="healthy", tool_name="healthy_tool")
    assert scorer.score(req, frozen, context).excluded is True
    assert scorer.score(req, healthy, context).excluded is False


def test_frozen_exclusion_scorer_is_inert_without_a_ledger():
    """Absent trust information must not exclude everything."""
    from leapflow.plugins.capability_resolver import (
        CapabilityCandidate,
        FrozenExclusionScorer,
        ResolverContext,
    )
    from leapflow.domain.environment_fingerprint import EnvironmentFingerprint

    context = ResolverContext(
        environment=EnvironmentFingerprint(
            platform_id="linux_gnome", os_version="x", platform_capabilities=(), workspace_root="/w"
        )
    )
    req = CapabilityRequirement.create("chat.reply", "task_contract")
    component = FrozenExclusionScorer().score(
        req, CapabilityCandidate(plugin_id="p", tool_name="t"), context
    )
    assert component.excluded is False


def test_frozen_exclusion_is_not_a_default_scorer():
    """Default resolution must be unchanged by this addition."""
    from leapflow.plugins import capability_resolver as cr

    names = {type(s).__name__ for s in cr._DEFAULT_SCORERS}
    assert "FrozenExclusionScorer" not in names
    assert "EnvironmentAffordanceScorer" not in names   # same opt-in contract
