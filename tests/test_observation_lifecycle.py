"""LF-11: the observation lifecycle is no longer write-only.

`JsonCapabilityObservationStore.mark_status` existed with **zero callers and zero
test coverage**, so `unresolved()` grew monotonically: evidence that motivated a
capability which later resolved kept being reported, and any consumer sizing work
from it would re-propose capabilities the system already had.
"""

from __future__ import annotations

from leapflow.domain.evolution_intent import WORLD_MODEL_INTENT, EvolutionIntent
from leapflow.learning.capability_observation import (
    CapabilityEvidenceClassifier,
    CapabilityObservationService,
)
from leapflow.storage.capability_observation_store import JsonCapabilityObservationStore

_UNKNOWN = {
    "error_type": "unknown_tool",
    "original_tool_name": "list_dir",
    "recovery_hint": "use file_list",
}


def _service(tmp_path, *, kinds=None):
    store = JsonCapabilityObservationStore(tmp_path / "observations.json")
    classifier = CapabilityEvidenceClassifier.from_kinds(kinds) if kinds else None
    return store, CapabilityObservationService(store, classifier=classifier)


def test_resolving_a_capability_retires_its_observation(tmp_path):
    store, service = _service(tmp_path)
    service.observe_result(_UNKNOWN)
    assert len(store.unresolved()) == 1
    assert [r.capability for r in service.requirements()] == ["list_dir"]

    retired = service.resolve_capability("list_dir", reason="acquired")
    assert len(retired) == 1
    assert store.unresolved() == []
    assert service.requirements() == ()          # no longer re-proposed


def test_resolution_reason_is_recorded(tmp_path):
    store, service = _service(tmp_path)
    service.observe_result(_UNKNOWN)
    service.resolve_capability("list_dir", reason="resolved in loop-7")
    record = store.list_observations(limit=10)[0]
    assert record["status"] == "resolved"
    assert record["status_reason"] == "resolved in loop-7"


def test_only_the_matching_capability_is_retired(tmp_path):
    store, service = _service(tmp_path)
    service.observe_result(_UNKNOWN)
    service.observe_result({
        "error_type": "unknown_tool",
        "original_tool_name": "grep_code",
    })
    assert len(store.unresolved()) == 2

    service.resolve_capability("list_dir")
    remaining = store.unresolved()
    assert len(remaining) == 1
    assert remaining[0]["result"]["original_tool_name"] == "grep_code"


def test_unrelated_capability_retires_nothing(tmp_path):
    store, service = _service(tmp_path)
    service.observe_result(_UNKNOWN)
    assert service.resolve_capability("something.else") == ()
    assert len(store.unresolved()) == 1


def test_blank_capability_is_a_no_op(tmp_path):
    store, service = _service(tmp_path)
    service.observe_result(_UNKNOWN)
    assert service.resolve_capability("") == ()
    assert service.resolve_capability("   ") == ()
    assert len(store.unresolved()) == 1


def test_world_model_intent_observations_are_retired_too(tmp_path):
    """The world-model path must not leak an ever-growing backlog either."""
    store, service = _service(tmp_path, kinds=[WORLD_MODEL_INTENT])
    intent = EvolutionIntent.create("chat.reply", "v2 send path unserved")
    service.observe_result(intent.to_observation_result())
    assert len(store.unresolved()) == 1

    retired = service.resolve_capability("chat.reply")
    assert len(retired) == 1
    assert store.unresolved() == []


def test_durable_round_trip_preserves_the_clamped_risk_ceiling(tmp_path):
    """The store must not widen a requirement's risk ceiling in transit.

    ``_safe_result`` filters the payload to a field whitelist. When that whitelist
    omitted ``max_risk_level``, a requirement rebuilt from a persisted observation
    inherited the domain default of ``external`` -- the *most permissive* ceiling --
    so an intent clamped to ``read_only`` came back able to select mutating tools.
    ``origin`` was lost the same way, making a world-model intent
    indistinguishable from an environment probe.

    This is the production path (engine -> observe_result -> store ->
    requirements), so an in-memory-only test cannot cover it.
    """
    store, service = _service(tmp_path, kinds=[WORLD_MODEL_INTENT])
    intent = EvolutionIntent.create(
        "chat.reply", "v2 send path unserved",
        max_risk_level="external",            # the model asked for the most permissive
        target_affordance="app.chat.v2",
        expected_effect="message appears in the thread",
    )
    payload = intent.to_observation_result()   # ...and was clamped to read_only
    assert payload["max_risk_level"] == "read_only"
    service.observe_result(payload)

    persisted = store.unresolved()[0]["result"]
    assert persisted["max_risk_level"] == "read_only"
    assert persisted["origin"] == "world_model"

    requirement = service.requirements()[0]
    assert requirement.max_risk_level == "read_only"   # not "external"
    assert requirement.origin == "world_model"
    meta = dict(requirement.metadata)
    assert meta["target_affordance"] == "app.chat.v2"
    assert meta["requested_max_risk_level"] == "external"   # denial stays auditable


def test_retiring_is_idempotent(tmp_path):
    store, service = _service(tmp_path)
    service.observe_result(_UNKNOWN)
    first = service.resolve_capability("list_dir")
    second = service.resolve_capability("list_dir")
    assert len(first) == 1
    assert second == ()                          # already retired, nothing to do
    assert len(store.list_observations(limit=10)) == 1


def test_recurrence_after_resolution_is_observed_again(tmp_path):
    """A gap that reopens must be visible again, not permanently silenced."""
    store, service = _service(tmp_path)
    service.observe_result(_UNKNOWN)
    service.resolve_capability("list_dir")
    assert store.unresolved() == []

    # The same failure happens again: dedup updates the existing record, so the
    # question is whether a retired observation can come back.
    service.observe_result(_UNKNOWN)
    reopened = store.unresolved()
    assert len(reopened) == 1, (
        "a recurring gap stayed retired; add_observation must reopen a resolved "
        "record or a real regression would be silently ignored"
    )
