# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for streaming usage telemetry and Anthropic cache-rate correction.

Covers:
1. Streaming text path telemetry recording (with and without usage data).
2. Anthropic usage semantic adaptation (effective prompt denominator).
3. OpenAI/DeepSeek backward compatibility (no change in behavior).
4. Edge-case tolerance (empty usage, missing keys, zero values).
"""
from __future__ import annotations

import types
from typing import Any, Dict

from leapflow.engine.turn_usage import TurnUsageTracker


# ═══════════════════════════════════════════════════════════════════════════
#  Streaming text path: telemetry with empty usage (no resp object)
# ═══════════════════════════════════════════════════════════════════════════

class TestStreamingTextPathTelemetry:
    """Streaming text path records API call even when usage is unavailable."""

    def test_empty_usage_records_api_call(self) -> None:
        """Empty usage dict increments api_calls but records zero tokens."""
        tracker = TurnUsageTracker(steady_state_skip_turns=0)
        tracker.record_api_call({}, provider="openai", model="gpt-4o")
        summary = tracker.summary()
        assert summary.api_calls == 1
        assert summary.prompt_tokens == 0
        assert summary.cached_tokens == 0
        assert summary.completion_tokens == 0
        assert summary.provider_name == "openai"
        assert summary.model == "gpt-4o"

    def test_missing_keys_in_usage_treated_as_zero(self) -> None:
        """Usage with missing keys falls back to 0, no crash."""
        tracker = TurnUsageTracker(steady_state_skip_turns=0)
        # Partial usage dict — only completion_tokens present
        usage: Dict[str, Any] = {
            "completion_tokens": 50,
        }
        tracker.record_api_call(usage, provider="test")
        summary = tracker.summary()
        assert summary.api_calls == 1
        assert summary.prompt_tokens == 0
        assert summary.cached_tokens == 0
        assert summary.completion_tokens == 50

    def test_streaming_resp_with_no_usage_attr(self) -> None:
        """SimpleNamespace with usage=None simulates streaming achat_stream resp."""
        tracker = TurnUsageTracker(steady_state_skip_turns=0)
        # This is what the engine creates for streaming text path
        stream_resp = types.SimpleNamespace(
            usage=None,
            model="qwen-turbo",
        )
        usage = getattr(stream_resp, "usage", None) or {}
        tracker.record_api_call(
            usage,
            provider="qwen",
            model=getattr(stream_resp, "model", "") or "",
        )
        summary = tracker.summary()
        assert summary.api_calls == 1
        assert summary.prompt_tokens == 0
        assert summary.cached_tokens == 0
        assert summary.model == "qwen-turbo"

    def test_streaming_resp_with_usage_dict(self) -> None:
        """When a streaming collapsed resp carries usage, tokens are recorded."""
        tracker = TurnUsageTracker(steady_state_skip_turns=0)
        stream_resp = types.SimpleNamespace(
            usage={"prompt_tokens": 500, "cached_tokens": 200,
                   "completion_tokens": 50, "total_tokens": 550},
            model="gpt-4o",
        )
        usage = getattr(stream_resp, "usage", None) or {}
        tracker.record_api_call(
            usage,
            provider="openai",
            model=getattr(stream_resp, "model", "") or "",
        )
        summary = tracker.summary()
        assert summary.api_calls == 1
        assert summary.prompt_tokens == 500
        assert summary.cached_tokens == 200
        assert summary.completion_tokens == 50

    def test_session_stats_unaffected_by_empty_usage_turn(self) -> None:
        """A streaming turn with empty usage doesn't corrupt session stats."""
        tracker = TurnUsageTracker(steady_state_skip_turns=0)

        # Turn 0: normal usage
        tracker.record_api_call(
            {"prompt_tokens": 1000, "cached_tokens": 800,
             "completion_tokens": 100, "total_tokens": 1100},
        )
        tracker.reset()

        # Turn 1: streaming path with empty usage
        tracker.record_api_call({})
        tracker.reset()

        # Turn 2: normal usage
        tracker.record_api_call(
            {"prompt_tokens": 2000, "cached_tokens": 1600,
             "completion_tokens": 200, "total_tokens": 2200},
        )

        stats = tracker.session_cache_stats()
        # Session totals should include turns 0 and 2 only for tokens
        assert stats.total_prompt_tokens == 3000  # 1000 + 0 + 2000
        assert stats.total_cached_tokens == 2400  # 800 + 0 + 1600
        assert stats.completed_turns == 3
        # Token-weighted rate uses total tokens
        assert stats.token_weighted_hit_rate == 0.8  # 2400/3000


# ═══════════════════════════════════════════════════════════════════════════
#  Anthropic usage semantic adaptation
# ═══════════════════════════════════════════════════════════════════════════

class TestAnthropicUsageAdaptation:
    """Anthropic usage: effective prompt = input + cache_read + cache_creation."""

    def _anthropic_usage(
        self,
        input_tokens: int = 200,
        cache_read: int = 800,
        cache_create: int = 100,
        output_tokens: int = 50,
    ) -> Dict[str, int]:
        """Build a usage dict as returned by Anthropic provider's _parse_usage."""
        usage: Dict[str, int] = {
            "prompt_tokens": input_tokens,  # = input_tokens (Anthropic mapping)
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cache_read_input_tokens": cache_read,
            "cached_tokens": cache_read,  # unified key
            "cache_creation_input_tokens": cache_create,
        }
        return usage

    def test_cache_hit_rate_within_100_percent(self) -> None:
        """Anthropic cache hit rate must not exceed 100%."""
        tracker = TurnUsageTracker(steady_state_skip_turns=0)
        usage = self._anthropic_usage(
            input_tokens=200, cache_read=800, cache_create=100,
        )
        tracker.record_api_call(usage, provider="anthropic", model="claude-sonnet-4-20250514")
        summary = tracker.summary()

        # Effective prompt = 200 + 800 + 100 = 1100
        assert summary.prompt_tokens == 1100
        assert summary.cached_tokens == 800

        # cache_hit_rate = 800 / 1100 ≈ 0.7273
        rate = summary.cache_hit_rate
        assert 0.0 <= rate <= 1.0, f"cache_hit_rate {rate} exceeds 100%"
        assert abs(rate - 800 / 1100) < 0.001

    def test_session_stats_anthropic_denominator(self) -> None:
        """Session-level token-weighted rate uses effective prompt for Anthropic."""
        tracker = TurnUsageTracker(steady_state_skip_turns=0)

        # Turn 0: Anthropic
        tracker.record_api_call(self._anthropic_usage(
            input_tokens=200, cache_read=800, cache_create=100,
        ))
        tracker.reset()

        # Turn 1: Anthropic with higher cache hit
        tracker.record_api_call(self._anthropic_usage(
            input_tokens=100, cache_read=900, cache_create=50,
        ))

        stats = tracker.session_cache_stats()
        # Turn 0 effective prompt = 200+800+100 = 1100, cached = 800
        # Turn 1 effective prompt = 100+900+50 = 1050, cached = 900
        # Total effective prompt = 2150, total cached = 1700
        assert stats.total_prompt_tokens == 2150
        assert stats.total_cached_tokens == 1700
        tw = stats.token_weighted_hit_rate
        expected = round(1700 / 2150, 4)
        assert tw == expected

    def test_anthropic_zero_cache_no_adjustment(self) -> None:
        """Anthropic usage with zero cache reads/writes: no denominator adjustment."""
        tracker = TurnUsageTracker(steady_state_skip_turns=0)
        usage: Dict[str, int] = {
            "prompt_tokens": 500,
            "completion_tokens": 50,
            "total_tokens": 550,
            "cache_read_input_tokens": 0,
            "cached_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        tracker.record_api_call(usage, provider="anthropic")
        summary = tracker.summary()
        # Both cache keys are 0, so no adjustment
        assert summary.prompt_tokens == 500
        assert summary.cache_hit_rate == 0.0

    def test_anthropic_only_cache_read(self) -> None:
        """Anthropic with cache_read but no cache_creation."""
        tracker = TurnUsageTracker(steady_state_skip_turns=0)
        usage: Dict[str, int] = {
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "total_tokens": 130,
            "cache_read_input_tokens": 400,
            "cached_tokens": 400,
        }
        tracker.record_api_call(usage)
        summary = tracker.summary()
        # effective prompt = 100 + 400 + 0 = 500
        assert summary.prompt_tokens == 500
        assert summary.cached_tokens == 400
        assert summary.cache_hit_rate == 0.8  # 400/500

    def test_anthropic_only_cache_creation(self) -> None:
        """Anthropic with cache_creation but no cache_read (first call, cold miss)."""
        tracker = TurnUsageTracker(steady_state_skip_turns=0)
        usage: Dict[str, int] = {
            "prompt_tokens": 300,
            "completion_tokens": 40,
            "total_tokens": 340,
            "cache_creation_input_tokens": 200,
            "cached_tokens": 0,
        }
        tracker.record_api_call(usage)
        summary = tracker.summary()
        # effective prompt = 300 + 0 + 200 = 500
        assert summary.prompt_tokens == 500
        assert summary.cached_tokens == 0
        assert summary.cache_hit_rate == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  OpenAI / DeepSeek backward compatibility
# ═══════════════════════════════════════════════════════════════════════════

class TestOpenAIBackwardCompatibility:
    """OpenAI/DeepSeek usage semantics remain unchanged."""

    def test_openai_usage_no_anthropic_keys(self) -> None:
        """Standard OpenAI usage without Anthropic keys: no adjustment."""
        tracker = TurnUsageTracker(steady_state_skip_turns=0)
        usage: Dict[str, int] = {
            "prompt_tokens": 1000,
            "cached_tokens": 800,
            "completion_tokens": 100,
            "total_tokens": 1100,
        }
        tracker.record_api_call(usage, provider="openai", model="gpt-4o")
        summary = tracker.summary()
        assert summary.prompt_tokens == 1000  # unchanged
        assert summary.cached_tokens == 800
        assert summary.cache_hit_rate == 0.8  # 800/1000

    def test_deepseek_usage_no_anthropic_keys(self) -> None:
        """DeepSeek usage: same semantics as OpenAI, no adjustment."""
        tracker = TurnUsageTracker(steady_state_skip_turns=0)
        usage: Dict[str, int] = {
            "prompt_tokens": 5000,
            "cached_tokens": 4000,
            "completion_tokens": 500,
            "total_tokens": 5500,
        }
        tracker.record_api_call(usage, provider="deepseek", model="deepseek-chat")
        summary = tracker.summary()
        assert summary.prompt_tokens == 5000
        assert summary.cached_tokens == 4000
        assert summary.cache_hit_rate == 0.8

    def test_mixed_providers_in_session(self) -> None:
        """Session with both OpenAI and Anthropic turns: each adjusted correctly."""
        tracker = TurnUsageTracker(steady_state_skip_turns=0)

        # Turn 0: OpenAI
        tracker.record_api_call({
            "prompt_tokens": 1000,
            "cached_tokens": 800,
            "completion_tokens": 100,
            "total_tokens": 1100,
        }, provider="openai")
        tracker.reset()

        # Turn 1: Anthropic
        tracker.record_api_call({
            "prompt_tokens": 200,  # input_tokens
            "cached_tokens": 800,  # cache_read
            "completion_tokens": 50,
            "total_tokens": 250,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 100,
        }, provider="anthropic")

        stats = tracker.session_cache_stats()
        # Turn 0: prompt=1000, cached=800
        # Turn 1: effective_prompt=200+800+100=1100, cached=800
        assert stats.total_prompt_tokens == 2100  # 1000 + 1100
        assert stats.total_cached_tokens == 1600  # 800 + 800
        tw = stats.token_weighted_hit_rate
        expected = round(1600 / 2100, 4)
        assert tw == expected
        # Both rates ≤ 1.0
        assert 0.0 <= tw <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  Edge cases: tolerance for malformed / partial usage dicts
# ═══════════════════════════════════════════════════════════════════════════

class TestUsageEdgeCases:
    """Robustness against unusual usage payloads."""

    def test_completely_empty_dict(self) -> None:
        tracker = TurnUsageTracker()
        tracker.record_api_call({})
        summary = tracker.summary()
        assert summary.api_calls == 1
        assert summary.prompt_tokens == 0
        assert summary.cache_hit_rate == 0.0

    def test_anthropic_keys_with_none_values(self) -> None:
        """Anthropic cache keys present but set to None: treated as 0."""
        tracker = TurnUsageTracker(steady_state_skip_turns=0)
        usage: Dict[str, Any] = {
            "prompt_tokens": 500,
            "cached_tokens": 0,
            "completion_tokens": 50,
            "cache_read_input_tokens": None,  # type: ignore[dict-item]
            "cache_creation_input_tokens": None,  # type: ignore[dict-item]
        }
        tracker.record_api_call(usage)
        summary = tracker.summary()
        # None or 0 → 0, so no Anthropic adjustment
        assert summary.prompt_tokens == 500

    def test_effective_prompt_tokens_with_anthropic(self) -> None:
        """TurnUsageSummary.effective_prompt_tokens works with corrected prompt."""
        tracker = TurnUsageTracker(steady_state_skip_turns=0)
        tracker.record_api_call({
            "prompt_tokens": 200,
            "cached_tokens": 800,
            "completion_tokens": 50,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 100,
        })
        summary = tracker.summary()
        # effective_prompt = 1100, cached = 800
        # miss = 1100 - 800 = 300
        # effective(ratio=0.1) = 300 + 800*0.1 = 380.0
        eff = summary.effective_prompt_tokens(cached_price_ratio=0.1)
        assert eff == 380.0


def test_capture_does_not_inherit_foreign_scope_or_task_identity() -> None:
    from leapflow.domain.events import RecordScope
    from leapflow.llm.services.telemetry import capture_inference

    with capture_inference(scope=RecordScope("p", "w1", "s1"), task_id="t1", tool_call_id="tc1") as parent:
        with capture_inference(scope=RecordScope("p", "w2", "s2")) as child:
            assert child.parent is None and child.task_id == "" and child.tool_call_id == ""
        with capture_inference() as child:
            assert child.parent is parent and child.task_id == "t1" and child.tool_call_id == "tc1"


def test_consultation_record_is_frozen_and_serialization_is_detached() -> None:
    import json
    from dataclasses import FrozenInstanceError
    import pytest
    from leapflow.domain.events import RecordScope
    from leapflow.llm.services.telemetry import ConsultationRecord

    record = ConsultationRecord("c", RecordScope("p", "w", "s"), "t", "tc", "specialist",
                                "eligible", 4, provenance_json=json.dumps({"evidence": ["a"]}),
                                timing_json=json.dumps({"acquisition_ms": 1}))
    detached = record.to_dict()
    detached["provenance"]["evidence"].append("b")
    detached["timing"]["acquisition_ms"] = 999
    assert record.to_dict()["provenance"]["evidence"] == ["a"]
    assert record.to_dict()["timing"] == {"acquisition_ms": 1}
    assert record.to_dict()["adoption"] == "unlinked"
    with pytest.raises(FrozenInstanceError):
        record.status = "failed"


def test_phase_timing_is_nested_owned_and_preserves_original_failure(monkeypatch) -> None:
    from types import SimpleNamespace
    import pytest
    from leapflow.domain.events import RecordScope
    from leapflow.llm.services import telemetry

    scope = RecordScope("p", "w", "s")
    ticks = iter([1.0, 1.025])
    monkeypatch.setattr(telemetry, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    failure = RuntimeError("original provider failure")
    with telemetry.capture_inference(scope=scope) as parent:
        with telemetry.capture_inference() as nested:
            with pytest.raises(RuntimeError) as error:
                with telemetry.measure_phase("cleanup_ms"):
                    raise failure
        assert error.value is failure
        assert nested.phase_timings == parent.phase_timings
        assert parent.phase_timings["cleanup_ms"] == pytest.approx(25)
        with telemetry.capture_inference(scope=RecordScope("p", "foreign", "s")) as foreign:
            telemetry.record_phase("cleanup_ms", 100)
        assert foreign.phase_timings == {"cleanup_ms": 100}
        assert parent.phase_timings["cleanup_ms"] == pytest.approx(25)


def test_phase_timing_defect_does_not_replace_execution_failure(monkeypatch) -> None:
    import pytest
    from leapflow.llm.services import telemetry

    def broken(*args):
        raise AttributeError("timing sink defect")

    failure = RuntimeError("original execution failure")
    with telemetry.capture_inference() as capture:
        monkeypatch.setattr(capture, "record_phase", broken)
        with pytest.raises(RuntimeError) as error:
            with telemetry.measure_phase("encoding_ms"):
                raise failure
    assert error.value is failure


def test_collaboration_feedback_is_admitted_bounded_scoped_and_survives_reset() -> None:
    import json
    from dataclasses import asdict, replace
    from leapflow.domain.events import RecordAdmission, RecordScope

    scope = RecordScope("p", "w", "s")
    tracker = TurnUsageTracker(max_feedback_records=2)
    receipts = []
    for index in range(3):
        value = {"call_id": str(index), "service_id": "specialist", "scope": asdict(scope),
                 "status": "eligible", "adoption": "unlinked", "elapsed_ms": index}
        receipt = RecordAdmission(True, "admitted", scope, "model.consultation",
                                  json.dumps({"value": value}))
        receipts.append(receipt)
        tracker.record_collaboration(receipt)
    tracker.record_collaboration(receipts[-1])
    tracker.record_collaboration(replace(receipts[-1], admitted=False))
    tracker.record_collaboration(replace(receipts[-1], payload_json="[]"))
    tracker.record_collaboration(replace(receipts[-1], payload_json="not json"))
    tracker.record_collaboration(replace(receipts[-1], scope=RecordScope("p", "w", "other")))
    tracker.reset()
    feedback = tracker.collaboration_feedback(scope)
    assert [r["call_id"] for r in feedback["records"]] == ["1", "2"]
    assert feedback["measured_task_outcome"] == "unverified" and not feedback["physical_authority"]
    assert tracker.collaboration_feedback(scope, service_id="another")["records"] == []
    assert tracker.collaboration_feedback(RecordScope("p", "w", "other"))["records"] == []


def test_receipt_restore_checks_inner_and_outer_identity() -> None:
    import json
    from dataclasses import asdict
    from leapflow.domain.events import RecordScope

    scope = RecordScope("p", "w", "s")
    value = {"scope": asdict(scope), "call_id": "c", "service_id": "specialist", "status": "refused"}
    row = {"event_type": "model.consultation", "record_id": "c", "scope": asdict(scope),
           "admitted": True, "lineage_eligible": True, "payload_json": json.dumps({"value": value})}
    tracker = TurnUsageTracker()
    tracker.restore_collaboration({"collaboration_signal_receipts": [
        {**row, "record_id": "foreign"}, {**row, "payload_json": "[]"},
        {**row, "scope": {**asdict(scope), "session_id": "other"}}, row]}, scope)
    assert len(tracker.collaboration_feedback(scope)["records"]) == 1


def test_plan_link_lookup_rejects_foreign_tasks_and_ineligible_adoption() -> None:
    import json
    from dataclasses import asdict
    from leapflow.domain.events import RecordAdmission, RecordScope
    from leapflow.llm.services.telemetry import ConsultationReference, capture_inference, record_governed_plan

    scope = RecordScope("p", "w", "s")
    tracker = TurnUsageTracker()
    for identity, task, status in [("a", "t", "eligible"), ("b", "foreign", "eligible"), ("c", "t", "refused")]:
        value = {"scope": asdict(scope), "call_id": identity, "task_id": task, "status": status}
        tracker.record_collaboration(RecordAdmission(True, "admitted", scope, "model.consultation",
                                                     json.dumps({"value": value})))
    refs = tuple(ConsultationReference(identity, "adopted") for identity in ("a", "b", "c", "missing"))
    with capture_inference(scope=scope, task_id="t", sink=tracker) as capture:
        record_governed_plan(workcell="node/cell", skill_id="motion.sequence", plan_digest="plan",
                             permit_digest="permit", references=refs, result={"status": "succeeded"})
    links = capture.plans[0].to_dict()["references"]
    assert [row["link_status"] for row in links] == ["linked", "unknown_or_foreign", "ineligible_advice", "unknown_or_foreign"]
    assert tracker.consultation_reference(RecordScope("p", "w", "other"), "t", "a") is None
    assert tracker.consultation_reference(scope, "", "a") is None
    assert capture.plans[0].to_dict()["measured_task_outcome"] == "unverified"


def test_lineage_sink_defect_never_fails_verified_execution() -> None:
    from leapflow.llm.services.telemetry import ConsultationReference, capture_inference, record_governed_plan

    class BrokenSink:
        def consultation_reference(self, scope, task_id, call_id):
            raise RuntimeError("lookup defect")

    with capture_inference(sink=BrokenSink()) as capture:
        record_governed_plan(workcell="node/cell", skill_id="motion.sequence", plan_digest="plan",
            permit_digest="permit", references=(ConsultationReference("a", "adopted"),), result={"status": "succeeded"})
    assert not capture.plans


def test_nested_plan_capture_forwards_identity_but_not_foreign_outcomes() -> None:
    from dataclasses import replace
    from leapflow.domain.events import RecordScope
    from leapflow.llm.services.telemetry import ConsultationReference, capture_inference, record_governed_outcome, record_governed_plan

    scope = RecordScope("p", "w", "s")
    with capture_inference(scope=scope, task_id="t") as parent:
        with capture_inference() as child:
            plan = record_governed_plan(workcell="node/cell", skill_id="motion.sequence", plan_digest="plan",
                permit_digest="permit", references=(ConsultationReference("a", "unused"),),
                result={"run_id": "run", "status": "running"})
        assert child.plans == parent.plans == [plan]
    outcome = {"run_id": "run", "workcell": "node/cell", "plan_digest": "plan", "status": "succeeded"}
    for owner, task in [(replace(scope, session_id="foreign"), "t"), (scope, "foreign")]:
        with capture_inference(scope=owner, task_id=task) as refused:
            record_governed_outcome(plan, outcome)
        assert not refused.plans
    with capture_inference(scope=scope, task_id="t", tool_call_id="status") as accepted:
        for field in ("run_id", "workcell", "plan_digest"):
            record_governed_outcome(plan, {**outcome, field: "foreign"})
        assert not accepted.plans
        record_governed_outcome(plan, outcome)
    assert accepted.plans[0].origin_record_id == plan.record_id
    assert accepted.plans[0].to_dict()["measured_task_outcome"] == "unverified"
