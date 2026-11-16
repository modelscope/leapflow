# Copyright (c) Alibaba, Inc. and its affiliates.
"""Result tiers: self-repair, feasibility refusal and real failure are told apart.

``ok: false`` alone conflated three events. These pin that each is graded,
shown, counted and charged to trust according to what it means: a repaired or
refused call is not a red failure, and a correct refusal never demotes a tool.
"""

from __future__ import annotations

from typing import Any

import pytest

from leapflow.cli.tui_app.stream import StreamRenderer
from leapflow.engine._message_helpers import _tool_result_metadata, _validate_tool_arguments
from leapflow.engine.tools.tool_execution import (
    TIER_FAILED,
    TIER_REFUSED,
    TIER_REPAIRED,
    TIER_SKIPPED,
    TIER_SUCCEEDED,
    result_tier,
)
from leapflow.engine.turn_usage import TurnUsageTracker
from leapflow.hardware.skill_results import outcome_fields, refusal_result
from leapflow.learning.plugin_stats import PluginUsageTracker, trust_verdict


# ── Grading ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("result", "tier"),
    [
        ({"ok": True}, TIER_SUCCEEDED),
        ("plain text output", TIER_SUCCEEDED),
        ({"ok": True, "argument_repair": {"message": "moved skill out of args"}}, TIER_REPAIRED),
        ({"ok": False, "counts_as_failure": False, "error": "missing x"}, TIER_REPAIRED),
        ({"ok": False, "failure_class": "invalid_request", "error": "bad"}, TIER_REPAIRED),
        ({"ok": False, "side_effect_state": "none", "failure_code": "unreachable"}, TIER_REFUSED),
        (
            {"ok": False, "side_effect_state": "none", "effect_uncertain": True},
            TIER_FAILED,
        ),
        (
            {"ok": False, "failure_class": "invalid_request", "effect_uncertain": True},
            TIER_FAILED,
        ),
        (
            {"ok": False, "side_effect_state": "none", "failure_class": "internal_defect"},
            TIER_FAILED,
        ),
        ({"ok": False, "error": "boom"}, TIER_FAILED),
        ({"ok": False, "side_effect_state": "unknown"}, TIER_FAILED),
        ({"ok": True, "duplicate_suppressed": True}, TIER_SKIPPED),
        ({"ok": True, "execution_skipped": True}, TIER_SKIPPED),
    ],
)
def test_result_tier_grades_by_meaning(result: Any, tier: str) -> None:
    assert result_tier(result) == tier


def test_pre_execution_argument_check_is_a_self_repair() -> None:
    class _Spec:
        name = "demo"
        required = frozenset({"target"})
        parameters = frozenset({"target", "speed"})

    invalid = _validate_tool_arguments(_Spec(), {"speed": 1})
    assert invalid is not None
    assert result_tier(invalid) == TIER_REPAIRED


def test_hardware_argument_refusals_declare_invalid_request() -> None:
    """A skill refusing its args before anything moved is a shape fix, not a verdict."""
    refused_args = outcome_fields(
        {"status": "failed", "failure_code": "invalid_args", "side_effect_state": "none"}
    )
    assert refused_args["failure_class"] == "invalid_request"
    assert result_tier(refused_args) == TIER_REPAIRED

    host_side = refusal_result("invalid_request", "workcell is required")
    assert host_side["failure_class"] == "invalid_request"
    assert result_tier(host_side) == TIER_REPAIRED

    infeasible = refusal_result("unreachable", "target out of reach")
    assert "failure_class" not in infeasible
    assert result_tier(infeasible) == TIER_REFUSED

    moved = outcome_fields(
        {"status": "failed", "failure_code": "guard_failed", "side_effect_state": "partial"}
    )
    assert result_tier(moved) == TIER_FAILED


def test_invalid_args_after_motion_is_not_downgraded() -> None:
    fields = outcome_fields(
        {"status": "failed", "failure_code": "invalid_args", "side_effect_state": "partial"}
    )
    assert "failure_class" not in fields
    assert result_tier(fields) == TIER_FAILED


def test_metadata_carries_the_tier_and_repair_note() -> None:
    refused = _tool_result_metadata(
        "hw_skill_run", {}, {"ok": False, "side_effect_state": "none", "failure_code": "unreachable"}
    )
    assert refused["result_tier"] == TIER_REFUSED
    assert refused["side_effect_state"] == "none"

    repaired = _tool_result_metadata(
        "hw_skill_run",
        {},
        {"ok": True, "argument_repair": {"message": "moved 'skill' beside args"}},
    )
    assert repaired["result_tier"] == TIER_REPAIRED
    assert "moved 'skill' beside args" in repaired["argument_repair"]


# ── TUI ─────────────────────────────────────────────────────────────────────


class _CapturingConsole:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, renderable="") -> None:
        self.lines.append(getattr(renderable, "plain", str(renderable)))

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def _render(result: dict[str, Any]) -> list[str]:
    console = _CapturingConsole()
    renderer = StreamRenderer(console)
    renderer.start()
    renderer.tool_started("hw_skill_run", {"tool_call_id": "c1", "normalized_tool_name": "hw_skill_run"})
    renderer.tool_finished(
        "hw_skill_run",
        metadata=_tool_result_metadata("hw_skill_run", {}, result, tool_call_id="c1"),
    )
    return console.lines


def test_refusal_renders_as_no_effect_not_failure() -> None:
    lines = _render(
        {
            "ok": False,
            "side_effect_state": "none",
            "failure_code": "unreachable",
            "error": "target out of reach",
        }
    )
    assert lines[0].strip().startswith("⊘ hw_skill_run")
    assert "no effect" in lines[0]
    assert "✗" not in lines[0]


def test_rejected_shape_renders_as_self_repair() -> None:
    lines = _render(
        {"ok": False, "failure_class": "invalid_request", "error": "skill is required"}
    )
    assert lines[0].strip().startswith("↺ hw_skill_run")
    assert "self-repair" in lines[0]


def test_repaired_success_prints_what_was_repaired() -> None:
    lines = _render({"ok": True, "argument_repair": {"message": "moved 'skill' beside args"}})
    assert lines[0].strip().startswith("↺ hw_skill_run")
    assert len(lines) == 2 and "repaired: moved 'skill' beside args" in lines[1]


def test_real_failure_stays_red_cross() -> None:
    lines = _render({"ok": False, "side_effect_state": "partial", "error": "guard failed"})
    assert lines[0].strip().startswith("✗ hw_skill_run")


# ── Statistics and trust ────────────────────────────────────────────────────


class _Ledger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def record_success(self, plugin_id: str) -> None:
        self.events.append(("success", plugin_id))

    def record_failure(self, plugin_id: str) -> None:
        self.events.append(("failure", plugin_id))


class _Quarantine:
    def __init__(self) -> None:
        self.events: list[bool] = []

    def record(self, plugin_id: str, tool_name: str, ok: bool) -> None:
        self.events.append(ok)


def _tracker() -> tuple[PluginUsageTracker, _Ledger, _Quarantine]:
    tracker = PluginUsageTracker()
    ledger, quarantine = _Ledger(), _Quarantine()
    tracker.set_trust_ledger(ledger)  # type: ignore[arg-type]
    tracker.set_quarantine_tracker(quarantine)
    tracker._tool_to_plugin = {"hw_skill_run": "hardware", "hw_skill_list": "hardware"}
    tracker._registry_version = 0
    tracker._get_reverse_index = lambda: tracker._tool_to_plugin  # type: ignore[method-assign]
    return tracker, ledger, quarantine


def test_trust_verdict_is_neutral_for_refusals_and_rejected_repairs() -> None:
    assert trust_verdict(False, TIER_FAILED) is False
    assert trust_verdict(False, TIER_REFUSED) is None
    assert trust_verdict(False, TIER_REPAIRED) is None
    assert trust_verdict(True, TIER_SKIPPED) is None
    assert trust_verdict(True, TIER_REPAIRED) is True
    assert trust_verdict(True, TIER_SUCCEEDED) is True
    # A legacy caller with no tier is graded by ok alone.
    assert trust_verdict(False, "") is False


def test_refusals_never_move_trust_or_quarantine() -> None:
    tracker, ledger, quarantine = _tracker()
    for _ in range(5):
        tracker.record("hw_skill_run", False, 10.0, tier=TIER_REFUSED, failure_code="unreachable")
    tracker.record("hw_skill_run", False, 1.0, tier=TIER_REPAIRED, failure_code="invalid_args")
    assert ledger.events == []
    assert quarantine.events == []

    tracker.record("hw_skill_run", False, 10.0, tier=TIER_FAILED, failure_code="guard_failed")
    tracker.record("hw_skill_run", True, 10.0, tier=TIER_SUCCEEDED)
    assert ledger.events == [("failure", "hardware"), ("success", "hardware")]
    assert quarantine.events == [False, True]


def test_stats_and_report_keep_the_three_rates_apart() -> None:
    tracker, _, _ = _tracker()
    tracker.record("hw_skill_run", True, 10.0, tier=TIER_SUCCEEDED)
    tracker.record("hw_skill_run", False, 10.0, tier=TIER_REFUSED, failure_code="unreachable")
    tracker.record("hw_skill_run", False, 10.0, tier=TIER_REFUSED, failure_code="unreachable")
    tracker.record("hw_skill_run", False, 1.0, tier=TIER_REPAIRED, failure_code="invalid_args")
    tracker.record("hw_skill_run", False, 10.0, tier=TIER_FAILED, failure_code="guard_failed")

    stats = tracker.stats_for_plugin("hardware")
    assert stats is not None
    assert stats.total_calls == 5
    assert stats.failures == 1 and stats.error_rate == 0.2
    assert stats.refusals == 2 and stats.refusal_rate == 0.4
    assert stats.repairs == 1 and stats.repair_rate == 0.2

    (row,) = tracker.tool_report("hardware")
    assert row["tool"] == "hw_skill_run" and row["calls"] == 5
    assert row["failure_rate"] == 0.2 and row["refusal_rate"] == 0.4 and row["repair_rate"] == 0.2
    assert row["failure_codes"] == {"unreachable": 2, "invalid_args": 1, "guard_failed": 1}


def test_persistence_round_trips_tiers_and_loads_legacy_rows() -> None:
    tracker, _, _ = _tracker()
    tracker.record("hw_skill_run", False, 10.0, tier=TIER_REFUSED, failure_code="unreachable")
    restored = PluginUsageTracker.load_state(tracker.to_state())
    (sample,) = restored._samples["hw_skill_run"]
    assert sample.tier == TIER_REFUSED and sample.failure_code == "unreachable"

    legacy = PluginUsageTracker.load_state(
        {"samples": {"shell_run": [[1.0, False, 5.0], [2.0, True, 3.0]]}}
    )
    tiers = [s.tier for s in legacy._samples["shell_run"]]
    assert tiers == [TIER_FAILED, TIER_SUCCEEDED]


def test_turn_summary_counts_tiers_and_forwards_them() -> None:
    tracker, ledger, _ = _tracker()
    usage = TurnUsageTracker()
    usage.set_plugin_stats_sink(tracker)
    usage.record_tool_call("hw_skill_run", False, 5.0, tier=TIER_REFUSED, failure_code="unreachable")
    usage.record_tool_call("hw_skill_run", False, 0.0, tier=TIER_REPAIRED)
    usage.record_tool_call("hw_skill_run", False, 5.0)  # untiered: graded by ok
    usage.record_tool_call("hw_skill_list", True, 5.0)

    summary = usage.summary()
    assert summary.tool_calls == 4
    assert summary.tool_failures == 1
    assert summary.tool_refusals == 1
    assert summary.tool_repairs == 1
    assert ledger.events == [("failure", "hardware"), ("success", "hardware")]
