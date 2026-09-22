# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for CalibrationManager — budget calibration and prefix commitment."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Dict


from leapflow.engine.budget import BudgetConfig, IterationBudget
from leapflow.engine.calibration import CalibrationManager
from leapflow.engine.context.context_disclosure import CacheBoundary
from leapflow.engine.prefix_commitment import PrefixCommitmentController


# ── Minimal engine stub ──────────────────────────────────────────────


def _stub_engine(
    *,
    scale_k: float = 1.0,
    max_iterations: int = 20,
    iter_ceiling: int = 0,
    calibration_enabled: bool = False,
    calibration_interval_turns: int = 0,
    cost_ceiling_multiple: float = 0.0,
    calibration_store: Any = None,
    calibration_event_store: Any = None,
    context_length: int = 128_000,
    last_context_snapshot: Dict[str, Any] | None = None,
    context_finalizing_ratio: float = 0.85,
    calibrated_finalizing_ratio: float | None = None,
) -> SimpleNamespace:
    """Build a minimal engine stub for CalibrationManager."""
    budget_config = BudgetConfig(
        max_iterations=max_iterations,
        iter_ceiling=iter_ceiling,
        scale_k=scale_k,
    )
    settings = SimpleNamespace(
        agent_calibration_enabled=calibration_enabled,
        agent_calibration_interval_turns=calibration_interval_turns,
        agent_calibration_difficulty_min_k=0.25,
        agent_calibration_difficulty_max_k=3.0,
        agent_calibration_min_confidence=0.3,
        agent_calibration_finalizing_min_ratio=0.6,
        agent_calibration_finalizing_max_ratio=0.98,
        agent_cost_ceiling_context_multiple=cost_ceiling_multiple,
        context_finalizing_ratio=context_finalizing_ratio,
        profile="default",
    )
    usage_summary = SimpleNamespace(effective_prompt_tokens=lambda: 50_000)
    engine = SimpleNamespace(
        _settings=settings,
        _budget_config=budget_config,
        _baseline_scale_k=scale_k,
        _calibration_store=calibration_store,
        _calibration_event_store=calibration_event_store,
        _last_context_snapshot=last_context_snapshot or {},
        _turns_since_calibration=0,
        _calibrated_finalizing_ratio=calibrated_finalizing_ratio,
        _context_governance_controller=SimpleNamespace(),
        _new_governance=lambda: SimpleNamespace(),
        _active_context_length=lambda: context_length,
        _usage_tracker=SimpleNamespace(summary=lambda: usage_summary),
        _research_ledger=SimpleNamespace(
            as_dict=lambda: {"findings": [], "open_questions": [], "decisions": [], "next_step": ""},
        ),
        # prefix commitment stubs
        _prefix_commitment=PrefixCommitmentController(),
        _current_cache_boundary=CacheBoundary.NONE,
        _last_disclosure_metadata={},
        _last_system_prompt="",
        _full_tools_tokens=None,
        _context_controller=SimpleNamespace(
            estimator=SimpleNamespace(
                estimate_tools=lambda tools: 500,
            ),
        ),
        _tool_dispatch=SimpleNamespace(
            _unified_tool_catalog=lambda: [],
        ),
    )
    return engine


# ── Construction ─────────────────────────────────────────────────────


class TestConstruction:
    def test_creates_with_engine_back_reference(self) -> None:
        engine = _stub_engine()
        mgr = CalibrationManager(engine)
        assert mgr._engine is engine


# ── recalibrate_difficulty ───────────────────────────────────────────


class TestRecalibrateDifficulty:
    def test_disabled_returns_not_applied(self) -> None:
        engine = _stub_engine(calibration_enabled=False)
        mgr = CalibrationManager(engine)
        result = mgr.recalibrate_difficulty(store=SimpleNamespace())
        assert result.applied is False
        assert "disabled" in result.reason

    def test_no_store_returns_not_applied(self) -> None:
        engine = _stub_engine(calibration_enabled=True)
        mgr = CalibrationManager(engine)
        result = mgr.recalibrate_difficulty(store=None)
        assert result.applied is False
        assert "no evolution store" in result.reason


# ── reset_calibration ────────────────────────────────────────────────


class TestResetCalibration:
    def test_resets_scale_k_to_baseline(self) -> None:
        engine = _stub_engine(scale_k=1.5)
        engine._budget_config = replace(engine._budget_config, scale_k=2.0)
        mgr = CalibrationManager(engine)
        mgr.reset_calibration()
        assert engine._budget_config.scale_k == 1.5


# ── recalibrate_thresholds ──────────────────────────────────────────


class TestRecalibrateThresholds:
    def test_disabled_returns_not_applied(self) -> None:
        engine = _stub_engine(calibration_enabled=False)
        mgr = CalibrationManager(engine)
        result = mgr.recalibrate_thresholds(store=SimpleNamespace())
        assert result.applied is False
        assert "disabled" in result.reason

    def test_no_store_returns_not_applied(self) -> None:
        engine = _stub_engine(calibration_enabled=True)
        mgr = CalibrationManager(engine)
        result = mgr.recalibrate_thresholds(store=None)
        assert result.applied is False


# ── _widen_budget_for_difficulty ─────────────────────────────────────


class TestWidenBudgetForDifficulty:
    def test_widens_when_difficulty_present(self) -> None:
        engine = _stub_engine(
            max_iterations=20,
            iter_ceiling=60,
            last_context_snapshot={"difficulty": 0.8},
        )
        mgr = CalibrationManager(engine)
        budget = IterationBudget(engine._budget_config)
        mgr._widen_budget_for_difficulty(budget)
        assert budget.effective_max > 20

    def test_no_op_when_difficulty_zero(self) -> None:
        engine = _stub_engine(
            max_iterations=20,
            iter_ceiling=60,
            last_context_snapshot={"difficulty": 0.0},
        )
        mgr = CalibrationManager(engine)
        budget = IterationBudget(engine._budget_config)
        mgr._widen_budget_for_difficulty(budget)
        assert budget.effective_max == 20

    def test_no_op_for_fixed_budget(self) -> None:
        engine = _stub_engine(
            max_iterations=20,
            iter_ceiling=0,  # fixed
            last_context_snapshot={"difficulty": 0.9},
        )
        mgr = CalibrationManager(engine)
        budget = IterationBudget(engine._budget_config)
        mgr._widen_budget_for_difficulty(budget)
        assert budget.effective_max == 20


# ── _task_progress_marker ────────────────────────────────────────────


class TestTaskProgressMarker:
    def test_returns_tuple(self) -> None:
        engine = _stub_engine(
            last_context_snapshot={
                "context_governance": {
                    "evidence_count": 5,
                    "sources_seen": 3,
                    "repeated_reads": 1,
                },
            },
        )
        mgr = CalibrationManager(engine)
        marker = mgr._task_progress_marker()
        assert isinstance(marker, tuple)
        assert len(marker) == 7
        assert marker[4] == 5  # evidence_count
        assert marker[5] == 3  # sources_seen

    def test_identical_state_produces_same_marker(self) -> None:
        engine = _stub_engine(last_context_snapshot={})
        mgr = CalibrationManager(engine)
        m1 = mgr._task_progress_marker()
        m2 = mgr._task_progress_marker()
        assert m1 == m2


# ── _cost_ceiling_notice ─────────────────────────────────────────────


class TestCostCeilingNotice:
    def test_disabled_returns_empty(self) -> None:
        engine = _stub_engine(cost_ceiling_multiple=0.0)
        mgr = CalibrationManager(engine)
        assert mgr._cost_ceiling_notice() == ""

    def test_not_exceeded_returns_empty(self) -> None:
        engine = _stub_engine(
            cost_ceiling_multiple=5.0,
            context_length=128_000,
        )
        # 50_000 < 128_000 * 5.0 = 640_000
        mgr = CalibrationManager(engine)
        assert mgr._cost_ceiling_notice() == ""

    def test_exceeded_returns_notice(self) -> None:
        engine = _stub_engine(
            cost_ceiling_multiple=0.3,
            context_length=128_000,
        )
        # 50_000 >= 128_000 * 0.3 = 38_400
        mgr = CalibrationManager(engine)
        notice = mgr._cost_ceiling_notice()
        assert "Cumulative cost budget reached" in notice
        assert "final answer" in notice


# ── _maybe_periodic_recalibration ────────────────────────────────────


class TestMaybePeriodicRecalibration:
    def test_does_nothing_when_disabled(self) -> None:
        engine = _stub_engine(calibration_enabled=False)
        mgr = CalibrationManager(engine)
        mgr._maybe_periodic_recalibration()
        assert engine._turns_since_calibration == 0

    def test_does_nothing_when_interval_zero(self) -> None:
        engine = _stub_engine(
            calibration_enabled=True,
            calibration_interval_turns=0,
        )
        mgr = CalibrationManager(engine)
        mgr._maybe_periodic_recalibration()
        assert engine._turns_since_calibration == 0

    def test_increments_counter_before_interval(self) -> None:
        engine = _stub_engine(
            calibration_enabled=True,
            calibration_interval_turns=5,
            calibration_store=SimpleNamespace(),
        )
        mgr = CalibrationManager(engine)
        mgr._maybe_periodic_recalibration()
        assert engine._turns_since_calibration == 1


# ── _cache_aware_plan_kwargs ─────────────────────────────────────────


class TestCacheAwarePlanKwargs:
    def test_empty_when_no_prior_snapshot(self) -> None:
        engine = _stub_engine()
        engine._last_context_snapshot = {}
        mgr = CalibrationManager(engine)
        result = mgr._cache_aware_plan_kwargs()
        assert result == {}

    def test_empty_when_no_message_tokens(self) -> None:
        engine = _stub_engine()
        engine._last_context_snapshot = {"message_tokens": 0, "tool_schema_tokens": 100}
        mgr = CalibrationManager(engine)
        result = mgr._cache_aware_plan_kwargs()
        assert result == {}
