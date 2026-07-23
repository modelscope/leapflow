"""Unit tests for the adaptive-depth core (mechanisms 1+2+3, S0 / W1).

Covers:
- Elastic IterationBudget: fixed-mode backward compatibility, difficulty->cap
  scaling (monotonic, bounded), monotonic retarget, bounded refund.
- Continuous difficulty signal: zero for trivial turns, monotonic in evidence,
  bounded to [0, 1].
- Bidirectional posture: high-end EXPANDING arm and low-end answer-ready arm.

All tests are hermetic (no network, no LLM, no disk).
"""
from __future__ import annotations

from leapflow.engine.budget import BudgetConfig, BudgetStatus, IterationBudget
from leapflow.engine.context_control import (
    ContextGovernanceController,
    DifficultyConfig,
    ToolEvidenceBuilder,
)


def _file_read(controller: ContextGovernanceController, path: str) -> None:
    controller.compact_tool_result(
        "file_read",
        {"path": path},
        {"ok": True, "path": path, "content": "print(1)", "mode": "raw"},
    )


# ── Elastic budget: fixed-mode backward compatibility ────────────────


def test_fixed_budget_is_byte_identical_to_legacy() -> None:
    budget = IterationBudget.for_react(
        BudgetConfig(max_iterations=20, soft_limit=14, warning_threshold=10)
    )
    assert budget._config.elastic is False
    statuses = [budget.consume() for _ in range(20)]
    assert statuses[0] == BudgetStatus.OK          # used=1
    assert statuses[8] == BudgetStatus.OK           # used=9
    assert statuses[9] == BudgetStatus.WARNING      # used=10
    assert statuses[13] == BudgetStatus.SOFT_LIMIT  # used=14
    assert statuses[19] == BudgetStatus.EXHAUSTED   # used=20


def test_for_tool_execution_stays_fixed() -> None:
    budget = IterationBudget.for_tool_execution(max_calls=30, soft=24)
    assert budget._config.elastic is False
    assert budget.effective_max == 30
    # difficulty must not widen a fixed budget
    budget.retarget(budget.elastic_max(1.0))
    assert budget.effective_max == 30


# ── Elastic budget: difficulty -> cap scaling (T3) ───────────────────


def test_elastic_max_is_monotonic_and_bounded() -> None:
    budget = IterationBudget.for_react(
        BudgetConfig(max_iterations=12, iter_ceiling=80, scale_k=1.0)
    )
    assert budget._config.elastic is True
    assert budget.elastic_max(0.0) == 12
    assert budget.elastic_max(1.0) == 80
    assert budget.elastic_max(-5.0) == 12       # clamped low
    assert budget.elastic_max(9.0) == 80        # clamped high

    prev = -1
    for i in range(0, 11):
        value = budget.elastic_max(i / 10.0)
        assert 12 <= value <= 80
        assert value >= prev                    # non-decreasing
        prev = value


# ── Progress-gated continuation: extension past the elastic ceiling (P0) ──


def test_grant_extension_pushes_past_ceiling_to_hard_cap() -> None:
    """P0: progress-gated extension widens the cap past the elastic ceiling up to
    the absolute hard cap, so a genuinely long task is bounded by progress and
    resources rather than a fixed iteration count."""
    cfg = BudgetConfig(max_iterations=12, iter_ceiling=80, hard_cap=200, scale_k=1.0)
    budget = IterationBudget.for_react(cfg)
    budget.retarget(budget.elastic_max(1.0))     # difficulty widens to the ceiling
    assert budget.effective_max == 80
    assert budget.can_extend is True             # 80 < 200 absolute ceiling

    budget.grant_extension(25)
    assert budget.effective_max == 105
    for _ in range(20):
        budget.grant_extension(25)               # saturate toward the hard cap
    assert budget.effective_max == 200           # capped at hard_cap
    assert budget.can_extend is False


def test_difficulty_retarget_bounded_by_ceiling_not_hard_cap() -> None:
    """Difficulty widening stays bounded by the elastic ceiling; only the
    progress-gated extension may go past it toward the hard cap."""
    cfg = BudgetConfig(max_iterations=12, iter_ceiling=80, hard_cap=200, scale_k=1.0)
    budget = IterationBudget.for_react(cfg)
    budget.retarget(999)
    assert budget.effective_max == 80
    assert budget.can_extend is True             # progress extension still available


def test_no_hard_cap_means_no_progress_extension() -> None:
    cfg = BudgetConfig(max_iterations=12, iter_ceiling=80)   # hard_cap defaults to 0
    budget = IterationBudget.for_react(cfg)
    budget.retarget(budget.elastic_max(1.0))
    assert budget.effective_max == 80
    assert budget.can_extend is False            # absolute_ceiling == elastic ceiling
    budget.grant_extension(25)
    assert budget.effective_max == 80            # no-op past the ceiling


def test_status_is_side_effect_free_and_extension_clears_exhaustion() -> None:
    budget = IterationBudget.for_react(
        BudgetConfig(max_iterations=5, iter_ceiling=40, hard_cap=100, scale_k=1.0)
    )
    assert budget.status() == BudgetStatus.OK
    assert budget.used == 0                       # status() consumes nothing
    for _ in range(5):
        budget.consume()                          # exhaust the floor cap
    assert budget.status() == BudgetStatus.EXHAUSTED
    budget.grant_extension(25)                    # progress-gated widening
    assert budget.status() != BudgetStatus.EXHAUSTED


def test_elastic_budget_capped_at_floor_without_retarget() -> None:
    budget = IterationBudget.for_react(
        BudgetConfig(max_iterations=12, iter_ceiling=80, scale_k=1.0)
    )
    statuses = [budget.consume() for _ in range(12)]
    assert statuses[-1] == BudgetStatus.EXHAUSTED   # used=12 hits the floor cap


def test_retarget_widens_horizon_for_hard_task() -> None:
    budget = IterationBudget.for_react(
        BudgetConfig(max_iterations=12, iter_ceiling=80, scale_k=1.0)
    )
    for _ in range(5):
        budget.consume()
    budget.retarget(budget.elastic_max(0.8))
    assert budget.effective_max == budget.elastic_max(0.8) > 12
    assert budget.exhausted is False
    # can now consume well past the original floor of 12
    for _ in range(20):
        budget.consume()
    assert budget.used == 25


# ── Elastic budget: retarget monotonic + physical floor (T4) ─────────


def test_retarget_is_monotonic_and_never_lowers() -> None:
    budget = IterationBudget.for_react(
        BudgetConfig(max_iterations=12, iter_ceiling=80, scale_k=1.0)
    )
    budget.retarget(60)
    assert budget.effective_max == 60
    budget.retarget(30)                 # lower target ignored
    assert budget.effective_max == 60
    budget.retarget(200)                # capped at ceiling
    assert budget.effective_max == 80


def test_retarget_never_below_consumed_or_floor() -> None:
    budget = IterationBudget.for_react(
        BudgetConfig(max_iterations=12, iter_ceiling=80, scale_k=1.0)
    )
    budget.retarget(80)
    for _ in range(20):
        budget.consume()
    assert budget.used == 20
    budget.retarget(1)                  # cannot drop below used/floor/current
    assert budget.effective_max == 80


# ── Elastic budget: bounded refund (T5) ──────────────────────────────


def test_refund_is_bounded_when_configured() -> None:
    budget = IterationBudget.for_react(
        BudgetConfig(max_iterations=12, iter_ceiling=80, max_refunds=2)
    )
    for _ in range(6):
        budget.consume()
    for _ in range(5):
        budget.refund("slow-tool")
    assert budget.used == 4             # 6 consumed - 2 (capped) refunds


def test_refund_is_unbounded_by_default() -> None:
    budget = IterationBudget.for_react(BudgetConfig(max_iterations=20))
    for _ in range(6):
        budget.consume()
    for _ in range(5):
        budget.refund()
    assert budget.used == 1             # 6 - 5


# ── Difficulty signal (T1/T2 core) ───────────────────────────────────


def test_difficulty_is_zero_for_trivial_turn() -> None:
    controller = ContextGovernanceController(
        evidence_builder=ToolEvidenceBuilder(max_content_chars=240),
    )
    snap = controller.snapshot()
    assert snap.difficulty == 0.0
    assert snap.posture == "baseline"


def test_difficulty_is_monotonic_in_evidence_and_bounded() -> None:
    controller = ContextGovernanceController(
        evidence_builder=ToolEvidenceBuilder(max_content_chars=240),
        difficulty_config=DifficultyConfig(ema_alpha=1.0),  # no smoothing for determinism
    )
    prev = -1.0
    for r in range(1, 9):
        _file_read(controller, f"/tmp/src_{r}_a.py")
        _file_read(controller, f"/tmp/src_{r}_b.py")
        difficulty = controller.snapshot(context_ratio=0.0, round_number=r).difficulty
        assert 0.0 <= difficulty <= 1.0
        assert difficulty >= prev            # non-decreasing as work accumulates
        prev = difficulty
    assert prev > 0.5                        # a broad, growing investigation is "hard"


# ── Bidirectional posture: high-end EXPANDING arm ────────────────────


def test_posture_expands_for_hard_healthy_task() -> None:
    controller = ContextGovernanceController(
        evidence_builder=ToolEvidenceBuilder(max_content_chars=240),
        difficulty_config=DifficultyConfig(ema_alpha=1.0),
    )
    posture = "baseline"
    for r in range(1, 8):
        _file_read(controller, f"/tmp/deep_{r}_a.py")
        _file_read(controller, f"/tmp/deep_{r}_b.py")
        snap = controller.snapshot(context_ratio=0.10, round_number=r)
        posture = snap.posture
    assert snap.difficulty >= 0.55
    assert posture == "expanding"


def test_posture_does_not_expand_when_context_unhealthy() -> None:
    controller = ContextGovernanceController(
        evidence_builder=ToolEvidenceBuilder(max_content_chars=240),
        difficulty_config=DifficultyConfig(ema_alpha=1.0),
    )
    for r in range(1, 8):
        _file_read(controller, f"/tmp/hot_{r}_a.py")
        _file_read(controller, f"/tmp/hot_{r}_b.py")
        snap = controller.snapshot(context_ratio=0.95, round_number=r)
    # high context pressure -> safety-first finalizing, never expanding
    assert snap.posture == "finalizing"


# ── Bidirectional posture: low-end answer-ready arm ──────────────────


def test_answer_ready_converges_simple_task_early() -> None:
    controller = ContextGovernanceController(
        evidence_builder=ToolEvidenceBuilder(max_content_chars=240),
    )
    # round 1: one retrieval (the deterministic KB-QA shape)
    _file_read(controller, "/tmp/answer.py")
    controller.snapshot(context_ratio=0.0, round_number=1)
    # round 2: no new evidence -> marginal drops -> answer-ready
    snap = controller.snapshot(context_ratio=0.0, round_number=2)
    assert snap.posture in {"baseline", "expanded"}   # disclosure stays minimal
    assert snap.should_converge is True
    assert snap.convergence_reason == "answer-ready"
    notice = controller.convergence_notice(2)
    assert "enough evidence" in notice


def test_answer_ready_not_triggered_before_min_round() -> None:
    controller = ContextGovernanceController(
        evidence_builder=ToolEvidenceBuilder(max_content_chars=240),
    )
    _file_read(controller, "/tmp/answer.py")
    snap = controller.snapshot(context_ratio=0.0, round_number=1)
    assert snap.should_converge is False              # round 1 < answer_ready_min_round(2)


def test_stale_round_peek_does_not_corrupt_signal() -> None:
    """A round-0 peek (the tool_metadata UI path) interleaved with authoritative
    round-N snapshots must not mutate the difficulty/marginal state.
    """
    controller = ContextGovernanceController(
        evidence_builder=ToolEvidenceBuilder(max_content_chars=240),
        difficulty_config=DifficultyConfig(ema_alpha=1.0),
    )
    for r in range(1, 6):
        _file_read(controller, f"/tmp/peek_{r}.py")
        controller.snapshot(context_ratio=0.1, round_number=r)
    authoritative = controller.snapshot(context_ratio=0.1, round_number=5).difficulty
    assert authoritative > 0.0
    for _ in range(5):                                # simulate stale round-0 peeks
        controller.snapshot()
    after_peeks = controller.snapshot(context_ratio=0.1, round_number=5).difficulty
    assert after_peeks == authoritative


# ── W2 slice 1: effective-cost accounting (mechanism 7 foundation) ──


def test_turn_usage_cache_hit_rate_and_effective_cost() -> None:
    from leapflow.engine.turn_usage import TurnUsageTracker

    tracker = TurnUsageTracker()
    tracker.record_api_call(
        {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100, "cached_tokens": 800},
        provider="primary",
        model="m",
    )
    s = tracker.summary()
    assert s.cached_tokens == 800
    assert s.cache_hit_rate == 0.8
    # effective = miss(200) + cached(800) * 0.1 = 280
    assert s.effective_prompt_tokens(0.1) == 280.0
    assert s.effective_prompt_tokens(cached_price_ratio=1.0) == 1000.0  # no discount
    assert tracker.to_learning_signal()["cache_hit_rate"] == 0.8


def test_turn_usage_without_cache_field_degrades_gracefully() -> None:
    from leapflow.engine.turn_usage import TurnUsageTracker

    tracker = TurnUsageTracker()
    tracker.record_api_call({"prompt_tokens": 500, "completion_tokens": 50, "total_tokens": 550})
    s = tracker.summary()
    assert s.cached_tokens == 0
    assert s.cache_hit_rate == 0.0
    assert s.effective_prompt_tokens() == 500.0


def test_extract_cached_tokens_openai_and_deepseek_shapes() -> None:
    from leapflow.llm.openai_provider import _extract_cached_tokens

    class _Details:
        cached_tokens = 640

    class _OpenAIUsage:
        prompt_tokens_details = _Details()

    class _DeepSeekUsage:
        prompt_cache_hit_tokens = 512

    class _NoCache:
        pass

    assert _extract_cached_tokens(_OpenAIUsage()) == 640
    assert _extract_cached_tokens(_DeepSeekUsage()) == 512
    assert _extract_cached_tokens(_NoCache()) == 0


# ── W2 slice 2: adaptive prefix-commitment decision (CL-1..CL-5) ──


def _commit_controller():
    from leapflow.engine.prefix_commitment import PrefixCommitmentController

    return PrefixCommitmentController()


def test_commit_when_hard_long_horizon() -> None:
    ctl = _commit_controller()
    assert ctl.should_commit(
        difficulty=0.70,
        posture="expanding",
        remaining_rounds=30,
        est_full_prefix_tokens=5000,
        est_pcd_prefix_tokens=2000,
    ) is True
    assert ctl.projected_savings(
        remaining_rounds=30, est_full_prefix_tokens=5000, est_pcd_prefix_tokens=2000
    ) > 0.0


def test_no_commit_low_difficulty() -> None:
    ctl = _commit_controller()
    assert ctl.should_commit(
        difficulty=0.40, posture="expanding", remaining_rounds=30,
        est_full_prefix_tokens=5000, est_pcd_prefix_tokens=2000,
    ) is False


def test_no_commit_on_convergence_posture() -> None:
    ctl = _commit_controller()
    # near-end postures must never newly commit a large prefix (7.2.2)
    for posture in ("converging", "finalizing", "baseline", "expanded"):
        assert ctl.should_commit(
            difficulty=0.90, posture=posture, remaining_rounds=40,
            est_full_prefix_tokens=8000, est_pcd_prefix_tokens=4000,
        ) is False


def test_no_commit_small_prefix() -> None:
    ctl = _commit_controller()
    assert ctl.should_commit(
        difficulty=0.80, posture="research", remaining_rounds=30,
        est_full_prefix_tokens=500, est_pcd_prefix_tokens=400,   # < min_prefix_tokens(1024)
    ) is False


def test_no_commit_near_end() -> None:
    ctl = _commit_controller()
    assert ctl.should_commit(
        difficulty=0.80, posture="expanding", remaining_rounds=2,   # < min_remaining_rounds(3)
        est_full_prefix_tokens=5000, est_pcd_prefix_tokens=2000,
    ) is False


def test_no_commit_when_amortization_negative() -> None:
    ctl = _commit_controller()
    # big full prefix, tiny churning prefix, short horizon -> committing loses
    assert ctl.should_commit(
        difficulty=0.90, posture="expanding", remaining_rounds=3,
        est_full_prefix_tokens=5000, est_pcd_prefix_tokens=100,
    ) is False


def test_commitment_is_monotonic() -> None:
    ctl = _commit_controller()
    state = ctl.evaluate(
        difficulty=0.70, posture="expanding", round_number=6, remaining_rounds=30,
        est_full_prefix_tokens=5000, est_pcd_prefix_tokens=2000,
    )
    assert state.committed is True
    assert state.committed_at_round == 6
    # later low-difficulty rounds must not un-commit
    later = ctl.evaluate(
        difficulty=0.10, posture="baseline", round_number=9, remaining_rounds=1,
        est_full_prefix_tokens=5000, est_pcd_prefix_tokens=2000,
    )
    assert later.committed is True
    assert later.committed_at_round == 6


def test_amortization_matches_manual_calc() -> None:
    from leapflow.engine.prefix_commitment import CachePriceModel, PrefixCommitmentController

    ctl = PrefixCommitmentController(price_model=CachePriceModel(price_miss=1.0, price_read=0.1, price_write=1.0))
    # cost_commit = 5000*1 + 29*5000*0.1 = 19500 ; *1.15 margin = 22425
    # cost_nocommit = 30*2000*1 = 60000 ; savings = 60000 - 22425 = 37575
    savings = ctl.projected_savings(
        remaining_rounds=30, est_full_prefix_tokens=5000, est_pcd_prefix_tokens=2000
    )
    assert round(savings, 1) == 37575.0


# ── W2 slice 3: cache-aware, SNR-preserving compression (CL-8) ──


def _turns(prefix: str, n: int):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"{prefix}{i}"}
        for i in range(n)
    ]


def test_summarize_append_only_freezes_prior_segments() -> None:
    """Prior summary segments are frozen (byte-stable) and never re-summarized:
    long-task findings are captured once at full fidelity (no summary-of-summary
    drift) and stay cacheable.
    """
    from leapflow.engine.context_compressor import SummarizeStage

    stage = SummarizeStage(threshold_messages=4, keep_recent=2, summarize_fn=None, append_only=True)
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1 find A"},
        {"role": "assistant", "content": "a1 found A"},
        {"role": "user", "content": "u2 find B"},
        {"role": "assistant", "content": "a2 found B"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "u4"},
    ]
    out1 = stage.apply(msgs, budget=100)
    seg1 = [m for m in out1 if m.get("_compressed_summary")]
    assert len(seg1) == 1
    frozen_content = seg1[0]["content"]
    assert "found A" in frozen_content              # early long-task finding retained

    out2 = stage.apply(out1 + _turns_named("C"), budget=100)
    seg2 = [m for m in out2 if m.get("_compressed_summary")]
    assert len(seg2) == 2                           # append-only: prior frozen + new appended
    assert seg2[0]["content"] == frozen_content     # first segment byte-stable (no drift, SNR + cache)


def test_summarize_legacy_mode_merges_segments() -> None:
    from leapflow.engine.context_compressor import SummarizeStage

    stage = SummarizeStage(threshold_messages=4, keep_recent=2, summarize_fn=None, append_only=False)
    msgs = [{"role": "system", "content": "sys"}] + _turns("m", 7)
    out1 = stage.apply(msgs, budget=100)
    out2 = stage.apply(out1 + _turns("n", 4), budget=100)
    seg2 = [m for m in out2 if m.get("_compressed_summary")]
    assert len(seg2) == 1                           # legacy: single evolving (re-summarized) segment


def _turns_named(letter: str):
    return [
        {"role": "user", "content": f"u5 find {letter}"},
        {"role": "assistant", "content": f"a5 found {letter}"},
        {"role": "user", "content": "u6"},
        {"role": "assistant", "content": "a6"},
    ]


# ── W2 slice 3 (A): global effective-cost ceiling (soft safety) ──


def test_cost_ceiling_exceeded_predicate() -> None:
    from leapflow.engine.turn_usage import cost_ceiling_exceeded

    # disabled cases
    assert cost_ceiling_exceeded(effective_prompt_tokens=1e9, context_length=1000, context_multiple=0.0) is False
    assert cost_ceiling_exceeded(effective_prompt_tokens=1e9, context_length=0, context_multiple=20.0) is False
    # enabled: ceiling = 1000 * 20 = 20000
    assert cost_ceiling_exceeded(effective_prompt_tokens=19_999, context_length=1000, context_multiple=20.0) is False
    assert cost_ceiling_exceeded(effective_prompt_tokens=20_000, context_length=1000, context_multiple=20.0) is True
    assert cost_ceiling_exceeded(effective_prompt_tokens=50_000, context_length=1000, context_multiple=20.0) is True


# ── S3-L1: adaptive-depth learning signal (observe-only calibration capture) ──


def test_build_adaptive_learning_signal() -> None:
    from leapflow.engine.turn_usage import build_adaptive_learning_signal

    snap = {
        "difficulty": 0.734159,
        "context_posture": "research",
        "prefix_committed": True,
        "open_questions": 2,
        "cumulative_effective_tokens": 12345.6,
    }
    sig = build_adaptive_learning_signal(snap)
    assert sig["final_difficulty"] == 0.7342          # rounded to 4dp
    assert sig["final_posture"] == "research"
    assert sig["prefix_committed"] is True
    assert sig["open_questions"] == 2
    assert sig["cumulative_effective_tokens"] == 12345.6


def test_build_adaptive_learning_signal_defaults_are_safe() -> None:
    from leapflow.engine.turn_usage import build_adaptive_learning_signal

    sig = build_adaptive_learning_signal({})
    assert sig["final_difficulty"] == 0.0
    assert sig["final_posture"] == ""
    assert sig["prefix_committed"] is False
    assert "open_questions" not in sig               # omitted when absent (no ledger signal)
    assert "cumulative_effective_tokens" not in sig  # omitted when zero/absent


# ── S3-L2: offline difficulty calibration (report-only) ──


def test_analyze_difficulty_calibration_well_calibrated() -> None:
    from leapflow.learning.difficulty_calibration import analyze_difficulty_calibration

    # Effort rises with predicted difficulty; all succeed → no adjustment.
    records = [
        {"context": {"final_difficulty": 0.1, "steps": 1}, "reward": 1.0},
        {"context": {"final_difficulty": 0.2, "steps": 2}, "reward": 1.0},
        {"context": {"final_difficulty": 0.5, "steps": 4}, "reward": 1.0},
        {"context": {"final_difficulty": 0.8, "steps": 8}, "reward": 1.0},
        {"context": {"final_difficulty": 0.9, "steps": 9}, "reward": 1.0},
    ]
    report = analyze_difficulty_calibration(records)
    assert report.sample_size == 5
    assert report.effort_monotonic is True
    assert report.suggested_weight_scale == 1.0
    assert report.confidence > 0.0


def test_analyze_difficulty_calibration_over_predicted() -> None:
    from leapflow.learning.difficulty_calibration import analyze_difficulty_calibration

    # High-difficulty turns cost no more than low ones → reduce sensitivity.
    records = [
        {"context": {"final_difficulty": 0.1, "steps": 3}, "reward": 1.0},
        {"context": {"final_difficulty": 0.15, "steps": 3}, "reward": 1.0},
        {"context": {"final_difficulty": 0.8, "steps": 2}, "reward": 1.0},
        {"context": {"final_difficulty": 0.9, "steps": 2}, "reward": 1.0},
    ]
    report = analyze_difficulty_calibration(records)
    assert report.suggested_weight_scale < 1.0
    assert 0.5 <= report.suggested_weight_scale <= 1.5
    assert "over-predicted" in report.rationale


def test_analyze_difficulty_calibration_insufficient_data() -> None:
    from leapflow.learning.difficulty_calibration import analyze_difficulty_calibration

    report = analyze_difficulty_calibration([{"context": {"final_difficulty": 0.5, "steps": 3}}])
    assert report.sample_size == 1
    assert report.suggested_weight_scale == 1.0     # no suggestion without evidence
    assert report.rationale == "insufficient data"


def test_build_calibration_report_from_store(tmp_path) -> None:
    from leapflow.learning.difficulty_calibration import build_calibration_report_from_store
    from leapflow.storage.evolution_store import DuckDBEvolutionStore

    store = DuckDBEvolutionStore(str(tmp_path / "evo.duckdb"))
    try:
        for i, (difficulty, steps) in enumerate([(0.1, 1), (0.3, 2), (0.5, 4), (0.8, 8)]):
            store.save_episode(
                episode_id=f"ep{i}", skill_name="turn_x", actions=[],
                outcome="completed", reward=1.0,
                context={"final_difficulty": difficulty, "steps": steps},
            )
        report = build_calibration_report_from_store(store, limit=100)
        assert report.sample_size == 4
        assert report.effort_monotonic is True
        summary = report.summary()
        assert summary["sample_size"] == 4
        assert len(summary["buckets"]) == 3
    finally:
        store.close()


# ── S3-L3: online difficulty calibration (bounded, gated, reversible) ──


def test_apply_calibration_applies_bounded_scale() -> None:
    from leapflow.learning.difficulty_calibration import DifficultyCalibrationReport, apply_calibration

    report = DifficultyCalibrationReport(
        sample_size=40, suggested_weight_scale=0.8, confidence=0.8, rationale="over-predicted",
    )
    result = apply_calibration(1.0, report, enabled=True)
    assert result.applied is True
    assert result.effective_k == 0.8          # baseline 1.0 * scale 0.8
    assert result.baseline_k == 1.0


def test_apply_calibration_disabled_is_noop() -> None:
    from leapflow.learning.difficulty_calibration import DifficultyCalibrationReport, apply_calibration

    report = DifficultyCalibrationReport(sample_size=40, suggested_weight_scale=0.8, confidence=0.8)
    result = apply_calibration(1.0, report, enabled=False)
    assert result.applied is False
    assert result.effective_k == 1.0
    assert "disabled" in result.reason


def test_apply_calibration_guards_thin_or_calibrated_data() -> None:
    from leapflow.learning.difficulty_calibration import DifficultyCalibrationReport, apply_calibration

    thin = apply_calibration(
        1.0, DifficultyCalibrationReport(sample_size=3, suggested_weight_scale=0.8, confidence=0.9), enabled=True,
    )
    assert thin.applied is False and "samples" in thin.reason
    low_conf = apply_calibration(
        1.0, DifficultyCalibrationReport(sample_size=40, suggested_weight_scale=0.8, confidence=0.1), enabled=True,
    )
    assert low_conf.applied is False and "confidence" in low_conf.reason
    calibrated = apply_calibration(
        1.0, DifficultyCalibrationReport(sample_size=40, suggested_weight_scale=1.0, confidence=0.9), enabled=True,
    )
    assert calibrated.applied is False and "well calibrated" in calibrated.reason


def test_apply_calibration_clamps_to_bounds() -> None:
    from leapflow.learning.difficulty_calibration import DifficultyCalibrationReport, apply_calibration

    report = DifficultyCalibrationReport(
        sample_size=40, suggested_weight_scale=1.5, confidence=0.9, rationale="under-predicted",
    )
    result = apply_calibration(2.5, report, enabled=True)   # 2.5 * 1.5 = 3.75 -> clamp to k_max 3.0
    assert result.applied is True
    assert result.effective_k == 3.0


# ── S3-L4: finalize-threshold calibration (posture self-tuning) ──


def test_analyze_threshold_calibration_premature_finalize() -> None:
    from leapflow.learning.difficulty_calibration import analyze_threshold_calibration

    # 20 finalizing-posture turns that all failed -> premature -> raise threshold.
    records = [{"context": {"final_posture": "finalizing"}, "reward": -0.5} for _ in range(20)]
    report = analyze_threshold_calibration(records)
    assert report.sample_size == 20
    assert report.premature_finalize_rate == 1.0
    assert report.suggested_weight_scale > 1.0
    assert "raise finalize threshold" in report.rationale


def test_analyze_threshold_calibration_healthy_finalize() -> None:
    from leapflow.learning.difficulty_calibration import analyze_threshold_calibration

    # 5/20 = 0.25 premature -> within [0.1, 0.4] healthy band -> no change.
    records = (
        [{"context": {"final_posture": "finalizing"}, "reward": 1.0} for _ in range(15)]
        + [{"context": {"final_posture": "finalizing"}, "reward": -0.5} for _ in range(5)]
    )
    report = analyze_threshold_calibration(records)
    assert report.suggested_weight_scale == 1.0


def test_analyze_threshold_calibration_low_premature_allows_earlier() -> None:
    from leapflow.learning.difficulty_calibration import analyze_threshold_calibration

    # 1/20 = 0.05 premature -> below 0.1 -> allow slightly earlier finalize.
    records = (
        [{"context": {"final_posture": "finalizing"}, "reward": 1.0} for _ in range(19)]
        + [{"context": {"final_posture": "finalizing"}, "reward": -0.5}]
    )
    report = analyze_threshold_calibration(records)
    assert report.suggested_weight_scale < 1.0
    assert "earlier finalize" in report.rationale


def test_analyze_threshold_calibration_ignores_non_finalizing() -> None:
    from leapflow.learning.difficulty_calibration import analyze_threshold_calibration

    # Only research-posture turns -> no finalizing samples -> insufficient.
    records = [{"context": {"final_posture": "research"}, "reward": -0.5} for _ in range(30)]
    report = analyze_threshold_calibration(records)
    assert report.sample_size == 0
    assert report.suggested_weight_scale == 1.0
    assert report.rationale == "insufficient data"


# ── S0/RB5: cacheable-prefix stability invariant (CL-6/7/9 intent) ──


def test_task_contract_render_is_deterministic_prefix_material() -> None:
    """RB5 (CL-9): the task contract sits in the cacheable prefix, so its render
    must be a pure function of stable fields (no volatile tokens) — keeping the
    prefix byte-stable across rounds/turns for prompt-cache reuse.
    """
    from leapflow.engine.engine import TaskContract

    contract = TaskContract(
        task_id="t1", original_request="do X",
        workspace_root="/ws", allowed_roots=("/ws", "/tmp"),
        research_protocol=("cite sources",),
    )
    rendered = contract.render()
    assert rendered == contract.render()               # deterministic (no time/round tokens)
    assert "## Task Contract" in rendered
    assert "do X" in rendered and "/ws" in rendered


# ── W3 slice 1: research ledger (durable long-task state, SNR) ──


def test_research_ledger_note_render_and_bounds() -> None:
    from leapflow.engine.research_ledger import ResearchLedger

    led = ResearchLedger(max_items=3)
    assert led.is_empty is True
    assert led.render() == ""
    assert led.note("finding", "A uses DuckDB") is True
    assert led.note("open_question", "does B cache?") is True
    assert led.note("decision", "excluded path X") is True
    assert led.note("next_step", "inspect B") is True
    assert led.note("bogus", "x") is False            # invalid kind
    assert led.note("finding", "   ") is False         # empty text
    rendered = led.render()
    for expected in ("A uses DuckDB", "does B cache?", "excluded path X", "inspect B"):
        assert expected in rendered
    assert led.open_question_count == 1
    for i in range(5):                                   # per-list cap keeps most recent
        led.note("finding", f"f{i}")
    findings = led.as_dict()["findings"]
    assert len(findings) == 3
    assert "f4" in findings and "f2" in findings and "f1" not in findings


def test_research_ledger_resolved_closes_open_question() -> None:
    from leapflow.engine.research_ledger import ResearchLedger

    led = ResearchLedger()
    led.note("open_question", "does B cache prompts?")
    assert led.open_question_count == 1
    led.note("resolved", "B cache prompts")              # substring match closes it
    assert led.open_question_count == 0
    assert "B cache prompts" in led.as_dict()["findings"]


def test_research_ledger_dedupe_moves_to_recent() -> None:
    from leapflow.engine.research_ledger import ResearchLedger

    led = ResearchLedger()
    led.note("finding", "X")
    led.note("finding", "Y")
    led.note("finding", "X")                             # dedupe -> X becomes most recent
    assert led.as_dict()["findings"] == ["Y", "X"]


def test_research_note_tool_handler_roundtrip() -> None:
    import asyncio

    from leapflow.engine.research_ledger import ResearchLedger
    from leapflow.tools import registry_bootstrap as rb

    led = ResearchLedger()
    rb.set_research_ledger(led)
    try:
        ok = asyncio.run(rb.TOOL_HANDLERS["research_note"]({"kind": "finding", "text": "cache reuses KV"}))
        assert ok["ok"] is True
        assert "cache reuses KV" in led.as_dict()["findings"]
        bad = asyncio.run(rb.TOOL_HANDLERS["research_note"]({"kind": "nope", "text": "x"}))
        assert bad["ok"] is False
    finally:
        rb.set_research_ledger(None)                      # avoid leaking global into other tests
    unset = asyncio.run(rb.TOOL_HANDLERS["research_note"]({"kind": "finding", "text": "y"}))
    assert unset["ok"] is False


def test_research_note_is_disclosed_tool() -> None:
    from leapflow.tools.registry_bootstrap import TOOL_DEFINITIONS

    names = {td.get("function", {}).get("name") for td in TOOL_DEFINITIONS}
    assert "research_note" in names


# ── W3: mechanism 6 sufficiency gated by ledger open questions ──


def test_answer_ready_suppressed_by_open_questions() -> None:
    controller = ContextGovernanceController(
        evidence_builder=ToolEvidenceBuilder(max_content_chars=240),
    )
    _file_read(controller, "/tmp/answer.py")
    controller.snapshot(context_ratio=0.0, round_number=1)
    # resolved (0 open questions) -> answer-ready fires
    resolved = controller.snapshot(context_ratio=0.0, round_number=2, open_questions=0)
    assert resolved.should_converge is True
    assert resolved.convergence_reason == "answer-ready"
    # open work remains -> early convergence suppressed (long task not cut short)
    open_work = controller.snapshot(context_ratio=0.0, round_number=2, open_questions=2)
    assert open_work.should_converge is False
    # no ledger signal -> falls back to marginal heuristic -> fires
    no_ledger = controller.snapshot(context_ratio=0.0, round_number=2, open_questions=None)
    assert no_ledger.should_converge is True


def test_convergence_notice_respects_open_questions() -> None:
    controller = ContextGovernanceController(
        evidence_builder=ToolEvidenceBuilder(max_content_chars=240),
    )
    _file_read(controller, "/tmp/a.py")
    controller.snapshot(context_ratio=0.0, round_number=1)
    assert controller.convergence_notice(2, open_questions=0) != ""       # done -> nudge
    assert controller.convergence_notice(2, open_questions=3) == ""        # open work -> silent


# ── S1: research ledger DuckDB persistence (durable Orient) ──


def test_research_ledger_store_roundtrip(tmp_path) -> None:
    from leapflow.engine.research_ledger import ResearchLedger
    from leapflow.storage.research_ledger_store import ResearchLedgerStore

    db = tmp_path / "leap.duckdb"
    store = ResearchLedgerStore(db)
    assert store.load("s1") is None                       # nothing persisted yet

    led = ResearchLedger()
    led.note("finding", "A uses DuckDB")
    led.note("open_question", "does B cache?")
    led.note("next_step", "inspect B")
    store.save("s1", led.to_state())
    store.close()                                          # simulate restart

    store2 = ResearchLedgerStore(db)
    try:
        state = store2.load("s1")
        assert state is not None
        restored = ResearchLedger()
        restored.load_state(state)
        assert "A uses DuckDB" in restored.as_dict()["findings"]
        assert restored.open_question_count == 1
        assert "inspect B" in restored.render()
        assert store2.load("other") is None               # session isolation
        store2.clear("s1")
        assert store2.load("s1") is None
    finally:
        store2.close()


def test_research_ledger_to_state_excludes_derived_count() -> None:
    from leapflow.engine.research_ledger import ResearchLedger

    led = ResearchLedger()
    led.note("open_question", "q1")
    state = led.to_state()
    assert "open_question_count" not in state
    assert state["open_questions"] == ["q1"]


def test_research_ledger_change_listener_fires_only_on_note() -> None:
    from leapflow.engine.research_ledger import ResearchLedger

    calls: list[int] = []
    led = ResearchLedger()
    led.set_change_listener(lambda: calls.append(1))
    led.note("finding", "x")               # valid -> fires
    led.note("bogus", "y")                 # invalid kind -> no fire
    led.note("finding", "   ")             # empty -> no fire
    led.load_state({"findings": ["z"]})    # load -> no fire
    led.reset()                            # reset -> no fire
    assert len(calls) == 1


# ── W4 slice 1: config-driven subagent governance ──


def test_subagent_governance_config_catalog() -> None:
    from leapflow.config import get_settings
    from leapflow.config_service import ConfigService

    settings = get_settings()
    assert settings.agent_subagent_max_depth == 2
    assert settings.agent_subagent_max_concurrent == 3
    assert settings.agent_subagent_max_iterations == 15
    keys = ConfigService(settings).writable_keys()
    for key in (
        "agent.subagent_max_depth",
        "agent.subagent_max_concurrent",
        "agent.subagent_max_iterations",
    ):
        assert key in keys


def test_subagent_manager_respects_max_depth() -> None:
    import asyncio

    from leapflow.engine.subagent import SubagentConfig, SubagentManager, SubagentResult

    class _Exec:
        async def execute_subagent(self, config: SubagentConfig) -> SubagentResult:
            return SubagentResult(session_id="x", goal=config.goal, summary="ok", status="completed")

    mgr = SubagentManager(executor=_Exec(), max_depth=1, max_concurrent=2)
    ran = asyncio.run(mgr.delegate(SubagentConfig(goal="g", depth=0)))
    assert ran.status == "completed"
    blocked = asyncio.run(mgr.delegate(SubagentConfig(goal="g", depth=1)))
    assert blocked.status == "failed" and blocked.error == "max_depth_exceeded"


def test_subagent_executor_uses_settings_iteration_budget() -> None:
    import asyncio

    from leapflow.engine.subagent import DefaultSubagentExecutor, SubagentConfig

    class _Resp:
        content = "working"
        model = "m"
        usage: dict = {}
        tool_calls = [type("TC", (), {"id": "1", "name": "noop", "arguments": {}})()]

    class _LLM:
        def __init__(self) -> None:
            self.calls = 0

        async def achat(self, messages, **kw):
            self.calls += 1
            return _Resp()

    class _Settings:
        agent_subagent_max_iterations = 3
        max_tool_result_chars = 1000

    llm = _LLM()
    ex = DefaultSubagentExecutor(llm=llm, tool_handlers={}, tool_definitions=[], settings=_Settings())
    # config asks for 99 iterations; settings floor (3) governs. Elastic budget:
    # a blocked tool yields no evidence -> difficulty 0 -> no widening; floor=3
    # means 2 productive rounds (the 3rd consume hits EXHAUSTED).
    res = asyncio.run(ex.execute_subagent(SubagentConfig(goal="g", max_iterations=99)))
    assert llm.calls == 2
    assert res.status == "completed"


def test_subagent_budget_widens_with_difficulty() -> None:
    import asyncio

    from leapflow.engine.subagent import DefaultSubagentExecutor, SubagentConfig

    class _LLM:
        def __init__(self) -> None:
            self.calls = 0

        async def achat(self, messages, **kw):
            self.calls += 1
            n = self.calls
            resp = type("R", (), {})()
            resp.content = "working"
            resp.model = "m"
            resp.usage = {}
            # each round reads a distinct file -> accumulating evidence -> rising difficulty
            resp.tool_calls = [
                type("TC", (), {"id": str(n), "name": "file_read", "arguments": {"path": f"/p/{n}.py"}})()
            ]
            return resp

    async def _file_read_handler(args):
        return {"ok": True, "path": args["path"], "content": "print(1)", "mode": "raw"}

    class _Settings:
        agent_subagent_max_iterations = 4   # floor 4 -> ceiling 8
        max_tool_result_chars = 500

    llm = _LLM()
    ex = DefaultSubagentExecutor(
        llm=llm,
        tool_handlers={"file_read": _file_read_handler},
        tool_definitions=[],
        settings=_Settings(),
    )
    asyncio.run(ex.execute_subagent(SubagentConfig(goal="investigate")))
    # floor 4 alone => 3 productive rounds; rising difficulty widens the cap => more
    assert llm.calls > 3


# ── W4-M1: AgentLoopFrame value object (per-frame state + depth governance) ──


def test_agent_loop_frame_depth_governance() -> None:
    from leapflow.engine.agent_loop import AgentLoopFrame

    root = AgentLoopFrame(user_text="q", depth=0)
    assert root.is_root is True
    assert root.child_depth == 1
    assert root.can_delegate(max_depth=2) is True     # child 1 < 2
    assert root.can_delegate(max_depth=1) is False    # child 1 < 1 == False

    child = AgentLoopFrame(user_text="q", depth=1)
    assert child.is_root is False
    assert child.child_depth == 2
    assert child.can_delegate(max_depth=2) is False   # child 2 < 2 == False
    assert child.can_delegate(max_depth=3) is True


def test_agent_loop_frame_tool_filter_and_defaults() -> None:
    from leapflow.engine.agent_loop import AgentLoopFrame

    unrestricted = AgentLoopFrame(user_text="q")
    assert unrestricted.allows_tool("anything") is True        # None -> all tools
    assert unrestricted.budget is None and unrestricted.ledger is None  # subsystems optional in M1
    assert unrestricted.metadata == {}

    restricted = AgentLoopFrame(user_text="q", tool_filter=frozenset({"file_read"}))
    assert restricted.allows_tool("file_read") is True
    assert restricted.allows_tool("shell_run") is False


def test_agent_loop_frame_carries_mutable_per_turn_state() -> None:
    from leapflow.engine.agent_loop import AgentLoopFrame

    frame = AgentLoopFrame(user_text="q", depth=1)
    # Per-turn state is reassigned during the loop (frame is mutable by design).
    frame.last_context_snapshot = {"difficulty": 0.5}
    assert frame.last_context_snapshot["difficulty"] == 0.5
    assert frame.recovery_coordinator is None            # optional until built by the engine
    assert frame.last_turn_tool_categories == frozenset()
    # Per-turn identity is carried on the frame (foundation for isolation).
    assert frame.session_id == "" and frame.turn_id == "" and frame.command_id == ""
    ided = AgentLoopFrame(user_text="q", session_id="s1", turn_id="t1", command_id="c1")
    assert (ided.session_id, ided.turn_id, ided.command_id) == ("s1", "t1", "c1")


# ── W4-A1: depth-gated subagent recursion (default-equivalent) ──


def test_delegate_task_depth_gating_is_default_equivalent() -> None:
    from leapflow.engine.subagent import (
        DELEGATE_BLOCKED_TOOLS,
        SubagentConfig,
        build_subagent_tool_filter,
    )

    tools = ["file_read", "delegate_task", "gp_delegate_task", "shell_run"]
    # default max_depth=2: a depth-1 subagent must NOT see delegate_task
    # (byte-equivalent to the previous hard-block => single-level delegation)
    depth1 = build_subagent_tool_filter(tools, SubagentConfig(goal="g", depth=1), max_depth=2)
    assert "delegate_task" not in depth1 and "gp_delegate_task" not in depth1
    assert "file_read" in depth1
    # max_depth=3 unlocks one more level: depth-1 keeps delegate_task, depth-2 drops it
    d1 = build_subagent_tool_filter(tools, SubagentConfig(goal="g", depth=1), max_depth=3)
    assert "delegate_task" in d1
    d2 = build_subagent_tool_filter(tools, SubagentConfig(goal="g", depth=2), max_depth=3)
    assert "delegate_task" not in d2
    # recursion is now gated by depth, not by an outright block
    assert "delegate_task" not in DELEGATE_BLOCKED_TOOLS


def test_subagent_depth_propagates_during_execution() -> None:
    import asyncio

    from leapflow.engine.subagent import (
        SubagentConfig,
        SubagentManager,
        SubagentResult,
        current_subagent_depth,
    )

    seen: dict[str, int] = {}

    class _Exec:
        async def execute_subagent(self, config: SubagentConfig) -> SubagentResult:
            seen["depth"] = current_subagent_depth()
            return SubagentResult(session_id="x", goal=config.goal, summary="ok", status="completed")

    mgr = SubagentManager(executor=_Exec(), max_depth=5, max_concurrent=2)
    assert current_subagent_depth() == 0                 # top-level default
    asyncio.run(mgr.delegate(SubagentConfig(goal="g", depth=1)))
    assert seen["depth"] == 1                              # propagated inside execution
    assert current_subagent_depth() == 0                  # reset after
