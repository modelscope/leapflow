# Copyright (c) Alibaba, Inc. and its affiliates.
"""Contracts for budget-estimator self-calibration.

The character heuristic (CJK 1:1, Latin 4:1) cannot match a real tokenizer, and
the residual error scales with the window: on a 1M budget a 15% underestimate is
~150K tokens, enough to either trip provider overflow at the hard gate or waste a
large slice of the window. Shipping a per-vendor tokenizer would add a dependency
that goes stale the same way a static capability table does, so the estimator
instead learns its correction from provider-reported prompt_tokens.

The subtle part is the feedback direction: the estimate handed back for
calibration already carries the current factor, so comparing it against the
actual makes the observed ratio approach 1.0 exactly when calibration starts
working — which drags the factor back to uncalibrated. Measured on a real run,
the factor drifted 0.649 -> 0.802 before this was fixed.
"""

from __future__ import annotations

from types import SimpleNamespace

from leapflow.engine.context.context_control import (
    _CALIBRATION_MAX_FACTOR,
    _CALIBRATION_MIN_FACTOR,
    ContextBudgetEstimator,
)

_LONG_CJK = [{"role": "user", "content": "推动经济增长的核心要素分析与产业结构变迁研究" * 40}]


def _converge(estimator: ContextBudgetEstimator, ratio: float, rounds: int = 8) -> int:
    """Feed a consistent provider/estimate ratio until the factor settles."""
    actual = int(estimator.estimate_messages(_LONG_CJK) * ratio)
    for _ in range(rounds):
        estimator.observe_actual(
            estimated=estimator.estimate_messages(_LONG_CJK), actual=actual,
        )
    return actual


def test_uncalibrated_estimator_is_unchanged() -> None:
    """No observations means no behaviour change from the previous heuristic."""
    estimator = ContextBudgetEstimator()

    assert estimator.calibration_samples == 0
    assert estimator.calibration_factor == 1.0
    assert estimator.estimate_messages(_LONG_CJK) > 0


def test_calibration_converges_when_the_heuristic_overestimates() -> None:
    """CJK 1:1 overestimates for BPE tokenizers that merge common words."""
    estimator = ContextBudgetEstimator()

    actual = _converge(estimator, 0.65)

    assert abs(estimator.calibration_factor - 0.65) < 0.02
    assert estimator.estimate_messages(_LONG_CJK) == actual


def test_calibration_converges_when_the_heuristic_underestimates() -> None:
    """The dangerous direction: underestimating overruns the hard gate."""
    estimator = ContextBudgetEstimator()

    actual = _converge(estimator, 1.4)

    assert abs(estimator.calibration_factor - 1.4) < 0.02
    assert estimator.estimate_messages(_LONG_CJK) == actual


def test_the_factor_does_not_drift_back_once_calibrated() -> None:
    """Regression: the corrected estimate must be un-corrected before comparing.

    Without dividing the factor back out, the observed ratio becomes 1.0 as soon
    as calibration takes effect and the factor climbs back toward 1.0.
    """
    estimator = ContextBudgetEstimator()
    actual = _converge(estimator, 0.65, rounds=2)
    settled = estimator.calibration_factor

    for _ in range(15):
        estimator.observe_actual(
            estimated=estimator.estimate_messages(_LONG_CJK), actual=actual,
        )

    assert abs(estimator.calibration_factor - settled) < 0.01
    assert estimator.calibration_factor < 0.7, "must not creep back toward 1.0"


def test_tiny_prompts_are_ignored() -> None:
    """Fixed per-request overhead dominates small prompts and would skew it."""
    estimator = ContextBudgetEstimator()

    estimator.observe_actual(estimated=50, actual=10_000)

    assert estimator.calibration_samples == 0
    assert estimator.calibration_factor == 1.0


def test_outlier_ratios_are_rejected() -> None:
    """Prompt caching or injected system content can distort a single sample."""
    estimator = ContextBudgetEstimator()

    estimator.observe_actual(estimated=1_000, actual=1_000_000)
    estimator.observe_actual(estimated=1_000, actual=1)

    assert estimator.calibration_samples == 0


def test_missing_actual_is_ignored() -> None:
    estimator = ContextBudgetEstimator()

    estimator.observe_actual(estimated=1_000, actual=0)
    estimator.observe_actual(estimated=1_000, actual=-5)

    assert estimator.calibration_samples == 0


def test_factor_stays_within_bounds() -> None:
    """A wildly wrong factor would be worse than the raw heuristic."""
    estimator = ContextBudgetEstimator()

    for ratio in (2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4):
        estimator.observe_actual(
            estimated=estimator.estimate_messages(_LONG_CJK),
            actual=int(estimator.estimate_messages(_LONG_CJK) * ratio),
        )

    assert _CALIBRATION_MIN_FACTOR <= estimator.calibration_factor <= _CALIBRATION_MAX_FACTOR


def test_snapshot_ratio_reflects_calibration() -> None:
    """Calibration must reach the gate decision, not just the raw estimate."""
    estimator = ContextBudgetEstimator()
    before = estimator.snapshot(_LONG_CJK, tools=None, context_length=10_000).ratio

    _converge(estimator, 0.65)
    after = estimator.snapshot(_LONG_CJK, tools=None, context_length=10_000).ratio

    assert after < before, "an overestimating heuristic must relax the gate ratio"


# ── Wiring: the engine must calibrate before overwriting the estimate ────
#
# These build the engine with object.__new__ and assign the attributes the method
# reads. That is deliberate for the ordering contracts below, but it cannot catch
# a wrong attribute *name* — an earlier version of these tests asserted against
# `_context_window_controller`, a name the engine never defines, so the feature
# raised AttributeError on every real turn while the suite stayed green. The real
# instance test at the end of this file is what closes that hole; keep it.


def test_engine_calibrates_from_provider_usage() -> None:
    """Covers the wiring: the pair must be captured before the overwrite.

    ``_record_provider_usage`` replaces the snapshot's token count with the
    provider value, so calibration has to read the estimate first — afterwards
    the signal is gone.
    """
    from leapflow.engine.engine import AgentEngine

    estimator = ContextBudgetEstimator()
    engine = object.__new__(AgentEngine)
    engine._context_controller = SimpleNamespace(estimator=estimator)
    engine._model_capabilities = None
    engine._last_context_snapshot = {"total_tokens": 10_000, "context_length": 1_000_000}
    engine._last_context_tokens = 0

    AgentEngine._record_provider_usage(engine, "qwen3.8-max", {"prompt_tokens": 6_500})

    assert estimator.calibration_samples == 1
    assert abs(estimator.calibration_factor - 0.65) < 0.01
    assert engine._last_context_tokens == 6_500


def test_engine_does_not_calibrate_against_a_provider_snapshot() -> None:
    """A snapshot already holding a provider count is not an estimate."""
    from leapflow.engine.engine import AgentEngine

    estimator = ContextBudgetEstimator()
    engine = object.__new__(AgentEngine)
    engine._context_controller = SimpleNamespace(estimator=estimator)
    engine._model_capabilities = None
    engine._last_context_snapshot = {
        "total_tokens": 6_500,
        "provider_prompt_tokens": 6_500,
        "context_length": 1_000_000,
    }
    engine._last_context_tokens = 0

    AgentEngine._record_provider_usage(engine, "qwen3.8-max", {"prompt_tokens": 6_500})

    assert estimator.calibration_samples == 0, "would calibrate against its own output"


def test_calibration_failure_never_breaks_the_turn() -> None:
    """Calibration is an optimisation; it must not be able to fail a request."""
    from leapflow.engine.engine import AgentEngine

    class _Exploding:
        def observe_actual(self, **kwargs):
            raise RuntimeError("boom")

    engine = object.__new__(AgentEngine)
    engine._context_controller = SimpleNamespace(estimator=_Exploding())
    engine._model_capabilities = None
    engine._last_context_snapshot = {"total_tokens": 10_000, "context_length": 1_000_000}
    engine._last_context_tokens = 0

    AgentEngine._record_provider_usage(engine, "m", {"prompt_tokens": 6_500})

    assert engine._last_context_tokens == 6_500, "usage recording must still complete"


# ── The hole the mocks left: a real engine, the production code path ─────


def _real_engine(tmp_path):
    """Build an actual AgentEngine, so attribute wiring is exercised for real."""
    from conftest import StubLLM, make_settings
    from leapflow.engine.engine import AgentEngine
    from leapflow.engine import build_default_registry
    from leapflow.engine.intent_classifier import Intent
    from leapflow.memory import (
        EpisodicMemoryProvider,
        SemanticMemoryProvider,
        WorkingMemoryProvider,
    )
    from leapflow.platform.mock import MockBridge

    class _Classifier:
        async def classify(self, user_text: str) -> Intent:
            return Intent(label="complex", reason="test")

    settings = make_settings(str(tmp_path))
    rpc = MockBridge()
    llm = StubLLM(["ok"])
    wm = WorkingMemoryProvider(max_tokens=1024)
    lt = SemanticMemoryProvider(source=settings.duckdb_path)
    imm = EpisodicMemoryProvider()
    registry = build_default_registry(rpc, llm, wm, lt)
    engine = AgentEngine(settings, rpc, llm, wm, lt, imm, registry, _Classifier())
    return engine, lt


def test_real_engine_calibrates_on_the_production_path(tmp_path) -> None:
    """The regression test for the outage: a real engine, a realistic snapshot.

    Every existing calibration test either mocked the controller attribute or
    left the snapshot empty, and an empty snapshot returns early before the
    controller is ever read. A turn with a real context estimate is the only
    shape that reaches the wiring, and it raised AttributeError on every round.
    """
    engine, store = _real_engine(tmp_path)
    try:
        engine._last_context_snapshot = {"total_tokens": 10_000, "context_length": 1_000_000}

        engine._record_provider_usage("qwen3.8-max", {"prompt_tokens": 6_500})

        assert engine._context_controller.estimator.calibration_samples == 1
        # The overwrite must also have happened; it used to be unreachable.
        assert engine._last_context_tokens == 6_500
        assert engine._last_context_snapshot["provider_prompt_tokens"] == 6_500
    finally:
        store.close()


def test_telemetry_helper_absorbs_its_own_defects(tmp_path) -> None:
    """A bookkeeping defect must not propagate into the provider-error path.

    The outage was a local AttributeError escaping into the LLM call's except
    block, where it was classified as a provider condition and driven through
    recovery. The helper now contains anything it raises.
    """
    engine, store = _real_engine(tmp_path)
    try:
        def _boom(*args, **kwargs):
            raise AttributeError("'AgentEngine' object has no attribute '_whatever'")

        engine._record_provider_usage = _boom
        recorded = []

        engine._record_llm_call_telemetry(
            SimpleNamespace(usage={"prompt_tokens": 6_500}, model="m"),
            recovery=SimpleNamespace(record_api_success=lambda: recorded.append(True)),
        )

        assert recorded == [True], "the success signal must survive a telemetry defect"
    finally:
        store.close()
