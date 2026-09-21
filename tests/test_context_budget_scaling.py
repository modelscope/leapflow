# Copyright (c) Alibaba, Inc. and its affiliates.
"""Contracts for context budget resolution and truncation-chain scaling.

Two problems these pin down:

- A stale registry entry must never shrink the window. Family-wide patterns like
  "qwen" carry whatever the vendor's line supported when the entry was written,
  so clamping to them silently runs a newer model at a fraction of its window —
  and every compression ratio is then computed against the wrong denominator.
  Model names always outrun a static table, so the table is advisory except for
  version-specific entries.
- The truncation chain is serial, so its tightest link decides what reaches the
  model. The tool-result budget used ``min(base, ...)``, which can only shrink:
  a 1M window still cut every tool result at 3000 chars.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from leapflow.engine.context.context_compressor import (
    _RESULT_CEILING_CHARS,
    _RESULT_FLOOR_CHARS,
    adaptive_tool_result_chars,
)
from leapflow.engine.engine import AgentEngine
from leapflow.llm.model_capabilities import ModelCapabilities, ModelCapabilityRegistry


def _active_length(model: str, budget: int, registry=None) -> int:
    """Run the real resolution logic without constructing a whole engine."""
    engine = object.__new__(AgentEngine)
    engine._settings = SimpleNamespace(llm_model=model, llm_context_length=budget)
    engine._model_capabilities = registry if registry is not None else ModelCapabilityRegistry()
    return AgentEngine._active_context_length(engine)


# ── Budget resolution ────────────────────────────────────────────────────


@pytest.mark.parametrize("model", ["qwen3.8-max", "qwen-next", "deepseek-v9", "claude-9-opus"])
def test_family_fallback_does_not_shrink_the_configured_budget(model: str) -> None:
    """A family pattern is not evidence about a specific model's window."""
    assert _active_length(model, 1_000_000) == 1_000_000


@pytest.mark.parametrize("model", ["gpt-5.5", "gpt-6-ultra", "brand-new-model-2030"])
def test_unrecognised_models_get_the_full_configured_budget(model: str) -> None:
    """New names are expected to miss the table; that must not cost them window.

    The registry is a convenience, not a gate: keeping it current can never be a
    correctness requirement because model names ship faster than this list.
    """
    assert _active_length(model, 1_000_000) == 1_000_000


@pytest.mark.parametrize(
    ("model", "expected"),
    [("gpt-3.5-turbo", 16_385), ("gpt-4o", 128_000), ("claude-4-opus", 200_000)],
)
def test_authoritative_capability_caps_an_oversized_budget(model: str, expected: int) -> None:
    """Version-specific entries still guard against configuring past the limit."""
    assert _active_length(model, 1_000_000) == expected


def test_a_smaller_configured_budget_is_always_respected() -> None:
    """Lowering the budget deliberately (e.g. to cut spend) must hold."""
    assert _active_length("gpt-4o", 32_000) == 32_000
    assert _active_length("qwen3.8-max", 200_000) == 200_000


def test_a_registered_override_is_authoritative() -> None:
    """An explicitly registered capability describes that exact model."""
    registry = ModelCapabilityRegistry()
    registry.register("custom-model", ModelCapabilities(context_length=64_000))

    assert _active_length("custom-model", 1_000_000, registry) == 64_000


def test_a_learned_length_becomes_authoritative() -> None:
    """A length observed from a real response outranks a family guess."""
    registry = ModelCapabilityRegistry()
    registry.update_from_usage("qwen3.8-max", {"prompt_tokens": 300_000, "completion_tokens": 1_000})

    caps = registry.resolve("qwen3.8-max")

    assert caps.authoritative is True
    assert caps.context_length > 300_000


def test_a_non_authoritative_entry_with_a_small_length_is_ignored() -> None:
    """The real guard: a stale family number must not cap the budget.

    The shipped family entries deliberately carry no context length, so a plain
    ``min()`` happens to be harmless against them today. This constructs the
    situation the flag exists for — a family entry that does carry a (stale)
    number — and pins that it cannot shrink the window.
    """
    registry = ModelCapabilityRegistry()
    stale = ModelCapabilities(context_length=131_072, authoritative=False)
    registry._overrides["vendor-next-gen"] = stale

    assert _active_length("vendor-next-gen", 1_000_000) == 1_000_000


def test_an_authoritative_entry_with_the_same_length_does_cap() -> None:
    """Same number, authoritative: now it must apply. Isolates the flag itself."""
    registry = ModelCapabilityRegistry()
    registry.register("pinned-model", ModelCapabilities(context_length=131_072))

    assert _active_length("pinned-model", 1_000_000, registry) == 131_072


def test_shipped_family_entries_carry_no_context_length_claim() -> None:
    """Family entries must not assert a window, only feature flags.

    Guards the table itself: adding a context length to a family row would
    re-create the original defect for every future model in that family.
    """
    from leapflow.llm.model_capabilities import _KNOWN_MODELS

    for pattern, caps in _KNOWN_MODELS:
        if not caps.authoritative:
            assert caps.context_length == ModelCapabilities().context_length, (
                f"family pattern {pattern!r} must not pin a context length"
            )


def test_missing_registry_falls_back_to_the_budget() -> None:
    engine = object.__new__(AgentEngine)
    engine._settings = SimpleNamespace(llm_model="anything", llm_context_length=512_000)
    engine._model_capabilities = None

    assert AgentEngine._active_context_length(engine) == 512_000


def test_a_broken_registry_does_not_break_budget_resolution() -> None:
    class _Broken:
        def resolve(self, model):
            raise RuntimeError("registry exploded")

    assert _active_length("qwen3.8-max", 1_000_000, _Broken()) == 1_000_000


# ── Truncation chain scaling ─────────────────────────────────────────────


def test_tool_result_budget_grows_with_a_large_window() -> None:
    """The old min() form pinned every large window to the base value."""
    assert adaptive_tool_result_chars(3000, 1_000_000) > 3000
    assert adaptive_tool_result_chars(3000, 1_000_000) == 25_000


def test_tool_result_budget_still_contracts_on_small_windows() -> None:
    """A 32K model must not spend a tenth of its window on one tool result."""
    assert adaptive_tool_result_chars(3000, 32_000) < 3000


def test_tool_result_budget_is_monotonic_and_bounded() -> None:
    windows = [16_000, 32_000, 128_000, 200_000, 1_000_000, 8_000_000]
    values = [adaptive_tool_result_chars(3000, w) for w in windows]

    assert values == sorted(values), "a bigger window must never yield a smaller budget"
    assert max(values) <= _RESULT_CEILING_CHARS
    assert min(values) >= _RESULT_FLOOR_CHARS


def test_tool_result_budget_holds_near_the_historical_value_at_128k() -> None:
    """Existing 128K setups should not regress; the divisor was picked for this."""
    assert 2_800 <= adaptive_tool_result_chars(3000, 128_000) <= 3_600


def test_tool_result_budget_handles_an_unknown_window() -> None:
    assert adaptive_tool_result_chars(3000, 0) == 3000
    assert adaptive_tool_result_chars(3000, -1) == 3000


# ── The scaling must actually be wired into the runtime ────────────────


def test_runtime_sync_applies_the_widened_budget_to_the_engine() -> None:
    """Covers the wiring, not just the formula.

    A correct helper is useless if the call site still clamps. This drives
    ``_sync_engine_runtime_budget`` and asserts what the engine actually receives,
    which is what a formula-only test cannot see.
    """
    from leapflow.cli.context import Context

    class _Engine:
        def __init__(self) -> None:
            self.result_budget = 0
            self.caps = None

        def set_tool_result_budget(self, value: int) -> None:
            self.result_budget = value

        def set_model_capabilities(self, registry) -> None:
            self.caps = registry

    ctx = object.__new__(Context)
    engine = _Engine()
    ctx.engine = engine
    ctx.llm_chain = None
    settings = SimpleNamespace(
        llm_model="qwen3.8-max",
        llm_context_length=1_000_000,
        max_tool_result_chars=3000,
        context_hard_limit_ratio=0.92,
        native_tool_calling_enabled=True,
    )

    Context._sync_engine_runtime_budget(ctx, settings)

    assert engine.result_budget == 25_000, "a 1M window must not stay pinned at 3000"
    assert engine.caps is not None, "capability registry must still be wired"


def test_runtime_sync_still_contracts_for_a_small_window() -> None:
    """The widening must not remove the small-window protection."""
    from leapflow.cli.context import Context

    class _Engine:
        def __init__(self) -> None:
            self.result_budget = 0

        def set_tool_result_budget(self, value: int) -> None:
            self.result_budget = value

        def set_model_capabilities(self, registry) -> None:
            pass

    ctx = object.__new__(Context)
    engine = _Engine()
    ctx.engine = engine
    ctx.llm_chain = None
    settings = SimpleNamespace(
        llm_model="tiny-model",
        llm_context_length=32_000,
        max_tool_result_chars=3000,
        context_hard_limit_ratio=0.92,
        native_tool_calling_enabled=True,
    )

    Context._sync_engine_runtime_budget(ctx, settings)

    assert engine.result_budget < 3000
